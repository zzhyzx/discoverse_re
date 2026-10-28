"""Real-only offline SmolVLA. No robot/Hub writes. --smoke tests two updates.
Resume restores optimizer/scheduler/RNG, but restarts the shuffled iterator.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'submodules/lerobot/src'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import save_checkpoint,load_training_state,update_last_checkpoint

DATA=ROOT/'data/real_ur5_pick_cube_v21'
BASE=ROOT/'policies/smolvla/pretrained/smolvla_base'
CAMERAS=['observation.images.d436','observation.images.d405']
KEYS=['observation.state',*CAMERAS,'action']

def dump(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)

def move(batch):
    return {k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()
            if k in KEYS+['action_is_pad','task']}

def main(simulation=False):
    DATA = ROOT/'data/ur5_lift_sim_v21' if simulation else globals()['DATA']
    p=argparse.ArgumentParser(description=('Simulation-only offline SmolVLA; no robot/Hub writes. --smoke tests two updates.' if simulation else __doc__))
    p.add_argument('--data',type=Path,help='Explicit verified dataset; required for low-speed baseline')
    p.add_argument('--control-profile',type=Path,help='Require matching converted-data control contract')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--steps',type=int,help='Total optimizer updates, NOT additional steps')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--output',type=Path)
    p.add_argument('--resume',type=Path,help='Trusted checkpoint directory, not pretrained_model')
    p.add_argument('--fork-output',type=Path,help='Resume into a NEW run directory, preserving original run')
    p.add_argument('--backbone',type=Path,help='Optional complete local SmolVLM2 directory')
    args=p.parse_args()
    if args.data: DATA=args.data.resolve()
    if args.control_profile:
        sys.path.insert(0,str(ROOT/'scripts'))
        from ur5_control_profile import load_profile
        expected_contract=load_profile(args.control_profile)
        saved=DATA/'conversion/control_contract.json'
        if not saved.exists() or json.loads(saved.read_text())!=expected_contract:
            raise ValueError('Dataset does not match requested low-speed control contract')
    if args.fork_output and not args.resume:
        p.error('--fork-output requires --resume')
    if args.fork_output and args.fork_output.exists():
        raise FileExistsError(args.fork_output)
    sys.argv=sys.argv[:1]  # old config loader also examines CLI
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if not (DATA/'conversion/COMPLETE.json').is_file(): raise RuntimeError('Verified dataset missing')
    split=json.loads((DATA/'conversion/episode_split.json').read_text())
    train_ids,val_ids=split['train'],split['validation']
    total = json.loads((DATA/'meta/info.json').read_text())['total_episodes']
    if (not train_ids or not val_ids or len(set(train_ids+val_ids)) != len(train_ids+val_ids)
            or set(train_ids+val_ids)!=set(range(total))):
        raise ValueError('Invalid disjoint episode split')
    if not simulation and (len(train_ids)!=40 or len(val_ids)!=10):
        raise ValueError('Expected disjoint 40/10 real split')
    checkpoint=args.resume.resolve() if args.resume else None
    if checkpoint:
        if args.smoke or args.output or args.batch_size or args.backbone:
            p.error('Resume uses saved settings; only --steps may override total updates')
        cfg=TrainPipelineConfig.from_pretrained(checkpoint/'pretrained_model/train_config.json',cli_args=[])
        old=json.loads((checkpoint/'real_training.json').read_text())
        if Path(old['data']).resolve()!=DATA.resolve(): raise ValueError('Checkpoint belongs to a different dataset')
        if old['train_episodes']!=train_ids or old['validation_episodes']!=val_ids: raise ValueError('Split changed')
        if args.steps is not None: cfg.steps=args.steps
        cfg.policy.pretrained_path=str(checkpoint/'pretrained_model')
        cfg.resume=True;cfg.output_dir=Path(cfg.output_dir)
        if args.fork_output:
            cfg.output_dir=args.fork_output.resolve()
    else:
        policy=SmolVLAConfig(device='cuda',push_to_hub=False)
        policy.pretrained_path=str(BASE)
        policy.freeze_vision_encoder=True;policy.train_expert_only=True;policy.train_state_proj=True
        policy.n_action_steps=10;policy.chunk_size=50;policy.use_amp=False
        if args.backbone:
            if not (args.backbone/'config.json').is_file(): raise ValueError('Incomplete backbone directory')
            policy.vlm_model_name=str(args.backbone.resolve())
        output=args.output or ROOT/'policies/smolvla/checkpoints'/(
            (('sim_ur5_smoke_' if args.smoke else 'sim_ur5_lift_') if simulation else
             ('real_ur5_smoke_' if args.smoke else 'real_ur5_'))+datetime.now().strftime('%Y%m%d_%H%M%S'))
        cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='local/'+DATA.name,root=str(DATA),
            episodes=train_ids,use_imagenet_stats=False,video_backend='pyav'),policy=policy,
            output_dir=output.resolve(),steps=args.steps if args.steps is not None else (2 if args.smoke else 10000),
            batch_size=args.batch_size if args.batch_size is not None else (1 if args.smoke else 4),
            num_workers=0 if args.smoke else 2,seed=42,save_freq=1000,log_freq=50,eval_freq=1000)
        cfg.validate()
    if cfg.steps<1 or cfg.batch_size<1: raise ValueError('Positive steps/batch size required')
    set_seed(cfg.seed);torch.backends.cuda.matmul.allow_tf32=True
    # This checkout incorrectly uses original episode IDs against subset offsets.
    # Keep full episode indexing intact; isolate train/validation with Subset.
    full=LeRobotDataset(cfg.dataset.repo_id,root=DATA,
        delta_timestamps={'action':[i/30 for i in range(cfg.policy.chunk_size)]},video_backend='pyav')
    def indices(ids):
        return [i for eid in ids for i in range(int(full.episode_data_index['from'][eid]),
                                               int(full.episode_data_index['to'][eid]))]
    training,validation=Subset(full,indices(train_ids)),Subset(full,indices(val_ids))
    if full.fps!=30: raise ValueError('Expected 30Hz')
    print(f'Checking action-chunk boundaries for {len(train_ids)+len(val_ids)} episodes...', flush=True)
    # Verify action chunks at both ends of every episode, not merely shapes.
    for eid in train_ids+val_ids:
        start=int(full.episode_data_index['from'][eid]);end=int(full.episode_data_index['to'][eid])
        for index in (start,end-1):
            sample=full[index]
            expected=torch.stack([full.hf_dataset[min(index+j,end-1)]['action'] for j in range(cfg.policy.chunk_size)])
            if not torch.equal(sample['action'],expected): raise RuntimeError(f'Action boundary mismatch: {eid}')
    print('Action-chunk boundaries verified.', flush=True)
    stats=aggregate_stats([full.meta.episodes_stats[i] for i in train_ids])
    features=dataset_to_policy_features({k:full.meta.features[k] for k in KEYS})
    cfg.policy.input_features={k:features[k] for k in KEYS if k!='action'}
    cfg.policy.output_features={'action':features['action']}
    if tuple(features['action'].shape)!=(7,) or tuple(features['observation.state'].shape)!=(7,):
        raise ValueError('Expected 7D state/action')
    policy=SmolVLAPolicy.from_pretrained(cfg.policy.pretrained_path,config=cfg.policy,dataset_stats=stats).to('cuda')
    if simulation:
        # Base checkpoint loading may overwrite constructor statistics. Explicitly
        # bind normalization to TRAIN episodes, and verify saved stats on resume.
        from lerobot.policies.normalize import Normalize, Unnormalize
        modules = {
            'normalize_inputs': Normalize(cfg.policy.input_features, cfg.policy.normalization_mapping, stats),
            'normalize_targets': Normalize(cfg.policy.output_features, cfg.policy.normalization_mapping, stats),
            'unnormalize_outputs': Unnormalize(cfg.policy.output_features, cfg.policy.normalization_mapping, stats),
        }
        for name, module in modules.items():
            if any(not torch.isfinite(v).all() for v in module.state_dict().values()):
                raise ValueError('Nonfinite training statistics: '+name)
            if checkpoint:
                saved = getattr(policy, name).state_dict()
                if any(k not in saved or not torch.equal(saved[k].cpu(), v.cpu()) for k,v in module.state_dict().items()):
                    raise ValueError('Resume normalization mismatch: '+name)
            else:
                setattr(policy, name, module.to('cuda'))
    for name,buf in policy.named_buffers():
        if name.startswith(('normalize','unnormalize')) and not torch.isfinite(buf).all():
            raise ValueError(f'Invalid normalization {name}')
    optimizer,scheduler=make_optimizer_and_scheduler(cfg,policy)
    step=0
    if checkpoint: step,optimizer,scheduler=load_training_state(checkpoint,optimizer,scheduler)
    if cfg.steps<=step: raise ValueError(f'Total steps must exceed saved step {step}')
    cfg.output_dir.mkdir(parents=True,exist_ok=bool(checkpoint) and not bool(args.fork_output))
    manifest=dict(train_episodes=train_ids,validation_episodes=val_ids,data=str(DATA),
        images='MuJoCo RGB; front d436/wrist d405' if simulation else 'raw real RGB; no rectification/simulation',camera_order=CAMERAS,
        action='absolute six joint radians + binary gripper',normalization='training episodes ONLY; reused for validation',
        validation='up to 40 evenly spaced frames per held-out episode, fixed flow-loss RNG seed',
        resume='optimizer/scheduler/RNG restored; shuffled iterator restarts',
        n_action_steps=cfg.policy.n_action_steps,chunk_size=cfg.policy.chunk_size,
        train_frames=len(training),validation_frames=len(validation))
    manifest['loss_revision']='action_is_pad_valid_timestep_mean_v1'
    if checkpoint:
        manifest['resumed_from']=str(checkpoint)
    if not checkpoint or args.fork_output:
        dump(cfg.output_dir/'real_training.json',manifest)
        if args.control_profile: dump(cfg.output_dir/'control_contract.json',expected_contract)
        dump(cfg.output_dir/'train_stats.json',{k:{s:np.asarray(v).tolist() for s,v in d.items()} for k,d in stats.items() if k in KEYS})
    log=cfg.output_dir/('metrics_resume_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.jsonl' if checkpoint else 'metrics.jsonl')
    loader=DataLoader(training,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,pin_memory=True)
    chosen=[];offset=0
    for eid in val_ids:
        n=full.meta.episodes[eid]['length']
        chosen.extend((offset+np.linspace(0,n-1,min(n,40),dtype=int)).tolist());offset+=n
    val_loader=DataLoader(Subset(validation,chosen),batch_size=cfg.batch_size,shuffle=False,num_workers=0)
    print(json.dumps(manifest,indent=2),flush=True)
    print(f'Output: {cfg.output_dir}\nUpdates: {step}->{cfg.steps}; batch={cfg.batch_size}',flush=True)
    iterator=iter(loader);params=[v for v in policy.parameters() if v.requires_grad]
    probe=params[0];before=probe.detach().cpu().clone() if args.smoke else None
    with log.open('x') as stream:
        while step<cfg.steps:
            started=time.monotonic()
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader);batch=next(iterator)
            batch=move(batch);policy.train();optimizer.zero_grad(set_to_none=True)
            loss,_=policy.forward(batch)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,cfg.optimizer.grad_clip_norm,error_if_nonfinite=True)
            optimizer.step()
            if scheduler: scheduler.step()
            step+=1
            row=dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),lr=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-started)
            if args.smoke or step%cfg.log_freq==0: print(json.dumps(row),flush=True)
            stream.write(json.dumps(row)+'\n');stream.flush()
            if step%cfg.eval_freq==0 or step==cfg.steps:
                policy.eval();total=0.;count=0
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
                    torch.manual_seed(12345);torch.cuda.manual_seed_all(12345)
                    for vb in val_loader:
                        vb=move(vb);vl,_=policy.forward(vb)
                        if not torch.isfinite(vl): raise RuntimeError('Nonfinite validation loss')
                        n=len(vb['action']);total+=float(vl)*n;count+=n
                        if args.smoke: break
                result=dict(step=step,validation_loss=total/count,validation_samples=count)
                print(json.dumps(result),flush=True);stream.write(json.dumps(result)+'\n');stream.flush()
            if step%cfg.save_freq==0 or step==cfg.steps:
                dest=cfg.output_dir/'checkpoints'/f'{step:06d}'
                if dest.exists(): raise FileExistsError(dest)
                save_checkpoint(dest,step,cfg,policy,optimizer,scheduler)
                dump(dest/'real_training.json',manifest);update_last_checkpoint(dest)
                print('Saved:',dest,flush=True)
    if args.smoke:
        delta=float((probe.detach().cpu()-before).abs().max())
        if delta==0: raise RuntimeError('Smoke: no measured parameter update')
        policy._load_as_safetensor(policy,str(dest/'pretrained_model/model.safetensors'),'cuda',False)
        dump(cfg.output_dir/'smoke_pass.json',dict(parameter_max_change=delta,checkpoint_reload=True,steps=step))
        print('SMOKE PASSED: forward/backward/update/validation/save/reload',flush=True)

if __name__=='__main__':
    main()
