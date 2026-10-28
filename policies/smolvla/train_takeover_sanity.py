"""Training-state memorization diagnostic, NOT held-out evaluation. No robot access."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from train_real_ur5 import ROOT,move,dump,TrainPipelineConfig,LeRobotDataset,SmolVLAPolicy,set_seed,save_checkpoint,update_last_checkpoint
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.argv=sys.argv[:1]
    if a.steps<1:raise ValueError('steps must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'policies/smolvla/checkpoints/sim_ur5_padding_fix_from023000_20261021/checkpoints/024000'
    root=ROOT/'data/ur5_takeover_recovery_20261026_v21'
    cfg=TrainPipelineConfig.from_pretrained(source/'pretrained_model/train_config.json',cli_args=[])
    cfg.resume=False;cfg.output_dir=a.output.resolve();cfg.steps=a.steps;cfg.batch_size=4;cfg.num_workers=2
    cfg.dataset.root=str(root);cfg.dataset.repo_id='local/'+root.name
    cfg.policy.pretrained_path=str(source/'pretrained_model');cfg.policy.push_to_hub=False;cfg.policy.device='cuda';cfg.policy.use_amp=False
    cfg.optimizer.lr=1e-5
    cfg.scheduler=CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=min(25,a.steps),num_decay_steps=a.steps,peak_lr=1e-5,decay_lr=2.5e-6)
    set_seed(42);torch.backends.cuda.matmul.allow_tf32=True
    ds=LeRobotDataset('local/'+root.name,root=root,delta_timestamps={'action':[i/30 for i in range(50)]},video_backend='pyav')
    assert ds.meta.total_episodes==16 and ds.fps==30
    probe=[]
    for eid in range(16):
        start=int(ds.episode_data_index['from'][eid]);end=int(ds.episode_data_index['to'][eid])
        probe.extend(np.linspace(start,end-1,8,dtype=int).tolist())
        for i in (start,end-1):
            s=ds[i];assert torch.equal(s['action_is_pad'],torch.arange(50)+i>=end)
            assert torch.equal(s['action'],torch.stack([ds.hf_dataset[min(i+j,end-1)]['action'] for j in range(50)]))
    draws=torch.randint(0,len(ds),(a.steps*4,),generator=torch.Generator().manual_seed(42)).tolist()
    loader=DataLoader(Subset(ds,draws),batch_size=4,num_workers=2,pin_memory=True)
    probes=DataLoader(Subset(ds,probe),batch_size=4,num_workers=0)
    model=SmolVLAPolicy.from_pretrained(cfg.policy.pretrained_path,config=cfg.policy).to('cuda')
    norm={k:v.cpu().clone() for k,v in model.state_dict().items() if k.startswith(('normalize_inputs.','normalize_targets.','unnormalize_outputs.'))}
    params=[p for p in model.parameters() if p.requires_grad];before=params[0].detach().cpu().clone()
    opt=cfg.optimizer.build(params);sched=cfg.scheduler.build(opt,a.steps)
    manifest=dict(purpose='memorization_only_no_generalization_claim',all_16_episodes_used_for_training=True,
        converter_split_ignored_for_this_explicit_memorization_experiment=True,source=str(source),dataset=str(root),
        steps=a.steps,batch_size=4,lr=1e-5,normalization='unchanged source',sampling='uniform frame',
        loss='unchanged 32D objective; report valid 7D and padded 25D separately',probe_indices=probe)
    dump(a.output/'sanity_training.json',manifest);dump(a.output/'sampling_indices.json',draws)
    def evaluate(step):
        model.eval();total=valid7=padded=0.;n=den=0
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            torch.manual_seed(12345);torch.cuda.manual_seed_all(12345)
            for batch in probes:
                batch=move(batch);loss,details=model.forward(batch);assert torch.isfinite(loss)
                total+=float(loss)*len(batch['action']);n+=len(batch['action'])
                mask=(~batch['action_is_pad']).unsqueeze(-1);raw=details['losses_after_forward']
                valid7+=float((raw[:,:,:7]*mask).sum());padded+=float((raw[:,:,7:]*mask).sum());den+=int(mask.sum())
        return dict(step=step,training_probe_loss=total/n,active7_flow_mse=valid7/(den*7),padding25_flow_mse=padded/(den*25),samples=n)
    with (a.output/'metrics.jsonl').open('x') as log:
        initial=evaluate(0);print(initial,flush=True);log.write(json.dumps(initial)+'\n');log.flush()
        for step,batch in enumerate(loader,1):
            model.train();opt.zero_grad(set_to_none=True);loss,_=model.forward(move(batch));assert torch.isfinite(loss)
            loss.backward();g=torch.nn.utils.clip_grad_norm_(params,cfg.optimizer.grad_clip_norm,error_if_nonfinite=True);opt.step();sched.step()
            row=dict(step=step,loss=float(loss.detach()),grad=float(g));log.write(json.dumps(row)+'\n');log.flush()
            if step%50==0:print(row,flush=True)
            if step==a.steps:
                final=evaluate(step);print(final,flush=True);log.write(json.dumps(final)+'\n')
                dest=cfg.output_dir/'checkpoints'/f'{step:06d}';save_checkpoint(dest,step,cfg,model,opt,sched);update_last_checkpoint(dest)
                dump(dest/'sanity_training.json',manifest)
    change=float((params[0].detach().cpu()-before).abs().max());assert change>0
    model._load_as_safetensor(model,str(dest/'pretrained_model/model.safetensors'),'cuda',False)
    assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in norm.items())
    dump(a.output/'COMPLETE.json',dict(initial=initial,final=final,parameter_change=change,reload=True,normalization_unchanged=True,checkpoint=str(dest)))
    print('COMPLETE',dest,flush=True)

if __name__=='__main__':main()
