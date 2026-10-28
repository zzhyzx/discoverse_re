"""Offline simulation mixed-data fine-tuning. No robot access or Hub uploads.

Steps are ADDITIONAL updates from --init, with a fresh optimizer/schedule.
Original and recovery validation episodes are never sampled for training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from train_real_ur5 import (
    ROOT, KEYS, CAMERAS, move, dump, TrainPipelineConfig, LeRobotDataset,
    SmolVLAPolicy, set_seed, save_checkpoint, update_last_checkpoint)
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

SOURCE = ROOT/'policies/smolvla/checkpoints/sim_ur5_padding_fix_from023000_20261021/checkpoints/024000'
ORIGINAL = ROOT/'data/ur5_lift_sim_v21'
RECOVERY = ROOT/'data/ur5_recovery_demos_20261025_v21'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--recovery-fraction', type=float, default=.30)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    sys.argv = sys.argv[:1]
    if a.output.exists():
        raise FileExistsError(a.output)
    if not 0 < a.recovery_fraction < 1 or not np.isfinite(a.lr) or not 0 < a.lr <= 1e-4:
        raise ValueError('Invalid mixture or learning rate')
    steps = 2 if a.smoke else a.steps
    if steps < 1 or a.batch_size < 1:
        raise ValueError('Positive steps/batch size required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cfg = TrainPipelineConfig.from_pretrained(a.init/'pretrained_model/train_config.json', cli_args=[])
    cfg.resume = False
    cfg.output_dir = a.output.resolve()
    cfg.steps = steps
    cfg.batch_size = a.batch_size
    cfg.num_workers = 0 if a.smoke else 2
    cfg.save_freq = 250
    cfg.eval_freq = 250
    cfg.policy.pretrained_path = str((a.init/'pretrained_model').resolve())
    cfg.policy.push_to_hub = False
    cfg.policy.device = 'cuda'
    cfg.policy.use_amp = False
    cfg.optimizer.lr = a.lr
    cfg.scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=1 if a.smoke else min(50,steps),
        num_decay_steps=steps, peak_lr=a.lr, decay_lr=a.lr*.25)
    assert cfg.policy.chunk_size == 50 and cfg.policy.n_action_steps == 10
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    training = []
    validation = {}
    sources = {}
    for name, root in [('original',ORIGINAL),('recovery',RECOVERY)]:
        assert (root/'conversion/COMPLETE.json').is_file()
        split = json.loads((root/'conversion/episode_split.json').read_text())
        full = LeRobotDataset('local/'+root.name, root=root,
                              delta_timestamps={'action':[i/30 for i in range(50)]}, video_backend='pyav')
        assert full.fps == 30
        ids = split['train']+split['validation']
        assert len(ids) == len(set(ids)) and set(ids) == set(range(full.meta.total_episodes))
        assert split['train'] and split['validation']
        if name == 'original':
            source_manifest = json.loads((a.init/'real_training.json').read_text())
            assert source_manifest['train_episodes'] == split['train']
            assert source_manifest['validation_episodes'] == split['validation']
        indices = []
        selected_val = []
        for eid in ids:
            start = int(full.episode_data_index['from'][eid])
            end = int(full.episode_data_index['to'][eid])
            for index in (start,end-1):
                sample = full[index]
                expected = torch.stack([full.hf_dataset[min(index+j,end-1)]['action'] for j in range(50)])
                assert torch.equal(sample['action'],expected), (name,eid,'action boundary')
                expected_pad = torch.arange(50)+index >= end
                assert torch.equal(sample['action_is_pad'].cpu(),expected_pad), (name,eid,'padding')
            if eid in split['train']:
                indices.extend(range(start,end))
            else:
                selected_val.extend(np.linspace(start,end-1,min(40,end-start),dtype=int).tolist())
        if a.smoke:
            selected_val = selected_val[:a.batch_size]
        training.append(Subset(full,indices))
        validation[name] = DataLoader(Subset(full,selected_val), batch_size=a.batch_size, num_workers=0)
        sources[name] = dict(root=str(root), train_episodes=split['train'], validation_episodes=split['validation'],
                             train_frames=len(indices), validation_probe_frames=len(selected_val),
                             split_sha256=hashlib.sha256((root/'conversion/episode_split.json').read_bytes()).hexdigest())
        print(f'{name}: {len(indices)} training frames; episode boundaries/padding verified', flush=True)
    policy = SmolVLAPolicy.from_pretrained(cfg.policy.pretrained_path,config=cfg.policy).to('cuda')
    normalization = {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()
                     if k.startswith(('normalize_inputs.','normalize_targets.','unnormalize_outputs.'))}
    assert normalization and all(torch.isfinite(v).all() for v in normalization.values())
    params = [v for v in policy.parameters() if v.requires_grad]
    optimizer = cfg.optimizer.build(params)
    scheduler = cfg.scheduler.build(optimizer,steps)
    # Keep pretrained normalization: changing it would change physical meanings
    # of learned inputs/outputs as well as the training data in this experiment.
    combined = ConcatDataset(training)
    weights = torch.cat([torch.full((len(training[0]),),(1-a.recovery_fraction)/len(training[0]),dtype=torch.double),
                         torch.full((len(training[1]),),a.recovery_fraction/len(training[1]),dtype=torch.double)])
    draws = torch.multinomial(weights,steps*a.batch_size,replacement=True,
                              generator=torch.Generator().manual_seed(42)).tolist()
    recovery_count = sum(i>=len(training[0]) for i in draws)
    loader = DataLoader(Subset(combined,draws),batch_size=a.batch_size,shuffle=False,
                        num_workers=cfg.num_workers,pin_memory=True)
    cfg.output_dir.mkdir(parents=True,exist_ok=False)
    manifest = dict(init=str(a.init.resolve()), additional_steps=steps, batch_size=a.batch_size,
                    sources=sources, recovery_fraction=a.recovery_fraction,
                    actual_recovery_samples=recovery_count, total_draws=len(draws), seed=42,
                    optimizer='fresh AdamW', lr_peak=a.lr, warmup=cfg.scheduler.num_warmup_steps,
                    normalization='unchanged source checkpoint; original training subset only',
                    cameras=CAMERAS, action='absolute six joint radians + gripper',
                    loss_revision='action_is_pad_valid_timestep_mean_v1',
                    validation='separate datasets, <=40 evenly spaced frames/episode, fixed flow-loss RNG',
                    cfg_dataset_note='train_config.dataset describes original source only; use this manifest for mixture',
                    resume_note='This entry supports fresh fine-tunes, not optimizer resume; checkpoints include training state')
    dump(cfg.output_dir/'mixed_training.json',manifest)
    dump(cfg.output_dir/'sampling_indices.json',draws)
    probe = params[0]
    before = probe.detach().cpu().clone()

    def evaluate(step, stream):
        policy.eval()
        result = dict(step=step)
        for name, val_loader in validation.items():
            total = 0.
            count = 0
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
                torch.manual_seed(12345)
                torch.cuda.manual_seed_all(12345)
                for batch in val_loader:
                    batch = move(batch)
                    loss,_ = policy.forward(batch)
                    assert torch.isfinite(loss)
                    n = len(batch['action'])
                    total += float(loss)*n
                    count += n
            result[name+'_validation_loss'] = total/count
            result[name+'_validation_samples'] = count
        print(json.dumps(result),flush=True)
        stream.write(json.dumps(result)+'\n');stream.flush()
        return result

    print(json.dumps(manifest,indent=2),flush=True)
    with (cfg.output_dir/'metrics.jsonl').open('x') as stream:
        baseline = evaluate(0,stream)
        for step,batch in enumerate(loader,1):
            started = time.monotonic()
            batch = move(batch)
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss,_ = policy.forward(batch)
            assert torch.isfinite(loss)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params,cfg.optimizer.grad_clip_norm,error_if_nonfinite=True)
            used_lr = optimizer.param_groups[0]['lr']
            optimizer.step();scheduler.step()
            row = dict(step=step,loss=float(loss.detach()),grad_norm=float(grad),lr=used_lr,
                       seconds=time.monotonic()-started)
            if a.smoke or step%50 == 0:
                print(json.dumps(row),flush=True)
            stream.write(json.dumps(row)+'\n');stream.flush()
            if step%250 == 0 or step == steps:
                latest = evaluate(step,stream)
                dest = cfg.output_dir/'checkpoints'/f'{step:06d}'
                assert not dest.exists()
                save_checkpoint(dest,step,cfg,policy,optimizer,scheduler)
                dump(dest/'mixed_training.json',manifest)
                update_last_checkpoint(dest)
                print('Saved:',dest,flush=True)
    change = float((probe.detach().cpu()-before).abs().max())
    assert change > 0
    # Verify serialized weights reload and all normalizers remain unchanged.
    policy._load_as_safetensor(policy,str(dest/'pretrained_model/model.safetensors'),'cuda',False)
    state = policy.state_dict()
    assert all(torch.equal(state[k].cpu(),v) for k,v in normalization.items())
    dump(cfg.output_dir/('smoke_pass.json' if a.smoke else 'COMPLETE.json'),
         dict(steps=steps,parameter_max_change=change,checkpoint_reload=True,
              normalization_unchanged=True,baseline=baseline,final=latest,checkpoint=str(dest)))
    print('SMOKE PASSED' if a.smoke else 'TRAINING COMPLETE',flush=True)


if __name__ == '__main__':
    main()
