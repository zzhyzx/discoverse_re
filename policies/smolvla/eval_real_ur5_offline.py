"""Logged-observation offline evaluation. No robot interfaces or commands.
Predict 50 actions every saved n_action_steps; score only the executed prefix.
This is teacher-forced open-loop evaluation, NOT closed-loop grasp success.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'submodules/lerobot/src'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import default_collate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CAMERAS = ['observation.images.d436', 'observation.images.d405']

def save_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)

def metrics(pred, target):
    error = np.degrees(pred[:, :6] - target[:, :6])
    closed, truth = pred[:, 6] >= .5, target[:, 6] >= .5
    tp = int(np.sum(closed & truth))
    fp = int(np.sum(closed & ~truth))
    fn = int(np.sum(~closed & truth))
    return dict(frames=len(pred), joint_mae_deg=float(np.abs(error).mean()),
        joint_rmse_deg=float(np.sqrt(np.mean(error**2))),
        joint_mae_per_axis_deg=np.abs(error).mean(axis=0).tolist(),
        joint_p95_per_axis_deg=np.percentile(np.abs(error), 95, axis=0).tolist(),
        joint_max_per_axis_deg=np.abs(error).max(axis=0).tolist(),
        gripper_accuracy=float(np.mean(closed == truth)),
        gripper_closed_fraction=float(truth.mean()),
        gripper_always_closed_accuracy=float(truth.mean()),
        gripper_f1=2*tp/max(2*tp+fp+fn, 1), gripper_tp=tp, gripper_fp=fp, gripper_fn=fn,
        gripper_raw_mae=float(np.abs(pred[:, 6]-target[:, 6]).mean()),
        gripper_raw_range=[float(pred[:, 6].min()), float(pred[:, 6].max())])

def transitions(values, timestamps):
    binary = values >= .5
    return dict(initial_closed=bool(binary[0]),
        close_seconds=timestamps[np.where(np.diff(binary.astype(int)) == 1)[0]+1].tolist(),
        open_seconds=timestamps[np.where(np.diff(binary.astype(int)) == -1)[0]+1].tolist())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'policies/smolvla/checkpoints/real_ur5_20260924_151943/checkpoints/010000/pretrained_model')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    if args.batch_size < 1:
        raise ValueError('Positive batch size required')
    checkpoint = args.checkpoint.resolve()
    run = checkpoint.parents[2]
    manifest = json.loads((run / 'real_training.json').read_text())
    ids = manifest['validation_episodes']
    if not ids or len(set(ids)) != len(ids) or set(ids) & set(manifest['train_episodes']):
        raise ValueError('Expected nonempty disjoint held-out episodes')
    stats = {key: {name: np.asarray(value) for name, value in entry.items()}
             for key, entry in json.loads((run / 'train_stats.json').read_text()).items()}
    torch.manual_seed(12345)
    np.random.seed(12345)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = 'cuda'
    config.push_to_hub = False
    stride = config.n_action_steps
    if config.n_obs_steps != 1 or not 1 <= stride <= config.chunk_size:
        raise ValueError('Evaluator requires one observation and valid action stride')
    dataset = LeRobotDataset('local/'+Path(manifest['data']).name, root=Path(manifest['data']), video_backend='pyav')
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, dataset_stats=stats).to('cuda').eval()
    for name, value in policy.named_buffers():
        if name.startswith(('normalize', 'unnormalize')) and not torch.isfinite(value).all():
            raise ValueError('Invalid normalization: ' + name)
    output = args.output or ROOT / 'data/smolvla_offline_predictions' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    print('OUTPUT:', output, flush=True)
    save_json(output / 'protocol.json', dict(checkpoint=str(checkpoint), validation_episodes=ids,
        seed=12345, action_stride=stride, chunk_size=config.chunk_size, fps=dataset.fps,
        images=manifest['images'],
        normalization='Saved training-only train_stats.json',
        interpretation='Logged actual observations at each chunk boundary, no simulated/real feedback from predictions; not success rate',
        gripper='Raw model output; threshold 0.5 diagnostic only, not a probability; no clipping',
        joints='Absolute joint radians, no angle wrapping', batch_size=args.batch_size))
    results, all_pred, all_target = [], [], []
    started = time.monotonic()
    for eid in ids:
        start = int(dataset.episode_data_index['from'][eid])
        end = int(dataset.episode_data_index['to'][eid])
        rows = [dataset.hf_dataset[i] for i in range(start, end)]
        target = torch.stack([r['action'] for r in rows]).numpy()
        state = torch.stack([r['observation.state'] for r in rows]).numpy()
        timestamps = np.array([float(r['timestamp']) for r in rows])
        timestamps -= timestamps[0]
        queries = list(range(0, end-start, stride))
        chunks = []
        for offset in range(0, len(queries), args.batch_size):
            samples = [dataset[start+q] for q in queries[offset:offset+args.batch_size]]
            batch = default_collate([{k: sample[k] for k in ['observation.state', *CAMERAS, 'task']} for sample in samples])
            batch = {k: value.to('cuda') if isinstance(value, torch.Tensor) else value for k, value in batch.items()}
            policy.reset()  # batch entries are independent logged observations
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(batch).cpu().numpy()
            if not np.isfinite(chunk).all():
                raise RuntimeError(f'Nonfinite predictions in episode {eid}')
            chunks.extend(chunk)
        chunks = np.stack(chunks)
        pred = np.concatenate([chunk[:min(stride, len(target)-q)] for q, chunk in zip(queries, chunks)])
        result = dict(episode=eid, **metrics(pred, target))
        result['target_transitions'] = transitions(target[:, 6], timestamps)
        result['predicted_transitions'] = transitions(pred[:, 6], timestamps)
        result['max_predicted_step_deg'] = np.max(np.abs(np.degrees(np.diff(pred[:, :6], axis=0))), axis=0).tolist()
        result['max_recorded_step_deg'] = np.max(np.abs(np.degrees(np.diff(target[:, :6], axis=0))), axis=0).tolist()
        result['max_target_from_actual_deg'] = np.max(np.abs(np.degrees(pred[:, :6]-state[:, :6])), axis=0).tolist()
        np.savez_compressed(output / f'ep{eid:03d}.npz', timestamps=timestamps, actual_state=state,
            recorded_action=target, predicted_action=pred, query_frame_indices=np.array(queries), predicted_chunks=chunks)
        fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
        for axis in range(7):
            ax = axes.flat[axis]
            scale = 180/np.pi if axis < 6 else 1
            ax.plot(timestamps, target[:, axis]*scale, label='Recorded target', linewidth=1.5)
            ax.plot(timestamps, pred[:, axis]*scale, label='Predicted target', linewidth=1, alpha=.8)
            ax.plot(timestamps, state[:, axis]*scale, label='Actual feedback', linewidth=.8, alpha=.65)
            ax.set_ylabel(f'Joint {axis+1} (deg)' if axis < 6 else 'Gripper (raw)')
            ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=8)
        axes.flat[7].plot(timestamps, np.abs(np.degrees(pred[:, :6]-target[:, :6])).mean(axis=1))
        axes.flat[7].set_ylabel('Mean absolute joint error (deg)')
        for ax in axes[-1]: ax.set_xlabel('Episode time (s)')
        fig.suptitle(f'Episode {eid}: logged-observation prediction, stride={stride}; NOT closed-loop execution')
        fig.tight_layout()
        fig.savefig(output / f'ep{eid:03d}.png', dpi=130)
        plt.close(fig)
        save_json(output / f'ep{eid:03d}.json', result)
        results.append(result); all_pred.append(pred); all_target.append(target)
        print(f'ep={eid} frames={len(pred)} joint_MAE_deg={result["joint_mae_deg"]:.3f} gripper_acc={result["gripper_accuracy"]:.3f} elapsed={time.monotonic()-started:.1f}s', flush=True)
    summary = dict(overall=metrics(np.concatenate(all_pred), np.concatenate(all_target)), episodes=results,
                   macro_joint_mae_deg=float(np.mean([r['joint_mae_deg'] for r in results])),
                   seconds=time.monotonic()-started, checkpoint=str(checkpoint))
    save_json(output / 'summary.json', summary)
    print(json.dumps(summary['overall'], indent=2), flush=True)
    print('COMPLETE:', output, flush=True)

if __name__ == '__main__':
    main()
