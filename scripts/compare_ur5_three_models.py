"""Paired MuJoCo-only benchmark. Never imports robot control interfaces."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ('real_010000', 'real_ur5_20260924_151943/checkpoints/010000',
     'pick up the cube and place it at the home position'),
    ('sim_023000', 'sim_ur5_lift_20261020_234246/checkpoints/023000',
     'Pick up the red block and hold it steadily above the table'),
    ('takeover16_000500', 'takeover_sanity_20261026/checkpoints/000500',
     'Pick up the red block and hold it steadily above the table'),
]

def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261027)
    a = p.parse_args()
    if a.episodes < 1: p.error('episodes must be positive')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(episodes=a.episodes, seed=a.seed, start='home',
                    timeout_seconds=20, success='block bottom >= 5cm; transient lifts count',
                    yaw_deg=0, action_stride=10, command_rate_rad_s=.9,
                    language='each checkpoint training instruction',
                    scope='simulation-only; real model score is real-to-sim transfer',
                    models=MODELS)
    mp = a.output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != json.loads(json.dumps(manifest)):
        raise RuntimeError('Output belongs to different benchmark')
    write(mp, manifest)
    environment = dict(os.environ, MUJOCO_GL='egl', HF_HUB_OFFLINE='1',
                       PYTHONWARNINGS='ignore::UserWarning', PYTHONUNBUFFERED='1')
    summaries = {}
    for label, relative, instruction in MODELS:
        dest = a.output/label
        summary_path = dest/'summary.json'
        previous = []
        run_dest = dest
        start_episode = 0
        if dest.exists():
            if summary_path.exists() and json.loads(summary_path.read_text())['completed'] == a.episodes:
                summaries[label] = json.loads(summary_path.read_text())
                continue
            if not summary_path.exists():
                raise RuntimeError(f'No completed summary at {dest}; inspect manually')
            previous = json.loads(summary_path.read_text())['episodes']
            start_episode = len(previous)
            assert [r['episode'] for r in previous] == list(range(start_episode))
            run_dest = a.output/(label + f'_resume{start_episode:03d}')
            if run_dest.exists():
                raise RuntimeError(f'Resume output exists: {run_dest}; inspect manually')
        checkpoint = ROOT/'policies/smolvla/checkpoints'/relative/'pretrained_model'
        if not (checkpoint/'model.safetensors').is_file(): raise FileNotFoundError(checkpoint)
        command = [sys.executable, str(ROOT/'policies/smolvla/eval_sim_ur5_lift.py'),
                   '--checkpoint', str(checkpoint), '--output', str(run_dest),
                   '--episode-start', str(start_episode),
                   '--episodes', str(a.episodes), '--seed', str(a.seed),
                   '--start-mode', 'home', '--block-yaw-deg', '0',
                   '--seconds', '20', '--action-steps', '10',
                   '--max-command-rate', '.9', '--success-mode', 'height',
                   '--task', instruction]
        write(a.output/'status.json', dict(state='running', model=label, started=time.time()))
        print('START', label, flush=True)
        with (a.output/(run_dest.name+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write(a.output/'status.json', dict(state='failed', model=label, code=result.returncode))
            raise RuntimeError(f'{label} failed; inspect log')
        current = json.loads((run_dest/'summary.json').read_text())
        if previous:
            rows = previous + current['episodes']
            for r in rows:
                r['artifact_dir'] = str((dest if r['episode'] < start_episode else run_dest)/f"{r['episode']:03d}")
            current = dict(completed=len(rows), successes=sum(r['success'] for r in rows),
                           success_rate=sum(r['success'] for r in rows)/len(rows), episodes=rows)
            write(dest/'summary_before_resume.json', json.loads(summary_path.read_text()))
            write(summary_path, current)
        summaries[label] = current
        if summaries[label]['completed'] != a.episodes: raise RuntimeError('Incomplete evaluation')
        print('DONE', label, summaries[label]['successes'], '/', a.episodes, flush=True)
    # Identical sampled scenes, actual starting joints, and rollout protocol.
    first = summaries[MODELS[0][0]]['episodes']
    common = ['seed','episodes','timeout_sim_seconds','fps','action_stride',
              'max_command_rate_rad_s','gravity_compensation','start_mode',
              'fixed_block_yaw_deg','success_mode']
    protocols = [json.loads((a.output/m[0]/'protocol.json').read_text()) for m in MODELS]
    assert all(all(p[k] == protocols[0][k] for k in common) for p in protocols)
    for label, summary in summaries.items():
        for original, current in zip(first, summary['episodes']):
            assert original['episode'] == current['episode']
            assert original['initialization'] == current['initialization'], label
            assert original['sampling'] == current['sampling'], label
    report = dict(paired_initial_states_verified=True, protocol=manifest,
                  results={key:dict(successes=s['successes'], episodes=s['completed'],
                           success_rate=s['success_rate'],
                           max_clearance_m=[r['max_clearance_m'] for r in s['episodes']],
                           failures=[dict(episode=r['episode'], reason=r['reason']) for r in s['episodes'] if not r['success']])
                           for key,s in summaries.items()})
    write(a.output/'comparison.json', report)
    lines = ['# Three-model simulation benchmark', '',
             'Same 20-layout protocol (or requested episode count), fixed yaw, common home, 5 cm bottom clearance.',
             'Transient lifts count; this does not establish stable grasp or real-world success.', '',
             '| Model | Success | Rate |', '|---|---:|---:|']
    for key,s in summaries.items():
        lines.append(f"| {key} | {s['successes']}/{s['completed']} | {s['success_rate']:.1%} |")
    (a.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    write(a.output/'status.json', dict(state='complete', paired_initial_states_verified=True))
    print(json.dumps(report,indent=2), flush=True)

if __name__ == '__main__': main()
