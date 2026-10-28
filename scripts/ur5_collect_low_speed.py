"""Simulation only: save EVERY attempt, including failures and rejected proposals.
Rejected proposals have one initial frame (no motion), plus a labelled candidate FK snapshot.
All outputs are diagnostic_only and must not be blindly converted to training data.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import av
import mujoco
import numpy as np
from ur5_deploy_core import Limiter
from ur5_control_profile import load_profile, SimCommandAdapter

from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import (
    SimNode, Expert, EpisodeFailure, make_config, new_buffers, record,
    save_episode, PyavImageEncoder, step_func,
)


def contacts(model, data):
    result = []
    for c in data.contact[:data.ncon]:
        if c.dist >= 0:
            continue
        def name(g):
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) + '/unnamed_geom')
        result.append({'geom1': name(c.geom1), 'geom2': name(c.geom2), 'distance_m': float(c.dist)})
    return result


def snapshot(env, expert):
    actual = env.mj_data.site_xpos[env.ee_site_id].copy()
    target = expert.target_tcp
    return {'time_s': float(env.mj_data.time), 'state': expert.state,
            'stage': 'hold' if expert.finished else expert.NAMES[expert.state],
            'approach_waypoint': getattr(expert, 'approach_index', None),
            'q_actual_rad': env.mj_data.qpos[:6].tolist(),
            'q_target_rad': env.target_control[:6].tolist(),
            'max_joint_error_rad': float(np.max(np.abs(env.mj_data.qpos[:6] - env.target_control[:6]))),
            'tcp_actual_m': actual.tolist(), 'tcp_target_m': None if target is None else target.tolist(),
            'tcp_error_m': None if target is None else float(np.linalg.norm(actual - target)),
            'block_pos_m': env.block_pos().tolist(), 'lift': env.success_info(),
            'contacts': contacts(env.mj_model, env.mj_data)}


def save_images(env, folder, prefix, data=None):
    # These are explicitly static images, never claimed as executed trajectories.
    with mujoco.Renderer(env.mj_model, height=480, width=640) as renderer:
        for cid in (0, 1):
            renderer.update_scene(env.mj_data if data is None else data, camera=cid)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            if not cv2.imwrite(str(folder / f'{prefix}_cam_{cid}.png'), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)):
                raise RuntimeError('Could not save diagnostic image')


def verify_saved(root):
    checked = []
    for item in sorted(Path(root).glob('*/diagnostic.json')):
        info = json.loads(item.read_text())
        payload = json.loads((item.parent / 'obs_action.json').read_text())
        times = np.asarray(payload['time'])
        if not len(times):
            raise ValueError(f'No frames: {item.parent}')
        assert payload['metadata']['diagnostic_only'] == (info['outcome'] != 'success')
        assert payload['metadata']['training_eligible'] == (info['outcome'] == 'success')
        assert payload['metadata']['success'] == (info['outcome'] == 'success')
        assert np.asarray(payload['act']).shape == (len(times), 7)
        if len(times) > 1:
            assert np.allclose(np.diff(times), 1 / 30, atol=1e-8)
        for cid in (0, 1):
            with av.open(str(item.parent / f'cam_{cid}.mp4')) as container:
                stamps = [float(f.pts * f.time_base) for f in container.decode(video=0)]
            assert len(stamps) == len(times), (item.parent, cid, len(stamps), len(times))
            assert np.allclose(stamps, times - times[0], atol=2e-5)
        checked.append({'folder': item.parent.name, 'outcome': info['outcome'], 'frames_each_camera': len(times)})
    (Path(root) / 'file_validation.json').write_text(json.dumps(checked, indent=2))
    print('Verified saved attempts:', len(checked), flush=True)
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempts', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261020)
    p.add_argument('--arm-gravity-compensation', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--motion-profile', type=Path, help='Simulation-only motion candidate JSON')
    p.add_argument('--profile-mode', choices=('candidate', 'baseline'), default='candidate')
    args = p.parse_args()
    profile = json.loads(args.motion_profile.read_text()) if args.motion_profile else None
    contract=load_profile()
    if profile: p.error("Low-speed baseline uses its fixed control contract")
    if profile and profile.get('status') != 'simulation_only_not_real_robot_approved':
        p.error('Expected simulation-only candidate profile')
    if args.attempts <= 0:
        p.error('attempts must be positive')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg = make_config(True)
    cfg.initial_camera_binding = True
    cfg.align_base_table_centerline = True
    cfg.render_shadows = False
    cfg.lift_only = True
    cfg.use_real_training_start = True
    cfg.success_height = .05
    cfg.hold_seconds = 1.
    cfg.sampling = 'tabletop'
    cfg.edge_margin = .12
    cfg.far_edge_inset = .14
    cfg.side_margin = .27
    cfg.arm_gravity_compensation = args.arm_gravity_compensation
    cfg.expert_state_timeout_s = 90.
    env = SimNode(cfg, np.random.default_rng(args.seed), randomize=True)
    expert = Expert(env)
    results = []
    for attempt in range(args.attempts):
        folder = root / f'{attempt:03d}'
        folder.mkdir()
        obs = env.reset()
        buffers = new_buffers()
        trace = []
        outcome = 'error'
        failure = None
        interrupted = False
        preflight = None
        encoders = [PyavImageEncoder(640, 480, str(folder), i, fps=30, crf=18, preset='veryfast') for i in (0, 1)]
        try:
            expert.reset()
            save_images(env, folder, 'initial')
            action = env.target_control.copy()
            adapter=SimCommandAdapter(env,contract)
            limiter = None
            if profile:
                speeds = profile['candidate_speed_rad_s'] if args.profile_mode == 'candidate' else profile['baseline']['speed_rad_s']
                accelerations = profile['candidate_acceleration_rad_s2'] if args.profile_mode == 'candidate' else profile['baseline']['acceleration_rad_s2']
                limiter = Limiter(env.mj_data.qpos[:6], speeds, accelerations)
                action[:6] = limiter.q
            try:
                expert.preflight()
            except EpisodeFailure as exc:
                outcome = 'preflight_rejected'
                failure = str(exc)
                record(env, obs, action, buffers, encoders)
                preflight = {'q_candidate_rad': expert.ik_data.qpos[:6].tolist(),
                             'contacts': contacts(env.mj_model, expert.ik_data),
                             'note': 'Static candidate from rejection; NEVER executed'}
                save_images(env, folder, 'rejected_candidate_NOT_EXECUTED', expert.ik_data)
            if outcome != 'preflight_rejected':
                outcome = 'execution_failed'
                settle_start = None
                while env.running:
                    if not adapter.busy and not (expert.state==4 and not adapter.closed):
                        expert.update()
                    desired=np.r_[env.target_control[:6],env.target_control[6]/255.]
                    action,_=adapter.step(desired,float(env.mj_data.time),env.delta_t)
                    trace.append(snapshot(env, expert))
                    record(env, obs, action, buffers, encoders)
                    obs, *_ = env.step(action)
                    if expert.table_collision(env.mj_data):
                        raise EpisodeFailure('dynamic arm/gripper contact with table/floor/pedestal')
                    if env.mj_data.time > contract['episode_timeout_s']:
                        raise EpisodeFailure('episode timed out')
                    if expert.finished:
                        if env.check_success():
                            if settle_start is None:
                                settle_start = float(env.mj_data.time)
                            elif env.mj_data.time - settle_start >= cfg.hold_seconds + env.delta_t:
                                outcome = 'success'
                                break
                        else:
                            settle_start = None
                            if env.mj_data.time - expert.state_start > cfg.hold_seconds + 5:
                                raise EpisodeFailure(f'lift/hold verification failed: {env.success_info()}')
                if outcome != 'success' and failure is None:
                    failure = 'simulation stopped'
        except KeyboardInterrupt:
            outcome, failure, interrupted = 'interrupted', 'user interrupt', True
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
        finally:
            for encoder in encoders:
                encoder.close()
        save_episode(str(folder), buffers, args.seed, attempt + 1, cfg,
                     success=outcome == 'success', failure=failure, diagnostic=outcome != 'success')
        payload=json.loads((folder/'obs_action.json').read_text())
        payload['metadata']['control_contract']=contract
        payload['metadata']['max_command_rate_rad_s']=contract['max_command_speed_rad_s']
        payload['metadata']['max_command_acceleration_rad_s2']=contract['max_command_acceleration_rad_s2']
        (folder/'obs_action.json').write_text(json.dumps(payload))
        save_images(env, folder, 'final')
        result = {'attempt': attempt + 1, 'folder': folder.name, 'outcome': outcome,
                  'reason': failure, 'frames_per_camera': len(buffers['time']),
                  'sampling': cfg.sampling_metadata, 'final': snapshot(env, expert),
                  'preflight': preflight,
                  'arm_gravity_compensation': cfg.arm_gravity_compensation,
                  'control_contract': contract, 'motion_profile': profile, 'profile_mode': args.profile_mode if profile else None,
                  'control_parity': 'shared joint limiter only; 30Hz expert, NOT full 125Hz deployment/gripper parity',
                  'video_semantics': 'one initial frame; no motion executed' if outcome == 'preflight_rejected' else 'executed simulation; final state also saved as PNG'}
        (folder / 'diagnostic.json').write_text(json.dumps(result, indent=2))
        (folder / 'trace.json').write_text(json.dumps(trace))
        results.append(result)
        (root / 'summary.json').write_text(json.dumps({'counts': dict(Counter(x['outcome'] for x in results)), 'attempts': results}, indent=2))
        print(f"SAVED {attempt + 1}/{args.attempts}: {outcome}: {failure or 'lift held'} -> {folder}", flush=True)
        if interrupted:
            break
    print('Counts:', dict(Counter(x['outcome'] for x in results)), flush=True)
    verify_saved(root)


if __name__ == '__main__':
    main()
