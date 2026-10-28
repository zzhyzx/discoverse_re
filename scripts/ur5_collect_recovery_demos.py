"""Simulation-only lateral correction demonstrations; no robot connections.

48 balanced proposals. Resume never overwrites attempts. Successful episodes
are eligible only after full video/state acceptance. No training is started.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policies.smolvla.eval_sim_ur5_lift import initialize_above_block
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import (
    SimNode, Expert, EpisodeFailure, make_config, new_buffers, record,
    save_episode, PyavImageEncoder)
from ur5_validate_lift_dataset import validate


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--stop-after', type=int, default=48)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    if not 1 <= args.stop_after <= 48:
        raise ValueError('stop-after must be 1..48')
    root = args.output.resolve()
    rng = np.random.default_rng(args.seed)
    plans = []
    for repeat in range(2):
        for low, high in ((.02,.04),(.04,.07),(.07,.10)):
            for height in (.12,.18):
                for name, direction in (('x_plus',(1,0)),('x_minus',(-1,0)),
                                        ('y_plus',(0,1)),('y_minus',(0,-1))):
                    distance = float(rng.uniform(low, high))
                    plans.append(dict(direction=name, range_m=[low,high], height_m=height,
                                      offset_xy_m=(np.array(direction)*distance).tolist(), repeat=repeat))
    if args.resume:
        protocol = json.loads((root/'protocol.json').read_text())
        assert protocol['seed'] == args.seed and protocol['plan'] == plans
    else:
        root.mkdir(parents=True, exist_ok=False)
        write(root/'protocol.json', dict(seed=args.seed, plan=plans, fps=30,
              max_command_rate_rad_s=.9, gravity_compensation=True,
              block_yaw_deg=0, timeout_s=20, expert_updates_hz=30,
              excluded_test_layout_radius_m=.02, task='lateral realignment then descent, close, lift/hold',
              limitation='No induced empty-close recovery; no model training; small first batch'))
    for name in ('episodes','attempts'):
        (root/name).mkdir(exist_ok=True)
    reference = json.loads((ROOT/'data/smolvla_translation_large_20261022/translation_summary.json').read_text())
    test_xy = []
    for row in reference['conditions'][0]['episodes']:
        init = json.loads((Path(row['source'])/'initialization.json').read_text())
        test_xy.append(init['block_pos_world_m'][:2])
    test_xy = np.asarray(test_xy)
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
    cfg.arm_gravity_compensation = True

    class NewLayoutSimNode(SimNode):
        def sample_block_pose(self):
            for _ in range(1000):
                xy, _ = super().sample_block_pose()
                if np.min(np.linalg.norm(test_xy-xy, axis=1)) >= .02:
                    self.config.sampling_metadata['yaw_deg'] = 0.
                    return xy, 0.
            raise RuntimeError('Could not sample away from validation layouts')

    env = NewLayoutSimNode(cfg, np.random.default_rng(args.seed+1), randomize=True)
    expert = Expert(env)
    home_reference = env.home_q.tolist()
    results = json.loads((root/'summary.json').read_text())['attempts'] if args.resume else []
    done = {r['index'] for r in results}
    for index, plan in enumerate(plans[:args.stop_after]):
        obs = env.reset()  # preserve the same layout sequence on resume
        if index in done:
            continue
        folder = root/'attempts'/f'{index:03d}'
        folder.mkdir(exist_ok=False)
        write(folder/'proposal.json', plan)
        write(folder/'scene_before_initialization.json', dict(
            block_pos_world_m=env.block_pos().tolist(),
            state=env.normalized_state().tolist(), sampling=cfg.sampling_metadata))
        reason = 'timeout'
        success = False
        buffers = new_buffers()
        trace = []
        peak = 0.
        try:
            obs, init = initialize_above_block(env, expert, plan['height_m'], plan['offset_xy_m'], 0.)
            init['expert_control_after_initialization'] = True
            write(folder/'initialization.json', init)
            cfg.start_reference_rad = env.normalized_state()[:6].tolist()
            action = env.target_control.copy()
            expert.state = 1
            expert.enter()
            settled = None
            encoders = [PyavImageEncoder(640,480,str(folder),c,fps=30,crf=18,preset='veryfast') for c in (0,1)]
            try:
                for step in range(600):
                    expert.update()
                    action[:6] += np.clip(env.target_control[:6]-action[:6], -.9*env.delta_t, .9*env.delta_t)
                    action[6] = env.target_control[6]
                    trace.append(dict(step=step, state=expert.state, finished=expert.finished))
                    record(env, obs, action, buffers, encoders)
                    obs, *_ = env.step(action)
                    info = env.success_info()
                    peak = max(peak, info['clearance_m'])
                    if expert.table_collision(env.mj_data):
                        raise EpisodeFailure('arm_table_floor_pedestal_collision')
                    if info['lifted'] and info['stationary']:
                        if settled is None:
                            settled = float(env.mj_data.time)
                        if env.mj_data.time-settled >= cfg.hold_seconds+env.delta_t:
                            success = True
                            reason = 'success'
                            break
                    else:
                        settled = None
            finally:
                for encoder in encoders:
                    encoder.close()
        except (EpisodeFailure, ValueError) as exc:
            reason = f'{type(exc).__name__}: {exc}'
        if buffers['time']:
            save_episode(str(folder), buffers, args.seed, index+1, cfg,
                         success=success, failure=None if success else reason, diagnostic=True)
            payload = json.loads((folder/'obs_action.json').read_text())
            payload['metadata'].update(max_command_rate_rad_s=.9, recovery_proposal=plan,
                                       home_reference_rad=home_reference, initial_pose_source='offset IK initialization')
            write(folder/'obs_action.json', payload)
            write(folder/'expert_trace.json', trace)
            if success:
                validation = validate(folder)
                validation['episode'] = str(root/'episodes'/folder.name)
                write(folder/'validation.json', validation)
                payload['metadata'].update(diagnostic_only=False, training_eligible=True)
                write(folder/'obs_action.json', payload)
                destination = root/'episodes'/folder.name
                assert not destination.exists()
                folder.rename(destination)
                folder = destination
        row = dict(index=index, proposal=plan, success=success, reason=reason,
                   frames=len(buffers['time']), peak_clearance_m=peak, output=str(folder))
        write(folder/'result.json', row)
        results.append(row)
        write(root/'summary.json', dict(planned=48, completed=len(results),
              successes=sum(r['success'] for r in results), attempts=results))
        print('ATTEMPT', json.dumps(row), flush=True)
    print(f'FINISHED {len(results)}/48 attempts; {sum(r["success"] for r in results)} accepted', flush=True)


if __name__ == '__main__':
    main()
