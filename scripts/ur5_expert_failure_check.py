"""MuJoCo-only paired expert check of previously failed policy trials.

No robot interfaces. Reuses policy initialization and collection Expert; skips
home execution, not the recorded starting state. Outputs are diagnostic only.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import av

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policies.smolvla.eval_sim_ur5_lift import initialize_above_block
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import (
    SimNode, Expert, EpisodeFailure, make_config, new_buffers, record,
    save_episode, PyavImageEncoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    reference = json.loads((args.reference/'translation_summary.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    expected = sum(not r['success'] for c in reference['conditions'] for r in c['episodes'])
    for condition in reference['conditions']:
        failures = {r['episode']: r for r in condition['episodes'] if not r['success']}
        if not failures:
            continue
        protocol = json.loads((Path(next(iter(failures.values()))['source']).parent/'protocol.json').read_text())
        assert protocol['fps'] == 30 and protocol['gravity_compensation']
        assert protocol['fixed_block_yaw_deg'] == 0 and protocol['initial_yaw_offset_deg'] == 0
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

        class FixedYawSimNode(SimNode):
            def sample_block_pose(self):
                xy, yaw = super().sample_block_pose()
                self.config.sampling_metadata['yaw_deg'] = 0.
                return xy, 0.

        env = FixedYawSimNode(cfg, np.random.default_rng(protocol['seed']), randomize=True)
        expert = Expert(env)
        for index in range(max(failures)+1):
            obs = env.reset()
            if index not in failures:
                continue
            source = Path(failures[index]['source'])
            folder = args.output/condition['condition']/f'{index:03d}'
            folder.mkdir(parents=True, exist_ok=False)
            obs, init = initialize_above_block(env, expert, protocol['above_height_m'],
                                              protocol['initial_offset_world_xy_m'], 0.)
            old_init = json.loads((source/'initialization.json').read_text())
            old_data = json.loads((source/'obs_action.json').read_text())
            errors = {
                'max_joint_difference_rad': float(np.max(np.abs(np.asarray(init['q_rad'])-old_init['q_rad']))),
                'tcp_difference_m': float(np.linalg.norm(np.asarray(init['actual_tcp_world_m'])-old_init['actual_tcp_world_m'])),
                'block_difference_m': float(np.linalg.norm(np.asarray(init['block_pos_world_m'])-old_init['block_pos_world_m'])),
                'gripper_difference': abs(init['gripper_state']-old_init['gripper_state']),
                'max_state_difference': float(np.max(np.abs(env.normalized_state()-np.asarray(old_data['obs']['qpos'][0])))),
            }
            assert max(errors.values()) < 1e-5, f'Initial state mismatch: {errors}'
            init.update(expert_control_after_initialization=True, paired_source=str(source), comparison=errors)
            (folder/'initialization.json').write_text(json.dumps(init, indent=2))
            # Capture initial applied hold BEFORE enter() selects a new IK target.
            action = env.target_control.copy()
            expert.state = 1
            expert.enter()
            buffers = new_buffers()
            trace = []
            settled = None
            success = False
            reason = 'timeout'
            peak = 0.
            encoders = [PyavImageEncoder(640,480,str(folder),i,fps=30,crf=18,preset='veryfast') for i in (0,1)]
            try:
                for step in range(round(protocol['timeout_sim_seconds']*30)):
                    expert.update()
                    # Identical per-axis target slew cap to policy evaluation.
                    amount = protocol['max_command_rate_rad_s']*env.delta_t
                    action[:6] += np.clip(env.target_control[:6]-action[:6], -amount, amount)
                    action[6] = env.target_control[6]
                    trace.append({'step': step, 'state': expert.state,
                                  'target_tcp': None if expert.target_tcp is None else expert.target_tcp.tolist(),
                                  'target': env.target_control.tolist(), 'applied': action.tolist()})
                    record(env, obs, action, buffers, encoders)
                    obs, *_ = env.step(action)
                    info = env.success_info()
                    peak = max(peak, info['clearance_m'])
                    if expert.table_collision(env.mj_data):
                        reason = 'arm_table_floor_pedestal_collision'
                        break
                    if info['lifted'] and info['stationary']:
                        if settled is None:
                            settled = float(env.mj_data.time)
                        if env.mj_data.time-settled >= cfg.hold_seconds+env.delta_t:
                            success = True
                            reason = 'success'
                            break
                    else:
                        settled = None
            except EpisodeFailure as exc:
                reason = 'expert_failure: '+str(exc)
            finally:
                for encoder in encoders:
                    encoder.close()
            save_episode(str(folder), buffers, protocol['seed'], index+1, cfg,
                         success=success, failure=None if success else reason, diagnostic=True)
            (folder/'expert_trace.json').write_text(json.dumps(trace))
            counts = []
            for camera in (0,1):
                with av.open(str(folder/f'cam_{camera}.mp4')) as video:
                    counts.append(sum(1 for _ in video.decode(video=0)))
            assert counts == [len(trace),len(trace)]
            row = dict(condition=condition['condition'], episode=index, policy_success=False,
                       policy_reason=failures[index]['reason'], success=success, reason=reason,
                       frames=len(trace), video_frames=counts, source=str(source), output=str(folder.resolve()),
                       initial_match=errors, peak_clearance_m=peak, final=env.success_info())
            rows.append(row)
            summary = dict(expected=expected, completed=len(rows), successes=sum(r['success'] for r in rows),
                           episodes=rows, configuration='Same initialization, 30Hz, gravity compensation, target slew and success predicate as reference. Expert updates at 30Hz, uses exact block pose and arrival gates; collection approach path retained, no home motion or preflight filtering.')
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(row), flush=True)
        # Renderer resources are owned by SimNode; explicit close if provided.
        if hasattr(env, 'close'):
            env.close()
        del expert, env
    assert len(rows) == expected
    print(f'COMPLETE {sum(r["success"] for r in rows)}/{expected}: {args.output}', flush=True)


if __name__ == '__main__':
    main()
