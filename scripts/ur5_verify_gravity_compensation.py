"""Simulation-only paired stationary-target test from saved failed snapshots."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import SimNode, make_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    cfg = make_config(True)
    cfg.align_base_table_centerline = True
    cfg.use_real_training_start = True
    env = SimNode(cfg, np.random.default_rng(0), randomize=False)
    results = []
    for key in ('001', '004', '008'):
        source = json.loads((a.baseline / key / 'diagnostic.json').read_text())['final']
        pair = {'sample': key, 'target_tcp_m': source['tcp_target_m']}
        for enabled in (False, True):
            env.reset()
            model, data = env.mj_model, env.mj_data
            data.qpos[:6] = source['q_actual_rad']
            data.qvel[:] = 0
            cfg.arm_gravity_compensation = enabled
            action = np.r_[source['q_target_rad'], 0.0]
            mujoco.mj_forward(model, data)
            peak = np.zeros(6)
            for _ in range(round(4 / model.opt.timestep)):
                env.updateControl(action)
                assert data.ctrl[6] == 0, 'Finger command changed'
                assert np.all(data.qfrc_applied == 0), 'Unexpected direct force injection'
                assert np.all(model.body_gravcomp == 0), 'Unexpected body anti-gravity'
                mujoco.mj_step(model, data)
                force = data.actuator_force[:6]
                assert np.all(force <= model.actuator_forcerange[:6, 1] + 1e-8)
                assert np.all(force >= model.actuator_forcerange[:6, 0] - 1e-8)
                peak = np.maximum(peak, np.abs(force))
            mujoco.mj_forward(model, data)
            pair['on' if enabled else 'off'] = {
                'tcp_error_mm': float(1000 * np.linalg.norm(data.site_xpos[env.ee_site_id] - source['tcp_target_m'])),
                'max_joint_error_deg': float(np.degrees(np.max(np.abs(data.qpos[:6] - action[:6])))),
                'peak_arm_actuator_force_Nm': peak.tolist(),
                'final_joint_speed_rad_s': float(np.max(np.abs(data.qvel[:6])))}
        assert pair['on']['tcp_error_mm'] < 1.0, pair
        assert pair['on']['tcp_error_mm'] < pair['off']['tcp_error_mm'] / 10, pair
        results.append(pair)
        print(json.dumps(pair), flush=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps({'duration_per_test_s': 4, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
