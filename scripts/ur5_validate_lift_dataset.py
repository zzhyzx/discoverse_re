"""Offline dataset/video acceptance and optional simulated empty-gripper sweep.
Never imports robot network/control interfaces.
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np


def validate(folder):
    payload = json.loads((folder / "obs_action.json").read_text())
    t = np.asarray(payload["time"])
    state = np.asarray(payload["obs"]["qpos"])
    action = np.asarray(payload["act"])
    meta = payload["metadata"]
    assert len(t) > 2 and np.isclose(t[0], 0)
    assert np.allclose(np.diff(t), 1 / 30, atol=1e-8)
    assert state.shape == action.shape == (len(t), 7)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert np.all((state[:, 6] >= 0) & (state[:, 6] <= 1))
    assert np.all((action[:, 6] >= 0) & (action[:, 6] <= 1))
    assert np.allclose(state[0, :6], meta["start_reference_rad"], atol=1e-6)
    # Legacy collection used 0.60; recovery collection explicitly records its cap.
    rate = float(meta.get("max_command_rate_rad_s", 0.60))
    assert np.isfinite(rate) and 0 < rate <= 1.2
    assert np.max(np.abs(np.diff(action[:, :6], axis=0))) <= rate / 30 + 1e-6
    assert meta["camera_feature_mapping"] == {
        "cam_0": "observation.images.d436", "cam_1": "observation.images.d405"}
    clearance = np.asarray(payload["obs"]["block_clearance_m"])
    criterion = meta["success_criterion"]
    tail = t >= t[-1] - criterion["continuous_hold_s"] - 1e-8
    assert tail.sum() >= 31
    assert np.all(clearance[tail] >= criterion["block_bottom_clearance_m"])
    if "lift_condition" in payload["obs"]:
        assert np.asarray(payload["obs"]["lift_condition"])[tail].all()
    assert np.all(action[tail, 6] == 1), "Gripper must stay commanded closed"
    for camera in (0, 1):
        with av.open(str(folder / f"cam_{camera}.mp4")) as container:
            stamps = []
            for frame in container.decode(video=0):
                assert frame.width == 640 and frame.height == 480
                stamps.append(float(frame.pts * frame.time_base))
            stamps = np.asarray(stamps)
            assert len(stamps) == len(t), (len(stamps), len(t))
            assert np.allclose(stamps, t - t[0], atol=2e-5)
    result = {"episode": str(folder), "frames_per_camera": len(t), "hz": 30,
              "initial_block_xy_m": payload["obs"]["block_pos"][0][:2],
              "sampling_grid_cell": meta.get("sampling", {}).get("proposal_grid_cell"),
              "recorded_duration_s": float(t[-1]),
              "minimum_final_hold_clearance_m": float(clearance[tail].min()),
              "max_action_step_deg": float(np.degrees(np.abs(np.diff(action[:, :6], axis=0))).max()),
              "final_sim_feedback": float(state[-1, 6]),
              "final_pad_gap_m": payload["obs"]["gripper_gap_m"][-1]}
    print(json.dumps(result, indent=2))
    return result


def sweep():
    import mujoco
    from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import SimNode, make_config
    cfg = make_config(True)
    cfg.use_real_training_start = True
    env = SimNode(cfg, np.random.default_rng(0), randomize=False)
    env.reset()
    model, data = env.mj_model, env.mj_data
    # Remove the cube from the gripper workspace, then settle empty commands.
    data.qpos[env.block_qpos_adr:env.block_qpos_adr + 3] = [3, 3, 0.6]
    data.ctrl[:6] = env.home_q
    results = []
    for command in (0, 64, 128, 192, 255):
        data.ctrl[6] = command
        for _ in range(720):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        results.append({"command_raw": command, "feedback_driver_fraction": float(env.normalized_state()[6]),
                        "pad_gap_m": env.gripper_gap()})
    assert np.all(np.diff([x["feedback_driver_fraction"] for x in results]) > 0)
    assert np.all(np.diff([x["pad_gap_m"] for x in results]) < 0)
    print("SIMULATION ONLY: increasing command closes gap; real POS-to-mm still uncalibrated")
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--gripper-sweep", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    episodes = sorted(args.dataset.glob("*/obs_action.json"))
    if not episodes:
        raise ValueError("No complete episodes found")
    report = {"episodes": [validate(p.parent) for p in episodes]}
    xy = np.asarray([e["initial_block_xy_m"] for e in report["episodes"]])
    report["coverage"] = {"successful_episodes": len(episodes), "xy_min_m": xy.min(axis=0).tolist(),
                          "xy_max_m": xy.max(axis=0).tolist(), "successes_per_grid_cell": {}}
    counts = report["coverage"]["successes_per_grid_cell"]
    for episode in report["episodes"]:
        cell = str(episode["sampling_grid_cell"])
        counts[cell] = counts.get(cell, 0) + 1
    attempt_file = args.dataset / "attempts.jsonl"
    if attempt_file.exists():
        attempts = [json.loads(line) for line in attempt_file.read_text().splitlines() if line.strip()]
        report["coverage"]["logged_attempts"] = len(attempts)
        report["coverage"]["logged_failures"] = sum(not a["success"] for a in attempts)
    print("Coverage:", json.dumps(report["coverage"]))
    if args.gripper_sweep:
        report["simulation_gripper_sweep"] = sweep()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
