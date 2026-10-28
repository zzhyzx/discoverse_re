"""Add real/MuJoCo/3DGS comparisons to saved offline UR5 reproduction runs.

Example (run from the project root, in the discoverse environment):
  MUJOCO_GL=egl python scripts/render_ur5_three_view_comparison.py

No robot interfaces are used. Existing videos are never overwritten. Every
replayed trajectory is checked against the reference metrics before delivery.
Limited-mode real frames repeat according to source progress, not wall time.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import av
import numpy as np
from discoverse.task_base.airbot_task_base import PyavImageEncoder


def verify_video(path, frames, fps):
    stamps = []
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            if (frame.width, frame.height) != (1920, 556):
                raise ValueError(f"Unexpected video size: {path}")
            stamps.append(float(frame.pts * frame.time_base))
    if len(stamps) != frames or not np.allclose(stamps, np.arange(frames)/fps, atol=2e-5):
        raise ValueError(f"Frame count/timestamps mismatch: {path}")
    return {"frames": frames, "fps": fps, "duration_s": frames/fps,
            "resolution": [1920, 556], "decoded_and_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path,
                        default=ROOT/"data/ur5_real_reproduction_20261027")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 17, 39])
    parser.add_argument("--modes", nargs="+", choices=["state", "direct", "limited"],
                        default=["direct", "state", "limited"])
    args = parser.parse_args()
    cases = []
    for mode in dict.fromkeys(args.modes):
        for episode in dict.fromkeys(args.episodes):
            folder = args.reference_root/f"ep_{episode}_{mode}_video"
            report = json.loads((folder/"report.json").read_text())
            if not report["complete"] or report["mode"] != mode or report["episode"] != episode:
                raise ValueError(f"Incomplete/inconsistent reference: {folder}")
            if report.get("real_rectification"):
                raise ValueError("This batch preserves raw real video; rectified reference not supported")
            cases.append((folder, report))
    output = (args.output or args.reference_root/(
        "three_view_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"complete": False, "columns": ["real", "mujoco", "3dgs"],
               "reference_root": str(args.reference_root.resolve()), "runs": [],
               "limitations": ["Cameras are not precisely calibrated",
                   "Cube initialization is inferred from the original pre-close FK",
                   "MuJoCo and 3DGS render exactly the same simulation state",
                   "Limited replay is source-progress retimed, not wall-time synchronized",
                   "State replay has static gripper/cube and is not grasp evidence"]}

    def save_summary():
        (output/"summary.json").write_text(json.dumps(summary, indent=2))

    save_summary()
    for index, (reference, original) in enumerate(cases, 1):
        episode, mode = original["episode"], original["mode"]
        destination = output/reference.name
        command = [sys.executable, "-u", str(ROOT/"scripts/ur5_replay_control_comparison.py"),
                   "--episode", str(episode), "--mode", mode, "--compare-renderers",
                   "--reference-run", str(reference.resolve()), "--output", str(destination),
                   "--damping-scale", str(original["damping_scale"]),
                   "--max-sim-seconds", str(max(180, original["simulation_duration_s"] + 10))]
        if original.get("interaction"):
            command.append("--interaction")
        if original["arm_gravity_compensation"]:
            command.append("--arm-gravity-compensation")
        if original["camera_binding"]["base_table_centerline_override"]:
            command.append("--align-base-table-centerline")
        if original["tcp_offset_confirmed"]:
            command.extend(["--tcp-offset", *map(str, original["tcp_offset"])])
        print(f"[{index}/{len(cases)}] Rendering episode {episode}, {mode}", flush=True)
        with (output/f"ep_{episode}_{mode}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((destination/"report.json").read_text())
        if not report["complete"] or not report["reference_check"]["passed"]:
            raise ValueError(f"Replay verification failed: {destination}")
        videos = {}
        for cid, role in enumerate(("front", "wrist")):
            path = destination/f"cam_{cid}.mp4"
            videos[role] = verify_video(path, report["frames"], report["fps"])
            videos[role]["file"] = str(path.relative_to(output))
        summary["runs"].append({"episode": episode, "mode": mode, "videos": videos,
                                "physics_matches_reference": True})
        save_summary()
        print(f"  Verified: {report['frames']} frames, both views, original metrics match", flush=True)

    # Short, original-speed overview of the selected direct episodes.
    direct = [r for r in summary["runs"] if r["mode"] == "direct"]
    if direct:
        overview = output/"overview_direct"
        overview.mkdir()
        summary["overview_direct"] = {}
        for cid, role in enumerate(("front", "wrist")):
            fps = direct[0]["videos"][role]["fps"]
            writer = PyavImageEncoder(1920, 556, str(overview), cid, fps=fps, crf=18, preset="veryfast")
            count = 0
            try:
                for item in direct:
                    if item["videos"][role]["fps"] != fps:
                        raise ValueError("Cannot concatenate different frame rates")
                    with av.open(str(output/item["videos"][role]["file"])) as video:
                        for frame in video.decode(video=0):
                            count += 1
                            writer.encode(frame.to_ndarray(format="rgb24"), count/fps)
            finally:
                writer.close()
            path = overview/f"cam_{cid}.mp4"
            summary["overview_direct"][role] = verify_video(path, count, fps)
            summary["overview_direct"][role]["file"] = str(path.relative_to(output))
    summary["complete"] = True
    save_summary()
    print("COMPLETE:", output, flush=True)


if __name__ == "__main__":
    main()
