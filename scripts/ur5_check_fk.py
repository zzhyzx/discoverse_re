"""Read-only UR5 pose recording and MuJoCo FK comparison.

No rtde_control import or robot motion commands.
Named captures default to:
    data/real_ur5/pick_place_01/<name>.json

TCP offset: active pendant TCP relative to tool0, XYZ in metres,
orientation as rotation vector in radians.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data/real_ur5/pick_place_01"
DEFAULT_XML = ROOT / "models/mjcf/ur5/scene_single_red_block.xml"

JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def six(values):
    array = np.asarray(values, dtype=float)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError("Expected six finite values")
    return array


def valid_name(value):
    # Exclude path separators and special path names.
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value):
        raise argparse.ArgumentTypeError(
            "Name must start with a letter and contain only "
            "letters, digits, underscores or hyphens; max 64 characters."
        )
    return value


def transform(values):
    values = six(values)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
    matrix[:3, 3] = values[:3]
    return matrix


def pose(matrix):
    return np.r_[
        matrix[:3, 3],
        Rotation.from_matrix(matrix[:3, :3]).as_rotvec(),
    ].tolist()


def capture(ip):
    from rtde_receive import RTDEReceiveInterface

    robot = RTDEReceiveInterface(ip)
    samples = []

    try:
        for _ in range(25):
            if not robot.isConnected():
                raise RuntimeError("RTDE disconnected")

            if robot.getRobotMode() not in (5, 7):
                raise RuntimeError("Robot state unsuitable for capture")

            if robot.getSafetyMode() not in (1, 2):
                raise RuntimeError("Safety state unsuitable for capture")

            if robot.getRobotStatus() & 6:
                raise RuntimeError(
                    "Stop other programs and release freedrive first"
                )

            before = six(robot.getActualQ())
            tcp = six(robot.getActualTCPPose())
            after = six(robot.getActualQ())
            velocity = six(robot.getActualQd())

            if (
                np.max(np.abs(velocity)) > 0.005
                or np.max(np.abs(after - before)) > 0.001
            ):
                raise RuntimeError("Robot moving; capture only at rest")

            samples.append(
                (float(robot.getTimestamp()), after, tcp)
            )
            time.sleep(0.04)

        stamps = np.asarray([item[0] for item in samples])
        joints = np.asarray([item[1] for item in samples])
        tcps = np.asarray([item[2] for item in samples])

        if (
            not np.isfinite(stamps).all()
            or np.any(np.diff(stamps) <= 0)
        ):
            raise RuntimeError("Stale/non-monotonic RTDE timestamps")

        if np.max(np.ptp(joints, axis=0)) > 0.001:
            raise RuntimeError("Joint drift during capture")

        position_drift = np.linalg.norm(
            tcps[:, :3] - tcps[0, :3], axis=1
        )
        if np.max(position_drift) > 0.001:
            raise RuntimeError("TCP position changed during capture")

        rotations = Rotation.from_rotvec(tcps[:, 3:])
        rotation_drift = (
            rotations[0].inv() * rotations
        ).magnitude()
        if np.max(rotation_drift) > 0.005:
            raise RuntimeError("TCP orientation changed during capture")

        return {
            "ip": ip,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "timestamp": samples[-1][0],
            "q_rad": samples[-1][1].tolist(),
            "actual_tcp_base": samples[-1][2].tolist(),
            "robot_mode": robot.getRobotMode(),
            "safety_mode": robot.getSafetyMode(),
        }
    finally:
        robot.disconnect()


def load_snapshot(path):
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    # Accept both FK reports and earlier standalone pose records.
    sample = dict(payload.get("snapshot", payload))
    sample["q_rad"] = six(sample["q_rad"]).tolist()
    return sample


def build_simulation(xml_path, joint_angles):
    model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mapping = []
    for name, angle in zip(JOINTS, joint_angles):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if (
            joint_id < 0
            or model.jnt_type[joint_id]
            != mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise ValueError(f"Missing hinge joint: {name}")

        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address] = angle

        mapping.append({
            "name": name,
            "qposadr": address,
            "rad": float(angle),
        })

        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            if not lower <= angle <= upper:
                print(
                    "WARNING: measured angle outside simulation limits:",
                    name,
                )

    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return model, data, mapping


def body_transform(model, data, name):
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, name
    )
    if body_id < 0:
        raise ValueError(f"Missing body: {name}")

    matrix = np.eye(4)
    matrix[:3, :3] = data.xmat[body_id].reshape(3, 3)
    matrix[:3, 3] = data.xpos[body_id]
    return matrix


def make_report(model, data, sample, mapping, xml_path, tcp_offset):
    # UR base frame, not MuJoCo world or ROS base_link.
    world_to_base = body_transform(model, data, "base")
    base_inverse = np.linalg.inv(world_to_base)
    zero_tool = base_inverse @ body_transform(model, data, "tool0")

    pinch = None
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch"
    )
    if site_id >= 0:
        matrix = np.eye(4)
        matrix[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
        matrix[:3, 3] = data.site_xpos[site_id]
        pinch = base_inverse @ matrix

    report = {
        "schema": 1,
        "name": sample.get("name"),
        "snapshot": sample,
        "xml": str(xml_path.resolve()),
        "joint_mapping": mapping,
        "sim_world_T_ur_base": world_to_base.tolist(),
        "sim_zero_tool_in_ur_base": pose(zero_tool),
        "sim_pinch_in_ur_base": (
            pose(pinch) if pinch is not None else None
        ),
    }

    print("Name:", sample.get("name", "(unnamed)"))
    print(
        "Joint degrees:",
        np.round(np.degrees(sample["q_rad"]), 4).tolist(),
    )
    print(
        "UR-base zero-tool [m, rotvec rad]:",
        np.round(pose(zero_tool), 6).tolist(),
    )
    if pinch is not None:
        print(
            "Sim pinch [m, rotvec rad]:",
            np.round(pose(pinch), 6).tolist(),
        )

    if "actual_tcp_base" in sample:
        actual = transform(sample["actual_tcp_base"])
        inferred = np.linalg.inv(zero_tool) @ actual
        report["inferred_tool_offset_including_model_error"] = pose(
            inferred
        )

        print("Actual TCP:", sample["actual_tcp_base"])
        print(
            "INFERRED offset (includes model errors; NOT calibration):",
            np.round(pose(inferred), 6).tolist(),
        )

        if tcp_offset is not None:
            offset = six(tcp_offset)
            predicted = zero_tool @ transform(offset)
            position_error = float(
                np.linalg.norm(
                    predicted[:3, 3] - actual[:3, 3]
                ) * 1000
            )
            rotation_error = float(np.degrees(
                Rotation.from_matrix(
                    predicted[:3, :3].T @ actual[:3, :3]
                ).magnitude()
            ))

            report.update({
                "tcp_offset_m_rotvec": offset.tolist(),
                "predicted_active_tcp": pose(predicted),
                "position_error_mm": position_error,
                "rotation_error_deg": rotation_error,
            })
            print(
                f"ACTIVE TCP error: {position_error:.3f} mm, "
                f"{rotation_error:.4f} deg"
            )
        else:
            print(
                "No calibrated TCP comparison: supply the actual "
                "active pendant offset with --tcp-offset."
            )

    return report


def show_snapshot(model, data):
    from mujoco import viewer as mj_viewer

    frozen_qpos = data.qpos.copy()
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.qpos[:] = frozen_qpos
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.03)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", action="store_true")
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--q", nargs=6, type=float)

    parser.add_argument("--name", type=valid_name)
    parser.add_argument("--poses-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--ip", default="192.168.1.101")
    parser.add_argument("--tcp-offset", nargs=6, type=float)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    if args.out is not None:
        output = args.out
    elif args.capture and args.name:
        output = args.poses_dir / f"{args.name}.json"
    else:
        output = (
            ROOT / "data/real_ur5/fk_checks"
            / f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        )

    # Check before connecting to the robot.
    if output.exists():
        raise FileExistsError(
            f"{output} already exists. Use a new name or directory."
        )

    if args.capture:
        sample = capture(args.ip)
    elif args.snapshot:
        sample = load_snapshot(args.snapshot)
    else:
        sample = {
            "q_rad": six(args.q).tolist(),
            "source": "offline_manual",
        }

    if args.name:
        sample["name"] = args.name

    joint_angles = six(sample["q_rad"])
    model, data, mapping = build_simulation(args.xml, joint_angles)
    report = make_report(
        model, data, sample, mapping, args.xml, args.tcp_offset
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("Saved:", output.resolve())
    print(
        "Read-only capture/FK. No arm motion, no dynamics, no 3DGS. "
        "Simulation gripper remains at its initial configuration."
    )

    if args.viewer:
        show_snapshot(model, data)


if __name__ == "__main__":
    main()