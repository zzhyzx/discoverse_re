"""Move from the current stationary pose to a named saved pose.

Default: live READ-ONLY preview; no control interface.
--current-file: fully offline preview, forbidden with --execute.
No collision checking, gripper control, or automatic return.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time
import math

from ur5_large_ab import (
    ROOT, TOL, Feedback, budget, confirm, deg, err,
    load, move, scaling, validate, vec,
)


def valid_name(value):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value):
        raise argparse.ArgumentTypeError(
            "Name must start with a letter; use 1-64 letters/digits/_/-"
        )
    return value


def load_target(path, name, ip):
    with path.open() as stream:
        document = json.load(stream)

    if not isinstance(document, dict):
        raise ValueError("Expected a pose JSON object")

    snapshot = document.get("snapshot", document)
    if not isinstance(snapshot, dict):
        raise ValueError("Expected a snapshot JSON object")

    # Support both old unnamed FK reports and new named snapshots.
    for item in (document, snapshot):
        recorded_name = item.get("name")
        if recorded_name is not None and recorded_name != name:
            raise ValueError(
                f"Saved name {recorded_name!r} conflicts with "
                f"--target {name!r}; names are case-sensitive"
            )

    if snapshot.get("ip") != ip:
        raise ValueError("Pose IP mismatch")

    values = snapshot.get("q_rad")
    if not isinstance(values, list) or any(
        isinstance(value, bool) for value in values
    ):
        raise ValueError("Expected six numeric joint radians in q_rad")

    return vec(values)

def validate_move(current, target):
    current = vec(current)
    target = vec(target)

    changes = [
        abs(b - a) for a, b in zip(current, target)
    ]

    if (
        max(changes) > math.radians(45)
        or sum(changes) > math.radians(90)
    ):
        raise ValueError(
            "Test envelope exceeded: max 45 deg/joint, 90 deg sum"
        )

    if max(changes) < math.radians(0.3):
        raise ValueError("Current and target poses are too close")

    # 保留原有保守关节限位，不自动转换角度分支。
    for joints in (current, target):
        for index, angle in enumerate(joints):
            limit = math.pi if index == 2 else 2 * math.pi
            if abs(angle) > limit:
                raise ValueError(
                    "Outside conservative joint bounds; "
                    "do not wrap angles automatically"
                )


def preview(current, target):
    print("Current degrees:", deg(current))
    print("Target degrees:", deg(target))
    print("Delta degrees:", deg([
        destination - source
        for source, destination in zip(current, target)
    ]))

    if err(current, target) <= TOL:
        print("Already within 0.15 deg; no movement will be sent.")
        return False

    # Preserve existing bounds; do not wrap joint angles.
    # validate(current, target)
    validate_move(current, target)
    nominal, timeout = budget(current, target)
    print(
        "Speed 0.03 rad/s; acceleration 0.05 rad/s^2; "
        f"nominal {nominal:.1f}s; timeout {timeout:.1f}s"
    )
    print("No collision detection. Inspect the FULL swept path.")
    return True


def idle(receive):
    if receive.getRobotStatus() & 6 or receive.getRuntimeState() != 1:
        raise RuntimeError(
            "Other program/freedrive active or program not stopped; "
            "inspect pendant"
        )
    scaling(receive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=valid_name, required=True)
    parser.add_argument(
        "--poses-dir",
        type=Path,
        default=ROOT / "data/real_ur5/pick_place_01",
        help="Directory containing <target>.json",
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        help="Explicit saved pose file; overrides --poses-dir",
    )
    parser.add_argument(
        "--current-file",
        type=Path,
        help="Saved current pose for OFFLINE preview only",
    )
    parser.add_argument("--ip", default="192.168.1.101")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if args.execute and args.current_file:
        parser.error(
            "--current-file cannot be used with --execute; "
            "live feedback is mandatory"
        )

    target_file = (
        args.target_file
        if args.target_file is not None
        else args.poses_dir / f"{args.target}.json"
    ).resolve()

    # Fail before connecting if the file is missing or invalid.
    target = load_target(target_file, args.target, args.ip)
    print("Target:", args.target)
    print("Source:", target_file)

    if args.current_file:
        current = load(args.current_file, [], args.ip)
        preview(current, target)
        print("OFFLINE PREVIEW ONLY: no robot connection.")
        return

    from rtde_receive import RTDEReceiveInterface

    receive = RTDEReceiveInterface(args.ip)
    control = None
    log_file = None

    try:
        feedback = Feedback(receive)
        measured, _ = feedback.sample()
        current = feedback.stationary(measured)
        idle(receive)

        if not preview(current, target):
            return

        if not args.execute:
            print(
                "READ-ONLY PREVIEW: receive connection only; "
                "no motion command."
            )
            return

        print(
            "Confirm correct robot/tool/payload, clear FULL swept path, "
            "no self-collision/cable snag, and operator at E-stop."
        )
        confirm("MOVE")

        # Reject movement after preview instead of adopting a new start.
        feedback.stationary(current)
        idle(receive)

        folder = ROOT / "data/real_ur5/move_to_pose_logs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl"
        )
        log_file = path.open("x", buffering=1)

        def log(event, **values):
            log_file.write(json.dumps({
                "event": event,
                "monotonic": time.monotonic(),
                **values,
            }) + "\n")

        log(
            "start",
            ip=args.ip,
            current=current,
            target=target,
            target_name=args.target,
            target_file=str(target_file),
        )
        print("Log:", path)

        # Control interface is created only after explicit confirmation.
        from rtde_control import RTDEControlInterface

        control = RTDEControlInterface(args.ip)
        try:
            move(control, feedback, current, target, log)
        except BaseException as exc:
            log("aborted", reason=str(exc))
            raise

        log("complete")
        print("COMPLETE: staying at target. No automatic return.")

    finally:
        if control is not None:
            try:
                control.stopScript()
            except Exception as exc:
                print("stopScript failed:", exc)
            try:
                control.disconnect()
            except Exception as exc:
                print("Control disconnect failed:", exc)

        if log_file is not None:
            log_file.close()

        receive.disconnect()


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(
            f"ABORTED: {exc}. No automatic return. Inspect pendant.",
            flush=True,
        )
        raise SystemExit(1)