"""Record nearby UR5 joint poses and perform operator-confirmed moveJ tests.
Not safety-rated. No scene collision checking or guaranteed network-failure stop.
No power-on, brake release, fault reset, TCP changes, or gripper commands.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "real_ur5" / "taught_poses"
SPEED, ACCEL, TIMEOUT = 0.03, 0.05, 15.0
TOL = math.radians(0.15)


def vec(x):
    x = list(x)
    if len(x) != 6 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in x):
        raise ValueError("Expected six finite numbers")
    return x


def distance(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def degrees(q):
    return [round(math.degrees(x), 4) for x in q]


def pair(a, b):
    delta = [abs(x - y) for x, y in zip(vec(a), vec(b))]
    if max(delta) > math.radians(2) or sum(delta) > math.radians(3):
        raise ValueError("A/B too far: max 2 deg per joint and 3 deg total; reteach nearby B")
    if max(delta) < math.radians(0.3):
        raise ValueError("A/B too close (less than 0.3 deg); insufficient test separation")


def load(folder, name, ip):
    with (folder / f"{name}.json").open() as f:
        result = json.load(f)
    if result.get("schema") != 1 or result.get("name") != name or result.get("ip") != ip:
        raise ValueError("Pose schema/name/IP mismatch; do not reuse another robot's poses")
    result["q_rad"] = vec(result["q_rad"])
    return result


class Feedback:
    def __init__(self, robot):
        self.robot = robot
        self.stamp = None
        self.fresh = time.monotonic()

    def sample(self, ready=True):
        r = self.robot
        if not r.isConnected():
            raise RuntimeError("Receive disconnected")
        stamp = r.getTimestamp()
        if not math.isfinite(stamp):
            raise RuntimeError("Invalid RTDE timestamp")
        if self.stamp is not None and stamp < self.stamp:
            raise RuntimeError("Controller timestamp went backwards")
        if self.stamp is None or stamp > self.stamp:
            self.stamp, self.fresh = stamp, time.monotonic()
        if time.monotonic() - self.fresh > 0.5:
            raise RuntimeError("RTDE feedback stale > 0.5 s")
        if r.getRobotMode() not in ((7,) if ready else (5, 7)):
            raise RuntimeError("Robot not ready; prepare via pendant, never bypass safety")
        if r.getSafetyMode() not in (1, 2):
            raise RuntimeError("Safety state is not NORMAL/REDUCED")
        return vec(r.getActualQ()), vec(r.getActualQd())

    def stationary(self, expected=None, ready=True):
        first, _ = self.sample(ready)
        expected = first if expected is None else expected
        initial_stamp = self.stamp
        for _ in range(30):
            time.sleep(0.02)
            q, qd = self.sample(ready)
            if distance(q, expected) > TOL or max(abs(v) for v in qd) > 0.005:
                raise RuntimeError("Robot moved/is not at expected pose; no automatic reposition")
        if self.stamp <= initial_stamp:
            raise RuntimeError("No fresh feedback")
        return q


def confirm(word):
    if input(f"Type {word} to proceed; anything else cancels: ").strip().upper() != word:
        raise RuntimeError("Cancelled; staying at current pose, no automatic return")


def motion(c, feedback, source, target):
    actual = feedback.stationary(source)
    if not c.isConnected():
        raise RuntimeError("Control disconnected")
    if not c.isJointsWithinSafetyLimits(target):
        raise RuntimeError("Target rejected by controller limits (not a collision test)")
    last_progress = c.getAsyncOperationProgress()
    pending = True
    try:
        if not c.moveJ(target, SPEED, ACCEL, True):
            raise RuntimeError("moveJ rejected")
        deadline = time.monotonic() + TIMEOUT
        stable = 0
        began = False
        while time.monotonic() < deadline:
            q, qd = feedback.sample()
            if not c.isConnected():
                raise RuntimeError("Control disconnected")
            for v, x, y in zip(q, actual, target):
                if not min(x, y) - math.radians(0.5) <= v <= max(x, y) + math.radians(0.5):
                    raise RuntimeError("Motion exceeded joint envelope")
            if max(abs(v) for v in qd) > 0.06:
                raise RuntimeError("Unexpected joint speed")
            progress = c.getAsyncOperationProgress()
            began = began or progress >= 0 or progress != last_progress
            done = began and progress < 0 and distance(q, target) < TOL and max(abs(v) for v in qd) < 0.005
            stable = stable + 1 if done else 0
            if stable >= 10:
                pending = False
                print("REACHED degrees:", degrees(q), flush=True)
                return q
            time.sleep(0.02)
        raise TimeoutError("Motion exceeded 15 seconds")
    finally:
        if pending:
            print("Attempting stopJ. If ineffective use physical E-stop!", flush=True)
            try:
                c.stopJ(0.5)
            except Exception as exc:
                print("STOP FAILED:", exc, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("record", "run"))
    p.add_argument("--name", choices=("A", "B"), help="record only")
    p.add_argument("--start", choices=("A", "B"), default="A")
    p.add_argument("--one-way", action="store_true", help="run only: omit return leg")
    p.add_argument("--execute", action="store_true", help="run only: allow confirmed motion")
    p.add_argument("--ip", default="192.168.1.101")
    p.add_argument("--poses-dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args(argv)
    folder = args.poses_dir.resolve()
    if args.command == "record" and (not args.name or args.execute or args.one_way or args.start != "A"):
        p.error("record requires --name A/B; no motion flags")
    if args.command == "run" and args.name:
        p.error("--name is only for record")
    a = b = None
    if args.command == "run":
        a, b = load(folder, "A", args.ip), load(folder, "B", args.ip)
        pair(a["q_rad"], b["q_rad"])
        src, dst = (a, b) if args.start == "A" else (b, a)
        print("Route:", src["name"], "->", dst["name"], "" if args.one_way else "-> " + src["name"])
        print("A degrees:", degrees(a["q_rad"]))
        print("B degrees:", degrees(b["q_rad"]))
        print("Speed:", SPEED, "rad/s; acceleration:", ACCEL, "rad/s^2")
        print("No collision checking. Confirm same robot, mounting, tool, payload and clear path.")
        if not args.execute:
            print("OFFLINE PREVIEW ONLY: no robot connection. Does NOT check current position.")
            return
    else:
        if (folder / f"{args.name}.json").exists():
            raise FileExistsError("Pose exists; use a NEW --poses-dir to preserve previous teaching")
        if args.name == "B":
            a = load(folder, "A", args.ip)

    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(args.ip)
    c = None
    try:
        f = Feedback(r)
        # No external program or freedrive is allowed during recording/execution.
        if r.getRobotStatus() & 6:
            raise RuntimeError("Program/freedrive active; stop/release via pendant first")
        if args.command == "record":
            q = f.stationary()
            if a:
                pair(a["q_rad"], q)
            print("Record", args.name, "degrees:", degrees(q))
            print("Record is NOT a safety certification. Verify pendant values and surroundings.")
            confirm("SAVE")
            q = f.stationary(q)
            data = dict(schema=1, name=args.name, ip=args.ip,
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                        q_rad=q, q_deg=degrees(q), tcp_base_m_rotvec=vec(r.getActualTCPPose()))
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"{args.name}.json").open("x") as out:
                json.dump(data, out, indent=2)
                out.write("\n")
            print("SAVED:", folder / f"{args.name}.json")
            return

        f.stationary(src["q_rad"])
        print("Confirm empty tool, correct payload, no other program, safe full swept path, operator at E-stop.")
        confirm("MOVE")
        f.stationary(src["q_rad"])
        if r.getRobotStatus() & 6:
            raise RuntimeError("Program/freedrive became active")
        from rtde_control import RTDEControlInterface
        c = RTDEControlInterface(args.ip)
        motion(c, f, src["q_rad"], dst["q_rad"])
        if not args.one_way:
            print("Inspect full return path before proceeding.")
            confirm("RETURN")
            motion(c, f, dst["q_rad"], src["q_rad"])
        print("COMPLETE")
    finally:
        # Never issue an unconditional move home in cleanup.
        if c is not None:
            try:
                c.stopScript()
            except Exception as exc:
                print("stopScript failed:", exc)
            try:
                c.disconnect()
            except Exception as exc:
                print("Control disconnect failed:", exc)
        r.disconnect()


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ABORTED: {exc}. No automatic return; inspect pendant.", flush=True)
        raise SystemExit(1)
