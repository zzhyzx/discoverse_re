"""UR5/CB3 small joint test. Default: receive-only preview.
Not safety-rated; no collision checking. Physical E-stop/operator required.
"""
import argparse
import math
import time


def vector(values):
    values = list(values)
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise RuntimeError("Invalid six-joint feedback")
    return values


def error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def check(robot, ready=False):
    if not robot.isConnected():
        raise RuntimeError("Receive connection lost")
    if robot.getSafetyMode() not in (1, 2):
        raise RuntimeError("Safety state is not NORMAL/REDUCED; inspect pendant")
    mode = robot.getRobotMode()
    if mode not in ((7,) if ready else (5, 7)):
        raise RuntimeError(f"Robot mode {mode}: motion requires mode 7; prepare via pendant")
    return vector(robot.getActualQ())


def stationary(robot, expected, ready=True):
    for _ in range(10):
        q = check(robot, ready)
        if error(q, expected) > math.radians(0.2):
            raise RuntimeError("Position changed since preview; restart")
        if max(abs(v) for v in vector(robot.getActualQd())) > 0.01:
            raise RuntimeError("Robot is not stationary")
        time.sleep(0.05)


def confirm(word, message):
    print(message, flush=True)
    if input(f"Type {word} to continue (anything else cancels): ").strip() != word:
        raise RuntimeError("Cancelled; no automatic return")


def move(control, robot, start, target):
    stationary(robot, start)
    if not control.isJointsWithinSafetyLimits(target):
        raise RuntimeError("Controller rejected target safety limits")
    # This is NOT an obstacle/path collision check.
    pending = True
    try:
        if not control.moveJ(target, 0.03, 0.05, True):
            raise RuntimeError("moveJ rejected")
        deadline = time.monotonic() + 15.0
        stamp = robot.getTimestamp()
        fresh = time.monotonic()
        settled = 0
        while time.monotonic() < deadline:
            q = check(robot, True)
            now_stamp = robot.getTimestamp()
            if now_stamp != stamp:
                stamp, fresh = now_stamp, time.monotonic()
            if time.monotonic() - fresh > 0.5:
                raise RuntimeError("Stale RTDE feedback")
            if any(v < min(a, b) - math.radians(0.5)
                   or v > max(a, b) + math.radians(0.5)
                   for v, a, b in zip(q, start, target)):
                raise RuntimeError("Joint left test motion envelope")
            stopped = max(abs(v) for v in vector(robot.getActualQd())) < 0.005
            done = control.getAsyncOperationProgress() < 0
            settled = settled + 1 if done and stopped and error(q, target) < math.radians(0.1) else 0
            if settled >= 10:
                pending = False
                print("Reached. Joint degrees:", [round(math.degrees(v), 3) for v in q])
                return q
            time.sleep(0.02)
        raise TimeoutError("Motion timed out")
    finally:
        if pending:
            print("Attempting stopJ; if communication fails use physical E-stop!", flush=True)
            try:
                control.stopJ(0.5)
            except Exception as exc:
                print("STOP FAILED:", exc, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.1.101")
    parser.add_argument("--joint", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--delta-deg", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.delta_deg) or not 0 < abs(args.delta_deg) <= 2:
        parser.error("delta-deg must be nonzero and within [-2, 2]")
    from rtde_receive import RTDEReceiveInterface
    robot = RTDEReceiveInterface(args.ip)
    control = None
    try:
        start = check(robot)
        stationary(robot, start, ready=False)
        target = start.copy()
        target[args.joint - 1] += math.radians(args.delta_deg)
        print("IP:", args.ip, "mode:", robot.getRobotMode(), "safety:", robot.getSafetyMode())
        print("Start degrees:", [round(math.degrees(v), 4) for v in start])
        print("Target degrees:", [round(math.degrees(v), 4) for v in target])
        print("Speed 0.03 rad/s; acceleration 0.05 rad/s^2; timeout 15 s")
        print("No collision detection. No gripper control. No automatic power/brake release.")
        if not args.execute:
            print("PREVIEW ONLY: no control interface or motion command created.")
            return
        confirm("MOVE", "Confirm empty gripper, correct payload, clear swept path, operator at E-stop, no other program.")
        stationary(robot, start)
        if robot.getRobotStatus() & 2:
            raise RuntimeError("Another robot program is running; stop it via pendant first")
        from rtde_control import RTDEControlInterface
        control = RTDEControlInterface(args.ip)
        reached = move(control, robot, start, target)
        confirm("RETURN", "Inspect robot/path before returning to the recorded start.")
        move(control, robot, reached, start)
        print("Round trip complete.")
    finally:
        if control is not None:
            try:
                control.stopScript()
            except Exception as exc:
                print("Control script cleanup failed:", exc)
            control.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ABORTED: {exc}. No automatic return. Inspect pendant.")
        raise SystemExit(1)
