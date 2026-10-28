"""Independent, bounded large A/B test. Default is OFFLINE preview.
NOT safety-rated; no collision detection. No automatic homing, brake release or gripper control.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
A = [math.radians(x) for x in [44.42, -81.5034, 70.1148, 256.2497, -87.3409, -9.7948]]
B = [0.3492, -1.0919, 1.22771, 4.53726, -1.52446, -0.17095]
SPEED, ACCEL = .03, .05
TOL = math.radians(.15)
MIN_SCALING = .1


def vec(x):
    x = list(x)
    if len(x) != 6 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in x):
        raise ValueError("Expected six finite joint radians")
    return x


def err(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def deg(q):
    return [round(math.degrees(x), 4) for x in q]


def load(path, fallback, ip):
    if path is None:
        return fallback.copy()
    with path.open() as f:
        d = json.load(f)
    d = d.get("snapshot", d)
    if d.get("ip") != ip:
        raise ValueError("Pose IP mismatch")
    return vec(d["q_rad"])


def validate(a, b):
    delta = [abs(x-y) for x, y in zip(vec(a), vec(b))]
    if max(delta) > math.radians(30) or sum(delta) > math.radians(60):
        raise ValueError("Test envelope exceeded: max 30 deg/joint, 60 deg sum")
    if max(delta) < math.radians(.3):
        raise ValueError("A and B are too close")
    # Conservative nominal model bounds; actual controller limits also checked.
    for q in (a,b):
        for i,v in enumerate(q):
            limit = math.pi if i == 2 else 2*math.pi
            if abs(v) > limit:
                raise ValueError("Outside conservative joint bounds; do not wrap angles automatically")


def budget(a,b):
    d = err(a,b)
    nominal = 2*math.sqrt(d/ACCEL) if d <= SPEED*SPEED/ACCEL else d/SPEED+SPEED/ACCEL
    return nominal, min(300.0, max(30.0, nominal/MIN_SCALING*1.5+10))


def confirm(word):
    if input(f"Type {word} (case-insensitive); anything else cancels: ").strip().upper() != word:
        raise RuntimeError("Cancelled; no automatic return")


class Feedback:
    def __init__(self, r):
        self.r = r
        self.stamp = None
        self.fresh = time.monotonic()

    def sample(self):
        r = self.r
        if not r.isConnected():
            raise RuntimeError("Receive disconnected")
        stamp = r.getTimestamp()
        if not math.isfinite(stamp) or (self.stamp is not None and stamp < self.stamp):
            raise RuntimeError("Invalid controller timestamp")
        if self.stamp is None or stamp > self.stamp:
            self.stamp, self.fresh = stamp, time.monotonic()
        if time.monotonic()-self.fresh > .5:
            raise RuntimeError("Stale feedback")
        if r.getRobotMode() != 7 or r.getSafetyMode() not in (1,2):
            raise RuntimeError("Robot/safety state not ready; inspect pendant")
        return vec(r.getActualQ()), vec(r.getActualQd())

    def stationary(self, expected):
        q,_ = self.sample()
        stamp = self.stamp
        for _ in range(30):
            time.sleep(.02)
            q,qd = self.sample()
            if err(q,expected) > TOL or max(abs(v) for v in qd) > .005:
                raise RuntimeError("Not stationary at selected start (0.15 deg tolerance); no automatic reposition")
        if self.stamp <= stamp:
            raise RuntimeError("No fresh data")
        return q


def scaling(r):
    # Zero is valid before a motion program starts. Validate each field,
    # but apply the lower bound only during active motion, with a grace period.
    factors = (r.getSpeedScaling(), r.getTargetSpeedFraction())
    if any(not math.isfinite(v) or not 0 <= v <= 1.001 for v in factors):
        raise RuntimeError("Invalid speed scaling feedback")
    return factors[0] * factors[1]


def wait_program(c, f, source):
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        q, qd = f.sample()
        if err(q, source) > TOL or max(abs(v) for v in qd) > .005:
            raise RuntimeError("Robot moved while waiting for control program")
        if not c.isConnected():
            raise RuntimeError("Control disconnected")
        state = f.r.getRuntimeState()
        if state == 2 and f.r.getRobotStatus() & 2:
            return
        if state not in (1, 2):
            raise RuntimeError("Control program paused/stopping; inspect pendant")
        time.sleep(.02)
    raise RuntimeError("Control program did not become ready within 3 seconds")


def move(c,f,source,target,log):
    actual = f.stationary(source)
    scaling(f.r)
    wait_program(c, f, source)
    if not c.isConnected() or not c.isJointsWithinSafetyLimits(target):
        raise RuntimeError("Control unavailable or target outside controller limits")
    before = c.getAsyncOperationProgress()
    nominal, timeout = budget(actual,target)
    print(f"Nominal time {nominal:.1f}s; timeout {timeout:.1f}s")
    pending = True
    try:
        started = time.monotonic()
        if not c.moveJ(target,SPEED,ACCEL,True):
            raise RuntimeError("moveJ rejected")
        best = err(actual,target)
        last_progress_time = started
        next_print = started
        begun = False
        stable_since = None
        low_scaling_since = None
        while time.monotonic()-started < timeout:
            now = time.monotonic()
            q,qd = f.sample()
            effective_scaling = scaling(f.r)
            if f.r.getRuntimeState() != 2 or not (f.r.getRobotStatus() & 2):
                raise RuntimeError("Control program stopped/paused during motion")
            if not c.isConnected():
                raise RuntimeError("Control disconnected")
            if any(v < min(x,y)-math.radians(.5) or v > max(x,y)+math.radians(.5)
                   for v,x,y in zip(q,actual,target)):
                raise RuntimeError("Joint exceeded motion envelope")
            if max(abs(v) for v in qd) > .06:
                raise RuntimeError("Unexpected joint velocity")
            e = err(q,target)
            if e < best-math.radians(.02):
                best,last_progress_time = e,now
            if e > TOL and now-last_progress_time > 10:
                raise RuntimeError("No measurable progress for 10 seconds")
            progress = c.getAsyncOperationProgress()
            begun = begun or progress >= 0 or progress != before
            done = begun and progress < 0 and e < TOL and max(abs(v) for v in qd) < .005
            # Startup may report zero briefly. Never resume a paused program.
            # A completed motion may also report zero: still require settled q.
            if not done and now-started >= 2.0 and effective_scaling < MIN_SCALING-1e-6:
                low_scaling_since = now if low_scaling_since is None else low_scaling_since
                if now-low_scaling_since >= .5:
                    raise RuntimeError("Effective scaling below 10% for 0.5s after startup; inspect pendant")
            else:
                low_scaling_since = None
            stable_since = (now if stable_since is None else stable_since) if done else None
            log("feedback",q_rad=q,qd=qd,target=target,error_deg=math.degrees(e),effective_scaling=effective_scaling)
            if now >= next_print:
                print(f"elapsed {now-started:.1f}s | max error {math.degrees(e):.3f} deg",flush=True)
                next_print = now+1
            if stable_since is not None and now-stable_since >= .3:
                pending = False
                log("reached",q_rad=q)
                print("REACHED:",deg(q))
                return
            time.sleep(.02)
        raise TimeoutError("Motion time budget exceeded")
    finally:
        if pending:
            print("Attempting stopJ. Physical E-stop required if ineffective!",flush=True)
            try:
                c.stopJ(.5)
            except Exception as exc:
                print("STOP FAILED:",exc,flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ip",default="192.168.1.101")
    p.add_argument("--start",choices=("A","B"),default="B")
    p.add_argument("--one-way",action="store_true")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--a-file",type=Path)
    p.add_argument("--b-file",type=Path)
    args = p.parse_args(argv)
    a,b = load(args.a_file,A,args.ip),load(args.b_file,B,args.ip)
    validate(a,b)
    src,dst = (a,b) if args.start == "A" else (b,a)
    print("A degrees:",deg(a))
    print("B degrees:",deg(b))
    print("B-A degrees:",deg([y-x for x,y in zip(a,b)]))
    print("Route:",args.start,"->",("B" if args.start=="A" else "A"),"" if args.one_way else "-> "+args.start)
    print("Speed .03 rad/s; acceleration .05 rad/s^2; estimated times:",budget(src,dst))
    print("Defaults come from USER-PRINTED rounded values; prefer original --a-file/--b-file snapshots.")
    print("NO collision checking. Preserve angle branches (wrist1 ~260 deg). No safety certification.")
    if not args.execute:
        print("OFFLINE PREVIEW: no RTDE imports, no connection or motion.")
        return
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(args.ip)
    c = None
    log_file = None
    try:
        f = Feedback(r)
        f.stationary(src)
        if r.getRobotStatus() & 6:
            raise RuntimeError("Other program/freedrive active")
        if r.getRuntimeState() != 1:
            raise RuntimeError("Expected stopped program before control startup; inspect pendant")
        scaling(r)
        print("Confirm same robot/tool/payload, clear FULL route and return route, no self-collision/cable snag, operator at E-stop.")
        confirm("MOVE")
        f.stationary(src)
        if r.getRobotStatus() & 6:
            raise RuntimeError("Other program/freedrive became active")
        if r.getRuntimeState() != 1:
            raise RuntimeError("Program state changed before control startup")
        folder = ROOT/"data/real_ur5/large_ab_logs"
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".jsonl")
        log_file = path.open("x",buffering=1)
        def log(event,**values):
            log_file.write(json.dumps(dict(event=event,monotonic=time.monotonic(),**values))+"\n")
        log("start",ip=args.ip,a=a,b=b,start=args.start,one_way=args.one_way)
        print("Log:",path)
        from rtde_control import RTDEControlInterface
        c = RTDEControlInterface(args.ip)
        move(c,f,src,dst,log)
        if not args.one_way:
            print("Stopped at destination. Inspect before return.")
            confirm("RETURN")
            move(c,f,dst,src,log)
        log("complete")
        print("COMPLETE")
    finally:
        if c is not None:
            try:
                c.stopScript()
            except Exception as exc:
                print("stopScript failed:",exc)
            try:
                c.disconnect()
            except Exception as exc:
                print("disconnect failed:",exc)
        if log_file:
            log_file.close()
        r.disconnect()


if __name__ == "__main__":
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        print(f"ABORTED: {exc}. No automatic return. Inspect pendant.",flush=True)
        raise SystemExit(1)
