"""Robotiq URCap gripper test via port 63352.

Default: read-only.
Closure percent: 0=open, 100=closed; raw command fraction, not mm.
No robot arm commands, automatic activation, reset or calibration.
"""

import argparse
import math
import select
import socket
import time


class Gripper:
    def __init__(self, ip, port):
        self.sock = socket.create_connection((ip, port), timeout=1)
        self.sock.settimeout(1)

    def close(self):
        self.sock.close()

    def request(self, command):
        self.sock.sendall((command + "\n").encode("ascii"))
        data = self.sock.recv(1024)
        if not data:
            raise ConnectionError("Gripper socket closed")

        deadline = time.monotonic() + 1
        while b"\n" not in data and time.monotonic() < deadline:
            if not select.select([self.sock], [], [], 0.05)[0]:
                break
            chunk = self.sock.recv(1024)
            if not chunk:
                raise ConnectionError("Socket closed mid-response")
            data += chunk
            if len(data) > 1024:
                raise RuntimeError("Oversized URCap response")

        return data.decode("ascii").strip()

    def get(self, name):
        text = self.request("GET " + name)
        parts = text.split()
        if len(parts) != 2 or parts[0] != name:
            raise RuntimeError(
                f"Unexpected response to GET {name}: {text!r}"
            )
        value = int(parts[1])
        if not 0 <= value <= 255:
            raise RuntimeError(f"Invalid register {name}: {value}")
        return value

    def set(self, **values):
        command = "SET " + " ".join(
            f"{key} {value}" for key, value in values.items()
        )
        if self.request(command) != "ack":
            raise RuntimeError(
                "SET not acknowledged; command may already have executed"
            )

    def status(self):
        result = {
            key: self.get(key)
            for key in ("ACT", "STA", "FLT", "POS", "PRE", "OBJ", "GTO")
        }
        if result["OBJ"] not in (0, 1, 2, 3):
            raise RuntimeError("Invalid OBJ status")
        return result


def ready(status):
    if (
        status["ACT"] != 1
        or status["STA"] != 3
        or status["FLT"] != 0
    ):
        raise RuntimeError(
            "Gripper not ready or fault present. Inspect pendant; "
            "no automatic activation."
        )
    if status["GTO"] == 1 and status["OBJ"] == 0:
        raise RuntimeError(
            "Gripper already moving; stop other control first"
        )


def percent(value):
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("close-percent must be within 0..100")
    return round(value * 255 / 100)


def check_direction(mode, target, position):
    if mode == "grasp" and target <= position + 3:
        raise RuntimeError(
            "Grasp requires a closing target beyond current position. "
            "Start with the object between open fingers."
        )
    if mode == "release" and target >= position - 3:
        raise RuntimeError(
            "Release requires an opening target below current position."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.1.101")
    parser.add_argument("--port", type=int, default=63352)
    parser.add_argument(
        "--mode",
        choices=("empty", "grasp", "release"),
        default="empty",
    )
    parser.add_argument("--close-percent", type=float)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if args.execute and args.close_percent is None:
        parser.error("--execute requires --close-percent")

    target = (
        percent(args.close_percent)
        if args.close_percent is not None
        else None
    )

    gripper = Gripper(args.ip, args.port)
    pending = False

    try:
        initial = gripper.status()
        print("Registers:", initial)
        print(
            f"POS={initial['POS']}, "
            f"raw closure fraction={initial['POS'] / 255 * 100:.1f}%"
        )

        if target is None:
            print("READ ONLY: no SET commands.")
            return

        ready(initial)
        check_direction(args.mode, target, initial["POS"])

        print(
            f"Mode={args.mode}, target={args.close_percent}% "
            f"(POS={target}), SPE=32, FOR=16"
        )
        print(
            "Raw percentages are not calibrated opening width. "
            "No arm commands or automatic activation."
        )

        if not args.execute:
            print("PREVIEW ONLY: no SET commands.")
            return

        if args.mode == "grasp":
            print(
                "Confirm object between fingers, hands clear, "
                "arm stationary. Contact does NOT prove secure grasp."
            )
        elif args.mode == "release":
            print(
                "Confirm object is supported at the placement surface, "
                "hands clear and arm stationary."
            )
        else:
            print(
                "Confirm EMPTY gripper, hands clear and arm stationary."
            )

        print("Do not operate the pendant gripper controls simultaneously.")
        if input("Type MOVE to execute: ").strip().upper() != "MOVE":
            print("Cancelled; no SET commands.")
            return

        latest = gripper.status()
        ready(latest)
        if abs(latest["POS"] - initial["POS"]) > 3:
            raise RuntimeError("Gripper position changed after preview")
        check_direction(args.mode, target, latest["POS"])

        pending = True
        started = time.monotonic()
        gripper.set(POS=target, SPE=32, FOR=16, GTO=1)

        observed_motion = False
        previous_position = None
        previous_terminal = None
        stable_count = 0

        while time.monotonic() - started < 15:
            status = gripper.status()
            print(status, flush=True)

            if (
                status["FLT"] != 0
                or status["ACT"] != 1
                or status["STA"] != 3
            ):
                raise RuntimeError("Gripper fault or lost activation")

            if status["PRE"] != target:
                # Wait briefly for this command to be reflected.
                if time.monotonic() - started > 2:
                    raise RuntimeError(
                        "Requested target not reflected in PRE; "
                        "possible competing controller"
                    )
                time.sleep(0.1)
                continue

            if status["GTO"] != 1:
                raise RuntimeError("Motion was stopped or overridden")

            obj = status["OBJ"]
            observed_motion = (
                observed_motion
                or obj == 0
                or abs(status["POS"] - latest["POS"]) > 2
            )

            terminal = None

            if obj == 1:
                raise RuntimeError(
                    "Contact while opening; inspect obstruction"
                )

            if obj == 2:
                if args.mode != "grasp":
                    raise RuntimeError(
                        "Unexpected closing contact outside grasp mode"
                    )
                # Avoid accepting an unchanged contact status from before
                # this command. If uncertain, timeout rather than lift.
                if observed_motion:
                    terminal = "contact"

            elif obj == 3:
                if observed_motion or abs(status["POS"] - target) <= 3:
                    terminal = "endpoint"

            if (
                terminal is not None
                and terminal == previous_terminal
                and previous_position is not None
                and abs(status["POS"] - previous_position) <= 1
            ):
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= 3:
                if args.mode == "grasp":
                    if terminal != "contact":
                        raise RuntimeError(
                            "Grasp failed: reached endpoint without "
                            "closing-contact detection. Do not lift."
                        )
                    pending = False
                    print(
                        "CONTACT DETECTED: holding current grasp. "
                        "No automatic arm motion. Verify grip before lifting."
                    )
                    return

                pending = False
                print(
                    f"DONE: stopped without contact; POS={status['POS']}. "
                    "Verify physical opening/release."
                )
                return

            previous_position = status["POS"]
            previous_terminal = terminal
            time.sleep(0.1)

        raise TimeoutError("No confirmed completion within 15 seconds")

    finally:
        if pending:
            print(
                "Attempting GTO=0 stop. No automatic reopening. "
                "Network failure may prevent stop.",
                flush=True,
            )
            try:
                gripper.set(GTO=0)
            except Exception as exc:
                print("STOP NOT CONFIRMED:", exc)
        gripper.close()


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print("ABORTED:", exc)
        raise SystemExit(1)