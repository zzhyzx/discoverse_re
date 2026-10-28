import argparse
import math
import time

from rtde_receive import RTDEReceiveInterface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.101")
    args = parser.parse_args()

    robot = RTDEReceiveInterface(args.ip)

    try:
        for i in range(5):
            if not robot.isConnected():
                raise RuntimeError("机器人连接已断开")

            q = robot.getActualQ()
            qd = robot.getActualQd()
            tcp = robot.getActualTCPPose()

            print(f"\n=== 状态 {i + 1}/5 ===")
            print("关节角 rad:", [round(v, 5) for v in q])
            print("关节角 deg:", [round(math.degrees(v), 2) for v in q])
            print("关节速度 rad/s:", [round(v, 5) for v in qd])
            print("TCP [x,y,z,rx,ry,rz]:", [round(v, 5) for v in tcp])
            print("Robot mode:", robot.getRobotMode())
            print("Safety mode:", robot.getSafetyMode())

            time.sleep(1)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()