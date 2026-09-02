import os
import json
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.utils import step_func, SimpleStateMachine

from discoverse.robots_env.ur5_base import UR5Cfg, UR5Base
from discoverse.task_base.airbot_task_base import PyavImageEncoder
from discoverse.robots.ur5.ur5_ik import UR5IK


cfg = UR5Cfg()
cfg.mjcf_file_path = "mjcf/ur5/scene.xml"
cfg.timestep = 1 / 240
cfg.decimation = 4
cfg.sync = True
cfg.headless = False
cfg.render_set = {"fps": 20, "width": 640, "height": 480}
cfg.obs_rgb_cam_id = [0]
cfg.use_gaussian_renderer = False


class SimNode(UR5Base):
    """UR5 夹取放置任务节点"""

    COLORS = ["red", "blue", "green"]
    INSTRUCTIONS = {
        "red":   "Pick up the red block and place it in the target zone",
        "blue":  "Pick up the blue block and place it in the target zone",
        "green": "Pick up the green block and place it in the target zone",
    }

    def __init__(self, config: UR5Cfg):
        super().__init__(config)

        self.target_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        self.block_ids = {
            c: mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{c}_block")
            for c in self.COLORS
        }

        self.current_color = "red"
        self.current_instruction = self.INSTRUCTIONS["red"]

        # 控制目标：6 关节 + 1 夹爪
        self.target_control = np.zeros(7)
        self.delay_cnt = 0
        self.reset_sig = True
        self.running = True

        # 记录各方块初始位置（用于 domain_randomization）
        self.block_init_pos = {}
        for c, bid in self.block_ids.items():
            adr = self.mj_model.jnt_qposadr[self.mj_model.body_jntadr[bid]]
            self.block_init_pos[c] = self.mj_data.qpos[adr:adr + 3].copy()

    def domain_randomization(self):
        # 随机各方块初始位置
        for c, bid in self.block_ids.items():
            adr = self.mj_model.jnt_qposadr[self.mj_model.body_jntadr[bid]]
            self.mj_data.qpos[adr:adr + 3] = self.block_init_pos[c] + np.array([
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.02, 0.02),
                0.0,
            ])
        # 随机选语言指令 + 目标方块
        self.current_color = np.random.choice(self.COLORS)
        self.current_instruction = self.INSTRUCTIONS[self.current_color]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def resetState(self):
        super().resetState()
        self.target_control[:6] = self.mj_data.qpos[:6]
        self.target_control[6] = 0.04  # 夹爪张开
        self.delay_cnt = 0
        self.domain_randomization()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.reset_sig = True

    def check_success(self):
        bid = self.block_ids[self.current_color]
        block = self.mj_data.xpos[bid]
        target = self.mj_data.xpos[self.target_id]
        return np.linalg.norm(block[:2] - target[:2]) < 0.05

    def checkActionDone(self):
        joint_diff = np.abs(self.mj_data.qpos[:6] - self.target_control[:6])
        joint_done = np.max(joint_diff) < 0.06
        self.delay_cnt -= 1
        delay_done = (self.delay_cnt <= 0)
        return joint_done and delay_done

    def _block_pos(self):
        return self.mj_data.xpos[self.block_ids[self.current_color]].copy()

    def _target_pos(self):
        return self.mj_data.xpos[self.target_id].copy()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0)
    parser.add_argument("--data_set_size", type=int, default=300)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", "ur5_pick_place")
    os.makedirs(save_dir, exist_ok=True)

    sim_node = SimNode(cfg)
    ik = UR5IK(sim_node.mj_model, sim_node.mj_data, ee_name="base_mount")
    down_rot = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()

    stm = SimpleStateMachine()
    stm.max_state_cnt = 8
    max_time = 20.0
    state_timeout = 5.0

    action = np.zeros(7)
    move_speed = 0.8

    obs = sim_node.reset()
    state_start_time = 0.0
    success_time = None
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            state_start_time = sim_node.mj_data.time
            success_time = None

        try:
            if stm.trigger():
                state_start_time = sim_node.mj_data.time
                if stm.state_idx < stm.max_state_cnt:
                    block_pos = sim_node._block_pos()
                    target_pos = sim_node._target_pos()
                    q = sim_node.mj_data.qpos[:6]

                    # 末端执行器目标位置 + 夹爪开合度
                    if stm.state_idx == 0:      # 移到物块上方
                        ee_pos = block_pos + np.array([0.0, 0.0, 0.15])
                        sim_node.target_control[6] = 0.04   # 张开
                    elif stm.state_idx == 1:    # 下降到物块（手指包住）
                        ee_pos = block_pos + np.array([0.0, 0.0, 0.02])
                    elif stm.state_idx == 2:    # 闭合夹爪抓取
                        ee_pos = block_pos + np.array([0.0, 0.0, 0.02])
                        sim_node.target_control[6] = 0.0    # 闭合
                        sim_node.delay_cnt = int(0.5 / sim_node.delta_t)
                    elif stm.state_idx == 3:    # 抬起物块
                        ee_pos = block_pos + np.array([0.0, 0.0, 0.15])
                    elif stm.state_idx == 4:    # 移到目标区上方
                        ee_pos = target_pos + np.array([0.0, 0.0, 0.15])
                    elif stm.state_idx == 5:    # 下降到目标区
                        ee_pos = target_pos + np.array([0.0, 0.0, 0.03])
                    elif stm.state_idx == 6:    # 张开夹爪释放
                        ee_pos = target_pos + np.array([0.0, 0.0, 0.03])
                        sim_node.target_control[6] = 0.04
                        sim_node.delay_cnt = int(0.5 / sim_node.delta_t)
                    elif stm.state_idx == 7:    # 抬起撤离
                        ee_pos = target_pos + np.array([0.0, 0.0, 0.15])

                    sim_node.target_control[:6] = ik.properIK(ee_pos, down_rot, q)
                    if stm.state_idx == 0:
                        print(f"[state0] block_pos={block_pos}, ee_pos={ee_pos}, q={q}, q_sol={sim_node.target_control[:6]}")
                    dif = np.abs(action[:6] - sim_node.target_control[:6])
                    sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")
            elif sim_node.mj_data.time - state_start_time > state_timeout:
                stm.next()
            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError:
            sim_node.reset()

        # 动作平滑（6 关节）
        for i in range(6):
            action[i] = step_func(
                action[i], sim_node.target_control[i],
                move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t,
            )
        action[6] = sim_node.target_control[6]  # 夹爪直接赋值

        obs, _, _, _, _ = sim_node.step(action)

        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            obs_lst.append(obs)
            act_lst.append(action.tolist().copy())

        # 成功后延时 2 秒保存
        if sim_node.check_success():
            if success_time is None:
                success_time = sim_node.mj_data.time
            elif sim_node.mj_data.time - success_time > 2.0:
                save_path = os.path.join(save_dir, f"{data_idx:03d}")
                os.makedirs(save_path, exist_ok=True)

                with open(os.path.join(save_path, "obs_action.json"), "w") as f:
                    json.dump({
                        "time": [o["time"] for o in obs_lst],
                        "obs": {
                            "qpos": [o["qpos"].tolist() for o in obs_lst],
                            "qvel": [o["qvel"].tolist() for o in obs_lst],
                        },
                        "act": act_lst,
                        "language_instruction": sim_node.current_instruction,
                    }, f)

                encoder = PyavImageEncoder(
                    cfg.render_set["width"], cfg.render_set["height"], save_path, 0
                )
                for o in obs_lst:
                    encoder.encode(o["img"][0], o["time"])
                encoder.close()

                print(f"\r✅ {data_idx}/{data_set_size} | {sim_node.current_instruction}", end="", flush=True)
                data_idx += 1
                if data_idx >= data_set_size:
                    break
                obs = sim_node.reset()
                success_time = None
        else:
            success_time = None

    print("\n数据采集完成")
