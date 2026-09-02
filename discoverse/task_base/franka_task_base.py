import numpy as np
import mujoco
import os
from scipy.spatial.transform import Rotation
import json

from discoverse.robots_env.franka_play_base import FrankaPlayBase, FrankaPlayCfg


class FrankaTaskBase(FrankaPlayBase):

    def __init__(self, config: FrankaPlayCfg):
        super().__init__(config)

        # ===== 控制目标 =====
        self.target_control = np.zeros(8)   # 7 joints + gripper
        self.joint_move_ratio = np.ones(7)

        # ===== 状态控制 =====
        self.delay_cnt = 0
        self.reset_sig = True
        self.running = True

        self.action_done_dict = {
            "joint": False,
            "gripper": False,
            "delay": False,
        }

    # =========================================
    # RESET
    # =========================================
    def resetState(self):
        super().resetState()

        # 初始化控制目标
        self.target_control[:] = 0
        self.target_control[:7] = self.mj_data.qpos[:7]
        self.target_control[7] = 0.04  # gripper open

        self.delay_cnt = 0

        # 随机化
        self.domain_randomization()

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.reset_sig = True

    # =========================================
    # DOMAIN RANDOMIZATION（可覆盖）
    # =========================================
    def domain_randomization(self):
        pass

    # =========================================
    # ACTION DONE 判定（核心）
    # =========================================
    def checkActionDone(self):

        # 关节误差
        joint_diff = np.abs(self.mj_data.qpos[:7] - self.target_control[:7])
        joint_done = np.max(joint_diff) < 0.03

        # 夹爪误差
        # gripper_diff = abs(self.mj_data.qpos[7] - self.target_control[7])
        # gripper_done = gripper_diff < 0.01

        gripper_done = True

        # 延时
        self.delay_cnt -= 1
        delay_done = (self.delay_cnt <= 0)

        self.action_done_dict = {
            "joint": joint_done,
            "gripper": gripper_done,
            "delay": delay_done,
        }

        return joint_done and gripper_done and delay_done

    # =========================================
    # SUCCESS 判定（必须 override）
    # =========================================
    def check_success(self):
        raise NotImplementedError

    # =========================================
    # DEBUG
    # =========================================
    def printMessage(self):
        print("target_control:", self.target_control)
        print("action_done:", self.action_done_dict)




def recoder_franka(save_path, act_lst, obs_lst, cfg: FrankaPlayCfg):
        os.makedirs(save_path, exist_ok=True)

        time = []
        qpos = []
        qvel = []
        mouse_pos = []
        target_pos = []

        for obs in obs_lst:
            time.append(obs["time"])
            qpos.append(obs["qpos"].tolist())
            qvel.append(obs["qvel"].tolist())

            # ⭐ 关键：你需要在 obs 里提前放这些
            mouse_pos.append(obs["mouse_pos"].tolist())
            target_pos.append(obs["target_pos"].tolist())

        data = {
            "time": time,
            "observations": {
                "qpos": qpos,
                "qvel": qvel,
                "mouse_pos": mouse_pos,
                "target_pos": target_pos,
            },
            "act": act_lst,
        }

        with open(os.path.join(save_path, "obs_action.json"), "w") as f:
            json.dump(data, f)

        print(f"✅ Saved to {save_path}")