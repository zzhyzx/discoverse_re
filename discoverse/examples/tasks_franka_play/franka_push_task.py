import os
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.utils import step_func, SimpleStateMachine, get_body_tmat

from discoverse.robots_env.franka_play_base import FrankaPlayCfg
from discoverse.task_base.franka_task_base import FrankaTaskBase, recoder_franka

from discoverse.task_base import recoder_airbot_play
from discoverse.task_base.airbot_task_base import PyavImageEncoder

from discoverse.robots.franka_play.franka_play_ik import FrankaIK

import time

cfg = FrankaPlayCfg()
cfg.mjcf_file_path = "mjcf/franka_emika_panda/franka_push_task.xml"

cfg.timestep = 1 / 240
cfg.decimation = 4
cfg.sync = True
cfg.headless = False

cfg.render_set = {
    "fps": 20,
    "width": 640,
    "height": 480
}
cfg.obs_rgb_cam_id = [0]

# =========================================================
# 任务定义
# =========================================================
class SimNode(FrankaTaskBase):
    def __init__(self, config: FrankaPlayCfg):
        super().__init__(config)

        self.mouse_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "mouse"
        )
        self.pad_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "mousepad"
        )

        self.target_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_zone"
        )


        self.base_name = "link0"

        # ⭐ 正确 randomization 用 qpos
        self.mouse_qpos_adr = self.mj_model.jnt_qposadr[
            self.mj_model.body_jntadr[self.mouse_id]
        ]

        self.mouse_init_pos = self.mj_data.qpos[
            self.mouse_qpos_adr:self.mouse_qpos_adr + 3
        ].copy()

    def domain_randomization(self):
        pos = self.mj_data.qpos[
            self.mouse_qpos_adr:self.mouse_qpos_adr + 3
        ]

        pos[:] = self.mouse_init_pos + np.array([
            np.random.uniform(-0.02, 0.02),
            np.random.uniform(-0.02, 0.02),
            0.0
        ])

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def check_success(self):
        mouse = self.mj_data.xpos[self.mouse_id]
        target = self.mj_data.xpos[self.target_id]

        print(np.linalg.norm(mouse[:2] - target[:2]))

        return np.linalg.norm(mouse[:2] - target[:2]) < 0.05


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":

    sim = SimNode(cfg)
    ik = FrankaIK(sim.mj_model, sim.mj_data, "hand")

    down_rot = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()

    stm = SimpleStateMachine()
    stm.max_state_cnt = 8

    action = np.zeros(8)
    move_speed = 0.8

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", "franka_push")
    os.makedirs(save_dir, exist_ok=True)

    data_idx = 0
    data_set_size = 100

    max_epoch_time = 20  # 每个epoch最多20秒

    sim.reset()

    while sim.running:

        print(stm.state_idx)

        if sim.reset_sig:
            episode_start_time = sim.mj_data.time
            sim.reset_sig = False
            stm.reset()

            action[:] = sim.target_control[:]

            act_lst = []
            obs_lst = []

            save_path = os.path.join(save_dir, f"{data_idx:03d}")
            os.makedirs(save_path, exist_ok=True)

            encoders = {
                cam_id: PyavImageEncoder(
                    cfg.render_set["width"],
                    cfg.render_set["height"],
                    save_path,
                    cam_id
                ) for cam_id in cfg.obs_rgb_cam_id
            }

        try:
            # ===============================
            # ⭐ base坐标
            # ===============================
            tmat_base = get_body_tmat(sim.mj_data, "link0")
            tmat_base_inv = np.linalg.inv(tmat_base)

            q = sim.mj_data.qpos[:7]

            if stm.trigger():


                # ===========================
                # ⭐ state 0 初始化轨迹
                # ===========================
                if stm.state_idx == 0:
                    tmat_mouse = get_body_tmat(sim.mj_data, "mouse")
                    t_local = tmat_base_inv @ tmat_mouse
                    t_local[:3, 3] += np.array([-0.07, 0.0, 0.15])  # ⭐ 更高

                elif stm.state_idx == 1:
                    t_local[:3, 3] += np.array([0.02, 0.0, -0.03])

                elif stm.state_idx == 2:
                    t_local[:3, 3] += np.array([0.05, 0.0, 0.0])

                elif stm.state_idx == 3:
                    t_local[:3, 3] += np.array([0.06, 0.0, 0.0])

                elif stm.state_idx == 4:
                    t_local[:3, 3] += np.array([0.06, 0.0, 0.0])

                elif stm.state_idx == 5:
                    t_local[:3, 3] += np.array([0.05, 0.0, 0.0])

                elif stm.state_idx == 6:
                    tmat_pad = get_body_tmat(sim.mj_data, "mousepad")
                    t_local = tmat_base_inv @ tmat_pad
                    t_local[:3, 3] += np.array([-0.03, 0.0, 0.15])

                elif stm.state_idx == 7:
                    t_local[:3, 3] += np.array([0.0, 0.0, 0.08])

                # ===========================
                # IK
                # ===========================
                sim.target_control[:7] = ik.properIK(
                    t_local[:3, 3],
                    down_rot,
                    q
                )

                dif = np.abs(action[:7] - sim.target_control[:7])
                sim.joint_move_ratio[:7] = dif / (np.max(dif) + 1e-6)

            else:
                stm.update()

            if sim.checkActionDone():
                stm.next()

        except Exception as e:
            print("❌ Error:", e)
            sim.reset()
            continue

        # ===============================
        # 平滑控制
        # ===============================
        for i in range(7):
            action[i] = step_func(
                action[i],
                sim.target_control[i],
                move_speed * sim.joint_move_ratio[i] * sim.delta_t
            )

        action[7] = 0.04

        obs, _, _, _, _ = sim.step(action)

        print(sim.mj_data.time, episode_start_time)

        if sim.mj_data.time - episode_start_time > max_epoch_time:

            print("⏰ Episode 超时，强制 reset")

            for encoder in encoders.values():
                encoder.close()
                encoder.remove_av_file()

            sim.reset()
            continue

        # 机械臂状态（有些环境可能已经有）
        obs["qpos"] = sim.mj_data.qpos[:7].copy()
        obs["qvel"] = sim.mj_data.qvel[:7].copy()

        # ⭐ 任务相关（必须！）
        obs["mouse_pos"] = sim.mj_data.xpos[sim.mouse_id].copy()
        obs["target_pos"] = sim.mj_data.xpos[sim.target_id].copy()

        # ===============================
        # 记录
        # ===============================
        if len(obs_lst) < sim.mj_data.time * cfg.render_set["fps"]:
            imgs = obs.pop("img")

            for cam_id, img in imgs.items():
                encoders[cam_id].encode(img, obs["time"])

            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        # ===============================
        # episode结束
        # ===============================
        if stm.state_idx >= stm.max_state_cnt:

            for encoder in encoders.values():
                encoder.close()

            if sim.check_success():
                recoder_franka(save_path, act_lst, obs_lst, cfg)
                print(f"✅ 成功 {data_idx}")
                data_idx += 1

                if data_idx >= data_set_size:
                    break
            else:
                print("❌ 失败")
                for encoder in encoders.values():
                    encoder.remove_av_file()

            sim.reset()