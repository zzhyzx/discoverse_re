import os
import json
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.utils import step_func, SimpleStateMachine, get_body_tmat

from discoverse.robots_env.franka_play_base import FrankaPlayCfg
from discoverse.task_base.franka_task_base import FrankaTaskBase
from discoverse.task_base.airbot_task_base import PyavImageEncoder
from discoverse.robots.franka_play.franka_play_ik import FrankaIK


cfg = FrankaPlayCfg()
cfg.mjcf_file_path = "mjcf/franka_emika_panda/franka_multiblock_push.xml"
cfg.timestep = 1 / 240
cfg.decimation = 4
cfg.sync = True
cfg.headless = False
cfg.render_set = {"fps": 20, "width": 640, "height": 480}
cfg.obs_rgb_cam_id = [0]
cfg.use_gaussian_renderer = False


class SimNode(FrankaTaskBase):
    """三色方块语言条件推物任务节点"""

    COLORS = ["red", "blue", "green"]
    INSTRUCTIONS = {
        "red":   "Push the red block to the target zone",
        "blue":  "Push the blue block to the target zone",
        "green": "Push the green block to the target zone",
    }

    def __init__(self, config: FrankaPlayCfg):
        super().__init__(config)

        self.target_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        self.block_ids = {
            c: mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{c}_block")
            for c in self.COLORS
        }

        self.current_color = "red"
        self.current_instruction = self.INSTRUCTIONS["red"]

        # 记录各方块初始位置（用于 domain_randomization）
        self.block_init_pos = {}
        for c, bid in self.block_ids.items():
            adr = self.mj_model.jnt_qposadr[self.mj_model.body_jntadr[bid]]
            self.block_init_pos[c] = self.mj_data.qpos[adr:adr + 3].copy()

    def domain_randomization(self):
        # 随机各方块初始位置（xy 小幅扰动）
        for c, bid in self.block_ids.items():
            adr = self.mj_model.jnt_qposadr[self.mj_model.body_jntadr[bid]]
            self.mj_data.qpos[adr:adr + 3] = self.block_init_pos[c] + np.array([
                np.random.uniform(-0.03, 0.03),
                np.random.uniform(-0.02, 0.02),
                0.0,
            ])
        # 随机选一条语言指令 + 对应目标方块
        self.current_color = np.random.choice(self.COLORS)
        self.current_instruction = self.INSTRUCTIONS[self.current_color]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def check_success(self):
        bid = self.block_ids[self.current_color]
        block = self.mj_data.xpos[bid]
        target = self.mj_data.xpos[self.target_id]
        dist = np.linalg.norm(block[:2] - target[:2])
        return dist < 0.05

    def checkActionDone(self):
        # 放宽阈值：推动时关节受反作用力，0.03 太严会卡住
        joint_diff = np.abs(self.mj_data.qpos[:7] - self.target_control[:7])
        return np.max(joint_diff) < 0.06

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

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", "franka_multiblock_push")
    os.makedirs(save_dir, exist_ok=True)

    sim_node = SimNode(cfg)
    ik = FrankaIK(sim_node.mj_model, sim_node.mj_data, "hand")
    down_rot = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()

    stm = SimpleStateMachine()
    stm.max_state_cnt = 8
    max_time = 20.0
    state_timeout = 5.0

    action = np.zeros(8)
    move_speed = 0.8
    process_list = []

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
                print(f"  → state {stm.state_idx} | {sim_node.current_color}", flush=True)
                block_pos = sim_node._block_pos()
                target_pos = sim_node._target_pos()
                dist_vec = target_pos[:2] - block_pos[:2]  # 方块→目标区的 xy 向量
                tmat_base = get_body_tmat(sim_node.mj_data, "link0")
                tmat_base_inv = np.linalg.inv(tmat_base)
                q = sim_node.mj_data.qpos[:7]

                # 末端执行器目标（base 坐标系），每状态基于当前方块位置重新计算（闭环）
                t_local = tmat_base_inv @ get_body_tmat(
                    sim_node.mj_data, f"{sim_node.current_color}_block"
                )

                if stm.state_idx == 0:      # 移到方块后方上方
                    t_local[:3, 3] += np.array([-0.07, 0.0, 0.15])
                elif stm.state_idx == 1:    # 下降到推动高度（方块上方）
                    t_local[:3, 3] += np.array([-0.05, 0.0, 0.10])
                elif stm.state_idx == 2:    # 推第一段（沿 xy 向量 40%）
                    push = dist_vec * 0.4
                    t_local[:3, 3] += np.array([push[0], push[1], 0.10])
                elif stm.state_idx == 3:    # 推第二段（沿 xy 向量 50%）
                    push = dist_vec * 0.5
                    t_local[:3, 3] += np.array([push[0], push[1], 0.10])
                elif stm.state_idx == 4:    # 推第三段（沿 xy 向量 70%）
                    push = dist_vec * 0.7
                    t_local[:3, 3] += np.array([push[0], push[1], 0.10])
                elif stm.state_idx == 5:    # 推到位（收尾，余量）
                    push = dist_vec + np.array([0.02, 0.0])
                    t_local[:3, 3] += np.array([push[0], push[1], 0.10])
                elif stm.state_idx == 6:    # 垂直抬起（先升高避免碰方块）
                    t_local[:3, 3] += np.array([0.0, 0.0, 0.25])
                elif stm.state_idx == 7:    # 水平撤离（高处）
                    t_local[:3, 3] += np.array([-0.06, 0.0, 0.25])

                sim_node.target_control[:7] = ik.properIK(t_local[:3, 3], down_rot, q)
                dif = np.abs(action[:7] - sim_node.target_control[:7])
                sim_node.joint_move_ratio[:7] = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")
            elif sim_node.mj_data.time - state_start_time > state_timeout:
                stm.next()  # 状态超时，强制进入下一状态
            else:
                stm.update()

            if sim_node.checkActionDone():
                jd = np.max(np.abs(sim_node.mj_data.qpos[:7] - sim_node.target_control[:7]))
                print(f"  ← state {stm.state_idx} 完成 (joint_diff={jd:.3f})", flush=True)
                stm.next()

        except ValueError:
            sim_node.reset()

        for i in range(7):
            action[i] = step_func(
                action[i], sim_node.target_control[i],
                move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t,
            )
        action[7] = 0.04

        obs, _, _, _, _ = sim_node.step(action)

        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            obs_lst.append(obs)
            act_lst.append(action.tolist().copy())

        # 成功后等待 2 秒再保存（多记录后续动作，让视频更完整）
        if sim_node.check_success():
            if success_time is None:
                success_time = sim_node.mj_data.time
            elif sim_node.mj_data.time - success_time > 2.0:
                save_path = os.path.join(save_dir, f"{data_idx:03d}")
                os.makedirs(save_path, exist_ok=True)

                # 保存低维数据（含语言指令）
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

                # 保存视频（用 PyavImageEncoder，与 franka_push 一致）
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
