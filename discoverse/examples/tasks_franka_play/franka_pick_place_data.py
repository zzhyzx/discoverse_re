import os
import argparse
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# 引入 discoverse 相关工具
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.utils import step_func, SimpleStateMachine
from discoverse.task_base import recoder_airbot_play

from discoverse.task_base.airbot_task_base import PyavImageEncoder


# 引入你已经调整好路径的 Franka 基础环境
# 请根据你的实际文件路径修改这里的 import
from discoverse.robots_env.franka_play_base import FrankaPlayCfg, FrankaPlayBase


# ==============================================================================
# 1. Franka 逆运动学求解器 (Numerical IK)
# ==============================================================================
class FrankaIK:
    def __init__(self, mj_model, mj_data, ee_name="hand"):
        """
        基于 MuJoCo 雅可比矩阵的数值逆运动学求解器
        :param ee_name: Franka 末端执行器的 body 名称，通常为 "hand" 或 "panda_hand"
        """
        self.model = mj_model
        self.data = mj_data
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

        # IK 迭代超参数
        self.step_size = 0.5
        self.tol = 1e-4
        self.max_iter = 100
        self.damping = 1e-4

    def properIK(self, pos, ori, ref_q):
        """
        求解 IK
        :param pos: 目标位置 [x, y, z]
        :param ori: 目标姿态 旋转矩阵 (3, 3)
        :param ref_q: 参考关节角度 (7,)，作为迭代的初始猜测值
        """
        qpos_orig = self.data.qpos.copy()
        self.data.qpos[:7] = ref_q[:7]
        mujoco.mj_kinematics(self.model, self.data)

        for _ in range(self.max_iter):
            curr_pos = self.data.xpos[self.ee_id]
            curr_mat = self.data.xmat[self.ee_id].reshape(3, 3)

            err_pos = pos - curr_pos
            R_err = ori @ curr_mat.T
            err_ori = Rotation.from_matrix(R_err).as_rotvec()
            # err = np.concatenate([err_pos, err_ori])
            err = np.concatenate([err_pos, 0.1 * err_ori])

            if np.linalg.norm(err) < self.tol:
                break

            mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.ee_id)
            J = np.vstack([self.jacp[:, :7], self.jacr[:, :7]])

            J_T = J.T
            lambda_I = self.damping * np.eye(6)
            delta_q = J_T @ np.linalg.inv(J @ J_T + lambda_I) @ err

            self.data.qpos[:7] += self.step_size * delta_q
            self.data.qpos[:7] = np.clip(self.data.qpos[:7], self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
            mujoco.mj_kinematics(self.model, self.data)

        result_q = self.data.qpos[:7].copy()
        self.data.qpos[:] = qpos_orig
        mujoco.mj_kinematics(self.model, self.data)
        return result_q


# ==============================================================================
# 2. 任务节点 (继承自 FrankaPlayBase)
# ==============================================================================
class SimNode(FrankaPlayBase):
    def __init__(self, config: FrankaPlayCfg):
        super().__init__(config)
        # 目标控制量：7个手臂关节 + 1个夹爪开合度
        self.target_control = np.zeros(8)
        self.delta_t = self.config.timestep * self.config.decimation

        # 缓存物体 ID (请确保 XML 中有这些名字)
        try:
            self.cup_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "coffee_cup")
            self.plate_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        except Exception as e:
            print(f"警告: 未在 XML 中找到 coffee_cup 或 plate，请检查模型文件！({e})")
            self.cup_id, self.plate_id = -1, -1

        # OpenVLA 语言指令池
        self.language_instructions = [
            "pick up the cup and place it on the plate",
            "grasp the coffee cup and move it to the dish",
            "put the mug onto the plate"
        ]
        self.current_instruction = ""
        self.reset_sig = True
        self.delay_cnt = 0

    def domain_randomization(self):
        """域随机化：随机化杯子位置和语言指令"""
        if self.cup_id != -1:
            cup_adr = self.mj_model.body_jntadr[self.cup_id]
            # 随机化杯子位置 ±3cm
            self.mj_data.qpos[cup_adr:cup_adr + 2] += np.random.uniform(-0.03, 0.03, 2)

        self.current_instruction = np.random.choice(self.language_instructions)

    def check_success(self):
        """检查任务是否成功：杯子和盘子 XY 距离小于 4cm"""
        if self.cup_id == -1 or self.plate_id == -1: return False
        cup_pos = self.mj_data.xpos[self.cup_id]
        plate_pos = self.mj_data.xpos[self.plate_id]
        return np.linalg.norm(cup_pos[:2] - plate_pos[:2]) < 0.04

    def checkActionDone(self):
        """检查机械臂是否已经运动到了目标关节角度"""
        if self.delay_cnt > 0:
            self.delay_cnt -= 1
            return False
        dif = np.abs(self.mj_data.qpos[:7] - self.target_control[:7])
        return np.max(dif) < 0.05


# ==============================================================================
# 3. 主循环与状态机
# ==============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="起始数据索引")
    parser.add_argument("--data_set_size", type=int, default=50, help="需要采集的数据集大小")
    args = parser.parse_args()

    # 初始化配置和环境
    cfg = FrankaPlayCfg()
    sim_node = SimNode(cfg)

    # 初始化 IK 求解器 (注意：Franka 的末端 body 名字通常是 "hand" 或 "panda_hand")
    arm_ik = FrankaIK(sim_node.mj_model, sim_node.mj_data, ee_name="hand")

    # 定义夹爪朝下的旋转矩阵 (Roll=180度, Pitch=0, Yaw=0)
    trmat_down = Rotation.from_euler("xyz", [np.pi, 0, 0], degrees=False).as_matrix()

    # 状态机初始化
    stm = SimpleStateMachine()
    stm.max_state_cnt = 8
    max_time = 15.0  # 最大超时时间 (秒)

    action = np.zeros(8)
    move_speed = 1.0

    # 数据保存路径
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", "franka_openvla_dataset")
    os.makedirs(save_dir, exist_ok=True)

    data_idx = args.data_idx
    data_set_size = args.data_idx + args.data_set_size

    sim_node.reset()

    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            sim_node.domain_randomization()

            # 初始化目标控制量为当前姿态，夹爪张开 (0.04)
            sim_node.target_control[:7] = sim_node.mj_data.qpos[:7]
            sim_node.target_control[7] = 0.04
            action[:] = sim_node.target_control[:]

            act_lst, obs_lst = [], []
            episode_instruction = sim_node.current_instruction

            # 视频编码器初始化
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {cam_id: PyavImageEncoder(cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id)
                        for cam_id in cfg.obs_rgb_cam_id}

        try:
            if stm.trigger():
                cup_pos = sim_node.mj_data.xpos[sim_node.cup_id].copy()
                plate_pos = sim_node.mj_data.xpos[sim_node.plate_id].copy()
                current_q = sim_node.mj_data.qpos[:7]

                if stm.state_idx == 0:  # 1. 移动到杯子正上方 (预抓取)
                    target_pos = cup_pos + np.array([0, 0, 0.20])
                    sim_node.target_control[:7] = arm_ik.properIK(target_pos, trmat_down, current_q)
                    sim_node.target_control[7] = 0.04  # 夹爪张开

                elif stm.state_idx == 1:  # 2. 下降到杯子位置
                    target_pos = cup_pos + np.array([0, 0, 0.02])  # 根据杯子实际高度微调 Z 轴
                    sim_node.target_control[:7] = arm_ik.properIK(target_pos, trmat_down, current_q)

                elif stm.state_idx == 2:  # 3. 闭合夹爪抓取
                    sim_node.target_control[7] = 0.0  # 夹爪闭合
                    sim_node.delay_cnt = int(0.5 / sim_node.delta_t)  # 延时 0.5 秒等待抓稳

                elif stm.state_idx == 3:  # 4. 提起杯子
                    target_pos = cup_pos + np.array([0, 0, 0.25])
                    sim_node.target_control[:7] = arm_ik.properIK(target_pos, trmat_down, current_q)

                elif stm.state_idx == 4:  # 5. 移动到盘子正上方
                    target_pos = plate_pos + np.array([0, 0, 0.25])
                    sim_node.target_control[:7] = arm_ik.properIK(target_pos, trmat_down, current_q)

                elif stm.state_idx == 5:  # 6. 下降到盘子上
                    target_pos = plate_pos + np.array([0, 0, 0.05])  # 盘子上方一点点
                    sim_node.target_control[:7] = arm_ik.properIK(target_pos, trmat_down, current_q)

                elif stm.state_idx == 6:  # 7. 张开夹爪放置
                    sim_node.target_control[7] = 0.04  # 夹爪张开
                    sim_node.delay_cnt = int(0.5 / sim_node.delta_t)  # 延时 0.5 秒等待松开

                elif stm.state_idx == 7:  # 8. 机械臂抬起撤离
                    target_pos = plate_pos + np.array([0, 0, 0.30])
                    sim_node.target_control[:7] = arm_ik.properIK(target_pos, trmat_down, current_q)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")
            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            # print(f"Episode Failed: {ve}")
            sim_node.reset_sig = True

        # ---------------------------------------------------------
        # 动作平滑插值 (Action Smoothing)
        # ---------------------------------------------------------
        # 7个手臂关节平滑
        for i in range(7):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.delta_t)
        # 夹爪动作直接赋值或快速插值
        action[7] = step_func(action[7], sim_node.target_control[7], 0.5 * sim_node.delta_t)

        # 步进物理环境
        obs, pri_obs, rew, ter, info = sim_node.step(action)

        # ---------------------------------------------------------
        # 记录数据 (OpenVLA 格式对齐)
        # ---------------------------------------------------------
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            img = obs.pop("img")

            # 处理图像并送入编码器
            if isinstance(img, np.ndarray):
                encoders[cfg.obs_rgb_cam_id[0]].encode(img, obs["time"])
            elif isinstance(img, dict):
                for cam_id, i in img.items():
                    encoders[cam_id].encode(i, obs["time"])

            # 注入语言指令
            obs["language_instruction"] = episode_instruction
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        # ---------------------------------------------------------
        # 回合结束判定与保存
        # ---------------------------------------------------------
        if stm.state_idx >= stm.max_state_cnt:
            for encoder in encoders.values():
                encoder.close()

            if sim_node.check_success():
                # 使用 discoverse 内置的保存函数
                recoder_airbot_play(save_path, act_lst, obs_lst, cfg)
                print(f"\r成功采集第 {data_idx + 1}/{data_set_size} 条数据 | 指令: {episode_instruction}")
                data_idx += 1
                if data_idx >= data_set_size:
                    break
            else:
                print(f"\r第 {data_idx + 1} 条数据采集失败，重试中...")
                for encoder in encoders.values():
                    encoder.remove_av_file()

            sim_node.reset_sig = True