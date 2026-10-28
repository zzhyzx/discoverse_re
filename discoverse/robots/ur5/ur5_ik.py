import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class UR5IK:
    def __init__(self, mj_model, mj_data, ee_name="gripper_pinch", is_site=True):
        """
        基于 MuJoCo 雅可比矩阵的数值逆运动学求解器 (Damped Least Squares)

        :param mj_model: mujoco 的 model
        :param mj_data: mujoco 的 data
        :param ee_name: 末端执行器的名字 (通常是 body 或 site（Robotiq 默认使用 site "gripper_pinch"）)
        :param is_site: ee_name 是否是一个 site
        """
        self.model = mj_model
        self.data = mj_data
        self.is_site = is_site

        if self.is_site:
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
        else:
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)

        # 预分配雅可比矩阵内存
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

        # IK 迭代超参数
        self.step_size = 0.5  # 迭代步长
        self.tol = 1e-4  # 误差容忍度 (米/弧度)
        self.max_iter = 200  # 最大迭代次数
        self.damping = 1e-3  # 提高阻尼，减少奇异位形附近的分支跳变

    def properIK(self, pos, ori, ref_q):
        """
        求解逆运动学
        :param pos: 目标位置 [x, y, z] (3,)
        :param ori: 目标姿态 旋转矩阵 (3, 3)
        :param ref_q: 参考关节角度 (6,)，作为迭代的初始猜测值
        :return: 求解出的 6 个关节角度 (6,)
        """
        assert len(pos) == 3 and ori.shape == (3, 3), "Invalid pos or ori shape"
        assert ref_q is not None and len(ref_q) >= 6, "ref_q must be provided for numerical IK"

        # 1. 备份当前的物理状态，以免 IK 求解破坏了主循环的仿真状态
        qpos_orig = self.data.qpos.copy()

        # 2. 将初始猜测值赋给模型
        self.data.qpos[:6] = ref_q[:6]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        # 3. 开始迭代求解
        for _ in range(self.max_iter):
            # 获取当前末端的位姿
            if self.is_site:
                curr_pos = self.data.site_xpos[self.ee_id]
                curr_mat = self.data.site_xmat[self.ee_id].reshape(3, 3)
            else:
                curr_pos = self.data.xpos[self.ee_id]
                curr_mat = self.data.xmat[self.ee_id].reshape(3, 3)

            # 计算位置误差
            err_pos = pos - curr_pos

            # 计算姿态误差 (使用轴角表示法)
            # R_err = R_target * R_current^T
            R_err = ori @ curr_mat.T
            err_ori = Rotation.from_matrix(R_err).as_rotvec()

            # 组合 6D 误差向量
            err = np.concatenate([
                err_pos * 1.0,
                err_ori * 0.1  # 降低姿态权重，避免 wrist 绕圈
            ])

            # 如果误差小于容忍度，提前结束迭代
            if np.linalg.norm(err) < self.tol:
                break

            # 计算雅可比矩阵
            if self.is_site:
                mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.ee_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.ee_id)

            # 我们只关心前 6 个关节 (UR5 手臂)
            J = np.vstack([self.jacp[:, :6], self.jacr[:, :6] * 0.1])

            # 使用阻尼最小二乘法 (Damped Least Squares) 计算关节增量
            # delta_q = J.T * (J * J.T + lambda * I)^-1 * err
            J_T = J.T
            lambda_I = self.damping * np.eye(6)
            delta_q = J_T @ np.linalg.solve(J @ J_T + lambda_I, err)
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > 0.2:
                delta_q *= 0.2 / delta_norm

            # 更新关节角度
            self.data.qpos[:6] += self.step_size * delta_q

            # 限制在关节物理限位内
            self.data.qpos[:6] = np.clip(
                self.data.qpos[:6],
                self.model.jnt_range[:6, 0],
                self.model.jnt_range[:6, 1]
            )

            # 更新运动学状态
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

        # 4. 提取求解结果
        result_q = self.data.qpos[:6].copy()

        # 角度归一化：wrap 到离 ref_q 最近，避免 ±2π 绕圈
        for i in range(6):
            while result_q[i] - ref_q[i] > np.pi:
                result_q[i] -= 2 * np.pi
            while result_q[i] - ref_q[i] < -np.pi:
                result_q[i] += 2 * np.pi

        # 5. 恢复原始的物理状态
        self.data.qpos[:] = qpos_orig
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        return result_q


# ---------------------------------------------------------
# 测试代码
# ---------------------------------------------------------
if __name__ == "__main__":
    xml_path = "models/mjcf/ur5/scene.xml"
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # 初始化 IK 求解器（UR5 + Robotiq 使用 pinch TCP site）
        arm_ik = UR5IK(model, data, ee_name="gripper_pinch", is_site=True)

        # 设定一个目标位置和姿态 (夹爪朝下)
        target_pos = np.array([0.5, 0.0, 0.4])
        target_rot = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()

        # 提供一个初始猜测值（UR5 home 位姿）
        current_q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])

        # 求解 IK
        q_sol = arm_ik.properIK(target_pos, target_rot, ref_q=current_q)

        print("求解出的关节角度:", q_sol)

    except Exception as e:
        print("请确保 XML 路径正确以运行测试:", e)
