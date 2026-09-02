import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig


class FrankaPlayCfg(BaseConfig):
    # 1. 指向我们刚才新建的包含杯子、盘子和主相机的 XML
    mjcf_file_path = "mjcf/franka_emika_panda/franka_pick_place.xml"
    decimation = 4
    timestep = 0.005
    sync = True
    headless = False
    # render_set = {
    #     "fps": 20,
    #     "width": 336,  # 2. 修改为 OpenVLA 推荐的分辨率
    #     "height": 336,
    # }
    render_set = {
        "fps": 30,
        "width": 1280,  # 2. 修改为 OpenVLA 推荐的分辨率
        "height": 720,
    }
    init_qpos = np.zeros(7)
    obs_rgb_cam_id = [0]  # 3. 开启相机渲染！(假设 primary_view 是第 0 个相机)
    rb_link_list = [
        # "panda_link0", "panda_link1", "panda_link2", "panda_link3",
        # "panda_link4", "panda_link5", "panda_link6", "panda_link7",
    ]
    obj_list = []
    use_gaussian_renderer = False
    gs_model_dict = {
        "background": "franka_emika_panda/background.ply",
        "mouse": "objects/mouse.ply"
    }


class FrankaPlayBase(SimulatorBase):
    def __init__(self, config):
        self.n_arm = 7
        # 4. 增加夹爪控制。Franka 通常有 2 个手指关节，这里我们预留 8 个控制维度 (7手臂 + 1夹爪开合度)
        self.n_act = 8
        super().__init__(config)

    def post_load_mjcf(self):
        self.qpos = self.mj_data.qpos
        self.qvel = self.mj_data.qvel
        self.ctrl = self.mj_data.ctrl

        # 保留你原有的 home keyframe 逻辑
        try:
            self.home_key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        except:
            self.home_key_id = -1  # 如果新的 XML 里没有定义 home，做个防错处理

        # 获取末端执行器 (夹爪) 的 ID，用于获取 XYZ 坐标
        # 注意：请确认你的 XML 中夹爪的 body 名字是 "hand" 还是 "panda_hand"
        self.ee_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.home_key_id)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        action = np.asarray(action)
        assert action.shape[0] >= self.n_act, f"Action shape {action.shape} must be >= {self.n_act}"

        # 控制 7 个手臂关节
        self.mj_data.ctrl[:self.n_arm] = np.clip(
            action[:self.n_arm],
            self.mj_model.actuator_ctrlrange[:self.n_arm, 0],
            self.mj_model.actuator_ctrlrange[:self.n_arm, 1]
        )

        # 控制夹爪 (假设第 8 和第 9 个 actuator 是左右手指)
        # 将 action[7] 的开合度同时赋给两个手指
        if len(self.mj_data.ctrl) >= 9:
            self.mj_data.ctrl[7] = action[7]
            self.mj_data.ctrl[8] = action[7]
        elif len(self.mj_data.ctrl) == 8:
            self.mj_data.ctrl[7] = action[7]

    def checkTerminated(self):
        return False

    def getObservation(self):
        # 5. 核心修改：返回 OpenVLA 必须的图像和末端位姿
        self.obs = {
            "time": float(self.mj_data.time),
            "qpos": self.qpos[:7].copy(),
            "qvel": self.qvel[:7].copy(),
            "gripper": self.qpos[7],  # 记录夹爪当前位置
            "ep": self.mj_data.xpos[self.ee_body_id].tolist(),  # 末端 XYZ
            "eq": self.mj_data.xquat[self.ee_body_id].tolist(),  # 末端四元数姿态
            "img": self.img_rgb_obs_s.copy()  # 必须有！主相机 RGB 图像
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.getObservation()

    def getReward(self):
        return None


if __name__ == "__main__":
    cfg = FrankaPlayCfg()
    env = FrankaPlayBase(cfg)

    obs = env.reset()
    # 测试时给一个 8 维的 0 向量
    action = np.zeros(8)

    while env.running:
        obs, *_ = env.step(action)