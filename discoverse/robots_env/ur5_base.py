import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig


class UR5Cfg(BaseConfig):
    mjcf_file_path = "mjcf/ur5/scene.xml"
    decimation = 4
    timestep = 0.005
    sync = True
    headless = False
    render_set = {
        "fps": 30,
        "width": 1280,
        "height": 720,
    }
    init_qpos = np.zeros(6)
    obs_rgb_cam_id = [0]
    rb_link_list = []
    obj_list = []
    use_gaussian_renderer = False
    gs_model_dict = {}


class UR5Base(SimulatorBase):
    def __init__(self, config):
        self.n_arm = 6   # UR5 6 自由度
        self.n_act = 7   # 6 关节 + 1 夹爪
        super().__init__(config)

    def post_load_mjcf(self):
        self.qpos = self.mj_data.qpos
        self.qvel = self.mj_data.qvel
        self.ctrl = self.mj_data.ctrl

        try:
            self.home_key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        except:
            self.home_key_id = -1

        # 末端执行器（Robotiq 夹爪挂载点）
        self.ee_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_mount")

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.home_key_id)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        action = np.asarray(action)
        assert action.shape[0] >= self.n_act, f"Action shape {action.shape} must be >= {self.n_act}"

        # 控制 6 个关节
        self.mj_data.ctrl[:self.n_arm] = np.clip(
            action[:self.n_arm],
            self.mj_model.actuator_ctrlrange[:self.n_arm, 0],
            self.mj_model.actuator_ctrlrange[:self.n_arm, 1]
        )

        # 夹爪：单 actuator fingers_actuator（ctrl[6]，范围 0~255）
        self.mj_data.ctrl[6] = action[6]

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "time": float(self.mj_data.time),
            "qpos": self.qpos[:6].copy(),
            "qvel": self.qvel[:6].copy(),
            "gripper": self.qpos[6],  # 夹爪 driver joint 位置（0~0.8）
            "ep": self.mj_data.xpos[self.ee_body_id].tolist(),  # 末端 XYZ
            "eq": self.mj_data.xquat[self.ee_body_id].tolist(),  # 末端四元数姿态
            "img": self.img_rgb_obs_s.copy()  # 主相机 RGB 图像
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.getObservation()

    def getReward(self):
        return None


if __name__ == "__main__":
    cfg = UR5Cfg()
    env = UR5Base(cfg)

    obs = env.reset()
    # 测试时给一个 7 维的 0 向量（6 关节 + 1 夹爪）
    action = np.zeros(7)

    while env.running:
        obs, *_ = env.step(action)
