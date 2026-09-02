# envs/franka_mujoco_env.py

import collections
import importlib
import time
import dm_env


class MujocoEnv(object):
    """
    Mujoco environment for Franka robot
    path: path to the script containing the SimNode class and config (mainly for data collection)
    """

    def __init__(self, path: str):
        # 动态导入任务脚本中的 SimNode 类和配置
        module_path = path.replace("/", ".").replace(".py", "")
        module = importlib.import_module(module_path)

        # 获取 SimNode 类（Franka 任务中应该定义了这个类）
        node_cls = getattr(module, "SimNode")

        # 获取配置对象（Franka 任务中应该定义了这个对象）
        cfg = getattr(module, "cfg")
        cfg.headless = False

        # 创建执行节点
        self.exec_node = node_cls(cfg)

        # 初始化状态
        self.reset_position = None
        self._camera_names = self.exec_node.config.obs_rgb_cam_id

        print("Franka MujocoEnv initialized")
        print(f"Camera names: {self._camera_names}")

    def set_reset_position(self, reset_position):
        """设置重置位置"""
        self.reset_position = reset_position
        print("Resetting to the given position: ", self.reset_position)

    def get_reward(self):
        """获取奖励（如果需要自定义奖励，可以在这里实现）"""
        return 0

    def check_success(self):
        return self.exec_node.check_success()

    def reset(self, fake=False, sleep_time=0):
        """重置环境"""
        # 执行域随机化（如果任务支持）
        if hasattr(self.exec_node, 'domain_randomization'):
            self.exec_node.domain_randomization()

        # 如果设置了重置位置，使用自定义重置
        if self.reset_position is not None:
            raw_obs = self.exec_node.reset(position=self.reset_position)
        else:
            raw_obs = self.exec_node.reset()

        time.sleep(sleep_time)

        # 处理观察数据
        return self._process_obs(raw_obs, self._camera_names)

    @staticmethod
    def _process_obs(raw_obs: dict, camera_names: list):
        """处理原始观察数据，转换为标准格式"""
        obs = collections.OrderedDict()

        # 根据 Franka 机器人的观察数据结构调整
        # 通常 Franka 的观察包含：
        # - eef_pos: 末端执行器位置 (3维)
        # - eef_quat: 末端执行器四元数 (4维)
        # - qpos: 关节位置 (7维 for Franka)

        if "eef_pos" in raw_obs and "eef_quat" in raw_obs:
            # 组合位置和姿态作为状态
            obs["qpos"] = list(raw_obs["eef_pos"]) + list(raw_obs["eef_quat"])
        elif "eef_pos" in raw_obs:
            obs["qpos"] = list(raw_obs["eef_pos"])
        elif "qpos" in raw_obs:
            obs["qpos"] = list(raw_obs["qpos"])
        else:
            # 如果没有找到标准键，尝试使用第一个数值观察
            for key, value in raw_obs.items():
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    obs["qpos"] = list(value)
                    break

        # 处理图像观察
        obs["images"] = {}
        if "img" in raw_obs:
            for camera_id in camera_names:
                if camera_id in raw_obs["img"]:
                    # BGR to RGB conversion (OpenCV format)
                    obs["images"][f"cam_{camera_id}"] = raw_obs["img"][camera_id][:, :, ::-1]

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0.0,
            discount=None,
            observation=obs,
        )

    def step(
            self,
            action,
            get_obs=True,
            sleep_time=0,
    ):
        """执行一步动作"""
        # 执行动作
        step_result = self.exec_node.step(action)

        # 根据返回值数量解包
        if len(step_result) == 5:
            raw_obs, pri_obs, rew, ter, info = step_result
        elif len(step_result) == 4:
            raw_obs, rew, ter, info = step_result
        elif len(step_result) == 3:
            raw_obs, ter, info = step_result
        else:
            raw_obs = step_result
            ter = False
            info = {}

        time.sleep(sleep_time)

        # 处理观察
        if get_obs:
            obs = self._process_obs(raw_obs, self._camera_names)
        else:
            obs = None

        return obs


def make_env(path):
    """创建 Franka MuJoCo 环境的工厂函数"""
    print(f"Creating Franka environment from: {path}")

    try:
        env = MujocoEnv(path)
        print("Franka environment created successfully!")
        return env

    except AttributeError as e:
        print(f"Error: Missing required class/configuration in {path}")
        print(f"Make sure the script defines 'SimNode' class and 'cfg' object")
        print(f"Specific error: {e}")

        # 尝试列出模块中可用的内容，帮助调试
        import importlib
        module_path = path.replace("/", ".").replace(".py", "")
        module = importlib.import_module(module_path)
        print(f"Available in module: {[x for x in dir(module) if not x.startswith('_')]}")

        raise

    except Exception as e:
        print(f"Error creating Franka environment: {e}")
        import traceback
        traceback.print_exc()
        raise