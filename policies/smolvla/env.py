"""
SmolVLA 推理环境适配器 — 连接 DISCOVERSE 仿真与 SmolVLA 策略。
支持多语言指令（三色方块推物任务）。
"""

import importlib
import numpy as np
import torch


class SmolVLAEnv:
    def __init__(self, task_config: dict):
        """
        task_config:
          task_path:       "discoverse.examples.tasks_franka_play.franka_multiblock_push_task"
          model_path:      checkpoint 目录（含 model.safetensors）
          cam_ids:         [0]
          state_dim:       7
          action_dim:      8
          current_color:   要推的方块颜色（"red"/"blue"/"green"），None 表示随机
          max_steps:       单 episode 最大步数
        """
        self.config = task_config

        # 1. 动态加载 DISCOVERSE 任务
        module = importlib.import_module(task_config["task_path"])
        SimNode = getattr(module, "SimNode")
        cfg = getattr(module, "cfg")
        cfg.headless = True
        cfg.sync = False
        cfg.use_gaussian_renderer = False
        self.cfg = cfg
        self.env = SimNode(cfg)

        # 2. 加载 SmolVLA 策略
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.policy = SmolVLAPolicy.from_pretrained(task_config["model_path"])
        self.policy.eval()
        self.policy.reset()

        self.cam_ids = task_config["cam_ids"]
        self.state_dim = task_config["state_dim"]
        self.action_dim = task_config["action_dim"]
        self.max_steps = task_config.get("max_steps", 150)
        self.current_color = task_config.get("current_color", None)

        self.task_desc = ""
        self.video_list = []

    def set_task(self, color: str):
        """设置要推的方块颜色（切换语言指令）"""
        self.current_color = color

    def reset(self):
        obs = self.env.reset()
        self.policy.reset()
        self.video_list = []

        # reset 会随机 current_color，这里按评估要求固定目标方块
        if self.current_color is not None:
            self.env.current_color = self.current_color
            self.task_desc = self.env.INSTRUCTIONS[self.current_color]
            self.env.current_instruction = self.task_desc
        else:
            self.task_desc = self.env.current_instruction

        return self._build_batch(obs)

    def _build_batch(self, obs: dict) -> dict:
        batch = {}
        device = next(self.policy.parameters()).device
        for cam_id in self.cam_ids:
            if cam_id in obs["img"]:
                img = obs["img"][cam_id]  # (H, W, 3) uint8
                # uint8 [0,255] → float [0,1]，转 CHW，移到策略设备
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_t = img_t.unsqueeze(0).to(device)
                batch[f"observation.images.cam_{cam_id}"] = img_t

        state = np.array(obs.get("qpos", np.zeros(self.state_dim)), dtype=np.float32)
        batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)
        batch["task"] = self.task_desc
        return batch

    def step(self, action: np.ndarray):
        obs, _, _, _, _ = self.env.step(action)
        self.video_list.append(obs["img"])
        return self._build_batch(obs), obs

    def run_episode(self) -> tuple[bool, list]:
        obs_dict = self.reset()
        t = 0
        success = False

        while t < self.max_steps:
            with torch.no_grad():
                action_tensor = self.policy.select_action(obs_dict)
            action = action_tensor.squeeze(0).cpu().numpy()[: self.action_dim]

            obs_dict, raw_obs = self.step(action)
            t += 1

            if t % 50 == 0:
                print(f"  step {t}/{self.max_steps}")

            if self.env.check_success():
                success = True
                break

        return success, self.video_list
