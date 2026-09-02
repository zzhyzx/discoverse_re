"""
SmolVLA 训练脚本 — 三色方块语言条件推物任务。

用法:
  HF_HUB_OFFLINE=1 PYTHONWARNINGS="ignore::UserWarning" python policies/smolvla/train_multiblock.py
"""

import sys
from pathlib import Path

# 定位 DISCOVERSE 根目录（本脚本位于 policies/smolvla/ 下）
DISCOVERSE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DISCOVERSE_ROOT / "submodules" / "lerobot" / "src"))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.train import train


def main():
    # 1. 数据集配置
    dataset_cfg = DatasetConfig(
        repo_id="franka_multiblock_push",
        root=str(DISCOVERSE_ROOT / "policies" / "smolvla" / "training_data" / "franka_multiblock_push"),
        use_imagenet_stats=False,
    )

    # 2. 策略配置（SmolVLA）
    policy_cfg = SmolVLAConfig()
    policy_cfg.pretrained_path = str(
        DISCOVERSE_ROOT / "policies" / "smolvla" / "pretrained" / "smolvla_base"
    )
    policy_cfg.freeze_vision_encoder = True
    policy_cfg.train_expert_only = True
    policy_cfg.resize_imgs_with_padding = (512, 512)
    policy_cfg.n_obs_steps = 1
    policy_cfg.chunk_size = 50
    policy_cfg.n_action_steps = 50
    policy_cfg.max_state_dim = 32
    policy_cfg.max_action_dim = 32
    policy_cfg.repo_id = "local/franka_multiblock_push_smolvla"
    policy_cfg.push_to_hub = False

    # 3. 训练配置
    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=DISCOVERSE_ROOT / "policies" / "smolvla" / "checkpoints" / "franka_multiblock_push_2",
        batch_size=8,
        steps=25000,
        save_freq=5000,
        num_workers=2,
        seed=42,
    )

    print(f"policy type: {cfg.policy.type}")
    print(f"dataset:     {cfg.dataset.repo_id}")
    print(f"output:      {cfg.output_dir}")
    print(f"batch_size:  {cfg.batch_size}")
    print(f"steps:       {cfg.steps}")
    print("starting training...")

    train(cfg)


if __name__ == "__main__":
    main()
