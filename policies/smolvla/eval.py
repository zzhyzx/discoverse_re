"""
SmolVLA 策略评估脚本（三色方块语言条件推物）。

用法:
  python policies/smolvla/eval.py \
      --model_path policies/smolvla/checkpoints/franka_multiblock_push/checkpoints/025000/pretrained_model \
      --episodes 5
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submodules" / "lerobot" / "src"))

from policies.smolvla.env import SmolVLAEnv


COLORS = ["red", "blue", "green"]


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5, help="每个颜色测试的 episode 数")
    parser.add_argument("--max_steps", type=int, default=150)
    args = parser.parse_args()

    task_config = {
        "task_path": "discoverse.examples.tasks_franka_play.franka_multiblock_push_task",
        "model_path": args.model_path,
        "cam_ids": [0],
        "state_dim": 7,
        "action_dim": 8,
        "max_steps": args.max_steps,
        "current_color": None,
    }

    print(f"模型: {args.model_path}")
    print(f"每个颜色 Episodes: {args.episodes}")

    env = SmolVLAEnv(task_config)

    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    save_dir = Path(f"policies/smolvla/eval_results/franka_multiblock_push/{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    color_results = {}
    for color in COLORS:
        print(f"\n{'=' * 50}")
        print(f"指令: Push the {color} block to the target zone")
        print(f"{'=' * 50}")

        env.set_task(color)
        success_count = 0

        for ep in range(args.episodes):
            print(f"\n=== [{color}] Episode {ep + 1}/{args.episodes} ===")
            success, frames = env.run_episode()
            if success:
                success_count += 1
                print("  ✅ 成功")
            else:
                print("  ❌ 超时")
            print(f"  [{color}] 成功率: {success_count}/{ep + 1} = {success_count / (ep + 1):.1%}")

            if frames:
                valid = [f[cam_id] for f in frames for cam_id in env.cam_ids if cam_id in f]
                if valid:
                    cam_id = env.cam_ids[0]
                    imgs = [f[cam_id] for f in frames if cam_id in f]
                    if imgs:
                        h, w = imgs[0].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(
                            str(save_dir / f"{color}_ep{ep}_cam{cam_id}.mp4"),
                            fourcc, env.cfg.render_set["fps"], (w, h),
                        )
                        for frame in imgs:
                            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        writer.release()

        color_results[color] = success_count / args.episodes
        print(f"\n[{color}] 最终成功率: {success_count}/{args.episodes} = {color_results[color]:.1%}")

    # 汇总
    total = sum(color_results.values())
    print(f"\n{'=' * 50}")
    print("最终结果汇总:")
    for color, rate in color_results.items():
        print(f"  {color}: {rate:.1%}")
    print(f"  平均: {total / len(COLORS):.1%}")
    print(f"视频保存到: {save_dir}")

    with open(save_dir / "result.txt", "w", encoding="utf-8") as f:
        f.write(f"模型: {args.model_path}\n")
        f.write(f"每个颜色 Episodes: {args.episodes}\n\n")
        for color, rate in color_results.items():
            f.write(f"{color}: {rate:.1%}\n")
        f.write(f"平均: {total / len(COLORS):.1%}\n")


if __name__ == "__main__":
    main()
