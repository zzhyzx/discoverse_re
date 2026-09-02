"""
DISCOVERSE Franka Push 数据 → LeRobot SmolVLA 格式转换。

用法:
  python policies/smolvla/scripts/raw_to_lerobot.py \
      --task_name franka_push \
      --raw_dir data/franka_push \
      --cam_ids 0 \
      --task_description "Push the mouse to the target zone on the pad"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from datasets import Dataset, Features, Value, Sequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="franka_push")
    parser.add_argument("--raw_dir", type=str, default="data/franka_push")
    parser.add_argument("--cam_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--task_description", type=str,
                        default="Push the mouse to the target zone on the pad\n")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def read_episode(ep_dir, cam_ids):
    """读取单个 episode，返回 images, states, actions"""
    json_path = os.path.join(ep_dir, "obs_action.json")
    with open(json_path) as f:
        data = json.load(f)

    states = np.array(data["obs"]["qpos"], dtype=np.float32)  # (T, 7)
    actions = np.array(data["act"], dtype=np.float32)                  # (T, 8)

    images = {}
    for cam_id in cam_ids:
        video_path = os.path.join(ep_dir, f"cam_{cam_id}.mp4")
        if not os.path.exists(video_path):
            continue
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        images[cam_id] = np.stack(frames, axis=0)  # (T, H, W, 3)

    return images, states, actions


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "training_data", args.task_name
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # 找到所有 episode 目录
    ep_dirs = sorted([
        d for d in os.listdir(args.raw_dir)
        if os.path.isdir(os.path.join(args.raw_dir, d))
        and os.path.exists(os.path.join(args.raw_dir, d, "obs_action.json"))
    ])
    print(f"找到 {len(ep_dirs)} 个 episodes")

    all_frames = []

    for ep_idx, ep_name in enumerate(tqdm(ep_dirs, desc="转换中")):
        ep_dir = os.path.join(args.raw_dir, ep_name)
        try:
            images, states, actions = read_episode(ep_dir, args.cam_ids)
        except Exception as e:
            print(f"  跳过 {ep_name}: {e}")
            continue

        # 对齐帧数
        T = min(len(states), min(im.shape[0] for im in images.values()))

        for t in range(T):
            frame = {
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": t / args.fps,
                "task": args.task_description,
                "observation.state": states[t].tolist(),   # 7-dim
                "action": actions[min(t + 1, T - 1)].tolist(),  # 8-dim
            }
            for cam_id, img_array in images.items():
                if t < img_array.shape[0]:
                    img = img_array[t]
                    success, enc = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    if success:
                        frame[f"observation.images.cam_{cam_id}"] = enc.tobytes()

            all_frames.append(frame)
        if len(all_frames) == 0:
            print("❌ 没有成功转换任何帧！请检查数据格式")
            return

    # 构造 Dataset
    features_dict = {
        "episode_index": Value("int32"),
        "frame_index": Value("int32"),
        "timestamp": Value("float32"),
        "task": Value("string"),
        "observation.state": Sequence(Value("float32")),
        "action": Sequence(Value("float32")),
    }
    for cam_id in args.cam_ids:
        features_dict[f"observation.images.cam_{cam_id}"] = Value("binary")

    dataset = Dataset.from_list(all_frames, features=Features(features_dict))

    save_path = os.path.join(args.output_dir, "data")
    dataset.save_to_disk(save_path)

    print(f"\n✅ 数据集已保存到 {save_path}")
    print(f"   总帧数: {len(dataset)}")
    print(f"   特征: {list(dataset.features.keys())}")


if __name__ == "__main__":
    main()
