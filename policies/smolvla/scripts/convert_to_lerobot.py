"""
DISCOVERSE 三色方块语言条件推物 → LeRobot v2.1 原生格式转换。
支持多语言指令（每个 episode 的 language_instruction 作为 task 字段）。
"""

import json, os
import cv2, numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR  = "/home/boluo/DISCOVERSE/data/franka_multiblock_push"
OUT_DIR  = "/home/boluo/DISCOVERSE/policies/smolvla/training_data/franka_multiblock_push"
FPS = 20

# 语言指令池（与采集脚本 SimNode.INSTRUCTIONS 一致）
INSTRUCTIONS = [
    "Push the red block to the target zone",
    "Push the blue block to the target zone",
    "Push the green block to the target zone",
]
TASK_TO_INDEX = {instr: i for i, instr in enumerate(INSTRUCTIONS)}

os.makedirs(f"{OUT_DIR}/meta", exist_ok=True)
os.makedirs(f"{OUT_DIR}/data/chunk-000", exist_ok=True)
os.makedirs(f"{OUT_DIR}/videos/chunk-000/observation.images.cam_0", exist_ok=True)

ep_dirs = sorted([
    d for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d))
    and os.path.exists(os.path.join(RAW_DIR, d, "obs_action.json"))
])
print(f"Found {len(ep_dirs)} episodes")

# ====== 1. meta/info.json (v2.1) ======
features = {
    "timestamp":            {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index":          {"dtype": "int64",   "shape": (1,), "names": None},
    "episode_index":        {"dtype": "int64",   "shape": (1,), "names": None},
    "index":                {"dtype": "int64",   "shape": (1,), "names": None},
    "task_index":           {"dtype": "int64",   "shape": (1,), "names": None},
    "observation.state":    {"dtype": "float32", "shape": (7,), "names": None},
    "action":               {"dtype": "float32", "shape": (8,), "names": None},
    "observation.images.cam_0": {"dtype": "video", "shape": (3, 480, 640), "names": ["channel", "height", "width"]},
    "task":                 {"dtype": "string",  "shape": (1,), "names": None},
}

info = {
    "codebase_version": "v2.1",
    "robot_type": "franka",
    "total_episodes": len(ep_dirs),
    "total_frames": 0,
    "total_tasks": len(INSTRUCTIONS),
    "total_videos": len(ep_dirs),
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": FPS,
    "splits": {},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": features,
}

with open(f"{OUT_DIR}/meta/info.json", "w") as f:
    json.dump(info, f, indent=2)

# ====== 2. meta/tasks.jsonl（3 个语言任务） ======
with open(f"{OUT_DIR}/meta/tasks.jsonl", "w") as f:
    for i, instr in enumerate(INSTRUCTIONS):
        f.write(json.dumps({"task_index": i, "task": instr}) + "\n")

# ====== 3. convert each episode ======
all_ep_states = []
all_ep_actions = []
total_frames = 0

for ep_idx, ep_name in enumerate(tqdm(ep_dirs, desc="Converting")):
    ep_dir = os.path.join(RAW_DIR, ep_name)

    with open(os.path.join(ep_dir, "obs_action.json")) as f:
        data = json.load(f)

    states  = np.array(data["obs"]["qpos"], dtype=np.float32)
    actions = np.array(data["act"], dtype=np.float32)

    # 每个 episode 的语言指令
    instruction = data.get("language_instruction", INSTRUCTIONS[0])
    task_index = TASK_TO_INDEX.get(instruction, 0)

    cap = cv2.VideoCapture(os.path.join(ep_dir, "cam_0.mp4"))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.stack(frames, axis=0)

    T = min(len(states), len(frames))

    all_ep_states.append(states[:T])
    all_ep_actions.append(actions[:T])

    # Parquet with FixedSizeListArray
    next_actions = actions[1:T] if T > 1 else actions[T-1:T]
    if len(next_actions) < T:
        next_actions = np.concatenate([next_actions, actions[T-1:T]], axis=0)

    table = pa.table({
        "timestamp":         pa.array(np.arange(T, dtype=np.float32) / FPS),
        "frame_index":       pa.array(np.arange(T, dtype=np.int64)),
        "episode_index":     pa.array(np.full(T, ep_idx, dtype=np.int64)),
        "index":             pa.array(np.arange(total_frames, total_frames + T, dtype=np.int64)),
        "task_index":        pa.array(np.full(T, task_index, dtype=np.int64)),
        "observation.state": pa.FixedSizeListArray.from_arrays(states[:T].ravel(), list_size=7),
        "action":            pa.FixedSizeListArray.from_arrays(next_actions.ravel(), list_size=8),
        "task":              pa.array([instruction] * T),
    })
    pq.write_table(table, f"{OUT_DIR}/data/chunk-000/episode_{ep_idx:06d}.parquet")

    # Video
    h, w = frames[0].shape[:2]
    video_path = f"{OUT_DIR}/videos/chunk-000/observation.images.cam_0/episode_{ep_idx:06d}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))
    for t in range(T):
        writer.write(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
    writer.release()

    # episodes.jsonl
    with open(f"{OUT_DIR}/meta/episodes.jsonl", "a") as f:
        f.write(json.dumps({
            "episode_index": ep_idx,
            "tasks": [instruction],
            "length": T,
        }) + "\n")

    # episodes_stats.jsonl (per-episode stats)
    ep_s = states[:T]
    ep_a = next_actions
    ep_stats = {
        "observation.state": {
            "mean":  ep_s.mean(axis=0).tolist(),
            "std":   ep_s.std(axis=0).tolist(),
            "min":   ep_s.min(axis=0).tolist(),
            "max":   ep_s.max(axis=0).tolist(),
            "count": [T],
        },
        "action": {
            "mean":  ep_a.mean(axis=0).tolist(),
            "std":   ep_a.std(axis=0).tolist(),
            "min":   ep_a.min(axis=0).tolist(),
            "max":   ep_a.max(axis=0).tolist(),
            "count": [T],
        },
    }
    with open(f"{OUT_DIR}/meta/episodes_stats.jsonl", "a") as f:
        f.write(json.dumps({"episode_index": ep_idx, "stats": ep_stats}) + "\n")

    total_frames += T

# ====== 4. update info.json ======
info["total_frames"] = total_frames
with open(f"{OUT_DIR}/meta/info.json", "w") as f:
    json.dump(info, f, indent=2)

# ====== 5. meta/stats.json (global stats) ======
all_states  = np.concatenate(all_ep_states, axis=0)
all_actions = np.concatenate(all_ep_actions, axis=0)

total_N = len(all_states)
stats = {
    "observation.state": {
        "mean":  all_states.mean(axis=0).tolist(),
        "std":   all_states.std(axis=0).tolist(),
        "min":   all_states.min(axis=0).tolist(),
        "max":   all_states.max(axis=0).tolist(),
        "count": [total_N],
    },
    "action": {
        "mean":  all_actions.mean(axis=0).tolist(),
        "std":   all_actions.std(axis=0).tolist(),
        "min":   all_actions.min(axis=0).tolist(),
        "max":   all_actions.max(axis=0).tolist(),
        "count": [total_N],
    },
}
with open(f"{OUT_DIR}/meta/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nDone. {len(ep_dirs)} episodes, {total_frames} frames → {OUT_DIR}")
