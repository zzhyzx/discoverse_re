"""Prepare real UR5 data for camera recovery. Offline only; source never modified.
Outputs are an analysis bundle, NOT a newly exported LeRobot dataset.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import av
import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
CAMERAS = {"front": "observation.images.d436", "wrist": "observation.images.d405"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def body_transform(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Missing body {name}")
    result = np.eye(4)
    result[:3, :3] = data.xmat[bid].reshape(3, 3)
    result[:3, 3] = data.xpos[bid]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "data/real_ur5_pick_cube_v2/ur5_pick_cube_v2")
    p.add_argument("--output", type=Path, default=ROOT / "data/real_ur5_pick_cube_v2_analysis")
    p.add_argument("--xml", type=Path, default=ROOT / "models/mjcf/ur5/ur5.xml")
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    output.mkdir(parents=True, exist_ok=False)
    info = json.loads((source / "meta/info.json").read_text())
    fps = float(info["fps"])
    episodes = []
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda e: e["episode_index"])
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata count mismatch")
    cache = {}
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    if min(joint_ids) < 0:
        raise ValueError("Joint mapping incomplete")
    addresses = [int(model.jnt_qposadr[j]) for j in joint_ids]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_right_driver_joint")
    # Gripper feedback is deliberately NOT mapped to joint angles without calibration.
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pylist()
    requests = defaultdict(list)
    report_eps, annotations = [], []
    all_delta, all_tcp_rot, offsets = [], [], []
    frame_total = 0
    first_indices = []
    for ep in episodes:
        eid, n = int(ep["episode_index"]), int(ep["length"])
        rel = info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
        if rel not in cache:
            cache[rel] = pq.read_table(source / rel).to_pydict()
        d = cache[rel]
        mask = np.asarray(d["episode_index"]) == eid
        order = np.argsort(np.asarray(d["frame_index"])[mask])
        get = lambda key: np.asarray(d[key])[mask][order]
        state, action, tcp, ts = (get(k) for k in ("observation.state", "action", "observation.tcp_pose", "timestamp"))
        if state.shape != (n, 7) or action.shape != (n, 7) or tcp.shape != (n, 6):
            raise ValueError(f"Episode {eid}: invalid shapes")
        if not all(np.isfinite(a).all() for a in (state, action, tcp, ts)):
            raise ValueError(f"Episode {eid}: nonfinite values")
        if not np.array_equal(get("frame_index"), np.arange(n)):
            raise ValueError(f"Episode {eid}: noncontiguous frame indices")
        if not np.allclose(ts, np.arange(n) / fps, atol=2e-6):
            raise ValueError(f"Episode {eid}: unexpected sample timestamps")
        if not np.array_equal(get("index"), np.arange(ep["dataset_from_index"], ep["dataset_to_index"])):
            raise ValueError(f"Episode {eid}: global index mismatch")
        q = state[:, :6].astype(np.float64)
        fk = np.empty((n, 4, 4))
        for i, qi in enumerate(q):
            data.qpos[addresses] = qi
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)  # Static FK only, no mj_step.
            fk[i] = np.linalg.inv(body_transform(model, data, "base")) @ body_transform(model, data, "tool0")
        actual = np.repeat(np.eye(4)[None], n, axis=0)
        actual[:, :3, :3] = Rotation.from_rotvec(tcp[:, 3:]).as_matrix()
        actual[:, :3, 3] = tcp[:, :3]
        inferred = np.linalg.inv(fk) @ actual
        offsets.append(inferred)
        all_delta.append(action[:, :6] - q)
        all_tcp_rot.append(tcp[:, 3:])
        speed = np.max(np.abs(np.gradient(q, ts.astype(float), axis=0)), axis=1)
        # Distributed keyframes plus open/close transitions; no claim of all being stationary.
        selected = set(np.linspace(0, n - 1, 5).round().astype(int).tolist())
        selected.update((np.flatnonzero(np.diff(action[:, 6]) != 0) + 1).tolist())
        folder = output / "episodes" / f"{eid:03d}"
        folder.mkdir(parents=True)
        arrays = dict(timestamp_s=ts, frame_index=get("frame_index"), global_index=get("index"),
                      q_actual_rad=q, q_target_rad=action[:, :6], gripper_feedback=state[:, 6],
                      gripper_command=action[:, 6], tcp_pose_recorded_m_rotvec=tcp,
                      T_base_tool0_nominal=fk, T_tool0_tcp_inferred_not_calibrated=inferred,
                      q_speed_finite_difference_rad_s=speed)
        videos = {}
        for camera, feature in CAMERAS.items():
            prefix = "videos/" + feature + "/"
            vrel = info["video_path"].format(video_key=feature,
                chunk_index=ep[prefix+"chunk_index"], file_index=ep[prefix+"file_index"])
            start, end = float(ep[prefix+"from_timestamp"]), float(ep[prefix+"to_timestamp"])
            if abs((end-start)*fps-n) > 0.02:
                raise ValueError(f"Episode {eid}: video duration mismatch")
            video_ts = start + ts.astype(float)
            arrays[camera+"_video_timestamp_s"] = video_ts
            videos[camera] = dict(source_relative_path=vrel, start_s=start, end_s=end)
            for fi in sorted(selected):
                image_rel = f"episodes/{eid:03d}/{camera}_{fi:05d}.jpg"
                requests[vrel].append((int(round(video_ts[fi]*fps)), image_rel, video_ts[fi]))
                annotations.append(dict(episode=eid, frame=fi, camera=camera, image=image_rel,
                    source_video=vrel, video_timestamp_s=float(video_ts[fi]),
                    estimated_max_joint_speed_rad_s=float(speed[fi]), points=[]))
        np.savez_compressed(folder / "trajectory.npz", **arrays)
        split = "fit" if eid < 40 else "validation"
        manifest = dict(episode=eid, length=n, split=split, source_data=rel,
                        selected_frames=sorted(selected), videos=videos,
                        synchronization="assumed aligned by user; not hardware-verified")
        write_json(folder / "manifest.json", manifest)
        report_eps.append(dict(episode=eid, frames=n, split=split,
                               selected_frames=len(selected), low_speed_frames=int(np.sum(speed < 0.01))))
        first_indices.append(frame_total)
        frame_total += n
        print(f"Prepared episode {eid}: {n} samples, {len(selected)} keyframes", flush=True)
    if frame_total != info["total_frames"]:
        raise ValueError("Total frame count mismatch")
    video_report = []
    for vrel, items in requests.items():
        wanted = defaultdict(list)
        for idx, name, timestamp in items:
            wanted[idx].append((name, timestamp))
        seen, count, previous = set(), 0, None
        with av.open(str(source / vrel)) as container:
            stream = container.streams.video[0]
            if stream.width != 640 or stream.height != 480:
                raise ValueError("Unexpected video resolution")
            for idx, frame in enumerate(container.decode(video=0)):
                pts = float(frame.pts * frame.time_base)
                if previous is not None and pts <= previous:
                    raise ValueError("Nonmonotonic video PTS")
                previous = pts
                if idx in wanted:
                    for name, timestamp in wanted[idx]:
                        if abs(pts-timestamp) > 0.51/fps:
                            raise ValueError("Requested frame/PTS mismatch")
                        if not cv2.imwrite(str(output / name), frame.to_ndarray(format="bgr24")):
                            raise RuntimeError("Image write failed")
                    seen.add(idx)
                count += 1
        if seen != set(wanted):
            raise ValueError("Missing requested video frames")
        video_report.append(dict(path=vrel, decoded_frames=count, last_pts_s=previous))
        print(f"Decoded {vrel}: {count} frames", flush=True)
    off = np.concatenate(offsets)
    translation = off[:, :3, 3]
    rotation = Rotation.from_matrix(off[:, :3, :3])
    mean_rot = rotation.mean()
    median_t = np.median(translation, axis=0)
    residual_mm = np.linalg.norm(translation-median_t, axis=1)*1000
    residual_deg = np.degrees((mean_rot.inv()*rotation).magnitude())
    tcp_rotation = Rotation.from_rotvec(np.concatenate(all_tcp_rot))
    angle_deg = np.degrees((tcp_rotation[0].inv()*tcp_rotation).magnitude())
    delta = np.abs(np.concatenate(all_delta))
    report = dict(source=str(source), source_format=info["codebase_version"],
        total_episodes=len(episodes), total_frames=frame_total, fps=fps, tasks=tasks,
        action_semantics="User confirmed absolute joint targets [q1..q6 rad, gripper command]",
        original_action_names=info["features"]["action"]["names"],
        corrected_action_names=JOINTS+["gripper_command"],
        synchronization_assumption="Aligned by user confirmation; original sensor timestamps unavailable",
        camera_mapping=CAMERAS, videos=video_report, episodes=report_eps,
        tcp_orientation_max_from_first_deg=float(angle_deg.max()),
        action_vs_state_abs_rad_quantiles=np.quantile(delta,[0.5,0.95,1],axis=0).tolist(),
        inferred_tool_offset_NOT_calibration=dict(
            translation_m=median_t.tolist(), rotation_vector_rad=mean_rot.as_rotvec().tolist(),
            translation_residual_mm_quantiles=np.quantile(residual_mm,[.5,.95,1]).tolist(),
            rotation_residual_deg_quantiles=np.quantile(residual_deg,[.5,.95,1]).tolist(),
            note="Assumes recorded TCP is in UR base, metres/rotvec radians; includes nominal FK/calibration/timing errors"),
        missing=["camera intrinsics/distortion", "active TCP configuration", "camera extrinsics",
                 "measured cube sizes/locations", "gripper feedback-to-opening calibration"],
        source_sha256={str(f.relative_to(source)):digest(f) for f in sorted(source.rglob("*")) if f.is_file()},
        model_xml=str(args.xml.resolve()), model_xml_sha256=digest(args.xml),
        status="Preparation/FK analysis only; no camera extrinsics solved")
    write_json(output/"report.json",report)
    write_json(output/"annotations.json",dict(
        instructions="Fill points with {id, uv:[u,v], xyz_base_m:[x,y,z]}; never guess moving object coordinates",
        samples=annotations))
    # Desk coordinates are only a local metric reference, NOT assumed UR base axes.
    write_json(output/"scene_reference.json",dict(
        table_size_m=[0.8,0.9], table_top_floor_z_m=0.497, robot_mount_floor_z_m=0.443,
        table_corners_in_table_frame=[[-.4,-.45,0],[.4,-.45,0],[.4,.45,0],[-.4,.45,0]],
        T_base_table=None, front_intrinsics=None, wrist_intrinsics=None,
        T_base_front_camera=None, T_tool0_wrist_camera=None,
        note="Unsolved values intentionally null. Existing simulation offsets are not calibration."))
    # Visual contact sheet: first and central frame of first episode, both cameras.
    first_ep = episodes[0]
    eid = int(first_ep["episode_index"])
    fi = int(round((int(first_ep["length"])-1)*0.5))
    rows = []
    for frame in (0,fi):
        pictures = []
        for camera in CAMERAS:
            a = cv2.imread(str(output/f"episodes/{eid:03d}/{camera}_{frame:05d}.jpg"))
            cv2.putText(a,f"{camera} ep={eid} frame={frame}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
            pictures.append(a)
        rows.append(np.concatenate(pictures,axis=1))
    if not cv2.imwrite(str(output/"contact_sheet.jpg"),np.concatenate(rows,axis=0)):
        raise RuntimeError("Contact sheet write failed")
    write_json(output/"COMPLETE.json",dict(complete=True,frames=frame_total,images=len(annotations)))
    print(json.dumps({k:report[k] for k in ["total_frames","tcp_orientation_max_from_first_deg","inferred_tool_offset_NOT_calibration"]},indent=2))
    print("Output:",output)


if __name__ == "__main__":
    main()
