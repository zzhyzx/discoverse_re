"""Offline actual-joint replay. No robot interfaces, action commands, or mj_step.

Gripper stays at simulation home and objects stay static: this validates arm FK
and camera geometry, NOT grasp dynamics. Default TCP comparison assumes zero
tool offset and is diagnostic only unless --tcp-offset is explicitly supplied.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import av
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import SimNode, make_config
from discoverse.task_base.airbot_task_base import PyavImageEncoder

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


class VideoReader:
    def __init__(self, path, start):
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.container.seek(max(0, int(start / self.stream.time_base)),
                            stream=self.stream, backward=True)
        self.iterator = self.container.decode(self.stream)
        self.frame = None

    def at(self, timestamp, tolerance):
        while self.frame is None or float(self.frame.pts*self.frame.time_base) < timestamp-tolerance:
            self.frame = next(self.iterator)
        actual = float(self.frame.pts*self.frame.time_base)
        if abs(actual-timestamp) > tolerance:
            raise ValueError(f"Video alignment mismatch: expected {timestamp}, got {actual}")
        return self.frame.to_ndarray(format="rgb24"), actual

    def close(self):
        self.container.close()


def pose_matrix(p):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(p[3:]).as_matrix()
    t[:3, 3] = p[:3]
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--analysis", type=Path, default=ROOT/"data/real_ur5_pick_cube_v2_analysis")
    p.add_argument("--source", type=Path, default=ROOT/"data/real_ur5_pick_cube_v2/ur5_pick_cube_v2")
    p.add_argument("--output", type=Path)
    p.add_argument("--max-frames", type=int, help="Optional short smoke test; omit for whole episode")
    p.add_argument("--tcp-offset", type=float, nargs=6, metavar=("X","Y","Z","RX","RY","RZ"),
                   help="Actual active pendant TCP relative to tool0: metres/rotvec radians")
    p.add_argument("--gaussian", action="store_true")
    p.add_argument("--rectify-real", action="store_true", help="Route A: remap real RGB to the simulation pinhole model")
    p.add_argument("--align-base-table-centerline", action="store_true",
                   help="Align base world Y with table center; keep base X/Z/yaw and external camera unchanged")
    args = p.parse_args()
    if args.rectify_real and args.gaussian:
        p.error("Route A currently verified with MuJoCo mesh rendering only")
    if args.episode < 0 or (args.max_frames is not None and args.max_frames < 1):
        p.error("episode >= 0 and max-frames >= 1 required")
    folder = args.analysis/f"episodes/{args.episode:03d}"
    manifest = json.loads((folder/"manifest.json").read_text())
    with np.load(folder/"trajectory.npz") as data:
        q = data["q_actual_rad"].copy()
        ts = data["timestamp_s"].astype(float)
        actual_tcp = data["tcp_pose_recorded_m_rotvec"].copy()
    fps = float(json.loads((args.source/"meta/info.json").read_text())["fps"])
    if q.shape != (len(ts),6) or actual_tcp.shape != (len(ts),6):
        raise ValueError("Unexpected trajectory shape")
    if not all(np.isfinite(x).all() for x in (q,ts,actual_tcp)):
        raise ValueError("Nonfinite trajectory")
    if len(ts) < 2 or not np.allclose(np.diff(ts),1/fps,atol=2e-6):
        raise ValueError("Nonuniform sample timestamps")
    offset = np.zeros(6) if args.tcp_offset is None else np.array(args.tcp_offset)
    if not np.isfinite(offset).all():
        raise ValueError("Invalid TCP offset")
    count = min(len(ts),args.max_frames or len(ts))
    output = args.output or ROOT/"data/real_ur5_state_replay"/(
        f"ep{args.episode:03d}_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True,exist_ok=False)
    cfg = make_config(True,args.gaussian)
    cfg.align_base_table_centerline = args.align_base_table_centerline
    cfg.render_set["fps"] = fps
    cfg.render_shadows = False
    env = SimNode(cfg,np.random.default_rng(42),randomize=False)
    readers, writers = {}, {}
    rectifier = None
    errors, video_times, transforms = [], [], []
    limits = []
    started = time.perf_counter()
    try:
        env.reset()
        if args.rectify_real:
            from ur5_camera_rectification import Rectification
            rectifier = Rectification(env, output, fps)
        ids = [mujoco.mj_name2id(env.mj_model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in JOINTS]
        if min(ids) < 0:
            raise ValueError("Missing arm joint")
        addresses = [int(env.mj_model.jnt_qposadr[j]) for j in ids]
        for j, name, column in zip(ids, JOINTS, q.T):
            if env.mj_model.jnt_limited[j]:
                lo, hi = env.mj_model.jnt_range[j]
                if np.any((column < lo)|(column > hi)):
                    limits.append(name)

        def body(name):
            bid = mujoco.mj_name2id(env.mj_model,mujoco.mjtObj.mjOBJ_BODY,name)
            if bid < 0:
                raise ValueError(name)
            t=np.eye(4)
            t[:3,:3]=env.mj_data.xmat[bid].reshape(3,3)
            t[:3,3]=env.mj_data.xpos[bid]
            return t

        for cid, role in enumerate(("front","wrist")):
            v=manifest["videos"][role]
            readers[role]=VideoReader(args.source/v["source_relative_path"],v["start_s"])
            writers[role]=PyavImageEncoder(1280,480,str(output),cid,fps=fps,crf=18,preset="veryfast")
        for i in range(count):
            # Named joints, actual measurements, original angle branches; no integration.
            env.mj_data.qpos[addresses]=q[i]
            env.mj_data.qvel[:]=0
            env.mj_data.time=float(ts[i])
            mujoco.mj_forward(env.mj_model,env.mj_data)
            zero=np.linalg.inv(body("base"))@body("tool0")
            predicted=zero@pose_matrix(offset)
            measured=pose_matrix(actual_tcp[i])
            mm=np.linalg.norm(predicted[:3,3]-measured[:3,3])*1000
            deg=np.degrees(Rotation.from_matrix(predicted[:3,:3].T@measured[:3,:3]).magnitude())
            errors.append([mm,deg]);transforms.append(predicted)
            if args.gaussian:
                env.update_gs_scene()
            actual_times=[]
            for cid,role in enumerate(("front","wrist")):
                target=manifest["videos"][role]["start_s"]+ts[i]
                real, stamp=readers[role].at(target,0.51/fps)
                if real.shape != (480,640,3):
                    raise ValueError("Video resolution differs from camera calibration")
                sim=env.getRgbImg(cid).copy()
                if rectifier is not None:
                    real=rectifier.apply(role,real,i)
                combined=np.concatenate([real,sim],axis=1)
                cv2.putText(combined,f"REAL {role} | ep {args.episode} frame {i} t={ts[i]:.3f}s",
                            (8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,0),1)
                cv2.putText(combined,"SIM actual q | gripper/objects STATIC",
                            (648,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,0),1)
                cv2.putText(combined,f"TCP diagnostic {mm:.2f}mm {deg:.3f}deg",
                            (648,45),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,0),1)
                writers[role].encode(combined,(i+1)/fps)
                actual_times.append(stamp)
                if i in (0,count//2,count-1):
                    cv2.imwrite(str(output/f"{role}_{i:05d}.jpg"),cv2.cvtColor(combined,cv2.COLOR_RGB2BGR))
            video_times.append(actual_times)
            if i%30==0:
                print(f"Replay {i+1}/{count}",flush=True)
    finally:
        if rectifier is not None:
            rectifier.close()
        for w in writers.values():
            w.close()
        for r in readers.values():
            r.close()
        env.renderer.close()
    errors=np.asarray(errors)
    np.savez_compressed(output/"metrics.npz",timestamp_s=ts[:count],q_actual_rad=q[:count],
                        predicted_tcp_base=np.asarray(transforms),recorded_tcp_pose=actual_tcp[:count],
                        position_error_mm=errors[:,0],rotation_error_deg=errors[:,1],
                        decoded_video_timestamp_s=np.asarray(video_times))
    report=dict(complete=True,episode=args.episode,frames=count,total_episode_frames=len(ts),fps=fps,
                real_rectification="rectified_real/projection.json" if args.rectify_real else None,
                wall_seconds=time.perf_counter()-started,
                mode="static_actual_joint_replay_no_mj_step",
                joint_names=JOINTS,joints_outside_sim_limits=limits,
                source_manifest=str((folder/"manifest.json").resolve()),
                tcp_offset_m_rotvec=offset.tolist(),tcp_offset_confirmed=args.tcp_offset is not None,
                tcp_comparison="diagnostic_only_no_pass_fail",
                camera_binding=getattr(cfg,"camera_binding_metadata",None),
                renderer="3dgs" if args.gaussian else "mujoco",
                limitations=["User-assumed synchronization, not hardware verified",
                             "Camera poses are prior-assisted initial estimates; FOV-only intrinsics",
                             "Gripper fixed open: feedback-to-opening mapping not calibrated",
                             "Objects static at simulation defaults, not reconstructed from video",
                             "No dynamics/contact/control validation; no safety certification"],
                videos={"front":"cam_0.mp4","wrist":"cam_1.mp4"},
                position_error_mm=dict(mean=float(errors[:,0].mean()),p95=float(np.percentile(errors[:,0],95)),max=float(errors[:,0].max())),
                rotation_error_deg=dict(mean=float(errors[:,1].mean()),p95=float(np.percentile(errors[:,1],95)),max=float(errors[:,1].max())))
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    print("Output:",output)


if __name__=="__main__":
    main()
