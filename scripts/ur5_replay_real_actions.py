"""Offline joint-target dynamics replay. NEVER imports robot/network interfaces.

At t[i], compare simulated state to actual[i], then hold action[i] until t[i+1].
Default: gripper open. --interaction enables recorded binary gripper commands
and one-time cube initialization from pre-close measured arm FK.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from ur5_replay_real_state import (ROOT, JOINTS, VideoReader, pose_matrix,
    SimNode, make_config, PyavImageEncoder, np, cv2, mujoco, Rotation)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--max-frames",type=int)
    p.add_argument("--output",type=Path)
    p.add_argument("--gaussian",action="store_true")
    p.add_argument("--rectify-real",action="store_true",help="Route A: remap real RGB to the simulation pinhole model")
    p.add_argument("--align-base-table-centerline",action="store_true",
                   help="Align base world Y with table center; keep base X/Z/yaw and external camera unchanged")
    p.add_argument("--no-video",action="store_true",help="Metrics-only batch test; identical physics")
    p.add_argument("--damping-scale",type=float,default=1.0,
                   help="Offline experiment: scale arm actuator velocity damping only")
    p.add_argument("--interaction",action="store_true",
                   help="Replay 0=open/1=closed; initialize cube once from pre-close actual FK")
    p.add_argument("--tcp-offset",type=float,nargs=6)
    a=p.parse_args()
    if a.rectify_real and (a.gaussian or a.no_video):
        p.error("Route A requires video and currently supports MuJoCo mesh rendering only")
    if not np.isfinite(a.damping_scale) or not 0.25<=a.damping_scale<=2:
        p.error("damping-scale must be within 0.25..2")
    if a.no_video and a.gaussian:
        p.error("no-video does not need gaussian rendering")
    camera_roles=() if a.no_video else ("front","wrist")
    folder=ROOT/f"data/real_ur5_pick_cube_v2_analysis/episodes/{a.episode:03d}"
    source=ROOT/"data/real_ur5_pick_cube_v2/ur5_pick_cube_v2"
    manifest=json.loads((folder/"manifest.json").read_text())
    with np.load(folder/"trajectory.npz") as d:
        actual=d["q_actual_rad"].copy()
        target=d["q_target_rad"].copy()
        ts=d["timestamp_s"].astype(float)
        tcp=d["tcp_pose_recorded_m_rotvec"].copy()
        gripper=d["gripper_command"].copy().reshape(-1)
    if a.interaction and (len(gripper)!=len(ts) or not np.isin(gripper,[0,1]).all()):
        raise ValueError("Interaction requires finite binary 0=open/1=closed commands")
    fps=float(json.loads((source/"meta/info.json").read_text())["fps"])
    if not all(np.isfinite(v).all() for v in (actual,target,ts,tcp)):
        raise ValueError("Nonfinite input")
    if actual.shape!=target.shape or actual.shape!=(len(ts),6) or tcp.shape!=(len(ts),6):
        raise ValueError("Invalid input shapes")
    if len(ts)<2 or not np.allclose(np.diff(ts),1/fps,atol=2e-6):
        raise ValueError("Unexpected sample timestamps")
    if a.max_frames is not None and a.max_frames<2:
        p.error("max-frames must be >= 2")
    n=min(len(ts),a.max_frames or len(ts))
    offset=np.zeros(6) if a.tcp_offset is None else np.asarray(a.tcp_offset)
    if not np.isfinite(offset).all():
        raise ValueError("Invalid TCP offset")
    out=a.output or ROOT/("data/real_ur5_interaction_replay" if a.interaction else "data/real_ur5_action_replay")/(
        f"ep{a.episode:03d}_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True,exist_ok=False)
    cfg=make_config(True,a.gaussian)
    cfg.align_base_table_centerline=a.align_base_table_centerline
    cfg.render_set["fps"]=fps
    cfg.render_shadows=False
    cfg.enable_render=not a.no_video
    env=SimNode(cfg,np.random.default_rng(42),False)
    readers={};writers={}
    rectifier=None
    qs=[];vs=[];poses=[];contacts=[];tcp_errors=[]
    block_positions=[];pinch_positions=[];openings=[];finger_contacts=[]
    interaction_info=None
    started=time.perf_counter()
    try:
        env.reset()
        if a.rectify_real:
            from ur5_camera_rectification import Rectification
            rectifier=Rectification(env,out,fps)
        model,data=env.mj_model,env.mj_data
        original_damping=model.actuator_biasprm[:6,2].copy()
        model.actuator_biasprm[:6,2]*=a.damping_scale
        ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,k) for k in JOINTS]
        if min(ids)<0:
            raise ValueError("Joint map missing")
        adr=[int(model.jnt_qposadr[j]) for j in ids]
        dof=[int(model.jnt_dofadr[j]) for j in ids]
        # Existing UR5Base control expects these six actuators in named-joint order.
        if not np.array_equal(model.actuator_trnid[:6,0],ids):
            raise ValueError("Actuator ordering differs from joint ordering")
        lo,hi=model.actuator_ctrlrange[:6].T
        if np.any((target<lo)|(target>hi)):
            raise ValueError("Target exceeds actuator limits; refusing silent clipping")
        steps=round(1/(fps*model.opt.timestep))
        if not np.isclose(steps*model.opt.timestep,1/fps):
            raise ValueError("Physics frequency not divisible by dataset frequency")
        pad_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"gripper_"+s+"_pad")
                 for s in ("left","right")]
        block_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"red_block_geom")
        if min(pad_ids+[block_geom])<0:
            raise ValueError("Missing gripper pad or cube geometry")
        if a.interaction:
            edges=np.flatnonzero((gripper[:-1]==0)&(gripper[1:]==1))+1
            if gripper[0]!=0 or len(edges)==0:
                raise ValueError("Need initially open gripper and a recorded 0->1 transition")
            close_index=int(edges[0]);reference=close_index-1
            ref=mujoco.MjData(model)
            ref.qpos[:]=data.qpos
            ref.qpos[adr]=actual[reference]
            mujoco.mj_forward(model,ref)
            pinch=ref.site_xpos[env.ee_site_id].copy()
            closing=ref.xpos[pad_ids[1]]-ref.xpos[pad_ids[0]]
            yaw=float(np.arctan2(closing[1],closing[0]))
            size=model.geom_size[block_geom].copy()
            if model.geom_type[block_geom]!=mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("Expected a box cube")
            pos=np.array([pinch[0],pinch[1],env.table_z+size[2]])
            adr_block=env.block_qpos_adr
            data.qpos[adr_block:adr_block+3]=pos
            data.qpos[adr_block+3:adr_block+7]=Rotation.from_euler("z",yaw).as_quat()[[3,0,1,2]]
            releases=np.flatnonzero((gripper[:-1]==1)&(gripper[1:]==0))+1
            interaction_info=dict(
                initialization="one-time pre-close actual-joint FK pinch XY; resting on table; yaw aligned to finger closing axis",
                close_frame=close_index,reference_frame=reference,release_frames=releases.tolist(),
                initial_cube_world_m=pos.tolist(),cube_yaw_deg=float(np.degrees(yaw)),
                reference_pinch_world_m=pinch.tolist(),
                cube_size_m=(2*size).tolist(),cube_mass_kg=float(model.body_mass[env.block_id]),
                cube_friction=model.geom_friction[block_geom].tolist(),
                gripper_mapping="0->0, 1->255; existing simulated actuator, NOT measured real speed/force",
                gravity=model.opt.gravity.tolist())
            print("INTERACTION INIT",json.dumps(interaction_info),flush=True)
        data.qpos[adr]=actual[0]
        # Initial velocity estimated, not measured, because source lacks actual qd.
        data.qvel[:]=0
        data.qvel[dof]=(actual[1]-actual[0])/(ts[1]-ts[0])
        data.time=0
        mujoco.mj_forward(model,data)
        def body(name):
            bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
            if bid<0: raise ValueError(name)
            t=np.eye(4);t[:3,:3]=data.xmat[bid].reshape(3,3);t[:3,3]=data.xpos[bid]
            return t
        for cid,role in enumerate(camera_roles):
            v=manifest["videos"][role]
            readers[role]=VideoReader(source/v["source_relative_path"],v["start_s"])
            writers[role]=PyavImageEncoder(1280,480,str(out),cid,fps=fps,crf=18,preset="veryfast")
        for i in range(n):
            mujoco.mj_forward(model,data)
            q=data.qpos[adr].copy()
            if not np.isfinite(q).all():
                raise RuntimeError("Simulation diverged")
            qs.append(q);vs.append(data.qvel[dof].copy());contacts.append(int(data.ncon))
            block_positions.append(data.xpos[env.block_id].copy())
            pinch_positions.append(data.site_xpos[env.ee_site_id].copy())
            openings.append(float(np.linalg.norm(data.xpos[pad_ids[1]]-data.xpos[pad_ids[0]])))
            touching=set()
            for k in range(data.ncon):
                con=data.contact[k]
                g1,g2=int(con.geom1),int(con.geom2)
                if block_geom not in (g1,g2) or con.dist>0.001:
                    continue
                other=g2 if g1==block_geom else g1
                bid=int(model.geom_bodyid[other])
                while bid>0:
                    if bid in pad_ids: touching.add(pad_ids.index(bid))
                    bid=int(model.body_parentid[bid])
            finger_contacts.append([0 in touching,1 in touching])
            predicted=np.linalg.inv(body("base"))@body("tool0")@pose_matrix(offset)
            poses.append(predicted)
            measured=pose_matrix(tcp[i])
            tcp_errors.append([np.linalg.norm(predicted[:3,3]-measured[:3,3])*1000,
                               np.degrees(Rotation.from_matrix(predicted[:3,:3].T@measured[:3,:3]).magnitude())])
            error=np.degrees(q-actual[i])
            if a.gaussian: env.update_gs_scene()
            for cid,role in enumerate(camera_roles):
                real,_=readers[role].at(manifest["videos"][role]["start_s"]+ts[i],.51/fps)
                sim=env.getRgbImg(cid).copy()
                if rectifier is not None:
                    real=rectifier.apply(role,real,i)
                img=np.concatenate([real,sim],axis=1)
                cv2.putText(img,f"REAL {role} frame {i} t={ts[i]:.3f}",(8,22),
                            cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,0),1)
                title="SIM PHYSICS | recorded gripper | free cube" if a.interaction else "SIM ACTION -> PHYSICS | gripper held open"
                cv2.putText(img,title,(648,22),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,0),1)
                cv2.putText(img,f"max joint error {abs(error).max():.2f} deg",(648,45),
                            cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,0),1)
                writers[role].encode(img,(i+1)/fps)
                if i in (0,n//2,n-1):
                    cv2.imwrite(str(out/f"{role}_{i:05d}.jpg"),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            if i<n-1:
                env.updateControl(np.r_[target[i],255.0*gripper[i] if a.interaction else 0.0])
                for _ in range(steps):
                    mujoco.mj_step(model,data)
                # No subsequent qpos forcing, IK, or expert controller.
            if i%30==0: print(f"Action replay {i+1}/{n}",flush=True)
    finally:
        if rectifier is not None: rectifier.close()
        for w in writers.values(): w.close()
        for r in readers.values(): r.close()
        if not a.no_video:
            env.renderer.close()
    qs=np.asarray(qs);error=qs-actual[:n];te=np.asarray(tcp_errors)
    np.savez_compressed(out/"metrics.npz",timestamp_s=ts[:n],q_sim_rad=qs,
                        q_actual_rad=actual[:n],q_target_rad=target[:n],q_error_rad=error,
                        qvel_sim_rad_s=vs,tcp_sim_base=poses,tcp_recorded=tcp[:n],
                        tcp_error_mm_deg=te,contact_count_at_samples=contacts,
                        gripper_command=gripper[:n],block_world_m=block_positions,
                        pinch_world_m=pinch_positions,pad_center_distance_m=openings,
                        cube_pad_contacts=finger_contacts)
    report=dict(complete=True,episode=a.episode,frames=n,fps=fps,physics_substeps=steps,
                real_rectification="rectified_real/projection.json" if a.rectify_real else None,
                video_enabled=not a.no_video,damping_scale=a.damping_scale,
                original_velocity_bias=original_damping.tolist(),
                mode="zero_order_hold_absolute_joint_targets",alignment="compare state[i], then apply target[i] for one interval",
                joint_names=JOINTS,joint_mae_deg=np.degrees(abs(error).mean(axis=0)).tolist(),
                joint_rmse_deg=np.degrees(np.sqrt((error**2).mean(axis=0))).tolist(),
                joint_max_abs_deg=np.degrees(abs(error).max(axis=0)).tolist(),
                tcp_mean_error_mm=float(te[:,0].mean()),tcp_max_error_mm=float(te[:,0].max()),
                tcp_mean_rotation_error_deg=float(te[:,1].mean()),
                tcp_offset_confirmed=a.tcp_offset is not None,tcp_offset=offset.tolist(),
                camera_binding=getattr(cfg,"camera_binding_metadata",None),
                wall_seconds=time.perf_counter()-started,
                limitations=["Offline only; no hardware connection",
                             "Initial qvel estimated from first two actual samples",
                             "Real controller interpolation/latency not known; targets held at 30Hz",
                             "Gripper held open; no payload/grasp reproduction",
                             "Scene contacts remain enabled but object placement is not reconstructed",
                             "No automatic pass/fail or real-world safety inference"])
    if a.interaction:
        bp=np.asarray(block_positions);fc=np.asarray(finger_contacts)
        rise=bp[:,2]-interaction_info["initial_cube_world_m"][2]
        active=(np.arange(n)>=interaction_info["close_frame"])&(gripper[:n]==1)
        held=active&(rise>0.02)&fc.all(axis=1)
        longest=run=0
        for flag in held:
            run=run+1 if flag else 0;longest=max(longest,run)
        interaction_info.update(
            max_center_rise_m=float(rise.max()),
            bilateral_contact_frames=int(fc.all(axis=1).sum()),
            longest_lifted_bilateral_contact_s=longest/fps,
            lift_evidence=bool(longest/fps>=0.1),
            final_cube_world_m=bp[-1].tolist(),
            final_xy_displacement_m=float(np.linalg.norm(bp[-1,:2]-bp[0,:2])),
            gripper_opening_metric="pad-center separation, not calibrated inner jaw width",
            release_command_observed=bool(any(r<n for r in interaction_info["release_frames"])),
            task_success=None,
            interpretation="Lift evidence requires >2cm center rise and both pad contacts for >=0.1s; no measured final target or object trajectory available.")
        report["interaction"]=interaction_info
        report["limitations"][3:5]=[
            "Binary gripper commands enabled; real speed/force and opening mapping unknown",
            "Cube XY/yaw inferred from successful real grasp, not independent object localization; no forced attachment",
            "Table/model/cube dimensions and friction remain provisional; other real cubes absent"]
    (out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print("Output:",out)


if __name__=="__main__":
    main()
