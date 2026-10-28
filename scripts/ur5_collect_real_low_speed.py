"""Offline real-reference -> low-speed simulation demonstrations. No robot access.
Retains original train/validation membership. Cube initialization is a post-hoc
FK estimate, not independent localization. No forced attachment or action shift.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ur5_control_profile import load_profile, SimCommandAdapter
from ur5_validate_lift_dataset import validate
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import (
    SimNode, Expert, make_config, new_buffers, record, save_episode, PyavImageEncoder)
SPLIT=ROOT/'policies/smolvla/checkpoints/real_ur5_20260924_151943/real_training.json'
SOURCE=ROOT/'data/real_ur5_pick_cube_v2_analysis/episodes'

def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    temp.replace(path)

def collect(eid,part,root,contract):
    folder=root/f'{eid:03d}'
    folder.mkdir(exist_ok=False)
    trajectory=SOURCE/f'{eid:03d}/trajectory.npz'
    with np.load(trajectory) as d:
        actual=d['q_actual_rad'].copy()
        target=d['q_target_rad'].copy()
        grip=d['gripper_command'].reshape(-1).copy()
        stamps=d['timestamp_s'].copy()
    if actual.shape!=target.shape or actual.shape!=(len(stamps),6):
        raise ValueError('Invalid source shape')
    if not all(np.isfinite(x).all() for x in (actual,target,grip,stamps)):
        raise ValueError('Nonfinite source')
    if not np.allclose(np.diff(stamps),1/30,atol=2e-6) or not np.isin(grip,[0,1]).all() or grip[0]!=0:
        raise ValueError('Source frequency/gripper mismatch')
    edges=np.flatnonzero((grip[:-1]==0)&(grip[1:]==1))+1
    if len(edges)==0 or np.any(np.diff(grip)<0):
        raise ValueError('Lift-only source requires close and no release')
    cfg=make_config(True)
    cfg.initial_camera_binding=True
    cfg.align_base_table_centerline=True
    cfg.render_shadows=False
    cfg.lift_only=True
    cfg.use_real_training_start=True
    cfg.success_height=.05
    cfg.hold_seconds=1.
    cfg.arm_gravity_compensation=True
    cfg.render_set['fps']=30
    env=SimNode(cfg,np.random.default_rng(42),randomize=False)
    buffers=new_buffers();encoders=[];trace=[]
    failure=None;success=False;accepted=False;source_complete=False
    try:
        env.reset()
        # Render exactly once per camera at the observation timestamp below.
        cfg.enable_render=False
        model,data=env.mj_model,env.mj_data
        pads=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'gripper_'+s+'_pad') for s in ('left','right')]
        geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'red_block_geom')
        ref=mujoco.MjData(model);ref.qpos[:]=data.qpos
        ref.qpos[:6]=actual[int(edges[0])-1]
        mujoco.mj_forward(model,ref)
        pinch=ref.site_xpos[env.ee_site_id].copy()
        closing=ref.xpos[pads[1]]-ref.xpos[pads[0]]
        yaw=float(np.arctan2(closing[1],closing[0]))
        cube=np.r_[pinch[:2],env.table_z+model.geom_size[geom,2]]
        data.qpos[env.block_qpos_adr:env.block_qpos_adr+3]=cube
        data.qpos[env.block_qpos_adr+3:env.block_qpos_adr+7]=Rotation.from_euler('z',yaw).as_quat()[[3,0,1,2]]
        data.qpos[:6]=actual[0];data.qvel[:]=0;data.time=0
        cfg.start_reference_rad=actual[0].tolist()
        cfg.sampling_metadata={'mode':'real_reference_fk','proposal_grid_cell':f'real_{eid:03d}',
                               'source_episode':eid,'cube_world_m':cube.tolist(),'yaw_rad':yaw,
                               'independent_localization':False}
        lo,hi=model.actuator_ctrlrange[:6].T
        if np.any((target<lo)|(target>hi)):raise ValueError('Source exceeds joint range')
        mujoco.mj_forward(model,data)
        adapter=SimCommandAdapter(env,contract);expert=Expert(env)
        encoders=[PyavImageEncoder(640,480,str(folder),c,fps=30,crf=18,preset='veryfast') for c in (0,1)]
        i=0;hold=0;end_start=None
        while data.time<contract['episode_timeout_s']:
            now=float(data.time)
            desired=np.r_[target[i],grip[i]]
            action,aligned=adapter.step(desired,now,1/30)
            # Fetch images after current FK; do not reuse reset images.
            obs=env.getObservation()
            obs['img']={c:env.getRgbImg(c).copy() for c in (0,1)}
            condition=bool(env.check_success() and adapter.has_contact() and adapter.closed and not adapter.busy)
            hold=hold+1 if condition else 0
            record(env,obs,action,buffers,encoders)
            buffers['lift_condition'][-1]=condition
            trace.append({'source_index':i,'source_time_s':float(stamps[i]),'simulation_time_s':now,
                          'aligned':bool(aligned),'gripper_busy':bool(adapter.busy)})
            if source_complete and hold>=32:
                success=True
                break
            if source_complete and end_start is not None and now-end_start>5:
                raise RuntimeError('Final target held 5s without stable bilateral lift')
            env.step(action)
            if expert.table_collision(data):
                raise RuntimeError('Arm/gripper collision with table/floor/pedestal')
            advance=aligned and not adapter.busy and adapter.decision.candidate is None and adapter.closed==bool(grip[i])
            if advance:
                if i<len(target)-1:i+=1
                elif not source_complete:
                    source_complete=True;end_start=float(data.time)
            if len(trace)%300==0:
                print(f'ep={eid:03d} source={i}/{len(target)} sim={data.time:.1f}s',flush=True)
        if not success:failure='simulation timeout'
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        for encoder in encoders:encoder.close()
        if getattr(env,'renderer',None) is not None:env.renderer.close()
    save_episode(str(folder),buffers,42,eid+1,cfg,success=success,failure=failure,diagnostic=True)
    payload=json.loads((folder/'obs_action.json').read_text())
    payload['metadata'].update(control_contract=contract,source_episode=eid,source_split=part,
        source_trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        max_command_rate_rad_s=.03,max_command_acceleration_rad_s2=.05,
        retiming='source progression waits for tracking/gripper feedback; final target hold up to 5s',
        source_complete=source_complete)
    write(folder/'obs_action.json',payload)
    write(folder/'source_progress.json',trace)
    audit=None
    if success:
        try:
            audit=validate(folder)
            actions=np.asarray(payload['act'])[:,:6]
            velocity=np.diff(np.vstack([actual[0],actions]),axis=0)*30
            acceleration=np.diff(np.vstack([np.zeros(6),velocity]),axis=0)*30
            if np.max(abs(velocity))>.0301 or np.max(abs(acceleration))>.051:
                raise ValueError('Command rate/acceleration acceptance failed')
            accepted=True
        except Exception as exc:failure=f'acceptance: {type(exc).__name__}: {exc}'
    payload['metadata'].update(training_eligible=accepted,diagnostic_only=not accepted,failure_reason=failure)
    write(folder/'obs_action.json',payload)
    result=dict(source_episode=eid,split=part,success=success,accepted=accepted,reason=failure,
                frames=len(buffers['time']),source_complete=source_complete,audit=audit)
    write(folder/'result.json',result)
    print('RESULT',json.dumps(result),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='*')
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    split=json.loads(SPLIT.read_text());contract=load_profile()
    train=split['train_episodes'];val=split['validation_episodes']
    if len(train)!=40 or len(val)!=10 or set(train)&set(val) or set(train+val)!=set(range(50)):
        raise ValueError('Original split mismatch')
    provenance={'train':train,'validation':val,'split_source':str(SPLIT),
                'split_sha256':hashlib.sha256(SPLIT.read_bytes()).hexdigest(),'control_contract':contract}
    manifest=root/'source_split.json'
    if manifest.exists() and json.loads(manifest.read_text())!=provenance:raise ValueError('Resume provenance mismatch')
    if not manifest.exists():write(manifest,provenance)
    for eid in (a.episodes if a.episodes is not None else range(50)):
        if eid not in train+val:raise ValueError(eid)
        result=root/f'{eid:03d}/result.json'
        if result.exists():continue
        collect(eid,'train' if eid in train else 'validation',root,contract)
        if a.episodes is not None:continue
        results=[json.loads(f.read_text()) for f in sorted(root.glob('*/result.json'))]
        write(root/'summary.json',{'completed':len(results),'accepted_train':sum(x['accepted'] and x['split']=='train' for x in results),
              'accepted_validation':sum(x['accepted'] and x['split']=='validation' for x in results),'results':results})
if __name__=='__main__':main()
