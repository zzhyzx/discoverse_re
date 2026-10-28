"""Simulation-only policy-state replay and expert recovery. No robot interfaces."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from policies.smolvla.eval_sim_ur5_lift import initialize_above_block
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import SimNode,Expert,make_config,new_buffers,record,save_episode,PyavImageEncoder
from ur5_validate_lift_dataset import validate

def write(p,v): p.write_text(json.dumps(v,indent=2),encoding='utf-8')

def select(reference):
    pools={k:[] for k in ('low_offset','high_nonapproach','aligned_control')}
    for c in json.loads((reference/'translation_summary.json').read_text())['conditions']:
        for r in c['episodes']:
            source=Path(r['source']);d=json.loads((source/'obs_action.json').read_text())
            ee,b,q,a=[np.asarray(v) for v in (d['obs']['ee_pos'],d['obs']['block_pos'],d['obs']['qpos'],d['act'])]
            e=np.linalg.norm(ee[:,:2]-b[0,:2],axis=1);h=ee[:,2]-.497
            kinds=['aligned_control'] if c['condition']=='aligned' and r['success'] else (['low_offset','high_nonapproach'] if not r['success'] else [])
            for kind in kinds:
                for i in range(1 if kind=='aligned_control' else 30,len(ee)-1):
                    if q[i,6]>.15 or np.any(a[:i,6]>.5) or np.linalg.norm(b[i]-b[0])>.003:continue
                    ok=(.03<e[i]<.15 and .055<h[i]<.12) if kind=='low_offset' else ((.03<e[i]<.15 and h[i]>=.12 and e[i-30]-e[i]<.005) if kind=='high_nonapproach' else e[i]<.015 and .055<h[i]<.12)
                    if ok:
                        pools[kind].append(dict(kind=kind,condition=c['condition'],episode=r['episode'],source=str(source),step=i,xy_error_m=float(e[i]),tcp_height_m=float(h[i])));break
    plans=[];used=set()
    for kind,n in [('low_offset',6),('high_nonapproach',6),('aligned_control',4)]:
        pool=pools[kind].copy();directions={}
        while pool and sum(p['kind']==kind for p in plans)<n:
            pool.sort(key=lambda p:(directions.get(p['condition'].rstrip('0123456789'),0),p['step'],p['source']))
            p=pool.pop(0)
            if p['source'] in used:continue
            direction=p['condition'].rstrip('0123456789');directions[direction]=directions.get(direction,0)+1
            used.add(p['source']);plans.append(p)
    return plans,{k:len(v) for k,v in pools.items()}

def run(plan,folder,policy=None):
    source=Path(plan['source']);protocol=json.loads((source.parent/'protocol.json').read_text())
    original=json.loads((source/'obs_action.json').read_text());prefix=json.loads((source/'prediction_trace.json').read_text())
    cfg=make_config(True)
    cfg.initial_camera_binding=True;cfg.align_base_table_centerline=True;cfg.render_shadows=False
    cfg.lift_only=True;cfg.use_real_training_start=True;cfg.success_height=.05;cfg.hold_seconds=1.
    cfg.sampling='tabletop';cfg.edge_margin=.12;cfg.far_edge_inset=.14;cfg.side_margin=.27;cfg.arm_gravity_compensation=True
    class FixedYaw(SimNode):
        def sample_block_pose(self):
            xy,_=super().sample_block_pose();self.config.sampling_metadata['yaw_deg']=0.;return xy,0.
    env=FixedYaw(cfg,np.random.default_rng(protocol['seed']),randomize=True);expert=Expert(env)
    for _ in range(plan['episode']+1):obs=env.reset()
    obs,_=initialize_above_block(env,expert,protocol['above_height_m'],protocol['initial_offset_world_xy_m'],0.)
    action=env.target_control.copy();error=0.
    for i in range(plan['step']+1):
        error=max(error,float(np.max(abs(env.normalized_state()-original['obs']['qpos'][i]))),float(np.max(abs(env.block_pos()-original['obs']['block_pos'][i]))),float(np.max(abs(env.mj_data.site_xpos[env.ee_site_id]-original['obs']['ee_pos'][i]))))
        if error>1e-5:raise RuntimeError(f'Replay mismatch {error}')
        if expert.table_collision(env.mj_data):raise RuntimeError('Collision in policy prefix')
        if i<plan['step']:
            action=np.asarray(prefix[i]['applied'],dtype=float);obs,*_=env.step(action)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(env.mj_model,spec));mujoco.mj_getState(env.mj_model,env.mj_data,state,spec)
    np.savez_compressed(folder/'takeover_state.npz',integration=state,spec=int(spec),target_control=env.target_control.copy(),last_applied=action.copy(),grasp_rotation=expert.grasp_rotation.copy())
    # Full integration state includes velocities, controls, forces and warmstart.
    # Reload the serialized snapshot into a separate MjData before restoring.
    with np.load(folder/'takeover_state.npz') as saved:
        restored=mujoco.MjData(env.mj_model)
        mujoco.mj_setState(env.mj_model,restored,saved['integration'],spec)
        restored_vector=np.empty_like(state)
        mujoco.mj_getState(env.mj_model,restored,restored_vector,spec)
        assert np.array_equal(state,restored_vector)
    mujoco.mj_setState(env.mj_model,env.mj_data,restored_vector,spec)
    check=np.empty_like(state);mujoco.mj_getState(env.mj_model,env.mj_data,check,spec);assert np.array_equal(state,check)
    mujoco.mj_forward(env.mj_model,env.mj_data)
    before=env.block_pos().copy();cfg.start_reference_rad=env.normalized_state()[:6].tolist()
    expert.grasp_xy=before[:2].copy()
    # Keep initialization's cube-aligned orientation; do not reset to home.
    expert.state=(2 if plan['tcp_height_m']>.085 else 3) if plan['kind']=='aligned_control' else 1
    if policy is None:
        expert.enter()
    else:
        policy.reset()
    route=('continue_aligned_descent' if plan['kind']=='aligned_control' else 'retreat_then_align_at_safe_height') if policy is None else 'policy_only_from_takeover_state'
    write(folder/'takeover.json',dict(**plan,route=route,replay_max_error=error,full_state_roundtrip=True,state_spec=int(spec),last_applied=action.tolist(),time_s=float(env.mj_data.time),prefix_sha256=hashlib.sha256((source/'prediction_trace.json').read_bytes()).hexdigest()))
    buffers=new_buffers();trace=[];settled=None;success=False;reason='timeout';origin=float(env.mj_data.time);maxstep=0.;closed=False
    encoders=[PyavImageEncoder(640,480,str(folder),i,fps=30,crf=18,preset='veryfast') for i in (0,1)]
    try:
        for step in range(600):
            if policy is None:
                expert.update()
            else:
                import torch
                from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import TASK_INSTRUCTION
                batch={'observation.state':torch.from_numpy(env.normalized_state()).unsqueeze(0).cuda(),'task':[TASK_INSTRUCTION]}
                for camera,key in enumerate(('observation.images.d436','observation.images.d405')):
                    batch[key]=torch.from_numpy(np.asarray(obs['img'][camera]).copy()).permute(2,0,1).unsqueeze(0).float().cuda()/255
                with torch.inference_mode():prediction=policy.select_action(batch)[0].cpu().numpy()
                if prediction.shape!=(7,) or not np.isfinite(prediction).all():raise ValueError('Invalid prediction')
                limits=env.mj_model.actuator_ctrlrange[:6]
                env.target_control[:6]=np.clip(prediction[:6],limits[:,0],limits[:,1])
                env.target_control[6]=255. if prediction[6]>=.5 else 0.
            previous=action.copy();action[:6]+=np.clip(env.target_control[:6]-action[:6],-.9*env.delta_t,.9*env.delta_t);action[6]=env.target_control[6]
            maxstep=max(maxstep,float(np.max(abs(action[:6]-previous[:6]))));closed=closed or action[6]>.5
            trace.append(dict(step=step,state=expert.state if policy is None else None,target_tcp=None if policy is not None or expert.target_tcp is None else expert.target_tcp.tolist(),applied=action.tolist(),prediction=None if policy is None else prediction.tolist()))
            record(env,obs,action,buffers,encoders);obs,*_=env.step(action)
            if expert.table_collision(env.mj_data):reason='arm_obstacle_collision';break
            if not closed and np.linalg.norm(env.block_pos()-before)>.005:reason='block_displaced_before_close';break
            info=env.success_info()
            if info['lifted'] and info['stationary']:
                if settled is None:settled=float(env.mj_data.time)
                if env.mj_data.time-settled>=cfg.hold_seconds+env.delta_t:success=True;reason='success';break
            else:settled=None
    except Exception as exc:reason=f'{type(exc).__name__}: {exc}'
    finally:
        for encoder in encoders:encoder.close()
    buffers['time']=[float(t-origin) for t in buffers['time']]
    save_episode(str(folder),buffers,protocol['seed'],plan['episode']+1,cfg,success=success,failure=None if success else reason,diagnostic=True)
    payload=json.loads((folder/'obs_action.json').read_text());payload['metadata'].update(max_command_rate_rad_s=.9,takeover=plan,split_usage='development_recovery_not_independent_test',policy_prefix_excluded=True,initial_pose_source='full-state replayed policy takeover')
    write(folder/'obs_action.json',payload);write(folder/'expert_trace.json',trace);assert maxstep<=.9/30+1e-6
    if success:
        write(folder/'validation.json',validate(folder));payload['metadata'].update(diagnostic_only=policy is not None,training_eligible=policy is None);write(folder/'obs_action.json',payload)
    row=dict(**plan,success=success,reason=reason,frames=len(trace),replay_max_error=error,max_applied_step_rad=maxstep,first_action_step_rad=float(np.max(abs(np.asarray(trace[0]['applied'])[:6]-np.asarray(prefix[plan['step']-1]['applied'])[:6]))),output=str(folder.resolve()),route=route)
    write(folder/'result.json',row)
    if hasattr(env,'close'):env.close()
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,default=ROOT/'data/smolvla_translation_large_20261022');p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=16);p.add_argument('--resume',action='store_true');args=p.parse_args();sys.argv=sys.argv[:1]
    root=args.output.resolve()
    if not args.resume:
        root.mkdir(parents=True,exist_ok=False);plans,counts=select(args.reference);write(root/'plan.json',dict(plans=plans,candidate_counts=counts,reference=str(args.reference.resolve()),limitation='Training on these states retires source evaluation layouts to development; use new held-out layouts.'))
    plans=json.loads((root/'plan.json').read_text())['plans'];rows=[]
    for i,plan in enumerate(plans[:args.limit]):
        folder=root/f'{i:03d}'
        if (folder/'result.json').exists():rows.append(json.loads((folder/'result.json').read_text()));continue
        folder.mkdir(exist_ok=False)
        try:row=run(plan,folder)
        except Exception as exc:
            row=dict(**plan,success=False,reason=f'{type(exc).__name__}: {exc}',output=str(folder));write(folder/'result.json',row)
        rows.append(row);write(root/'summary.json',dict(planned=len(plans),completed=len(rows),successes=sum(r['success'] for r in rows),episodes=rows));print('RESULT',json.dumps(row),flush=True)
    print('FINISHED',len(rows),sum(r['success'] for r in rows),flush=True)

if __name__=='__main__':main()
