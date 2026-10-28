"""MuJoCo-only policy closed-loop evaluation. Never connects to robot interfaces.

Inference runs in one background worker; simulation is paced at most 30Hz.
Rendering may slow physics: not a hard real-time deployment benchmark.
All sampled positions are attempted. Optional IK initialization is documented;
the policy controls every evaluated step after initialization.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'submodules/lerobot/src')]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
from ur5_control_profile import load_profile, SimCommandAdapter
from ur5_deploy_core import ExecutionProgress, interpolate, validate_actions, takeover_reason
import mujoco
import torch
from scipy.spatial.transform import Rotation
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import (
    SimNode, Expert, make_config, new_buffers, record, save_episode,
    PyavImageEncoder, TASK_INSTRUCTION)

def initialize_above_block(env, checker, height, offset_xy=(0.,0.), yaw_deg=0.):
    """Set an open, cube-aligned pose via IK, then settle before policy time.

    Follow the collection IK reference sequence to select the same joint branch.
    No approach actions, stage labels or object positions enter the policy batch.
    """
    checker.reset()
    reference=env.mj_data.qpos[:6].copy()
    target=np.r_[env.block_pos()[:2], env.table_z+height]
    positions=checker.approach_path()[:-1]+[target]
    for position in positions:
        reference=checker.solve(position,reference)
    # Establish the same nominal branch, then perturb just the starting pose.
    target[:2]+=np.asarray(offset_xy,dtype=float)
    checker.grasp_rotation=Rotation.from_euler('z',yaw_deg,degrees=True).as_matrix() @ checker.grasp_rotation
    if np.any(np.asarray(offset_xy)!=0) or yaw_deg!=0:
        reference=checker.solve(target,reference)
    env.mj_data.qpos[:6]=reference
    env.mj_data.qvel[:]=0
    mujoco.mj_forward(env.mj_model,env.mj_data)
    if checker.table_collision(env.mj_data):
        raise ValueError('Above-block initial pose collides with table/floor/pedestal')
    env.target_control[:6]=reference
    env.target_control[6]=0.
    before=env.block_pos().copy()
    # Same controller/gravity compensation as collection, only a fixed hold.
    for _ in range(round(.5/env.mj_model.opt.timestep)):
        env.updateControl(env.target_control)
        mujoco.mj_step(env.mj_model,env.mj_data)
    mujoco.mj_forward(env.mj_model,env.mj_data)
    actual=env.mj_data.site_xpos[env.ee_site_id].copy()
    error=float(np.linalg.norm(actual-target))
    speed=float(np.max(np.abs(env.mj_data.qvel[:6])))
    if error>.003 or speed>.01 or checker.table_collision(env.mj_data):
        raise ValueError(f'Above-block settling failed: TCP error={error}, joint speed={speed}')
    if env.normalized_state()[6]>.05 or np.linalg.norm(env.block_pos()-before)>.003:
        raise ValueError('Initialization moved block or did not leave gripper open')
    env.mj_data.time=0.
    env.render_cnt=0
    env.render()
    obs=env.getObservation()
    metadata={'mode':'above-block','height_above_table_m':height,
              'requested_offset_world_xy_m':list(offset_xy),'requested_yaw_offset_deg':yaw_deg,
              'target_tcp_world_m':target.tolist(),'actual_tcp_world_m':actual.tolist(),
              'tcp_error_m':error,'max_joint_speed_rad_s':speed,
              'q_rad':env.normalized_state()[:6].tolist(),
              'block_pos_world_m':env.block_pos().tolist(),
              'gripper_state':float(env.normalized_state()[6]),
              'settling_seconds':.5,'expert_control_after_initialization':False,
              'method':'collection cube-aligned IK branch; static initialization, no approach rollout'}
    return obs,metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--reference-data',type=Path,help='Use original held-out real-reference simulation initial scenes')
    p.add_argument('--reference-descent-start',action='store_true',help='Start from held-out expert open-gripper aligned onset of descent; partial task')
    p.add_argument('--episode-start',type=int,default=0,
                   help='Skip earlier scene resets, preserving scene RNG and episode seeds; use a new output directory')
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--seconds',type=float,default=180.)
    p.add_argument('--success-mode',choices=['stable','height'],default='stable',
                   help='height: block bottom reaches 5cm; no hold or speed requirement (may include transient lift)')
    p.add_argument('--task',default=TASK_INSTRUCTION,help='Language instruction; use the model training instruction')
    p.add_argument('--start-mode',choices=['home','above-block'],default='home')
    p.add_argument('--above-height',type=float,default=.18,
                   help='Above-block initial TCP height above tabletop in meters (default .18)')
    p.add_argument('--initial-offset-xy',type=float,nargs=2,default=(0.,0.),metavar=('DX','DY'),
                   help='Initial TCP offset from block in WORLD X/Y meters; requires above-block mode')
    p.add_argument('--initial-yaw-deg',type=float,default=0.,
                   help='Initial orientation offset around WORLD Z from cube alignment')
    p.add_argument('--block-yaw-deg',type=float,default=None,
                   help='Fix block world yaw (0 aligns faces to table axes); default random yaw')
    p.add_argument('--fixed-timing',action='store_true',help='Offline 30Hz action consumption independent of arm tracking; gripper waits retained')
    p.add_argument('--fixed-steps',type=int,choices=range(1,51),help='Offline fixed action progress before replacement (1..50); no wall-clock expiry')
    p.add_argument('--fixed-five',action='store_true',help='Simulation experiment: finish five feedback-gated actions before replacement; no early takeover')
    p.add_argument('--action-steps',type=int,default=None,
                   help='Actions executed before replanning; default: checkpoint setting')
    p.add_argument('--cache-close-intent',action='store_true',help='Simulation-only: remember confirmed close events until their joint goal is reached; requires fixed timing')
    p.add_argument('--hold-after-grasp',action='store_true',help='Simulation-only: latch closed after contact/settling confirmation')
    p.add_argument('--max-command-acceleration',type=float,default=.05)
    p.add_argument('--max-command-rate',type=float,default=.03,
                   help='Simulation joint target slew limit in rad/s (0 < value <= 1.2); not actual joint velocity')
    a=p.parse_args();sys.argv=sys.argv[:1]
    if a.fixed_five and a.fixed_steps not in (None,5):
        raise ValueError('--fixed-five conflicts with --fixed-steps')
    if a.fixed_five: a.fixed_steps=5
    a.fixed_five=a.fixed_steps is not None  # backwards-compatible internal mode flag
    if a.fixed_timing and not a.fixed_five: raise ValueError('--fixed-timing requires --fixed-steps')
    if a.cache_close_intent and not a.fixed_timing: raise ValueError('--cache-close-intent requires --fixed-timing')
    if a.episodes<1 or not 0<a.seconds<=(300 if a.fixed_five else 180): raise ValueError('Invalid episodes or timeout')
    if not 0<=a.episode_start<a.episodes: raise ValueError('Invalid episode-start')
    if not np.isfinite(a.max_command_rate) or not 0<a.max_command_rate<=1.2:
        raise ValueError('Simulation max-command-rate must be finite and in (0, 1.2]')
    if not np.isfinite(a.above_height) or not .12<=a.above_height<=.24:
        raise ValueError('--above-height must be in [.12, .24] meters above table')
    if not np.isfinite(a.initial_offset_xy).all() or np.max(np.abs(a.initial_offset_xy))>.1:
        raise ValueError('Initial X/Y offsets must be finite and within +/-0.1m')
    if not np.isfinite(a.initial_yaw_deg) or abs(a.initial_yaw_deg)>45:
        raise ValueError('Initial yaw offset must be finite and within +/-45 degrees')
    if a.start_mode!='above-block' and (np.any(np.asarray(a.initial_offset_xy)!=0) or a.initial_yaw_deg!=0):
        raise ValueError('Initial pose offsets require --start-mode above-block')
    if a.reference_descent_start and not a.reference_data:
        raise ValueError('--reference-descent-start requires --reference-data')
    contract=load_profile()
    if not np.isfinite(a.max_command_acceleration) or not 0<a.max_command_acceleration<=2.4:
        raise ValueError('Invalid simulation acceleration')
    effective=dict(contract)
    effective.pop('sha256')
    effective.update(id='simulation_evaluation_override',
        max_command_speed_rad_s=a.max_command_rate,
        max_command_acceleration_rad_s2=a.max_command_acceleration,
        hold_after_grasp=a.hold_after_grasp,cache_close_intent=a.cache_close_intent,
        episode_timeout_s=a.seconds)
    if a.action_steps not in (None,contract['minimum_execution_steps']):
        raise ValueError('Fixed baseline replans after at least 10 executed steps')
    a.action_steps=a.fixed_steps if a.fixed_five else contract['minimum_execution_steps']
    references=None
    if a.reference_data:
        if a.start_mode!='home' or a.block_yaw_deg is not None:
            raise ValueError('Reference scenes cannot be combined with pose/yaw overrides')
        split=json.loads((a.reference_data/'source_split.json').read_text())
        if split['control_contract']!=contract:raise ValueError('Reference control contract mismatch')
        run=a.checkpoint.resolve().parents[2]
        if json.loads((run/'control_contract.json').read_text())!=contract:
            raise ValueError('Checkpoint control contract mismatch')
        references=split['validation']
        if a.episodes>len(references):raise ValueError('More episodes than held-out references')
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    if a.block_yaw_deg is not None and not np.isfinite(a.block_yaw_deg):
        raise ValueError('Block yaw must be finite')
    config=PreTrainedConfig.from_pretrained(a.checkpoint)
    saved_action_steps=config.n_action_steps
    if a.action_steps is not None:
        if not 1<=a.action_steps<=config.chunk_size:
            raise ValueError('--action-steps must be in [1, checkpoint chunk_size]')
        config.n_action_steps=a.action_steps
    a.output.mkdir(parents=True,exist_ok=False)
    config.device='cuda';config.push_to_hub=False
    policy=SmolVLAPolicy.from_pretrained(a.checkpoint,config=config).to('cuda').eval()
    cfg=make_config(True)
    cfg.initial_camera_binding=True;cfg.align_base_table_centerline=True;cfg.render_shadows=False
    cfg.lift_only=True;cfg.use_real_training_start=True;cfg.success_height=.05;cfg.hold_seconds=1.
    cfg.sampling='tabletop';cfg.edge_margin=.12;cfg.far_edge_inset=.14;cfg.side_margin=.27
    cfg.arm_gravity_compensation=True
    class EvaluationSimNode(SimNode):
        def sample_block_pose(self):
            xy,yaw=super().sample_block_pose()
            if a.block_yaw_deg is not None:
                yaw=np.deg2rad(a.block_yaw_deg)
                self.config.sampling_metadata['yaw_deg']=float(a.block_yaw_deg)
            return xy,yaw
    env=EvaluationSimNode(cfg,np.random.default_rng(a.seed),randomize=True)
    # Optional initialization uses IK; evaluated actions are always policy outputs.
    checker=Expert(env)
    protocol={'checkpoint':str(a.checkpoint.resolve()),'seed':a.seed,'episodes':a.episodes,'episode_start':a.episode_start,
              'timeout_sim_seconds':a.seconds,'fps':30,'action_stride':config.n_action_steps,
              'checkpoint_action_stride':saved_action_steps,
              'policy_seed_mode':'reseed torch/numpy with seed + episode after each reset',
              'max_command_rate_rad_s':a.max_command_rate,'gripper':'threshold 0.5, open=0 closed=255',
              'gravity_compensation':True,'expert_control':False,'preflight_filter':False,
              'start_mode':a.start_mode,'above_height_m':a.above_height if a.start_mode=='above-block' else None,
              'initialization_uses_object_pose_and_ik':a.start_mode=='above-block',
              'initial_offset_world_xy_m':list(a.initial_offset_xy),'initial_yaw_offset_deg':a.initial_yaw_deg,
              'fixed_block_yaw_deg':a.block_yaw_deg,
              'interpretation':'partial task from initialized pregrasp' if a.start_mode=='above-block' else 'full task from home',
              'timing':'simulation pauses during inference; not real-time',
              'task':a.task,'success_mode':a.success_mode,
              'success':('block bottom >= 0.05m; no hold/speed requirement, transient lifts count' if a.success_mode=='height' else 'block bottom >= 0.05m, speed <=0.03m/s, angular speed <=0.3rad/s and bilateral pad contact continuously >=1s'),
              'collision_scope':'same arm-vs-table/floor/pedestal predicate as collection; not full self collision'}
    protocol.update(reference_source=str(a.reference_data) if a.reference_data else None,
        source_validation_episodes=references,control_contract=contract,effective_control=effective,max_command_rate_rad_s=a.max_command_rate,
        gripper='shared .65/.35 thresholds, .2s dwell, feedback wait',
        timing='background single-worker inference; wall-clock freshness; simulation paced at most 30Hz, may run slower with rendering',
        scheduling_revision='async_early_takeover_v1',max_observation_joint_drift_deg=3.,early_takeover={'minimum_dwell_s':.25,'stall_s':.30,'expiry_margin_s':.45})
    if a.fixed_five:
        protocol.update(scheduling_revision='fixed_steps_offline_v3',early_takeover=None,
                        evaluation_override={'minimum_execution_steps':a.action_steps,
                        'endpoint':('replace after N clocked actions; wait only for gripper or next prediction' if a.fixed_timing else 'hold action N until aligned; do not consume N+1'),
                        'freshness':'offline only: no wall-clock expiry; state-drift and target checks retained'})
    protocol['action_progress_mode']='30Hz simulation clock; only gripper pauses' if a.fixed_timing else 'feedback gated'
    from ur5_sim_close_intent import SimCloseIntent
    protocol['close_intent_buffer']=SimCloseIntent().configuration() if a.cache_close_intent else None
    protocol['reference_descent_start']=a.reference_descent_start
    if a.reference_descent_start:
        protocol['interpretation']='partial task: expert aligned descent onset, velocities zeroed'
        protocol['initialization_uses_expert_state']=True
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    for episode in range(a.episodes):
        if episode<a.episode_start:
            env.reset()
            continue
        folder=a.output/f'{episode:03d}';folder.mkdir()
        obs=env.reset();policy.reset()
        initialization={'mode':'home','q_rad':env.normalized_state()[:6].tolist(),
                        'block_pos_world_m':env.block_pos().tolist(),'gripper_state':float(env.normalized_state()[6])}
        if references is not None:
            source_episode=references[episode]
            raw=json.loads((a.reference_data/f'{source_episode:03d}/obs_action.json').read_text())
            meta=raw['metadata']
            if meta['source_split']!='validation' or not meta['training_eligible']:
                raise ValueError('Reference must be an accepted held-out episode')
            frame=0
            if a.reference_descent_start:
                ee=np.asarray(raw['obs']['ee_pos'])
                blocks=np.asarray(raw['obs']['block_pos'])
                states=np.asarray(raw['obs']['qpos'])
                commands=np.asarray(raw['act'])
                eligible=(np.linalg.norm(ee[:,:2]-blocks[:,:2],axis=1)<.005)
                eligible &= (ee[:,2]<ee[0,2]-.002) & (ee[:,2]>env.table_z+.12)
                eligible &= (states[:,6]<.05) & (commands[:,6]<.05)
                candidates=np.flatnonzero(eligible)
                if not len(candidates): raise ValueError('No open aligned descent-onset reference state')
                frame=int(candidates[0])
            data=env.mj_data
            data.qpos[:6]=np.asarray(raw['obs']['qpos'][frame])[:6]
            data.qvel[:]=0.;data.time=0.
            sampling=meta['sampling']
            data.qpos[env.block_qpos_adr:env.block_qpos_adr+3]=raw['obs']['block_pos'][frame]
            data.qpos[env.block_qpos_adr+3:env.block_qpos_adr+7]=Rotation.from_euler('z',sampling['yaw_rad']).as_quat()[[3,0,1,2]]
            env.target_control[:6]=data.qpos[:6];env.target_control[6]=0.
            cfg.start_reference_rad=data.qpos[:6].tolist()
            cfg.sampling_metadata=dict(sampling)
            mujoco.mj_forward(env.mj_model,data)
            env.render_cnt=0;env.render();obs=env.getObservation()
            initialization={'mode':'held-out real-reference initial state','source_episode':source_episode,
                'q_rad':env.normalized_state()[:6].tolist(),'block_pos_world_m':env.block_pos().tolist(),
                'block_yaw_rad':sampling['yaw_rad'],'expert_control_after_initialization':False,
                'cube_position_is_posthoc_fk_estimate':True,
                'reference_frame':frame,'reference_time_s':raw['time'][frame],
                'velocities_zeroed':True,
                'partial_task':a.reference_descent_start}
            if a.reference_descent_start:
                expected=np.asarray(raw['obs']['ee_pos'][frame])
                actual_tcp=env.mj_data.site_xpos[env.ee_site_id].copy()
                error=float(np.linalg.norm(actual_tcp-expected))
                if error>.001 or checker.table_collision(data):
                    raise ValueError(f'Reference initialization mismatch/collision: TCP error {error}')
                initialization.update(mode='expert aligned descent onset',
                    reference_tcp_world_m=expected.tolist(),actual_tcp_world_m=actual_tcp.tolist(),
                    tcp_error_m=error)
                print('DESCENT INITIALIZATION:',json.dumps(initialization),flush=True)
        if a.start_mode=='above-block':
            try:
                obs,initialization=initialize_above_block(env,checker,a.above_height,a.initial_offset_xy,a.initial_yaw_deg)
            except (ValueError,RuntimeError) as exc:
                initialization={'mode':'above-block','error':str(exc)}
                (folder/'initialization.json').write_text(json.dumps(initialization,indent=2))
                row={'episode':episode,'success':False,'reason':'initialization_failed: '+str(exc),
                     'frames':0,'sampling':cfg.sampling_metadata,'initialization':initialization}
                results.append(row)
                (a.output/'summary.json').write_text(json.dumps({'completed':len(results),
                    'successes':sum(r['success'] for r in results),
                    'success_rate':sum(r['success'] for r in results)/len(results),'episodes':results},indent=2))
                print(json.dumps(row),flush=True)
                continue
        def capture():
            return (time.monotonic(),env.normalized_state().copy(),
                    [np.asarray(obs['img'][cid]).copy() for cid in (0,1)])
        def infer(snapshot):
            stamp,state,images=snapshot
            begin=time.monotonic()
            batch={'observation.state':torch.from_numpy(state).unsqueeze(0).cuda(),'task':[a.task]}
            for cid,key in enumerate(['observation.images.d436','observation.images.d405']):
                batch[key]=torch.from_numpy(images[cid]).permute(2,0,1).unsqueeze(0).float().cuda()/255
            policy.reset()
            with torch.inference_mode(): actions=policy.predict_action_chunk(batch)[0].cpu().numpy()[:a.action_steps if a.fixed_five else 30]
            return {'observed':stamp,'state':state,'actions':actions,'latency':time.monotonic()-begin}
        executor=ThreadPoolExecutor(max_workers=1)
        # Warm CUDA handles in the actual worker before starting control time.
        executor.submit(infer,capture()).result()
        (folder/'initialization.json').write_text(json.dumps(initialization,indent=2))
        action=env.target_control.copy()
        adapter=SimCommandAdapter(env,effective,hold_after_grasp=a.hold_after_grasp,cache_close_intent=a.cache_close_intent)
        progress=ExecutionProgress(30,a.action_steps,np.radians(.5))
        active=None;observed=0.;latency=0.;initial=action[:6].copy()
        future=None;newest=None;adopted=last_progress=time.monotonic();wait_start=adopted
        transitions=[];sequence=0
        # Separate scene RNG; episode length must not affect later policy seeds.
        episode_seed=a.seed+episode
        torch.manual_seed(episode_seed);np.random.seed(episode_seed)
        cfg.sampling_metadata['filter']='none; no expert preflight'
        buffers=new_buffers();trace=[];settled=None;success=False;reason='timeout';peak=0.;clipped=0
        encoders=[PyavImageEncoder(640,480,str(folder),i,fps=30,crf=18,preset='veryfast') for i in (0,1)]
        try:
            while env.mj_data.time < a.seconds:
                step=len(trace)
                start=time.monotonic()
                if future is not None and future.done():
                    newest=future.result();future=None;sequence+=1;newest['sequence']=sequence
                    if not a.fixed_five and newest['latency']>1.05:raise RuntimeError('Inference exceeds configured deadline')
                if future is None:
                    future=executor.submit(infer,capture())
                now=time.monotonic()
                actual=env.normalized_state()[:6]
                previous=None if active is None else interpolate(active,initial,progress.elapsed,30)
                aligned=previous is None or max(np.max(abs(actual-previous[:6])),
                    np.max(abs(adapter.limiter.q-previous[:6])))<=progress.tolerance
                reason_takeover='initial' if active is None and newest is not None else (
                    takeover_reason(progress,now,observed,adopted,last_progress,1.65,aligned,
                        adapter.busy,adapter.decision.candidate is not None,newest is not None)
                    if active is not None else None)
                if a.fixed_five and active is not None:
                    reason_takeover=('fixed_steps_complete' if newest is not None
                        and progress.steps>=a.action_steps-1e-8 and (a.fixed_timing or aligned) and not adapter.busy
                        and adapter.decision.candidate is None and not adapter.pending_close else None)
                replanned=reason_takeover is not None
                if replanned:
                    if not a.fixed_five and now-newest['observed']>1.65:raise RuntimeError('Next prediction observation too old')
                    if np.max(abs(actual-newest['state'][:6]))>np.radians(3.):
                        raise RuntimeError('Robot moved too far since prediction observation')
                    limits=env.mj_model.actuator_ctrlrange[:6]
                    validate_actions(newest['actions'],actual,limits[:,0],limits[:,1],np.radians(12.))
                    transitions.append({'time_s':float(env.mj_data.time),'reason':reason_takeover,
                        'previous_steps':progress.steps,'observation_age_s':now-newest['observed'],
                        'sequence':newest['sequence'],'latency_s':newest['latency']})
                    active=newest['actions'];observed=newest['observed'];latency=newest['latency'];newest=None
                    progress.elapsed=0.;initial=adapter.limiter.q.copy()
                    adapter.decision.candidate=adapter.decision.since=None
                    adopted=last_progress=now
                if active is None:
                    if not a.fixed_five and now-wait_start>1.65:raise RuntimeError('No initial fresh prediction')
                    time.sleep(.005)
                    continue
                if not a.fixed_five and now-observed>1.65:raise RuntimeError('Active prediction expired')
                prediction=(active[min(int(round(progress.steps)),len(active)-1)].copy()
                    if a.fixed_timing else interpolate(active,initial,progress.elapsed,30))
                action,_=adapter.step(prediction,float(env.mj_data.time),env.delta_t,
                    action_key=(transitions[-1]['sequence'],min(int(round(progress.steps)),len(active)-1)))
                for event in adapter.intent_events:
                    if event['event']!='evidence':
                        print('CLOSE_INTENT:',json.dumps(dict(episode=episode,**event)),flush=True)
                before_progress=progress.elapsed
                paused=adapter.busy or adapter.decision.candidate is not None or adapter.pending_close
                if a.fixed_timing:
                    lag=float(max(np.max(abs(prediction[:6]-actual)),np.max(abs(prediction[:6]-action[:6]))))
                    if not paused: progress.elapsed+=env.delta_t
                else:
                    lag=progress.advance(prediction[:6],actual,action[:6],env.delta_t,paused=paused)
                if a.fixed_five: progress.elapsed=min(progress.elapsed,a.action_steps/30)
                if progress.elapsed>before_progress:last_progress=time.monotonic()
                clipped+=int(lag>progress.tolerance)
                trace.append({'step':step,'prediction':prediction.tolist(),'applied':action.tolist(),
                              'replanned':bool(replanned),'gripper_progress_paused':bool(paused),'executed_steps':progress.steps,
                              'inference_s':latency,'observation_age_s':now-observed,'takeover_reason':reason_takeover,
                              'gripper_busy':adapter.busy,'grasp_confirmed':adapter.grasp_confirmed,
                              'release_suppressed':adapter.release_suppressed,
                               'pending_close':adapter.pending_close,'close_intent_events':adapter.intent_events,
                               'effective_prediction':adapter.effective_desired.tolist(),
                               'actual_q':actual.tolist(),
                              'tcp_world_m':env.mj_data.site_xpos[env.ee_site_id].tolist()})
                record(env,obs,action,buffers,encoders)
                obs,*_=env.step(action)
                info=env.success_info();peak=max(peak,info['clearance_m'])
                trace[-1]['post_step_success_info']=info
                if checker.table_collision(env.mj_data): reason='arm_table_floor_pedestal_collision';break
                if a.success_mode=='height' and info['lifted']:
                    success=True;reason='height_reached';break
                if info['lifted'] and info['stationary'] and adapter.closed and adapter.has_contact():
                    if settled is None: settled=float(env.mj_data.time)
                    if env.mj_data.time-settled>=cfg.hold_seconds+env.delta_t:
                        success=True;reason='success';break
                else: settled=None
                time.sleep(max(0.,env.delta_t-(time.monotonic()-start)))
        except Exception as exc:
            reason=f'{type(exc).__name__}: {exc}'
        finally:
            executor.shutdown(wait=True,cancel_futures=True)
            for encoder in encoders: encoder.close()
        save_episode(str(folder),buffers,a.seed,episode+1,cfg,success=success,failure=None if success else reason,diagnostic=True)
        payload=json.loads((folder/'obs_action.json').read_text())
        payload['metadata']['control_contract']=contract
        payload['metadata']['max_command_rate_rad_s']=a.max_command_rate
        payload['metadata']['effective_control']=effective
        (folder/'obs_action.json').write_text(json.dumps(payload))
        (folder/'prediction_trace.json').write_text(json.dumps(trace))
        (folder/'transitions.json').write_text(json.dumps(transitions,indent=2))
        row={'episode':episode,'success':success,'reason':reason,'frames':len(trace),
             'policy_seed':episode_seed,'action_steps':config.n_action_steps,
             'initialization':initialization,
             'sampling':cfg.sampling_metadata,'max_clearance_m':peak,'final':env.success_info(),
             'rate_limited_frames':clipped,'transitions':len(transitions),
             'early_takeovers':sum(t['reason'] in ('stalled_progress','expiry_margin') for t in transitions)}
        results.append(row)
        summary={'completed':len(results),'successes':sum(r['success'] for r in results),
                 'success_rate':sum(r['success'] for r in results)/len(results),'episodes':results}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(row),flush=True)
        if reason.startswith('AcceleratorError:'):
            env.renderer.close()
            raise RuntimeError('CUDA evaluation interrupted; episode saved, restart in a new process')
        if references is None and reason.startswith(('ValueError:','RuntimeError:','KeyError:')):
            env.renderer.close()
            raise RuntimeError(reason)
    env.renderer.close()
    print('COMPLETE:',a.output,flush=True)

if __name__=='__main__': main()
