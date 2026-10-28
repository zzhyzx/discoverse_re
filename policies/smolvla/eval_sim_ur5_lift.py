"""MuJoCo-only policy closed-loop evaluation. Never connects to robot interfaces.

Simulation clock pauses for inference: not a real-time deployment benchmark.
All sampled positions are attempted. Optional IK initialization is documented;
the policy controls every evaluated step after initialization.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'submodules/lerobot/src')]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import (
    SimNode, Expert, make_config, new_buffers, record, save_episode,
    PyavImageEncoder, TASK_INSTRUCTION)
from scripts.ur5_sim_grasp_hold import GraspHold, BilateralPadContact

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
    p.add_argument('--episode-start',type=int,default=0,
                   help='Skip earlier scene resets, preserving scene RNG and episode seeds; use a new output directory')
    p.add_argument('--episode-ids',type=int,nargs='+',default=None,
                   help='Evaluate only these original scene IDs; still reset skipped scenes to preserve layouts and seeds')
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--success-mode',choices=['stable','height'],default='stable',
                   help='height: block bottom reaches 5cm; no hold or speed requirement (may include transient lift)')
    p.add_argument('--hold-after-grasp',action='store_true',
                   help='Lift-only: after bilateral pad contact and settled closure, ignore open predictions until episode ends')
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
    p.add_argument('--y-side',choices=['any','positive','negative'],default='any',
                   help='Reflect sampled world Y into one half; same seed gives paired X and |Y| across sides')
    p.add_argument('--action-steps',type=int,default=None,
                   help='Actions executed before replanning; default: checkpoint setting')
    p.add_argument('--max-command-rate',type=float,default=.9,
                   help='Simulation joint target slew limit in rad/s (0 < value <= 1.2); not actual joint velocity')
    a=p.parse_args();sys.argv=sys.argv[:1]
    if a.episodes<1 or not 0<a.seconds<=60: raise ValueError('Invalid episodes or timeout')
    if not 0<=a.episode_start<a.episodes: raise ValueError('Invalid episode-start')
    selected_ids=None if a.episode_ids is None else set(a.episode_ids)
    if selected_ids is not None and (len(selected_ids)!=len(a.episode_ids)
            or any(i<a.episode_start or i>=a.episodes for i in selected_ids)):
        raise ValueError('--episode-ids must be unique and within [episode-start, episodes)')
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
            if a.y_side!='any':
                source_y=float(xy[1])
                if source_y==0.:
                    raise ValueError('Cannot assign a zero-Y sample to a strict half-plane')
                xy[1]=abs(source_y)*(1. if a.y_side=='positive' else -1.)
                self.config.sampling_metadata.update(sampled_xy_m=xy.tolist(),
                    source_y_before_reflection_m=source_y,requested_y_side=a.y_side)
            if a.block_yaw_deg is not None:
                yaw=np.deg2rad(a.block_yaw_deg)
                self.config.sampling_metadata['yaw_deg']=float(a.block_yaw_deg)
            return xy,yaw
    env=EvaluationSimNode(cfg,np.random.default_rng(a.seed),randomize=True)
    # Optional initialization uses IK; evaluated actions are always policy outputs.
    checker=Expert(env)
    protocol={'checkpoint':str(a.checkpoint.resolve()),'seed':a.seed,'episodes':a.episodes,'episode_start':a.episode_start,
              'selected_episode_ids':sorted(selected_ids) if selected_ids is not None else None,
              'timeout_sim_seconds':a.seconds,'fps':30,'action_stride':config.n_action_steps,
              'checkpoint_action_stride':saved_action_steps,
              'policy_seed_mode':'reseed torch/numpy with seed + episode after each reset',
              'max_command_rate_rad_s':a.max_command_rate,'gripper':'threshold 0.5, open=0 closed=255',
              'hold_after_grasp':a.hold_after_grasp,
              'grasp_hold_confirmation':({'bilateral_pad_contact':True,'settled_feedback_delta':.002,
                  'consecutive_frames':3,'release':'episode reset only',
                  'feedback_source':'MuJoCo contacts; no object pose or height used for command',
                  'scope':'simulation execution aid; does not change success criteria'} if a.hold_after_grasp else None),
              'gravity_compensation':True,'expert_control':False,'preflight_filter':False,
              'start_mode':a.start_mode,'above_height_m':a.above_height if a.start_mode=='above-block' else None,
              'initialization_uses_object_pose_and_ik':a.start_mode=='above-block',
              'initial_offset_world_xy_m':list(a.initial_offset_xy),'initial_yaw_offset_deg':a.initial_yaw_deg,
              'fixed_block_yaw_deg':a.block_yaw_deg,
              'requested_y_side':a.y_side,
              'interpretation':'partial task from initialized pregrasp' if a.start_mode=='above-block' else 'full task from home',
              'timing':'simulation pauses during inference; not real-time',
              'task':a.task,'success_mode':a.success_mode,
              'success':('block bottom >= 0.05m; no hold/speed requirement, transient lifts count' if a.success_mode=='height' else 'block bottom >= 0.05m, speed <=0.03m/s, angular speed <=0.3rad/s continuously >=1s'),
              'collision_scope':'same arm-vs-table/floor/pedestal predicate as collection; not full self collision'}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    for episode in range(a.episodes):
        if episode<a.episode_start or (selected_ids is not None and episode not in selected_ids):
            env.reset()
            continue
        folder=a.output/f'{episode:03d}';folder.mkdir()
        obs=env.reset();policy.reset()
        initialization={'mode':'home','q_rad':env.normalized_state()[:6].tolist(),
                        'block_pos_world_m':env.block_pos().tolist(),'gripper_state':float(env.normalized_state()[6])}
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
        (folder/'initialization.json').write_text(json.dumps(initialization,indent=2))
        action=env.target_control.copy()
        grasp_hold=GraspHold() if a.hold_after_grasp else None
        pad_contact=BilateralPadContact(env.mj_model) if a.hold_after_grasp else None
        # Separate scene RNG; episode length must not affect later policy seeds.
        episode_seed=a.seed+episode
        torch.manual_seed(episode_seed);np.random.seed(episode_seed)
        cfg.sampling_metadata['filter']='none; no expert preflight'
        buffers=new_buffers();trace=[];settled=None;success=False;reason='timeout';peak=0.;clipped=0
        encoders=[PyavImageEncoder(640,480,str(folder),i,fps=30,crf=18,preset='veryfast') for i in (0,1)]
        try:
            for step in range(int(a.seconds*30)):
                batch={'observation.state':torch.from_numpy(env.normalized_state()).unsqueeze(0).cuda(),
                       'task':[a.task]}
                for camera,key in enumerate(['observation.images.d436','observation.images.d405']):
                    image=np.asarray(obs['img'][camera]).copy()
                    batch[key]=torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float().cuda()/255
                start=time.monotonic()
                with torch.inference_mode(): prediction=policy.select_action(batch)[0].cpu().numpy()
                if prediction.shape!=(7,) or not np.isfinite(prediction).all(): raise ValueError('Invalid action')
                limits=env.mj_model.actuator_ctrlrange[:6]
                target=np.clip(prediction[:6],limits[:,0],limits[:,1])
                delta=target-action[:6]
                max_step=a.max_command_rate*env.delta_t
                clipped+=int(np.any(np.abs(delta)>max_step))
                action[:6]+=np.clip(delta,-max_step,max_step)
                contact=pad_contact(env.mj_data) if pad_contact is not None else None
                was_confirmed=grasp_hold.confirmed if grasp_hold is not None else False
                action[6]=(grasp_hold.command(float(prediction[6]),float(env.normalized_state()[6]),
                           contact,float(env.mj_data.time)) if grasp_hold is not None
                           else (255. if prediction[6]>=.5 else 0.))
                if grasp_hold is not None and grasp_hold.confirmed and not was_confirmed:
                    print(f'episode {episode:03d}: grasp confirmed at {env.mj_data.time:.3f}s; holding closed',flush=True)
                trace.append({'step':step,'prediction':prediction.tolist(),'applied':action.tolist(),
                              'replanned':step % config.n_action_steps == 0,
                              'inference_s':time.monotonic()-start})
                if grasp_hold is not None:
                    trace[-1].update(bilateral_pad_contact=contact,grasp_confirmed=grasp_hold.confirmed,
                                     release_suppressed=grasp_hold.release_suppressed)
                record(env,obs,action,buffers,encoders)
                obs,*_=env.step(action)
                info=env.success_info();peak=max(peak,info['clearance_m'])
                trace[-1]['post_step_success_info']=info
                if checker.table_collision(env.mj_data): reason='arm_table_floor_pedestal_collision';break
                if a.success_mode=='height' and info['lifted']:
                    success=True;reason='height_reached';break
                if info['lifted'] and info['stationary']:
                    if settled is None: settled=float(env.mj_data.time)
                    if env.mj_data.time-settled>=cfg.hold_seconds+env.delta_t:
                        success=True;reason='success';break
                else: settled=None
        except Exception as exc:
            reason=f'{type(exc).__name__}: {exc}'
        finally:
            for encoder in encoders: encoder.close()
        save_episode(str(folder),buffers,a.seed,episode+1,cfg,success=success,failure=None if success else reason,diagnostic=True)
        (folder/'prediction_trace.json').write_text(json.dumps(trace))
        row={'episode':episode,'success':success,'reason':reason,'frames':len(trace),
             'policy_seed':episode_seed,'action_steps':config.n_action_steps,
             'initialization':initialization,
             'sampling':cfg.sampling_metadata,'max_clearance_m':peak,'final':env.success_info(),
             'rate_limited_frames':clipped}
        if grasp_hold is not None:
            row.update(grasp_confirmed_at_s=grasp_hold.confirmed_at,
                       release_suppressed_frames=grasp_hold.suppressed_frames)
        results.append(row)
        summary={'completed':len(results),'successes':sum(r['success'] for r in results),
                 'success_rate':sum(r['success'] for r in results)/len(results),'episodes':results}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(row),flush=True)
        if reason.startswith(('ValueError:','RuntimeError:','KeyError:')): raise RuntimeError(reason)
    print('COMPLETE:',a.output,flush=True)

if __name__=='__main__': main()
