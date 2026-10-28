"""Offline matched-state chunk diagnosis. No robot/network interfaces."""
import sys,json,gc,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from policies.smolvla.eval_sim_ur5_low_speed import (
 np,torch,mujoco,Rotation,SimNode,Expert,make_config,load_profile,
 SimCommandAdapter,ExecutionProgress,interpolate,PreTrainedConfig,SmolVLAPolicy,TASK_INSTRUCTION)
OUT=ROOT/'data/ur5_descent_chunk_diagnosis_v1'
OUT.mkdir(exist_ok=False)
cfg=make_config(True)
cfg.initial_camera_binding=True;cfg.align_base_table_centerline=True;cfg.render_shadows=False
cfg.lift_only=True;cfg.use_real_training_start=True;cfg.arm_gravity_compensation=True
env=SimNode(cfg,np.random.default_rng(123),randomize=True)
profile=load_profile();checker=Expert(env)
m=env.mj_model
fk=mujoco.MjData(m)
cases=[]
def restore(case):
 env.reset()
 d=env.mj_data;d.qpos[:]=case['qpos'];d.qvel[:]=0;d.time=0
 env.target_control[:6]=d.qpos[:6];env.target_control[6]=0
 mujoco.mj_forward(m,d)
def tcp(q,case):
 fk.qpos[:]=case['qpos'];fk.qpos[:6]=q[:6];fk.qvel[:]=0
 mujoco.mj_forward(m,fk)
 return fk.site_xpos[env.ee_site_id].copy()
for source in [5,17]:
 raw=json.loads((ROOT/f'data/ur5_real_reference_low_speed_v1/{source:03d}/obs_action.json').read_text())
 ee=np.array(raw['obs']['ee_pos']);block=np.array(raw['obs']['block_pos'])
 state=np.array(raw['obs']['qpos']);acts=np.array(raw['act'])
 for phase,height in [('onset',ee[0,2]-.002),('middle',.60),('near_grasp',.535)]:
  mask=(np.linalg.norm(ee[:,:2]-block[:,:2],axis=1)<.005)&(ee[:,2]<height)&(state[:,6]<.05)&(acts[:,6]<.05)
  idx=int(np.flatnonzero(mask)[0])
  env.reset();d=env.mj_data
  d.qpos[:6]=state[idx,:6];d.qvel[:]=0;d.time=0
  d.qpos[env.block_qpos_adr:env.block_qpos_adr+3]=block[idx]
  d.qpos[env.block_qpos_adr+3:env.block_qpos_adr+7]=Rotation.from_euler('z',raw['metadata']['sampling']['yaw_rad']).as_quat()[[3,0,1,2]]
  env.target_control[:6]=d.qpos[:6];env.target_control[6]=0
  mujoco.mj_forward(m,d);env.render_cnt=0;env.render();obs=env.getObservation()
  case=dict(id=f'{source:03d}_{phase}',qpos=d.qpos.copy(),state=env.normalized_state().copy(),
   images=[np.asarray(obs['img'][c]).copy() for c in (0,1)],frame=idx,time=raw['time'][idx],
   expert=acts[idx:idx+50].copy(),task=raw['language_instruction'])
  assert np.linalg.norm(tcp(case['state'],case)-ee[idx])<.001
  np.savez_compressed(OUT/(case['id']+'_input.npz'),qpos=case['qpos'],state=case['state'],
     front=case['images'][0],wrist=case['images'][1],expert=case['expert'])
  cases.append(case)
def layers(actions,case,feedback=False):
 restore(case);adapter=SimCommandAdapter(env,profile)
 progress=ExecutionProgress(30,10,np.radians(.5))
 initial=case['state'][:6];start=tcp(initial,case)
 commanded=[];actual=[];used=[];error=None
 for tick in range(300 if feedback else len(actions)):
  target=interpolate(actions,initial,progress.elapsed,30) if feedback else actions[tick]
  before=env.normalized_state()[:6]
  try: command,_=adapter.step(target,float(env.mj_data.time),env.delta_t)
  except Exception as exc: error=str(exc);break
  if feedback:
   progress.advance(target[:6],before,command[:6],env.delta_t,paused=adapter.busy or adapter.decision.candidate is not None)
   progress.elapsed=min(progress.elapsed,10/30)
  commanded.append(tcp(command,case));used.append(target.tolist())
  for _ in range(env.decimation):
   env.updateControl(command);mujoco.mj_step(m,env.mj_data)
  mujoco.mj_forward(m,env.mj_data)
  actual.append(env.mj_data.site_xpos[env.ee_site_id].copy())
  if checker.table_collision(env.mj_data):error='collision';break
  if feedback and progress.steps>=10-1e-8:break
 result=dict(error=error,ticks=len(actual),progress=progress.steps if feedback else len(actual))
 if actual:
  result.update(command_end_dz_mm=float((commanded[-1][2]-start[2])*1000),
    actual_end_dz_mm=float((actual[-1][2]-start[2])*1000),
    actual_min_dz_mm=float((np.min(np.array(actual)[:,2])-start[2])*1000))
 return result,dict(command_tcp=commanded,actual_tcp=actual,used=used)
models=[
 ('low_010000','sim_real_reference_low_speed_v1','010000'),
 ('real_010000','real_ur5_20260924_151943','010000'),
 ('sim_010000','sim_ur5_lift_20261020_234246','010000'),
 ('sim_023000','sim_ur5_lift_20261020_234246','023000'),
 ('recovery_001000','sim_ur5_recovery_mix_20261025','001000')]
rows=[];manifest=[]
for label,run,step in models:
 path=ROOT/f'policies/smolvla/checkpoints/{run}/checkpoints/{step}/pretrained_model'
 assert (path/'model.safetensors').exists(),path
 config=PreTrainedConfig.from_pretrained(path);config.device='cuda';config.push_to_hub=False
 policy=SmolVLAPolicy.from_pretrained(path,config=config).cuda().eval()
 manifest.append(dict(label=label,checkpoint=str(path),chunk_size=config.chunk_size))
 for case in cases:
  batch={'observation.state':torch.from_numpy(case['state']).unsqueeze(0).cuda(),'task':[case['task']]}
  for key,img in zip(['observation.images.d436','observation.images.d405'],case['images']):
   batch[key]=torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda()/255
  for seed in [101,102,103]:
   torch.manual_seed(seed);policy.reset()
   with torch.inference_mode(): actions=policy.predict_action_chunk(batch)[0].cpu().numpy()
   assert actions.shape==(50,7) and np.isfinite(actions).all()
   start=tcp(case['state'],case);points=np.array([tcp(q,case) for q in actions])
   expert=np.array([tcp(q,case) for q in case['expert']])
   nominal,nt=layers(actions,case)
   gated,gt=layers(actions,case,True)
   row=dict(model=label,case=case['id'],seed=seed,
     raw_dz_mm={str(h):float((points[h-1,2]-start[2])*1000) for h in [1,5,10,30,50]},
     expert_dz_mm={str(h):float((expert[h-1,2]-start[2])*1000) for h in [1,5,10,30,50]},
     raw_xy_end_mm=float(np.linalg.norm(points[-1,:2]-start[:2])*1000),
     raw_grip_max=float(actions[:,6].max()),expert_grip_max=float(case['expert'][:,6].max()),
     nominal_30hz=nominal,feedback_first10=gated)
   rows.append(row)
   np.savez_compressed(OUT/f'{label}_{case["id"]}_{seed}.npz',actions=actions,raw_tcp=points,expert_tcp=expert,
       nominal_command_tcp=nt['command_tcp'],nominal_actual_tcp=nt['actual_tcp'],
       gated_command_tcp=gt['command_tcp'],gated_actual_tcp=gt['actual_tcp'])
   (OUT/'results.json').write_text(json.dumps(rows,indent=2))
  print(label,case['id'],'done',flush=True)
 del policy;gc.collect();torch.cuda.empty_cache()
(OUT/'manifest.json').write_text(json.dumps(dict(models=manifest,seeds=[101,102,103],
 cases=[dict(id=c['id'],frame=c['frame'],time=c['time'],state=c['state'].tolist(),task=c['task']) for c in cases],
 note='Same rendered observations across models. Static open-gripper initialization; velocities zeroed. Original checkpoint normalization retained. Real/older models evaluated out of native data domain; not success-rate ranking.'),indent=2))
env.renderer.close()
print('COMPLETE',OUT,flush=True)
