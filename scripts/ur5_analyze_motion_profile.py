"""Offline motion statistics; outputs SIMULATION candidates, never robot limits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
def stats(values):
    a=np.asarray(values)
    return {str(p):np.percentile(np.abs(a),p,axis=0).tolist() for p in (50,95,99,100)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'data/ur5_motion_profile_20261027')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/real_ur5_pick_cube_v21'
    split=json.loads((ROOT/'policies/smolvla/checkpoints/real_ur5_20260924_151943/real_training.json').read_text())
    selected=set(split['train_episodes'])
    groups={name:{k:[] for k in ('action_velocity','state_velocity','action_acceleration_smoothed','state_acceleration_smoothed','action_acceleration_raw')}
            for name in ('train','validation')}
    episodes=[]
    for path in sorted((data/'data').rglob('*.parquet')):
        d=pq.read_table(path).to_pydict()
        ids=np.asarray(d['episode_index'])
        for ep in np.unique(ids):
            ix=ids==ep
            t=np.asarray(d['timestamp'])[ix].astype(float)
            a=np.asarray(d['action'])[ix].astype(float)
            q=np.asarray(d['observation.state'])[ix].astype(float)
            if len(t)<15 or not np.isfinite(np.r_[t,a.ravel(),q.ravel()]).all():
                raise ValueError(f'Invalid episode {ep}')
            dt=np.diff(t)
            if np.any(dt<=0) or not np.allclose(dt,1/30,atol=2e-5):
                raise ValueError(f'Nonuniform timestamps episode {ep}')
            group='train' if int(ep) in selected else 'validation'
            for label,x in [('action',a),('state',q)]:
                v=np.diff(x[:,:6],axis=0)/dt[:,None]
                groups[group][label+'_velocity'].extend(v.tolist())
                # Resample only for smoothing derivatives; never join episodes.
                uniform=np.arange(len(t))/30+t[0]
                u=np.stack([np.interp(uniform,t,x[:,j]) for j in range(6)],axis=1)
                acc=savgol_filter(u,9,3,deriv=2,delta=1/30,axis=0)[4:-4]
                groups[group][label+'_acceleration_smoothed'].extend(acc.tolist())
                if label=='action':
                    raw=np.diff(v,axis=0)/((dt[1:]+dt[:-1])/2)[:,None]
                    groups[group]['action_acceleration_raw'].extend(raw.tolist())
            closing=np.flatnonzero((a[1:,6]>=.5)&(a[:-1,6]<.5))+1
            entries=[]
            for i in closing:
                response=np.flatnonzero(np.abs(q[i:,6]-q[i,6])>=.05)
                entries.append(dict(command_time_s=float(t[i]-t[0]),
                    feedback_change_05_delay_s=float(t[i+response[0]]-t[i]) if len(response) else None,
                    note='feedback change is NOT contact confirmation'))
            tcp=np.asarray(d['observation.tcp_pose'])[ix] if 'observation.tcp_pose' in d else None
            if tcp is not None:
                for item,i in zip(entries,closing):
                    lift=np.flatnonzero(tcp[i:,2]>=tcp[i,2]+.02)
                    item['tcp_rise_2cm_delay_s']=float(t[i+lift[0]]-t[i]) if len(lift) else None
            episodes.append(dict(episode=int(ep),split=group,frames=len(t),duration_s=float(t[-1]-t[0]),close_events=entries))
    result=dict(dataset=str(data),train_episodes=split['train_episodes'],
        validation_episodes=split['validation_episodes'],statistics={g:{k:stats(v) for k,v in ds.items()} for g,ds in groups.items()},
        episodes=episodes,method='30Hz finite differences; acceleration SG window9/order3, four boundary samples excluded. No recorded joint velocities in converted schema. Pooled per-frame percentiles.',
        limitations=['Not hardware limits or certified safe limits','TCP rise does not prove object lift','No contact/force labels; gripper speed/force cannot be inferred','Candidate derived from training split only'])
    (args.output/'motion_statistics.json').write_text(json.dumps(result,indent=2))
    tr=result['statistics']['train']
    # Rounded P95 candidates, clamped to current entry-point bounds; not auto-applied.
    v=np.minimum(.2,np.maximum(.03,np.ceil(np.array(tr['action_velocity']['95'])*100)/100))
    ac=np.minimum(.5,np.maximum(.05,np.ceil(np.array(tr['action_acceleration_smoothed']['95'])*100)/100))
    candidate=dict(status='simulation_only_not_real_robot_approved',action_hz=30,
        joint_names=['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3'],
        candidate_speed_rad_s=v.tolist(),candidate_acceleration_rad_s2=ac.tolist(),
        derivation='training action P95 rounded up .01; clipped [.03,.2] speed and [.05,.5] acceleration. Engineering candidate, not inferred controller settings.',
        baseline=dict(speed_rad_s=.03,acceleration_rad_s2=.05),
        gripper=dict(open=0,closed=1,physical_calibration=False,speed_force_known=False))
    (args.output/'simulation_candidate.json').write_text(json.dumps(candidate,indent=2))
    for label in ('action_velocity','state_velocity','action_acceleration_smoothed','action_acceleration_raw'):
        print(label,'P95',np.round(tr[label]['95'],4).tolist())
    print('CANDIDATE',v.tolist(),ac.tolist())
    print('episodes',len(episodes),'train',len(selected))
if __name__=='__main__':main()
