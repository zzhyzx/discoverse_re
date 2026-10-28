"""Validated 30Hz lift episodes -> LeRobot v2.1. No action shift or video re-encoding.

New output only. COMPLETE marker written after real LeRobot loading checks.
Image statistics use every 10th decoded frame, spatially sampled every 8 pixels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import contextlib
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'submodules/lerobot/src'))
os.environ['HF_HUB_OFFLINE'] = '1'
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ur5_validate_lift_dataset import validate
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERAS = ['observation.images.d436', 'observation.images.d405']

def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def stats(a, count, image=False):
    def value(v):
        return v.reshape(3,1,1).tolist() if image else v.tolist()
    return {**{k:value(v) for k,v in dict(mean=a.mean(0),std=a.std(0),min=a.min(0),max=a.max(0)).items()}, 'count':[count]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/ur5_lift_sim_train300_conservative_v1')
    p.add_argument('--output',type=Path,default=ROOT/'data/ur5_lift_sim_v21')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--preserve-source-split',action='store_true',help='Use original real episode membership; never re-split successful replays')
    p.add_argument('--control-profile',type=Path,help='Require this control contract on every selected episode')
    p.add_argument('--successful-only',action='store_true',help='Explicitly exclude failed diagnostic attempts')
    args=p.parse_args(); source=args.source.resolve(); out=args.output.resolve()
    folders=sorted(d for d in source.iterdir() if d.is_dir() and d.name.isdigit())
    contract=None
    if args.control_profile:
        from ur5_control_profile import load_profile
        contract=load_profile(args.control_profile)
    if args.successful_only:
        folders=[d for d in folders if (d/'obs_action.json').is_file() and json.loads((d/'obs_action.json').read_text())['metadata'].get('training_eligible')]
    if contract:
        for folder in folders:
            meta=json.loads((folder/'obs_action.json').read_text())['metadata']
            if meta.get('control_contract')!=contract:raise ValueError(f'Control contract mismatch: {folder}')
    if len(folders)<10: raise ValueError('Need at least ten episodes')
    if len(folders)>1000: raise ValueError('This single-chunk converter supports at most 1000 episodes')
    out.mkdir(parents=True,exist_ok=False)
    for sub in ['meta','conversion','data/chunk-000']:
        (out/sub).mkdir(parents=True)
    features={k:{'dtype':'int64','shape':[1],'names':None} for k in ['frame_index','episode_index','index','task_index']}
    features['timestamp']={'dtype':'float32','shape':[1],'names':None}
    names=['shoulder_pan','shoulder_lift','elbow','wrist_1','wrist_2','wrist_3','gripper']
    for k in ['observation.state','action']:
        features[k]={'dtype':'float32','shape':[7],'names':names}
    for k in CAMERAS:
        features[k]={'dtype':'video','shape':[480,640,3],'names':['height','width','channels'],
                     'info':{'video.fps':30,'video.height':480,'video.width':640,'video.channels':3,'video.is_depth_map':False,'video.has_audio':False,'video.codec':'h264','video.pix_fmt':'yuv420p'}}
    rows=[]; stat_rows=[]; reports=[]; total=0; task=None; groups={}
    for eid,folder in enumerate(folders):
        with contextlib.redirect_stdout(io.StringIO()): report=validate(folder)
        raw=(folder/'obs_action.json').read_bytes(); data=json.loads(raw); meta=data['metadata']
        if not meta.get('training_eligible') or meta.get('diagnostic_only') or not meta.get('success'):
            raise ValueError(f'Not successful training data: {folder}')
        if meta.get('renderer')!='mujoco': raise ValueError('Expected ordinary simulation RGB')
        instruction=data['language_instruction']
        if task is None: task=instruction
        if task!=instruction: raise ValueError('Mixed instructions')
        n=len(data['time']); state=np.asarray(data['obs']['qpos'],dtype=np.float32); action=np.asarray(data['act'],dtype=np.float32)
        columns={'timestamp':pa.array(np.asarray(data['time'],dtype=np.float32)),
                 'frame_index':pa.array(np.arange(n,dtype=np.int64)),
                 'episode_index':pa.array(np.full(n,eid,dtype=np.int64)),
                 'index':pa.array(np.arange(total,total+n,dtype=np.int64)),
                 'task_index':pa.array(np.zeros(n,dtype=np.int64))}
        for key,a in [('observation.state',state),('action',action)]:
            columns[key]=pa.FixedSizeListArray.from_arrays(pa.array(a.ravel()),7)
        pq.write_table(pa.table(columns),out/f'data/chunk-000/episode_{eid:06d}.parquet')
        st={'observation.state':stats(state.astype(float),n),'action':stats(action.astype(float),n)}
        for camera,key in enumerate(CAMERAS):
            src=folder/f'cam_{camera}.mp4'; dest=out/f'videos/chunk-000/{key}/episode_{eid:06d}.mp4'
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
            if hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(dest.read_bytes()).digest():
                raise ValueError('Video copy mismatch')
            pixels=[]
            with av.open(str(dest)) as video:
                for i,frame in enumerate(video.decode(video=0)):
                    if i%10==0: pixels.append(frame.to_ndarray(format='rgb24')[::8,::8].reshape(-1,3))
            st[key]=stats(np.concatenate(pixels).astype(float)/255,n,image=True)
        rows.append({'episode_index':eid,'tasks':[task],'length':n})
        stat_rows.append({'episode_index':eid,'stats':st})
        reports.append({**report,'source_json_sha256':hashlib.sha256(raw).hexdigest(),'episode_index':eid})
        groups.setdefault(meta['sampling']['proposal_grid_cell'],[]).append(eid)
        total+=n
        print(f'Converted {eid+1}/{len(folders)}: {folder.name}, frames={n}',flush=True)
    if args.preserve_source_split:
        original=json.loads((source/'source_split.json').read_text())
        train=[];val=[];mapping=[]
        for eid,folder in enumerate(folders):
            meta=json.loads((folder/'obs_action.json').read_text())['metadata']
            sid=meta['source_episode']
            expected_part='train' if sid in original['train'] else 'validation' if sid in original['validation'] else None
            if expected_part is None or meta['source_split']!=expected_part:raise ValueError('Source split mismatch')
            (train if expected_part=='train' else val).append(eid)
            mapping.append({'episode_index':eid,'source_episode':sid,'split':expected_part})
        if len({m['source_episode'] for m in mapping})!=len(mapping):raise ValueError('Duplicate source episode')
        if not train or not val:raise ValueError('Both original partitions must contain accepted episodes')
        write(out/'conversion/source_episode_mapping.json',mapping)
        write(out/'conversion/source_split.json',original)
    else:
        rng=np.random.default_rng(args.seed); chosen=[]; leftovers=[]
        for ids in groups.values():
            ids=rng.permutation(ids).tolist(); take=int(len(ids)*.1)
            chosen.extend(ids[:take]); leftovers.extend(ids[take:])
        target=max(1,round(len(folders)*.1)); rng.shuffle(leftovers)
        chosen+=leftovers[:target-len(chosen)]; val=sorted(chosen)
        train=sorted(set(range(len(folders)))-set(val))
    info=dict(codebase_version='v2.1',robot_type='ur5',total_episodes=len(rows),total_frames=total,
              total_tasks=1,total_videos=2*len(rows),total_chunks=1,chunks_size=1000,fps=30,
              splits={'train':f'0:{len(rows)}'},features=features,
              data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
              video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
    write(out/'meta/info.json',info)
    for name,items in [('episodes',rows),('episodes_stats',stat_rows),('tasks',[{'task_index':0,'task':task}])]:
        with (out/f'meta/{name}.jsonl').open('x') as f:
            for row in items: f.write(json.dumps(row,allow_nan=False)+'\n')
    write(out/'conversion/episode_split.json',{'train':train,'validation':val,'seed':args.seed,'method':('preserved original real 40/10 membership after acceptance filtering' if args.preserve_source_split else 'episode split, spatial grid proportional allocation then remainder')})
    write(out/'conversion/validation.json',reports)
    dataset=LeRobotDataset('local/'+out.name,root=out,video_backend='pyav')
    for eid,folder in enumerate(folders):
        original=json.loads((folder/'obs_action.json').read_text())
        start=int(dataset.episode_data_index['from'][eid]); end=int(dataset.episode_data_index['to'][eid])
        for idx,rawidx in [(start,0),(end-1,-1)]:
            sample=dataset[idx]
            assert np.array_equal(sample['action'].numpy(),np.asarray(original['act'][rawidx],dtype=np.float32))
            assert np.array_equal(sample['observation.state'].numpy(),np.asarray(original['obs']['qpos'][rawidx],dtype=np.float32))
            assert all(tuple(sample[c].shape)==(3,480,640) for c in CAMERAS)
            assert sample['task']==task
    if contract: write(out/'conversion/control_contract.json',contract)
    write(out/'conversion/COMPLETE.json',{'episodes':len(rows),'frames':total,'train':len(train),'validation':len(val),
          'source':str(source),'action_shift':0,'videos':'byte-identical copies',
          'image_stats':'every tenth frame, every eighth pixel; per-episode',
          'gripper_physical_calibration':False})
    print('COMPLETE',out,flush=True)

if __name__=='__main__': main()
