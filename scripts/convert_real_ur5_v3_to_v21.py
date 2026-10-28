"""Offline v3 -> local LeRobot v2.1 export. Never overwrite source or output.

Preserves absolute joint targets, raw RGB, language, and episode boundaries.
Videos: lossless libx264rgb, verified against source decoded RGB hashes.
No calibration, action unwrapping, gripper rescaling, or robot connection.
"""
import argparse
import copy
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'submodules/lerobot/src'))
os.environ['HF_HUB_OFFLINE'] = '1'
import av
import datasets
import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_hf_features_from_features
from lerobot.datasets.compute_stats import get_feature_stats

JOINTS = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
          'wrist_1_joint','wrist_2_joint','wrist_3_joint']


def write_json(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def export_video(source, dest, start, n, fps):
    dest.parent.mkdir(parents=True, exist_ok=True)
    hist = np.zeros((3,256), dtype=np.int64)
    digest = hashlib.sha256()
    max_time_error = 0.
    with av.open(str(source)) as inp, av.open(str(dest), 'w') as out:
        stream = inp.streams.video[0]
        if (stream.width,stream.height)!=(640,480):
            raise ValueError('Unexpected input resolution')
        inp.seek(max(0,int(start/stream.time_base)),stream=stream,backward=True)
        encoder = out.add_stream('libx264rgb',rate=fps)
        encoder.width,encoder.height,encoder.pix_fmt=640,480,'rgb24'
        encoder.options={'crf':'0','preset':'veryfast','g':'2'}
        encoder.time_base=Fraction(1,fps)
        iterator=inp.decode(stream)
        for i in range(n):
            desired=start+i/fps
            frame=next(iterator)
            stamp=float(frame.pts*frame.time_base)
            while stamp<desired-.51/fps:
                frame=next(iterator);stamp=float(frame.pts*frame.time_base)
            if abs(stamp-desired)>.51/fps:
                raise ValueError(f'Video sync error {source}: {stamp} vs {desired}')
            max_time_error=max(max_time_error,abs(stamp-desired))
            rgb=frame.to_ndarray(format='rgb24')
            digest.update(rgb.tobytes())
            for c in range(3):
                hist[c]+=np.bincount(rgb[:,:,c].ravel(),minlength=256)
            encoded=av.VideoFrame.from_ndarray(rgb,format='rgb24')
            encoded.pts=i;encoded.time_base=Fraction(1,fps)
            for packet in encoder.encode(encoded): out.mux(packet)
        for packet in encoder.encode(): out.mux(packet)
    verified=hashlib.sha256();count=0
    with av.open(str(dest)) as check:
        for i,frame in enumerate(check.decode(video=0)):
            if abs(float(frame.pts*frame.time_base)-i/fps)>1e-6:
                raise ValueError('Output PTS mismatch')
            verified.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
    if count!=n or verified.digest()!=digest.digest():
        raise ValueError('Lossless RGB verification failed')
    values=np.arange(256,dtype=np.float64)/255
    mean=(hist*values).sum(axis=1)/hist.sum(axis=1)
    var=(hist*values**2).sum(axis=1)/hist.sum(axis=1)-mean**2
    stats=dict(mean=mean.reshape(3,1,1),std=np.sqrt(np.maximum(var,0)).reshape(3,1,1),
               min=np.array([values[np.flatnonzero(row)[0]] for row in hist]).reshape(3,1,1),
               max=np.array([values[np.flatnonzero(row)[-1]] for row in hist]).reshape(3,1,1),
               count=np.array([n]))
    return stats,dict(frames=n,decoded_rgb_sha256=digest.hexdigest(),lossless_verified=True,
                      max_source_timestamp_error_s=max_time_error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/real_ur5_pick_cube_v2/ur5_pick_cube_v2')
    p.add_argument('--output',type=Path,default=ROOT/'data/real_ur5_pick_cube_v21')
    p.add_argument('--max-episodes',type=int,help='Smoke test only; default exports all episodes')
    a=p.parse_args();source=a.source.resolve();output=a.output.resolve()
    if output==source or source in output.parents or output in source.parents:
        p.error('Source/output must be separate, non-nested paths')
    if output.exists(): p.error('Output already exists; choose a new --output. No overwrite/resume.')
    info=json.loads((source/'meta/info.json').read_text())
    if info['codebase_version']!='v3.0': raise ValueError('Expected v3.0')
    fps=int(info['fps'])
    if fps!=info['fps']: raise ValueError('Expected integer FPS')
    episodes=[]
    for path in sorted((source/'meta/episodes').rglob('*.parquet')):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda e:e['episode_index'])
    if [e['episode_index'] for e in episodes]!=list(range(info['total_episodes'])):
        raise ValueError('Episode IDs/count mismatch')
    if a.max_episodes is not None:
        if a.max_episodes<1: p.error('max-episodes must be positive')
        episodes=episodes[:a.max_episodes]
    tasks=sorted(pq.read_table(source/'meta/tasks.parquet').to_pylist(),key=lambda t:t['task_index'])
    if [t['task_index'] for t in tasks]!=list(range(len(tasks))): raise ValueError('Task IDs mismatch')
    features=copy.deepcopy(info['features'])
    features['action']['names']=JOINTS+['gripper_command']
    cameras=[k for k,f in features.items() if f['dtype']=='video']
    if set(cameras)!={'observation.images.d436','observation.images.d405'}:
        raise ValueError('Unexpected camera keys')
    # Old metadata helper fails to recognize gbrp as three-channel RGB.
    # Describe the actual lossless stream explicitly; verify with the real loader.
    for key in cameras:
        features[key]['info']={'video.height':480,'video.width':640,
            'video.codec':'h264','video.pix_fmt':'gbrp','video.fps':fps,
            'video.channels':3,'video.is_depth_map':False,'has_audio':False}
    meta=LeRobotDatasetMetadata.create('local/real_ur5_pick_cube_v21',fps,features,
                                     root=output,robot_type='ur5',use_videos=True)
    for t in tasks: meta.add_task(t['task'])
    provenance=output/'conversion';provenance.mkdir()
    write_json(provenance/'source_info.json',info)
    write_json(provenance/'started.json',dict(source=str(source),note='Only COMPLETE.json indicates a verified export'))
    hf=get_hf_features_from_features(meta.features)
    cache={};reports=[]
    for ep in episodes:
        eid=int(ep['episode_index']);n=int(ep['length'])
        rel=info['data_path'].format(chunk_index=ep['data/chunk_index'],file_index=ep['data/file_index'])
        if rel not in cache: cache[rel]=pq.read_table(source/rel).to_pydict()
        raw=cache[rel];mask=np.asarray(raw['episode_index'])==eid
        order=np.argsort(np.asarray(raw['frame_index'])[mask])
        data={key:np.asarray(raw[key])[mask][order] for key in hf}
        if len(data['index'])!=n or not np.array_equal(data['frame_index'],np.arange(n)):
            raise ValueError(f'Episode {eid}: frame mismatch')
        if not np.array_equal(data['index'],np.arange(ep['dataset_from_index'],ep['dataset_to_index'])):
            raise ValueError('Global indices mismatch')
        if not np.allclose(data['timestamp'],np.arange(n)/fps,atol=2e-6):
            raise ValueError('Sample timestamps mismatch')
        for key,arr in data.items():
            if not np.isfinite(arr).all(): raise ValueError(f'Nonfinite {key}')
        if data['action'].shape!=(n,7) or data['observation.state'].shape!=(n,7):
            raise ValueError('Expected 7D state/action')
        if not np.isin(data['action'][:,-1],[0,1]).all(): raise ValueError('Nonbinary gripper command')
        if not np.isin(data['task_index'],range(len(tasks))).all(): raise ValueError('Unknown task')
        dest=output/meta.get_data_file_path(eid);dest.parent.mkdir(parents=True,exist_ok=True)
        datasets.Dataset.from_dict(data,features=hf).to_parquet(str(dest))
        reread=pq.read_table(dest).to_pydict()
        for key,arr in data.items():
            if not np.array_equal(np.asarray(reread[key]),arr): raise ValueError(f'Numeric roundtrip changed {key}')
        stats={key:get_feature_stats(arr.astype(np.float64),axis=0,keepdims=arr.ndim==1)
               for key,arr in data.items()}
        videos={}
        for key in cameras:
            relvideo=info['video_path'].format(video_key=key,chunk_index=ep[f'videos/{key}/chunk_index'],
                                              file_index=ep[f'videos/{key}/file_index'])
            start=float(ep[f'videos/{key}/from_timestamp']);end=float(ep[f'videos/{key}/to_timestamp'])
            if abs(end-start-n/fps)>1e-4: raise ValueError('Video metadata duration mismatch')
            stats[key],videos[key]=export_video(source/relvideo,output/meta.get_video_file_path(eid,key),start,n,fps)
            videos[key].update(source=relvideo,source_start_s=start,source_end_s=end)
        texts=[tasks[int(i)]['task'] for i in sorted(set(data['task_index']))]
        meta.save_episode(eid,n,texts,stats)
        reports.append(dict(episode=eid,frames=n,numeric_roundtrip_exact=True,videos=videos))
        print(f'Converted {eid+1}/{len(episodes)}: {n} frames, two RGB-exact videos',flush=True)
    if a.max_episodes is None and meta.total_frames!=info['total_frames']:
        raise ValueError('Total frame count mismatch')
    # Test the actual local training loader, both cameras and action chunk boundaries.
    ds=LeRobotDataset(meta.repo_id,root=output,video_backend='pyav',
                     delta_timestamps={'action':[i/fps for i in range(50)]})
    tested=0;offset=0
    for ep in episodes:
        for idx in sorted({offset,offset+ep['length']//2,offset+ep['length']-1}):
            sample=ds[idx]
            if tuple(sample['action'].shape)!=(50,7): raise ValueError('Action chunk shape mismatch')
            if tuple(sample['observation.state'].shape)!=(7,): raise ValueError('State shape mismatch')
            for key in cameras:
                if tuple(sample[key].shape)!=(3,480,640): raise ValueError('Image loader shape mismatch')
            tested+=1
        offset+=ep['length']
    ids=np.random.default_rng(42).permutation(len(episodes));nv=max(1,round(len(ids)*.2)) if len(ids)>1 else 0
    val=sorted(ids[:nv].tolist());train=sorted(ids[nv:].tolist())
    write_json(provenance/'episode_split.json',dict(seed=42,train=train,validation=val,
        note='Episode-disjoint random split; not stratified by outcome/position. Select train episodes explicitly; compute training stats from train only.'))
    write_json(provenance/'report.json',dict(complete=True,source=str(source),output=str(output),
        format='v2.1',episodes=len(episodes),frames=meta.total_frames,fps=fps,
        action='absolute six joint targets in radians + binary gripper 0=open 1=closed',
        state='six measured joint angles in radians + unchanged recorded gripper feedback',
        tcp='observation.tcp_pose retained for diagnostics; exclude explicitly from policy inputs',
        images='original decoded RGB; lossless libx264rgb; no rectification/resize/crop',
        stats='recomputed per episode; float64 numeric stats; all-pixel RGB stats; count=frames',
        synchronization='source timestamps preserved; hardware synchronization assumed, not measured',
        loader_samples_tested=tested,video_backend='pyav',datasets_version=datasets.__version__,
        tasks=tasks,episode_reports=reports))
    # Completion marker is separate: partial exports never acquire this file.
    write_json(provenance/'COMPLETE.json',dict(complete=True,episodes=len(episodes),frames=meta.total_frames))
    print('VERIFIED OUTPUT:',output,flush=True)


if __name__=='__main__':
    main()
