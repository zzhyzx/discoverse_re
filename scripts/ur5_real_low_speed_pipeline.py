"""Offline resumable 50-episode collection -> acceptance -> conversion -> training.
No robot/network control imports. Stage failures never fall through to training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable
def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'data/ur5_real_reference_low_speed_v1')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    logs=root/'pipeline_logs';logs.mkdir(exist_ok=True)
    dataset=ROOT/'data/ur5_real_reference_low_speed_v21'
    output=ROOT/'policies/smolvla/checkpoints/sim_real_reference_low_speed_v1'
    smoke=ROOT/'policies/smolvla/checkpoints/sim_real_reference_low_speed_v1_smoke'
    profile=ROOT/'configs/control/ur5_low_speed_v1.json'
    env=dict(os.environ,MUJOCO_GL='egl',PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',
             PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    status={'started_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),
            'dataset':str(dataset),'training_output':str(output),'steps':10000,'batch_size':4,
            'hardware_access':False}
    def update(stage,**values):
        status.update(stage=stage,**values);write(root/'pipeline_status.json',status)
        print(stage,values,flush=True)
    def run(name,args):
        with (logs/f'{name}.log').open('a') as f:
            subprocess.run([PYTHON,*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    def episode(eid):
        folder=root/f'{eid:03d}'
        if (folder/'result.json').exists():return eid
        # Interrupted partial outputs are retained under a timestamped name.
        if folder.exists():
            resolved=folder.resolve()
            if resolved.parent!=root or not folder.name.isdigit():raise ValueError('Unexpected partial path')
            folder.rename(root/f'partial_{eid:03d}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}')
        try:
            run(f'episode_{eid:03d}',[ROOT/'scripts/ur5_collect_real_low_speed.py','--output',root,'--episodes',eid])
        except Exception as exc:
            folder.mkdir(exist_ok=True)
            if not (folder/'result.json').exists():
                split=json.loads((root/'source_split.json').read_text())
                write(folder/'result.json',{'source_episode':eid,'split':'train' if eid in split['train'] else 'validation',
                      'accepted':False,'success':False,'reason':repr(exc),'frames':None})
        return eid
    try:
        update('collecting')
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for future in as_completed([pool.submit(episode,i) for i in range(50)]):
                future.result()
                results=[json.loads(f.read_text()) for f in sorted(root.glob('[0-9][0-9][0-9]/result.json'))]
                update('collecting',completed=len(results),
                       accepted_train=sum(x['accepted'] and x['split']=='train' for x in results),
                       accepted_validation=sum(x['accepted'] and x['split']=='validation' for x in results))
        results=[json.loads((root/f'{i:03d}/result.json').read_text()) for i in range(50)]
        write(root/'summary.json',{'results':results,'accepted_train':status['accepted_train'],
                                  'accepted_validation':status['accepted_validation'],'completed':50})
        # Require most of both original partitions, not just a tiny surviving subset.
        if status['accepted_train']<32 or status['accepted_validation']<8:
            raise RuntimeError('Acceptance gate: need >=32/40 train and >=8/10 validation; inspect failures before training')
        update('converting')
        if not (dataset/'conversion/COMPLETE.json').exists():
            run('conversion',[ROOT/'scripts/convert_sim_ur5_lift.py','--source',root,'--output',dataset,
                              '--successful-only','--preserve-source-split','--control-profile',profile])
        update('training_smoke')
        common=[ROOT/'policies/smolvla/train_sim_ur5.py','--data',dataset,'--control-profile',profile]
        if not (smoke/'smoke_pass.json').exists():run('training_smoke',[*common,'--smoke','--output',smoke])
        update('training')
        if output.exists():
            checkpoints=sorted((output/'checkpoints').glob('[0-9][0-9][0-9][0-9][0-9][0-9]'))
            if not checkpoints:raise RuntimeError('Training output exists without a resumable checkpoint')
            if int(checkpoints[-1].name)<10000:
                run('training',[*common,'--resume',checkpoints[-1],'--steps',10000])
        else:
            run('training',[*common,'--output',output,'--steps',10000,'--batch-size',4])
        if not (output/'checkpoints/010000/pretrained_model/model.safetensors').exists():
            raise RuntimeError('Final checkpoint missing')
        update('complete')
    except BaseException as exc:
        update('blocked',error=f'{type(exc).__name__}: {exc}')
        raise
if __name__=='__main__':main()
