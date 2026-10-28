"""Read-only bridge timing check, optionally with offline GPU model load. No RTDE."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from ur5_camera_wire import CameraClient

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--with-model',action='store_true')
    p.add_argument('--duration',type=float,default=20)
    p.add_argument('--host')
    args=p.parse_args()
    session=json.loads((ROOT/'data/real_ur5/camera_bridge/session.json').read_text())
    route=subprocess.check_output(['ip','route','show','default'],text=True).split()
    client=CameraClient(args.host or route[route.index('via')+1],session,timeout=2)
    stop=threading.Event(); rows=[]; errors=[]
    def collect():
        try:
            while not stop.is_set():
                # Diagnostic thresholds only. Never produces control input.
                m,_=client.get(10,10)
                rows.append([m['max_receipt_age_s'],m['transfer_seconds'],
                    m['receipt_skew_s'],m.get('server_wait_seconds',0)])
                stop.wait(.02)
        except Exception as exc: errors.append(str(exc))
        finally: client.close()
    worker=threading.Thread(target=collect,daemon=True);worker.start()
    try:
        if args.with_model:
            spec=importlib.util.spec_from_file_location('deploy',ROOT/'policies/smolvla/deploy_real_ur5.py')
            deploy=importlib.util.module_from_spec(spec);spec.loader.exec_module(deploy)
            cfg=deploy.configuration(ROOT/'policies/smolvla/configs/ur5_real_deploy.json')
            policy,_,_=deploy.load_model(cfg)
            deploy.model_check(policy,cfg)
        else: stop.wait(args.duration)
    finally:
        stop.set();worker.join(3)
    if rows:
        a=np.asarray(rows);total=a[:,0]+a[:,1]-a[:,3]
        print('samples:',len(a),'columns: server_age/request_total/skew/server_wait')
        print('median/p95/max:',np.quantile(a,[.5,.95,1],axis=0).round(4).tolist())
        print('age>250ms:',int((total>.25).sum()),'skew>80ms:',int((a[:,2]>.08).sum()))
    print('errors:',errors)
    return 1 if errors or not rows else 0

if __name__=='__main__': raise SystemExit(main())
