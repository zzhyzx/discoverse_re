"""Audit the fixed 16 training-state policy comparison; no model or robot calls."""
import json
from pathlib import Path
from collections import Counter
import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    base=ROOT/'data/ur5_takeover_sanity_baseline_20261026'
    trained=ROOT/'data/ur5_takeover_sanity_trained_20261026'
    a=json.loads((base/'summary.json').read_text());b=json.loads((trained/'summary.json').read_text())
    assert a['completed']==b['completed']==16
    pa=json.loads((base/'protocol.json').read_text());pb=json.loads((trained/'protocol.json').read_text())
    assert {k:v for k,v in pa.items() if k!='checkpoint'}=={k:v for k,v in pb.items() if k!='checkpoint'}
    rows=[];video_count=0
    for i,(old,new) in enumerate(zip(a['episodes'],b['episodes'])):
        assert (old['source'],old['step'],old['kind'])==(new['source'],new['step'],new['kind'])
        with np.load(base/f'{i:03d}'/'takeover_state.npz') as x,np.load(trained/f'{i:03d}'/'takeover_state.npz') as y:
            assert all(np.array_equal(x[k],y[k]) for k in x.files)
        row=dict(id=i,kind=old['kind'],source=old['source'],old_success=old['success'],new_success=new['success'])
        for label,folder,result in [('old',base/f'{i:03d}',old),('new',trained/f'{i:03d}',new)]:
            d=json.loads((folder/'obs_action.json').read_text());meta=d['metadata']
            assert meta['diagnostic_only'] and not meta['training_eligible']
            t=np.array(d['time']);ee=np.array(d['obs']['ee_pos']);block=np.array(d['obs']['block_pos']);act=np.array(d['act'])
            assert np.allclose(np.diff(t),1/30,atol=1e-8)
            err=np.linalg.norm(ee[:,:2]-block[0,:2],axis=1);close=np.flatnonzero(act[:,6]>=.5)
            end=int(close[0])+1 if len(close) else len(t)
            row[label]=dict(reason=result['reason'],frames=len(t),duration=float(t[-1]),first_close=float(t[close[0]]) if len(close) else None,
                initial_xy_cm=float(err[0]*100),best_preclose_xy_cm=float(err[:end].min()*100),
                peak_clearance_cm=float(np.max(d['obs']['block_clearance_m'])*100))
            for c in (0,1):
                with av.open(str(folder/f'cam_{c}.mp4')) as video:count=sum(1 for _ in video.decode(video=0))
                assert count==len(t)==result['frames'];video_count+=1
        rows.append(row)
    groups={}
    for kind in ('low_offset','high_nonapproach','aligned_control'):
        rs=[r for r in rows if r['kind']==kind]
        groups[kind]=dict(n=len(rs),old=sum(r['old_success'] for r in rs),new=sum(r['new_success'] for r in rs))
    report=dict(groups=groups,old_successes=a['successes'],new_successes=b['successes'],
        gained=[r['id'] for r in rows if r['new_success'] and not r['old_success']],
        lost=[r['id'] for r in rows if r['old_success'] and not r['new_success']],
        old_reasons=dict(Counter(r['old']['reason'] for r in rows)),new_reasons=dict(Counter(r['new']['reason'] for r in rows)),
        full_initial_states_identical=True,verified_videos=video_count,
        limitation='All 16 episodes were used in training. Memorization test only; not held-out or full-task performance.',episodes=rows)
    (trained/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))

if __name__=='__main__':main()
