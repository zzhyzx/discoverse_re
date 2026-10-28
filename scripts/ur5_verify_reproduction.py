"""Validate saved offline comparison outputs; no hardware access."""
from pathlib import Path
import json
import numpy as np
import av
ROOT=Path(__file__).resolve().parents[1]/'data/ur5_real_reproduction_20261027'
rows=[]
for ep in [0,17,39]:
    initial=None
    for mode in ['state','direct','limited']:
        folder=ROOT/f'ep_{ep}_{mode}_video'
        r=json.loads((folder/'report.json').read_text())
        with np.load(folder/'metrics.npz') as d:
            n=len(d['source_index'])
            assert n==r['frames'] and r['complete']
            assert np.all(np.diff(d['source_index'])>=0)
            assert np.allclose(np.diff(d['simulation_time_s']),1/30)
            assert np.isfinite(d['q_sim_rad']).all()
            pos=np.array(r['interaction']['initial_cube_world_m'])
            if initial is None: initial=pos
            assert np.array_equal(initial,pos)
            if mode=='limited':
                v=np.diff(d['executed_target'],axis=0)*30
                assert abs(v).max()<=.030001
                assert abs(np.diff(v,axis=0)*30).max()<=.050001
            elif mode=='state':
                assert np.max(abs(d['q_sim_rad']-d['q_actual_rad']))<1e-12
            if mode!='state':
                old=ROOT/f'ep_{ep}_{mode}_metrics/metrics.npz'
                with np.load(old) as m:
                    assert np.allclose(d['q_sim_rad'],m['q_sim_rad'])
                    assert np.allclose(d['block_world_m'],m['block_world_m'])
        for camera in [0,1]:
            with av.open(str(folder/f'cam_{camera}.mp4')) as video:
                stamps=[float(f.pts*f.time_base) for f in video.decode(video=0)]
            assert len(stamps)==n,(folder,len(stamps),n)
            assert np.allclose(np.diff(stamps),1/30,atol=2e-5)
        row=dict(episode=ep,mode=mode,seconds=r['simulation_duration_s'],frames=n,
                 mean_joint_error_deg=float(np.mean(r['joint_mae_deg'])),
                 max_bottom_clearance_m=r['interaction']['max_bottom_clearance_m'],
                 hold_5cm_s=r['interaction']['longest_5cm_bilateral_hold_s'],
                 lift_evidence=r['interaction']['lift_evidence'],
                 success=r['interaction']['task_success'],folder=str(folder))
        rows.append(row)
        print(ep,mode,'verified',flush=True)
(ROOT/'summary.json').write_text(json.dumps(rows,indent=2))
with (ROOT/'README.md').open('a') as f:
    f.write('\n## 已完成对照结果\n\n|轨迹|方式|仿真秒|平均关节误差°|最大底部离桌cm|5cm双侧接触保持秒|\n|---|---|---|---|---|---|\n')
    for r in rows:
        f.write(f"|{r['episode']}|{r['mode']}|{r['seconds']:.2f}|{r['mean_joint_error_deg']:.3f}|{100*r['max_bottom_clearance_m']:.2f}|{r['hold_5cm_s']:.2f}|\n")
    f.write('\n9组18个视频逐帧数量与时间戳验证通过；两组动态回放的视频/无视频物理结果一致；受限命令速度、加速度通过检查。\n')
