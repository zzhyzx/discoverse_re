"""Offline lateral-recovery audit; never connects to robot hardware."""
import argparse
import json
from pathlib import Path
import numpy as np
import av


def analyze(root):
    conditions = []
    reference = {}
    grouped = {}
    reference_protocol = None
    for summary_path in sorted(root.glob('*/summary.json')):
        folder = summary_path.parent
        protocol = json.loads((folder/'protocol.json').read_text())
        summary = json.loads(summary_path.read_text())
        controls={k:protocol[k] for k in ('checkpoint','seed','fps','action_stride','max_command_rate_rad_s','gravity_compensation','timeout_sim_seconds','above_height_m')}
        if reference_protocol is None:
            reference_protocol=controls
        assert controls == reference_protocol, 'Conditions have different control settings'
        rows = []
        assert protocol['fixed_block_yaw_deg'] == 0
        assert protocol['initial_yaw_offset_deg'] == 0
        for result in summary['episodes']:
            episode = folder/f"{result['episode']:03d}"
            layout=result['sampling']['sampled_xy_m']
            index=result['episode']
            if index in reference:
                assert np.allclose(layout,reference[index])
            reference[index]=layout
            assert result['sampling']['yaw_deg'] == 0
            row = dict(episode=result['episode'], success=result['success'], reason=result['reason'], source=str(episode.resolve()))
            if result['frames'] == 0:
                rows.append(row)
                continue
            data = json.loads((episode/'obs_action.json').read_text())
            assert result['policy_seed']==protocol['seed']+result['episode']
            ee = np.asarray(data['obs']['ee_pos'])
            block = np.asarray(data['obs']['block_pos'])
            actions = np.asarray(data['act'])
            times = np.asarray(data['time'])
            # Use the INITIAL block position so contact-induced block motion cannot
            # masquerade as a successful approach by the end effector.
            error = np.linalg.norm(ee[:, :2]-block[0, :2], axis=1)
            initial_offset = ee[0, :2]-block[0, :2]
            requested = np.asarray(protocol['initial_offset_world_xy_m'])
            assert np.max(np.abs(initial_offset-requested)) < .001
            closes = np.flatnonzero(actions[:, 6] >= .5)
            close = int(closes[0]) if len(closes) else None
            end = close+1 if close is not None else len(ee)
            best = int(np.argmin(error[:end]))
            direction = -initial_offset/max(np.linalg.norm(initial_offset), 1e-12)
            toward = (ee[:end, :2]-ee[0, :2]) @ direction
            lift=np.flatnonzero(np.asarray(data['obs']['block_clearance_m'])>=.05)
            recovery_end=int(lift[0])+1 if len(lift) else len(ee)
            recovery_best=int(np.argmin(error[:recovery_end]))
            closed=actions[:,6]>=.5
            closure_events=np.flatnonzero(closed & ~np.r_[False,closed[:-1]])
            row.update(initial_error_cm=float(error[0]*100),
                       closure_count=len(closure_events),
                       first_lift_time_s=float(times[lift[0]]) if len(lift) else None,
                       best_prelift_error_cm=float(error[recovery_best]*100),
                       best_prelift_time_s=float(times[recovery_best]),
                       meaningful_correction_prelift=bool(error[0]>=.05 and error[recovery_best]<=error[0]/2 and error[0]-error[recovery_best]>=.03),
                       aligned_within_3cm_prelift=bool(error[recovery_best]<=.03),
                       best_preclose_error_cm=float(error[best]*100),
                       reduction_preclose_cm=float((error[0]-error[best])*100),
                       max_toward_displacement_cm=float(toward.max()*100),
                       best_preclose_time_s=float(times[best]),
                       best_preclose_tcp_height_above_block_cm=float((ee[best,2]-block[0,2])*100),
                       first_close_s=float(times[close]) if close is not None else None,
                       close_xy_error_cm=float(error[close]*100) if close is not None else None,
                       close_tcp_height_above_block_cm=float((ee[close,2]-block[0,2])*100) if close is not None else None,
                       initial_block_max_xy_motion_preclose_cm=float(np.linalg.norm(block[:end,:2]-block[0,:2],axis=1).max()*100),
                       meaningful_correction=bool(error[0]>=.05 and error[best]<=error[0]/2 and error[0]-error[best]>=.03),
                       aligned_within_3cm_before_close=bool(error[best]<=.03),
                       peak_clearance_cm=result['max_clearance_m']*100,
                       video_frames={})
            for camera in (0,1):
                with av.open(str(episode/f'cam_{camera}.mp4')) as video:
                    count=sum(1 for frame in video.decode(video=0))
                assert count == len(times) == result['frames']
                row['video_frames'][str(camera)] = count
            rows.append(row)
        key=tuple(protocol['initial_offset_world_xy_m'])
        group=grouped.setdefault(key,dict(name=folder.name,rows=[]))
        assert not ({r['episode'] for r in group['rows']} & {r['episode'] for r in rows}), 'Duplicate completed episodes'
        group['rows'].extend(rows)
    for key,group in grouped.items():
        rows=sorted(group['rows'],key=lambda r:r['episode'])
        valid=[r for r in rows if 'initial_error_cm' in r]
        conditions.append(dict(condition=group['name'], offset_xy_m=list(key), completed=len(rows), successes=sum(r['success'] for r in rows),
            initialization_failures=len(rows)-len(valid),
            meaningful_corrections=sum(r['meaningful_correction'] for r in valid),
            meaningful_corrections_prelift=sum(r['meaningful_correction_prelift'] for r in valid),
            aligned_prelift=sum(r['aligned_within_3cm_prelift'] for r in valid),
            aligned_before_close=sum(r['aligned_within_3cm_before_close'] for r in valid),
            median_best_preclose_error_cm=float(np.median([r['best_preclose_error_cm'] for r in valid])) if valid else None,
            episodes=rows))
    report=dict(definition='Meaningful correction: distance to initial block XY decreases by >=50% and >=3cm. Report both before first close and before first 5cm lift (or timeout if no lift). The latter includes retries. Not proof of visual causality; policy may descend simultaneously.',
                conditions=conditions)
    (root/'translation_summary.json').write_text(json.dumps(report,indent=2))
    lines=['# Fixed-yaw lateral recovery test', '',
           'Block yaw=0, initial gripper yaw offset=0. Same five layouts and episode seeds per condition.',
           '024000 checkpoint, 30 Hz, action stride 10, slew 0.9 rad/s, 20 simulated seconds.',
           'All valid trials have two decoded videos matching log frame counts.', '',
           '| Condition | Success | Correction before first close | Correction including retries before lift | Within 3cm before closing | Median best preclose XY error (cm) | Init failures |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for c in conditions:
        lines.append(f"| {c['condition']} | {c['successes']}/{c['completed']} | {c['meaningful_corrections']} | {c['meaningful_corrections_prelift']} | {c['aligned_before_close']} | {c['median_best_preclose_error_cm']} | {c['initialization_failures']} |")
    lines+=['',report['definition'], '',
            'Distances use the original block center, avoiding false improvement due to pushing. Five repeated layouts are not independent samples across conditions. New zero-yaw baseline is required; prior random-yaw success rates are not paired controls.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    # Scientific diagnostic: actual end-effector XY paths relative to the
    # original object, including the first closure point and original block.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(16,12),sharex=True,sharey=True)
    for c in conditions:
        dx,dy=c['offset_xy_m']
        if dx==dy==0:
            continue
        magnitude=round(max(abs(dx),abs(dy))*100)
        if magnitude not in (6,8,10):
            continue
        col=0 if dx>0 else 1 if dx<0 else 2 if dy>0 else 3
        ax=axes[(6,8,10).index(magnitude),col]
        ax.add_patch(plt.Rectangle((-2.5,-2.5),5,5,color='red',alpha=.15))
        for r in c['episodes']:
            if 'initial_error_cm' not in r:
                continue
            data=json.loads((Path(r['source'])/'obs_action.json').read_text())
            xy=(np.asarray(data['obs']['ee_pos'])[:,:2]-np.asarray(data['obs']['block_pos'])[0,:2])*100
            closes=np.flatnonzero(np.asarray(data['act'])[:,6]>=.5)
            end=int(closes[0])+1 if len(closes) else len(xy)
            line,=ax.plot(xy[:end,0],xy[:end,1],label=f"ep{r['episode']} {'OK' if r['success'] else 'fail'}")
            ax.scatter(*xy[0],marker='o',color=line.get_color(),s=20)
            if len(closes):
                ax.scatter(*xy[end-1],marker='x',color=line.get_color(),s=40)
        ax.set_title(f"{c['condition']}: {c['successes']}/{c['completed']} success")
        ax.set_aspect('equal');ax.grid(alpha=.3);ax.legend(fontsize=7)
        ax.set_xlabel('World X relative to initial block (cm)')
        ax.set_ylabel('World Y relative to initial block (cm)')
    fig.suptitle('Actual TCP before first close: circle=start, cross=first close; red square=initial block')
    fig.tight_layout()
    fig.savefig(root/'preclose_xy_paths.png',dpi=160)
    plt.close(fig)
    print(json.dumps([{k:v for k,v in c.items() if k!='episodes'} for c in conditions],indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
