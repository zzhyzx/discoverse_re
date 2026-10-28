"""Compare completed offline simulation suites; no robot connections."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(old_root, new_root):
    old = json.loads((old_root / 'translation_summary.json').read_text())
    new = json.loads((new_root / 'translation_summary.json').read_text())
    old_conditions = {tuple(c['offset_xy_m']): c for c in old['conditions']}
    rows, pairs = [], []
    keys = ('seed', 'fps', 'action_stride', 'max_command_rate_rad_s',
            'gravity_compensation', 'timeout_sim_seconds', 'start_mode',
            'above_height_m', 'initial_offset_world_xy_m',
            'initial_yaw_offset_deg', 'fixed_block_yaw_deg', 'success')
    for c in new['conditions']:
        previous = old_conditions[tuple(c['offset_xy_m'])]
        before = {r['episode']: r for r in previous['episodes']}
        gained, lost = [], []
        for r in c['episodes']:
            b = before[r['episode']]
            bp, rp = Path(b['source']), Path(r['source'])
            pb = json.loads((bp.parent / 'protocol.json').read_text())
            pr = json.loads((rp.parent / 'protocol.json').read_text())
            for key in keys:
                assert pb[key] == pr[key], (c['condition'], key)
            ib = json.loads((bp / 'initialization.json').read_text())
            ir = json.loads((rp / 'initialization.json').read_text())
            for key in ('q_rad', 'block_pos_world_m', 'actual_tcp_world_m'):
                assert np.allclose(ib[key], ir[key], atol=1e-9, rtol=0), (c['condition'], key)
            if r['success'] and not b['success']:
                gained.append(r['episode'])
            if b['success'] and not r['success']:
                lost.append(r['episode'])
            pairs.append(dict(condition=c['condition'], episode=r['episode'],
                              old=b, new=r))
        rows.append(dict(condition=c['condition'], offset_xy_m=c['offset_xy_m'],
                         n=c['completed'], old_successes=sum(before[r['episode']]['success'] for r in c['episodes']),
                         new_successes=c['successes'], gained=gained, lost=lost))
    assert len(pairs) == 65, f'Expected complete 65-pair suite, got {len(pairs)}'
    report = dict(conditions=rows, pairs=pairs,
                  matched_initial_states=True,
                  old_successes=sum(r['old_successes'] for r in rows),
                  new_successes=sum(r['new_successes'] for r in rows),
                  gained=sum(len(r['gained']) for r in rows),
                  lost=sum(len(r['lost']) for r in rows))
    for name in ('old', 'new'):
        offsets = [p[name] for p in pairs if p['condition'] != 'aligned']
        report[name + '_offset_metrics'] = dict(
            success=sum(r['success'] for r in offsets),
            correction_before_close=sum(r.get('meaningful_correction', False) for r in offsets),
            correction_before_lift=sum(r.get('meaningful_correction_prelift', False) for r in offsets),
            collisions=sum('collision' in r['reason'] for r in offsets))
    (new_root / 'paired_comparison.json').write_text(json.dumps(report, indent=2))
    lines = ['# Paired checkpoint comparison', '',
             'Same 65 initial states and evaluation settings, except checkpoint. Two videos per trial audited separately.', '',
             '| Condition | Old | New | Newly successful episodes | Lost successes |',
             '| --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['old_successes']}/{r['n']} | {r['new_successes']}/{r['n']} | {r['gained']} | {r['lost']} |")
    lines.extend(['', f"Total: {report['old_successes']}/65 -> {report['new_successes']}/65; gained {report['gained']}, lost {report['lost']}.",
                  '', 'Repeated five layouts across offsets: not 65 independent layouts. This is initialized pregrasp recovery, not the full task from home. No causal attribution to vision or new data alone; matched old-data-only fine-tuning control not performed.', '',
                  'Offset metrics:', '```json', json.dumps({n: report[n] for n in ('old_offset_metrics', 'new_offset_metrics')}, indent=2), '```'])
    (new_root / 'paired_comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('pairs', 'conditions')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_root', type=Path)
    parser.add_argument('new_root', type=Path)
    args = parser.parse_args()
    compare(args.old_root, args.new_root)
