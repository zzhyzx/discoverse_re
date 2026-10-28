"""Aggregate saved offline diagnostic measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
out=root/'data/ur5_descent_chunk_diagnosis_v1'
rows=json.loads((out/'results.json').read_text())
summary=[]
fig,axes=plt.subplots(1,2,figsize=(12,5))
for label in dict.fromkeys(r['model'] for r in rows):
 rs=[r for r in rows if r['model']==label]
 means=[np.mean([r['raw_dz_mm'][str(h)] for r in rs]) for h in [1,5,10,30,50]]
 summary.append(dict(model=label,n=len(rs),raw_mean_dz_mm=dict(zip([1,5,10,30,50],means)),
   first10_up_count=sum(r['raw_dz_mm']['10']>1 for r in rs),
   end50_down_count=sum(r['raw_dz_mm']['50']< -1 for r in rs),
   nominal_command_dz_mm=np.mean([r['nominal_30hz'].get('command_end_dz_mm',float('nan')) for r in rs]),
   nominal_actual_dz_mm=np.mean([r['nominal_30hz'].get('actual_end_dz_mm',float('nan')) for r in rs]),
   gated_actual_dz_mm=np.mean([r['feedback_first10'].get('actual_end_dz_mm',float('nan')) for r in rs]),
   errors=[r['case'] for r in rs if r['nominal_30hz']['error'] or r['feedback_first10']['error']]))
 axes[0].plot([1,5,10,30,50],means,'o-',label=label)
 axes[1].bar(label,summary[-1]['gated_actual_dz_mm'])
base=rows[:18]
axes[0].plot([1,5,10,30,50],[np.mean([r['expert_dz_mm'][str(h)] for r in base]) for h in [1,5,10,30,50]],'k--',label='expert reference')
for ax in axes:ax.axhline(0,color='gray');ax.set_ylabel('Vertical displacement (mm; negative = down)')
axes[0].set_xlabel('Predicted action index');axes[0].legend(fontsize=8);axes[0].set_title('Raw target FK: six matched states, three seeds')
axes[1].set_title('Actual physics after feedback-gated first 10');axes[1].tick_params(axis='x',rotation=40)
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
