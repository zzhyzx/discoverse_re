"""Paired cold-start policy evaluation on training takeover states, simulation only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from ur5_collect_takeover_recovery import ROOT,run,write
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.argv=sys.argv[:1]
    a.output.mkdir(parents=True,exist_ok=False)
    plans=json.loads((ROOT/'data/ur5_takeover_recovery_20261026/plan.json').read_text())['plans']
    config=PreTrainedConfig.from_pretrained(a.checkpoint);config.device='cuda';config.n_action_steps=10;config.push_to_hub=False
    model=SmolVLAPolicy.from_pretrained(a.checkpoint,config=config).to('cuda').eval()
    write(a.output/'protocol.json',dict(checkpoint=str(a.checkpoint.resolve()),seed=20261026,training_states=True,policy_queue_reset_at_takeover=True,action_stride=10,hz=30,max_command_rate=.9,timeout=20,early_stop='arm obstacle collision or block displaced >5mm before first close',purpose='memorization not held-out generalization'))
    rows=[]
    for i,plan in enumerate(plans):
        folder=a.output/f'{i:03d}';folder.mkdir();torch.manual_seed(20261026+i);np.random.seed(20261026+i)
        try:row=run(plan,folder,model)
        except Exception as exc:
            row=dict(**plan,success=False,reason=f'{type(exc).__name__}: {exc}',output=str(folder));write(folder/'result.json',row)
        rows.append(row);write(a.output/'summary.json',dict(completed=len(rows),successes=sum(r['success'] for r in rows),episodes=rows));print('RESULT',json.dumps(row),flush=True)
    print('COMPLETE',sum(r['success'] for r in rows),len(rows),flush=True)

if __name__=='__main__':main()
