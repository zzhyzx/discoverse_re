"""Offline multi-episode comparison; writes new result directory, no real robot."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes",type=int,nargs="+",default=[0,5,10,15,20])
    p.add_argument("--damping-scales",type=float,nargs="+",default=[1.0])
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    out=a.output or ROOT/"data/real_ur5_batch_checks"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for scale in a.damping_scales:
        for ep in a.episodes:
            dest=out/f"damping_{scale:g}_ep{ep:03d}"
            command=[sys.executable,str(ROOT/"scripts/ur5_replay_real_actions.py"),
                     "--episode",str(ep),"--interaction","--no-video",
                     "--damping-scale",str(scale),"--output",str(dest)]
            result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
            (out/f"{dest.name}.log").write_text(result.stdout+"\n"+result.stderr)
            row=dict(episode=ep,damping_scale=scale,exit_code=result.returncode)
            if result.returncode==0:
                r=json.loads((dest/"report.json").read_text())
                row.update(tcp_mean_mm=r["tcp_mean_error_mm"],tcp_max_mm=r["tcp_max_error_mm"],
                           joint_max_deg=max(r["joint_max_abs_deg"]),
                           lift=r["interaction"]["lift_evidence"],
                           rise_m=r["interaction"]["max_center_rise_m"],
                           bilateral_lift_s=r["interaction"]["longest_lifted_bilateral_contact_s"],
                           release=r["interaction"]["release_command_observed"])
            rows.append(row)
            (out/"summary.json").write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
    print("Output:",out)


if __name__=="__main__":
    main()
