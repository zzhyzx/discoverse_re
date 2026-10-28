"""Export the calibrated background, removing the filmed stationary robot.
All coordinates here are asset-local; the task adds (-.01,0,.497) once.
"""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement
ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'data/point_cloud/background_integration_20261028'
ASSETS=ROOT/'models/3dgs/ur5_single_red_pick_place'
a=json.loads((QA/'alignment_initial.json').read_text())
src=Path(a['source']);v=PlyData.read(src)['vertex'].data.copy();n=len(v)
r=np.array(a['rotation']);s=a['scale'];t=np.array(a['translation'])
p=np.column_stack([v[k] for k in ['x','y','z']]);q=s*(p@r.T)+t
for i,k in enumerate(['x','y','z']):v[k]=q[:,i];v['scale_'+str(i)]+=np.log(s)
quat=np.column_stack([v['rot_'+str(i)] for i in range(4)])
quat=(Rotation.from_matrix(r)*Rotation.from_quat(quat[:,[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
for i in range(4):v['rot_'+str(i)]=quat[:,i]
if all(k in v.dtype.names for k in ['nx','ny','nz']):
 normals=np.column_stack([v[k] for k in ['nx','ny','nz']])@r.T
 for i,k in enumerate(['nx','ny','nz']):v[k]=normals[:,i]
helper=ROOT.parent/'DISCOVERSE-Real2Sim/scripts/align_table_gaussians.py'
spec=importlib.util.spec_from_file_location('align_table',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.rotate_sh(v,r)
# Volume covers the observed folded UR5, gripper and nearby attached cables.
# Pedestal and tabletop are outside the selected volume.
lo=np.array([-.40,-.12,-.045]);hi=np.array([.19,.43,1.4])
remove=np.all((q>=lo)&(q<=hi),axis=1)
assert 1000<remove.sum()<n*.3
full=QA/'background_aligned_full.ply';clean=ASSETS/'background_20261028_aligned_clean.ply'
for path,records in [(full,v),(QA/'removed_robot.ply',v[remove]),(clean,v[~remove])]:
 assert all(np.isfinite(records[k]).all() for k in records.dtype.names)
 PlyData([PlyElement.describe(records,'vertex')],text=False).write(path)
np.savez_compressed(QA/'removed_indices.npz',indices=np.flatnonzero(remove))
a.update(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),output=str(clean),input_count=n,removed_count=int(remove.sum()),retained_count=int((~remove).sum()),removal_box_local_m={'min':lo.tolist(),'max':hi.tolist()},preserved_sh_degree=3,convention='p_local = scale * R @ p_scan + t; task adds world_offset once',notes=['Manual table-corner calibration; not surveyed extrinsic calibration.','Robot-occluded surfaces cannot be recovered by deletion; verify actual task views.'])
(QA/'asset_manifest.json').write_text(json.dumps(a,indent=2))
print(json.dumps({k:a[k] for k in ['output','scale','rms_px','input_count','removed_count','retained_count']},indent=2))
