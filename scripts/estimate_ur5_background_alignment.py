from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from plyfile import PlyData
root=Path('/home/boluo/DISCOVERSE')
source=Path('/home/boluo/DISCOVERSE-Real2Sim/outputs/background_20261028')
out=root/'data/point_cloud/background_integration_20261028'
out.mkdir(parents=True,exist_ok=True)
cams={c['img_name']:c for c in json.loads((source/'cameras.json').read_text())}
# Back-left, back-right, front-right, front-left viewed from robot-opposite side.
annotations={'000001.png':[[418,253],[808,264],[839,450],[304,431]],'000051.png':[[372,121],[721,192],[499,375],[98,256]],'000355.png':[[459,72],[935,157],[1094,326],[361,191]]}
local=np.array([[.2,-.45,0],[.2,.45,0],[1,.45,0],[1,-.45,0]])
Ps=[]
for name in annotations:
 c=cams[name];R=np.array(c['rotation']);pos=np.array(c['position']);K=np.array([[c['fx'],0,c['width']/2],[0,c['fy'],c['height']/2],[0,0,1]])
 Ps.append(K@np.column_stack((R.T,-R.T@pos)))
xyz=[]
for i in range(4):
 A=[]
 for P,xy in zip(Ps,annotations.values()):
  u,v=xy[i];A.extend([u*P[2]-P[0],v*P[2]-P[1]])
 _,_,vt=np.linalg.svd(A);p=vt[-1];xyz.append(p[:3]/p[3])
xyz=np.array(xyz)
x=local-local.mean(0);y=xyz-xyz.mean(0)
u,d,vt=np.linalg.svd(x.T@y);r=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
scale=np.sum((x@r.T)*y)/np.sum(x*x);t=xyz.mean(0)-scale*r@local.mean(0)
def residual(v):
 R=Rotation.from_rotvec(v[:3]).as_matrix();p=np.exp(v[6])*(local@R.T)+v[3:6]
 result=[]
 for P,uv in zip(Ps,annotations.values()):
  clip=np.column_stack((p,np.ones(4)))@P.T;result.extend(((clip[:,:2]/clip[:,2,None])-uv).ravel())
 return result
fit=least_squares(residual,np.r_[Rotation.from_matrix(r).as_rotvec(),t,np.log(scale)],loss='soft_l1',f_scale=2,max_nfev=300)
r=Rotation.from_rotvec(fit.x[:3]).as_matrix();s=np.exp(fit.x[6]);t=fit.x[3:6]
rout=r.T;sout=1/s;tout=-rout@t/s
record={'source':str(source/'point_cloud/iteration_30000/point_cloud.ply'),'annotations':annotations,'local_corners':local.tolist(),'triangulated_corners_scan':xyz.tolist(),'scale':float(sout),'rotation':rout.tolist(),'translation':tout.tolist(),'reprojection_residuals_px':np.asarray(residual(fit.x)).reshape(-1,2).tolist(),'rms_px':float(np.sqrt(np.mean(np.square(residual(fit.x))))),'local_table_center':[.6,0,0],'world_offset':[-.01,0,.497]}
(out/'alignment_initial.json').write_text(json.dumps(record,indent=2))
ply=PlyData.read(record['source']);v=ply['vertex'].data
p=np.column_stack([v[k] for k in ['x','y','z']]);q=sout*(p@rout.T)+tout
rgb=np.clip(np.column_stack([v['f_dc_'+str(i)] for i in range(3)])*.28209479177+.5,0,1)
alpha=1/(1+np.exp(-np.clip(v['opacity'],-50,50)))
np.savez_compressed(out/'aligned_diagnostic.npz',xyz=q.astype('float32'),rgb=rgb.astype('float32'),alpha=alpha)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rng=np.random.default_rng(8);sel=np.flatnonzero((alpha>.25)&(np.linalg.norm(q-[.6,0,0],axis=1)<3));sel=rng.choice(sel,min(len(sel),100000),replace=False)
fig,axs=plt.subplots(1,3,figsize=(18,6))
for ax,(i,j),limits in zip(axs,[(0,1),(0,2),(1,2)],[((-.7,1.7),(-1.2,1.2)),((-.7,1.7),(-.7,1.2)),((-1.2,1.2),(-.7,1.2))]):
 ax.scatter(q[sel,i],q[sel,j],c=rgb[sel],s=.5);ax.plot(local[[0,1,2,3,0],i],local[[0,1,2,3,0],j],c='red');ax.set(xlim=limits[0],ylim=limits[1],xlabel='XYZ'[i],ylabel='XYZ'[j]);ax.set_aspect('equal');ax.grid()
fig.tight_layout();fig.savefig(out/'alignment_initial.png',dpi=150)
print(json.dumps(record,indent=2))
