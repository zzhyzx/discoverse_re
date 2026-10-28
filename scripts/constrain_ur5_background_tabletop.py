"""Constrain the textureless tabletop to its measured plane, using video colors."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement
import cv2
ROOT=Path(__file__).resolve().parents[1];QA=ROOT/'data/point_cloud/background_integration_20261028';ASSET=ROOT/'models/3dgs/ur5_single_red_pick_place'
a=json.loads((QA/'asset_manifest.json').read_text());v=PlyData.read(a['output'])['vertex'].data.copy()
q=np.column_stack([v[k] for k in ['x','y','z']]);sc=np.exp(np.column_stack([v['scale_'+str(i)] for i in range(3)]));quat=np.column_stack([v['rot_'+str(i)] for i in range(4)])
rot=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix();extent=3*np.sqrt(np.einsum('nij,nj->ni',rot*rot,sc*sc))
# Remove splats that intrude into the robot/task volume, not distant scenery.
lo=np.array([-.42,-.5,.008]);hi=np.array([1.025,.5,1.3])
intruding=np.all((q+extent>lo)&(q-extent<hi),axis=1)
interior=(q[:,0]>.205)&(q[:,0]<.995)&(abs(q[:,1])<.445)&(q[:,2]>-.16)&(q[:,2]<.04)
remove=intruding|interior
assert remove.sum()<len(v)*.15
# A dense, thin splat sheet. Rounded corners follow the real tabletop.
xs=np.arange(.2015,1,.003);ys=np.arange(-.4485,.45,.003);xx,yy=np.meshgrid(xs,ys)
xy=np.column_stack((xx.ravel(),yy.ravel()));edge=np.maximum(np.abs(xy-[.6,0])-[.385,.435],0);inside=np.linalg.norm(edge,axis=1)<=.015
xy=xy[inside];p=np.column_stack((xy,np.full(len(xy),-.0008)))
r=np.array(a['rotation']);s=a['scale'];t=np.array(a['translation']);scan=(p-t)@r/s
src=ROOT.parent/'DISCOVERSE-Real2Sim';cams={c['img_name']:c for c in json.loads((src/'outputs/background_20261028/cameras.json').read_text())}
colors=[];used=[]
for i in [1,25,51,75,102,125,152,175,203,225,253,275,304,330,355]:
 name=f'{i:06d}.png';c=cams[name];view=(scan-np.array(c['position']))@np.array(c['rotation'])
 uv=view[:,:2]/view[:,2,None]*[c['fx'],c['fy']]+[c['width']/2,c['height']/2]
 im=cv2.cvtColor(cv2.imread(str(src/'datasets/background_20261028/images'/name)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
 rgb=np.concatenate([cv2.remap(im,part[:,0].astype('float32').reshape(-1,1),part[:,1].astype('float32').reshape(-1,1),cv2.INTER_LINEAR).reshape(-1,3) for part in np.array_split(uv,8)])
 valid=(view[:,2]>0)&(uv[:,0]>2)&(uv[:,0]<c['width']-3)&(uv[:,1]>2)&(uv[:,1]<c['height']-3)&(rgb.min(1)>.35)
 rgb[~valid]=np.nan;colors.append(rgb);used.append(name)
colors=np.stack(colors);color=np.nanmedian(colors,axis=0);coverage=np.isfinite(colors[:,:,0]).sum(0)
if np.isnan(color).any():raise ValueError('Unobserved tabletop texels; do not invent colors')
patch=np.zeros(len(p),dtype=v.dtype)
for i,k in enumerate(['x','y','z']):patch[k]=p[:,i];patch['scale_'+str(i)]=np.log([.0028,.0028,.0002][i]);patch['f_dc_'+str(i)]=(color[:,i]-.5)/.28209479177387814
patch['rot_0']=1;patch['opacity']=np.log(.97/.03);patch['nz']=1
result=np.concatenate((v[~remove],patch));dest=ASSET/'background_20261028_ready.ply'
PlyData([PlyElement.describe(result,'vertex')],text=False).write(dest)
np.savez_compressed(QA/'surface_repair_indices.npz',removed_indices=np.flatnonzero(remove))
texture=np.full((len(ys)*len(xs),3),.15,dtype=np.float32);texture[inside]=color;texture=texture.reshape(len(ys),len(xs),3)
cv2.imwrite(str(QA/'tabletop_texture.png'),cv2.cvtColor((texture*255).astype('uint8'),cv2.COLOR_RGB2BGR))
report={'input':a['output'],'output':str(dest),'removed_intruding':int(intruding.sum()),'removed_old_surface_or_intruding':int(remove.sum()),'added_surface_points':len(p),'output_count':len(result),'surface_z_local_m':-.0008,'sigma_m':[.0028,.0028,.0002],'color_sources':used,'minimum_valid_views_per_texel':int(coverage.min()),'median_valid_views_per_texel':float(np.median(coverage)),'color_rgb_mean':color.mean(0).tolist(),'method':'Multi-view median of rectified real-video RGB on a measured tabletop plane. Not a uniform-color fill.','limitation':'Static appearance, not dynamic relighting.'}
(QA/'surface_repair.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
