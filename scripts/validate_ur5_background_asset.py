"""Validate background asset math, tabletop coverage and saved simulation rollout."""
import sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np,cv2,torch
from plyfile import PlyData
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));QA=ROOT/'data/point_cloud/background_integration_20261028'
a=json.loads((QA/'asset_manifest.json').read_text());repair=json.loads((QA/'surface_repair.json').read_text());dest=Path(repair['output'])
v=PlyData.read(dest)['vertex'].data
assert all(np.isfinite(v[k]).all() for k in v.dtype.names)
assert len(v)==repair['output_count']
report={'output':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'count':len(v),'all_fields_finite':True,'table_world_center_m':[.59,0,.497],'table_size_xy_m':[.8,.9],'corner_fit_rms_px':a['rms_px'],'background_sh_degree':3}
# Check similarity covariance and directional color invariance on a random subset.
source=PlyData.read(a['source'])['vertex'].data;aligned=PlyData.read(QA/'background_aligned_full.ply')['vertex'].data
ids=np.random.default_rng(12).choice(len(source),100,replace=False);r=np.array(a['rotation']);s=a['scale']
def covariance(vertices):
 quat=np.column_stack([vertices['rot_'+str(i)] for i in [1,2,3,0]]);R=Rotation.from_quat(quat).as_matrix();sc=np.exp(np.column_stack([vertices['scale_'+str(i)] for i in range(3)]));return np.einsum('nij,nj,nkj->nik',R,sc*sc,R)
expected=s*s*np.einsum('ij,njk,lk->nil',r,covariance(source[ids]),r)
np.testing.assert_allclose(covariance(aligned[ids]),expected,rtol=2e-5,atol=1e-8)
helper=ROOT.parent/'DISCOVERSE-Real2Sim/scripts/align_table_gaussians.py';spec=importlib.util.spec_from_file_location('align',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
d=np.random.default_rng(10).normal(size=(100,3));d/=np.linalg.norm(d,axis=1,keepdims=True)
def color(vertices,direction):
 dc=np.column_stack([vertices['f_dc_'+str(i)] for i in range(3)])[:,:,None];rest=np.column_stack([vertices['f_rest_'+str(i)] for i in range(45)]).reshape(-1,3,15)
 return np.einsum('nck,nk->nc',np.concatenate((dc,rest),axis=2),mod.basis(direction,3))+.5
err=np.max(abs(color(source[ids],d)-color(aligned[ids],d@r.T)));assert err<1e-5
report.update(covariance_transform_verified=True,sh_rotation_max_color_error=float(err),source_preserved=hashlib.sha256(Path(a['source']).read_bytes()).hexdigest()==a['source_sha256'])
del v,source,aligned
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import SimNode,make_config
from diff_gaussian_rasterization import GaussianRasterizer
cfg=make_config(True,True);cfg.align_base_table_centerline=True
env=SimNode(cfg,np.random.default_rng(0),False);env.reset();env.update_gs_scene();rdr=env.gs_renderer.renderer;g=rdr.gaussians;n=env.gs_renderer.gaussians_size['background'];orig=g.opacity.clone();g.opacity[:]=0;g.opacity[n-repair['added_surface_points']:n]=orig[n-repair['added_surface_points']:n]
report['tabletop_coverage']={}
for cid in [0,1]:
 env.getRgbImg(cid)
 with torch.no_grad():rgb,_,depth,alpha=GaussianRasterizer(rdr.raster_settings)(means3D=g.xyz,means2D=None,shs=g.sh,colors_precomp=None,opacities=g.opacity,scales=g.scale,rotations=g.rot,cov3D_precomp=None)
 aa=alpha.squeeze().cpu().numpy();corners=np.array([[.24,-.39,.4962,1],[.94,-.39,.4962,1],[.94,.39,.4962,1],[.24,.39,.4962,1]])
 clip=corners@rdr.raster_settings.projmatrix.cpu().numpy();pixels=(clip[:,:2]/clip[:,3,None]+1)*[320,240]-.5
 mask=np.zeros(aa.shape,np.uint8);cv2.fillConvexPoly(mask,np.round(pixels).astype(np.int32),1);mask=cv2.erode(mask,np.ones((11,11),np.uint8))
 values=aa[mask>0];assert len(values)>100;assert values.min()>.99
 report['tabletop_coverage'][str(cid)]={'min_alpha':float(values.min()),'mean_alpha':float(values.mean()),'pixels':len(values)}
 cv2.imwrite(str(QA/f'tabletop_alpha_cam{cid}.png'),(aa*255).astype('uint8'))
env.renderer.close()
ep=QA/'rollout_validation/000';obs=json.loads((ep/'obs_action.json').read_text());report['rollout_success']=bool(obs['metadata']['success']);assert report['rollout_success'];frames=len(obs['time']);report['rollout_frames']=frames;report['videos']={}
from PIL import Image,ImageDraw
canvas=Image.new('RGB',(1280,4*510));draw=ImageDraw.Draw(canvas)
for cid in [0,1]:
 cap=cv2.VideoCapture(str(ep/f'cam_{cid}.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert count==frames
 report['videos'][str(cid)]={'frames':count,'fps':float(cap.get(cv2.CAP_PROP_FPS))}
 for row,idx in enumerate(np.linspace(0,count-1,4,dtype=int)):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,im=cap.read();assert ok
  canvas.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(cid*640,row*510+30));draw.text((cid*640+8,row*510+7),f'camera {cid}, frame {idx}',fill='white')
 cap.release()
canvas.save(QA/'rollout_contact_sheet.jpg',quality=93)
(QA/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
