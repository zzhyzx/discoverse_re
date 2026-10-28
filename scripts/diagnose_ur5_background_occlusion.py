import sys,os,json
from pathlib import Path
import numpy as np,torch,cv2
ROOT=Path('/home/boluo/DISCOVERSE');sys.path.insert(0,str(ROOT))
from discoverse.examples.tasks_ur5.ur5_single_red_pick_place_task import SimNode,make_config
from diff_gaussian_rasterization import GaussianRasterizer
out=ROOT/'data/point_cloud/background_integration_20261028';env=SimNode(make_config(True,True),np.random.default_rng(0),False);env.reset();env.update_gs_scene();env.getRgbImg(0)
r=env.gs_renderer.renderer;g=r.gaussians;opacity=g.opacity.clone();n=env.gs_renderer.gaussians_size['background'];idx=env.gs_renderer.gaussians_idx['red_block'];size=env.gs_renderer.gaussians_size['red_block'];report={}
for mode in ['all','dynamic_only','background_only','block_only']:
 g.opacity.copy_(opacity)
 if mode=='dynamic_only':g.opacity[:n]=0
 if mode=='background_only':g.opacity[n:]=0
 if mode=='block_only':g.opacity[:]=0;g.opacity[idx:idx+size]=opacity[idx:idx+size]
 with torch.no_grad():color,_,depth,alpha=GaussianRasterizer(r.raster_settings)(means3D=g.xyz,means2D=None,shs=g.sh,colors_precomp=None,opacities=g.opacity,scales=g.scale,rotations=g.rot,cov3D_precomp=None)
 rgb=color.permute(1,2,0).cpu().numpy();a=alpha.squeeze().cpu().numpy();cv2.imwrite(str(out/(mode+'.png')),cv2.cvtColor((np.clip(rgb,0,1)*255).astype('uint8'),cv2.COLOR_RGB2BGR));np.save(out/(mode+'_alpha.npy'),a)
 report[mode]={'center_rgb':rgb[242,316].tolist(),'center_alpha':float(a[242,316]),'center_depth':float(depth.squeeze()[242,316].cpu())}
print(json.dumps(report,indent=2));(out/'occlusion_diagnostic.json').write_text(json.dumps(report,indent=2));env.renderer.close()
