"""Route A: RGB remapping only. No hardware access or source-video changes."""
import json
from pathlib import Path
import cv2
import numpy as np
from estimate_real_ur5_cameras import rays
from discoverse.task_base.airbot_task_base import PyavImageEncoder


class Rectification:
    def __init__(self, env, output, fps):
        root = Path(__file__).resolve().parents[1]
        source = root / 'models/mjcf/ur5/camera_initial_v3.json'
        payload = json.loads(source.read_text())
        self.folder = Path(output) / 'rectified_real'
        self.folder.mkdir(exist_ok=False)
        self.maps, self.writers = {}, {}
        report = dict(source=str(source), method='numerical inversion of RealSense inverse-Brown deprojection',
                      border='black; no crop or extra zoom', pixel_centers='integer; symmetric raster center=(319.5,239.5)',
                      cameras={}, limitations=['Provided intrinsics assumed to match original unresized video',
                      'Extrinsics unchanged; not an extrinsic calibration', 'MuJoCo mesh renderer only verified'])
        for role, name, cid in [('front','D436',0),('wrist','D405',1)]:
            camera = next(c for c in payload['intrinsics']['cameras'] if name in c['name'])
            p = next(p for p in camera['profiles'] if p['stream']=='color' and p['format']=='format.rgb8')
            w,h = p['width'],p['height']
            if (w,h)!=(640,480):
                raise ValueError('Expected 640x480 calibration')
            f = float(p['fy'])
            center = np.array([(w-1)/2,(h-1)/2])
            yy,xx = np.indices((h,w))
            ideal = np.c_[xx.ravel(),yy.ravel()].astype(float)
            desired = (ideal-center)/f
            focal = np.array([p['fx'],p['fy']])
            raw = desired*focal + [p['cx'],p['cy']]
            for _ in range(30):
                raw += (desired-rays(raw,p)[:,:2])*focal
            residual = np.linalg.norm((rays(raw,p)[:,:2]-desired)*f,axis=1)
            if not np.isfinite(raw).all() or residual.max()>0.01:
                raise ValueError('Remap inversion failed pixel residual check')
            mx=raw[:,0].reshape(h,w).astype(np.float32)
            my=raw[:,1].reshape(h,w).astype(np.float32)
            valid=(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)
            self.maps[role]=(mx,my)
            np.savez_compressed(self.folder/f'{role}_remap.npz',map_x=mx,map_y=my,valid=valid)
            cv2.imwrite(str(self.folder/f'{role}_valid_mask.png'),valid.astype(np.uint8)*255)
            env.mj_model.cam_fovy[cid]=np.degrees(2*np.arctan(h/(2*f)))
            report['cameras'][role]=dict(serial=camera['serial_number'],source_profile=p,
                K_output=[[f,0,float(center[0])],[0,f,float(center[1])],[0,0,1]],
                distortion_output=[0,0,0,0,0],fovy_deg=float(env.mj_model.cam_fovy[cid]),
                valid_fraction=float(valid.mean()),max_inverse_residual_px=float(residual.max()),
                video=f'cam_{cid}.mp4')
        (self.folder/'projection.json').write_text(json.dumps(report,indent=2))
        for cid,role in enumerate(('front','wrist')):
            self.writers[role]=PyavImageEncoder(640,480,str(self.folder),cid,fps=fps,crf=18,preset='veryfast')
        self.fps=fps

    def apply(self, role, image, index):
        if image.shape!=(480,640,3):
            raise ValueError('Input image does not match calibration resolution')
        corrected=cv2.remap(image,*self.maps[role],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        self.writers[role].encode(corrected,(index+1)/self.fps)
        if index in (0,130,259):
            cv2.imwrite(str(self.folder/f'{role}_{index:05d}.png'),cv2.cvtColor(corrected,cv2.COLOR_RGB2BGR))
            pair=np.concatenate([image,corrected],axis=1)
            cv2.imwrite(str(self.folder/f'{role}_raw_vs_rectified_{index:05d}.jpg'),cv2.cvtColor(pair,cv2.COLOR_RGB2BGR))
        return corrected

    def close(self):
        for writer in self.writers.values():
            writer.close()
