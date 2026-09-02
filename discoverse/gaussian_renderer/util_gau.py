import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
import torch
import glm

def multiple_quaternion_vector3d(qwxyz, vxyz):
    qw = qwxyz[..., 0]
    qx = qwxyz[..., 1]
    qy = qwxyz[..., 2]
    qz = qwxyz[..., 3]
    vx = vxyz[..., 0]
    vy = vxyz[..., 1]
    vz = vxyz[..., 2]
    qvw = -vx*qx - vy*qy - vz*qz
    qvx =  vx*qw - vy*qz + vz*qy
    qvy =  vx*qz + vy*qw - vz*qx
    qvz = -vx*qy + vy*qx + vz*qw
    vx_ =  qvx*qw - qvw*qx + qvz*qy - qvy*qz
    vy_ =  qvy*qw - qvz*qx - qvw*qy + qvx*qz
    vz_ =  qvz*qw + qvy*qx - qvx*qy - qvw*qz
    return torch.stack([vx_, vy_, vz_], dim=-1).cuda().requires_grad_(False)

def multiple_quaternions(qwxyz1, qwxyz2):
    q1w = qwxyz1[..., 0]
    q1x = qwxyz1[..., 1]
    q1y = qwxyz1[..., 2]
    q1z = qwxyz1[..., 3]

    q2w = qwxyz2[..., 0]
    q2x = qwxyz2[..., 1]
    q2y = qwxyz2[..., 2]
    q2z = qwxyz2[..., 3]

    qw_ = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
    qx_ = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
    qy_ = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
    qz_ = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w

    return torch.stack([qw_, qx_, qy_, qz_], dim=-1).cuda().requires_grad_(False)

class Camera:
    def __init__(self, h, w):
        self.znear = 1e-6
        self.zfar = 100
        self.h = h
        self.w = w
        self.fovy = 1.05
        self.position = np.array([0.0, 0.0, -2.0]).astype(np.float32)
        self.target = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        self.up = np.array([0.0, 0.0, 1.0]).astype(np.float32)
        self.yaw = -np.pi / 2
        self.pitch = 0

        self.is_pose_dirty = True
        self.is_intrin_dirty = True
        
        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True
        
        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False
        
        self.rot_sensitivity = 0.02
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity = 0.08
        self.roll_sensitivity = 0.03
        self.target_dist = 3.
    
    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)

    def get_view_matrix(self):
        # return np.array(glm.lookAt(self.position, self.target, self.up))
        return np.array(glm.lookAt(glm.vec3(self.position), glm.vec3(self.target), glm.vec3(self.up)))

    def get_project_matrix(self):
        project_mat = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        return self.h / (2 * np.tan(self.fovy / 2))

    def update_target_distance(self):
        _dir = self.target - self.position
        _dir = _dir / np.linalg.norm(_dir)
        self.target = self.position + _dir * self.target_dist
        
    def update_resolution(self, height, width):
        self.h = max(height, 1)
        self.w = max(width, 1)
        self.is_intrin_dirty = True
    
    def update_fovy(self, fovy):
        self.fovy = fovy
        self.is_intrin_dirty = True

@dataclass
class GaussianData:
    def __init__(self, xyz, rot, scale, opacity, sh):
        self.xyz = xyz
        self.rot = rot
        self.scale = scale
        self.opacity = opacity
        self.sh = sh

        self.origin_xyz = np.zeros(3)
        self.origin_rot = np.array([1., 0., 0., 0.])

    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]

def gamma_shs(shs, gamma):
    C0 = 0.28209479177387814
    new_shs = ((np.clip(shs * C0 + 0.5, 0.0, 1.0) ** gamma) - 0.5) / C0
    return new_shs

def load_ply(path, gamma=1):
    max_sh_degree = 0
    plydata = PlyData.read(path)

    super_splat_format = True
    try:
        plydata['chunk']
    except KeyError:
        super_splat_format = False
    
    if super_splat_format:
        vtx = plydata['vertex'].data  # structured array
        chk = plydata['chunk'].data   # structured array

        # 每256个vertex对应一个chunk（按顺序）
        num_vertex = vtx.shape[0]
        if num_vertex == 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, np.zeros((0, 4), dtype=np.float32), empty.copy(), empty.copy(), np.zeros((0,), dtype=np.float32)

        chunk_idx = (np.arange(num_vertex) // 256).astype(np.int64)
        # 防御性裁剪（以防最后一个 chunk 未满或越界情况）
        chunk_idx = np.clip(chunk_idx, 0, chk.shape[0] - 1)

        # 拉取每个点对应 chunk 的标量边界
        def gather_chunk(field):
            return chk[field][chunk_idx]

        # 解码位置（11/10/11）
        ppos = vtx['packed_position'].astype(np.uint32)
        xbits = (ppos >> 21) & 0x7FF
        ybits = (ppos >> 11) & 0x3FF
        zbits = ppos & 0x7FF
        fx = xbits.astype(np.float32) / 2047.0 * (gather_chunk('max_x') - gather_chunk('min_x')) + gather_chunk('min_x')
        fy = ybits.astype(np.float32) / 1023.0 * (gather_chunk('max_y') - gather_chunk('min_y')) + gather_chunk('min_y')
        fz = zbits.astype(np.float32) / 2047.0 * (gather_chunk('max_z') - gather_chunk('min_z')) + gather_chunk('min_z')
        positions = np.stack([fx, fy, fz], axis=1).astype(np.float32)

        # 解码尺度（11/10/11），并指数还原
        pscale = vtx['packed_scale'].astype(np.uint32)
        sxb = (pscale >> 21) & 0x7FF
        syb = (pscale >> 11) & 0x3FF
        szb = pscale & 0x7FF
        sx = sxb.astype(np.float32) / 2047.0 * (gather_chunk('max_scale_x') - gather_chunk('min_scale_x')) + gather_chunk('min_scale_x')
        sy = syb.astype(np.float32) / 1023.0 * (gather_chunk('max_scale_y') - gather_chunk('min_scale_y')) + gather_chunk('min_scale_y')
        sz = szb.astype(np.float32) / 2047.0 * (gather_chunk('max_scale_z') - gather_chunk('min_scale_z')) + gather_chunk('min_scale_z')
        scales = np.exp(np.stack([sx, sy, sz], axis=1)).astype(np.float32)

        # 解码颜色和不透明度（8/8/8/8）
        pcol = vtx['packed_color'].astype(np.uint32)
        r8 = (pcol >> 24) & 0xFF
        g8 = (pcol >> 16) & 0xFF
        b8 = (pcol >> 8) & 0xFF
        a8 = pcol & 0xFF
        fr = r8.astype(np.float32) / 255.0 * (gather_chunk('max_r') - gather_chunk('min_r')) + gather_chunk('min_r')
        fg = g8.astype(np.float32) / 255.0 * (gather_chunk('max_g') - gather_chunk('min_g')) + gather_chunk('min_g')
        fb = b8.astype(np.float32) / 255.0 * (gather_chunk('max_b') - gather_chunk('min_b')) + gather_chunk('min_b')
        SH_C0 = 0.28209479177387814
        fr = (fr - 0.5) / SH_C0
        fg = (fg - 0.5) / SH_C0
        fb = (fb - 0.5) / SH_C0
        opacity = a8.astype(np.float32) / 255.0
        # opacity = 1.0 / (1.0 + np.exp(-opacity))
        colors = np.stack([fr, fg, fb], axis=1).astype(np.float32)
        opacities = opacity.astype(np.float32)

        # 解码旋转（最大分量索引 + 3×10bit）
        prot = vtx['packed_rotation'].astype(np.uint32)
        largest = (prot >> 30) & 0x3  # 0..3
        v0 = (prot >> 20) & 0x3FF
        v1 = (prot >> 10) & 0x3FF
        v2 = prot & 0x3FF
        norm = np.sqrt(2.0) * 0.5
        vals = np.stack([v0, v1, v2], axis=1).astype(np.float32)
        vals = (vals / 1023.0 - 0.5) / norm
        # 映射到四元数的非最大分量（顺序依 index 增序，略过 largest）
        q = np.zeros((num_vertex, 4), dtype=np.float32)

        # Masks for largest index
        m0 = (largest == 0)
        m1 = (largest == 1)
        m2 = (largest == 2)
        m3 = (largest == 3)

        # 对应关系见说明：
        # largest=0: (1,2,3) <= (v0,v1,v2)
        q[m0, 1] = vals[m0, 0]
        q[m0, 2] = vals[m0, 1]
        q[m0, 3] = vals[m0, 2]
        # largest=1: (0,2,3) <= (v0,v1,v2)
        q[m1, 0] = vals[m1, 0]
        q[m1, 2] = vals[m1, 1]
        q[m1, 3] = vals[m1, 2]
        # largest=2: (0,1,3) <= (v0,v1,v2)
        q[m2, 0] = vals[m2, 0]
        q[m2, 1] = vals[m2, 1]
        q[m2, 3] = vals[m2, 2]
        # largest=3: (0,1,2) <= (v0,v1,v2)
        q[m3, 0] = vals[m3, 0]
        q[m3, 1] = vals[m3, 1]
        q[m3, 2] = vals[m3, 2]

        # 复原最大分量
        sum_sq = np.sum(q * q, axis=1)
        max_comp = np.sqrt(np.clip(1.0 - sum_sq, 0.0, 1.0)).astype(np.float32)
        # 写回到对应的 largest 位置（0:w, 1:x, 2:y, 3:z）
        q[m0, 0] = max_comp[m0]
        q[m1, 1] = max_comp[m1]
        q[m2, 2] = max_comp[m2]
        q[m3, 3] = max_comp[m3]
        quats = q.astype(np.float32)

        # return positions, quats, scales, colors, opacities
        return GaussianData(positions, quats, scales, opacities, colors)

    else:
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        # assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
        features_extra = features_extra.reshape((features_extra.shape[0], 3, len(extra_f_names)//3))
        features_extra = features_extra[:, :, :(max_sh_degree + 1) ** 2 - 1]
        features_extra = np.transpose(features_extra, [0, 2, 1])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = xyz.astype(np.float32)
        rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
        rots = rots.astype(np.float32)
        scales = np.exp(scales)

        # if 2dgs
        if len(scale_names) == 2:
            print(f"len(scale_names) = {len(scale_names)} (2dgs ply model)")
            scales = np.hstack([scales, 1e-9 * np.ones_like(scales[:, :1])])

        scales = scales.astype(np.float32)
        opacities = 1/(1 + np.exp(-opacities))
        opacities = opacities.astype(np.float32)

        if abs(gamma - 1.0) > 1e-3:
            features_dc = gamma_shs(features_dc, gamma)
            features_extra[...,:] = 0.0
            opacities *= 0.8

        shs = np.concatenate([features_dc.reshape(-1, 3), 
                            features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
        shs = shs.astype(np.float32)
        return GaussianData(xyz, rots, scales, opacities, shs)
