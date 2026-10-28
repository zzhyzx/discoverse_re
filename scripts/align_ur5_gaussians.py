"""Convert merged mesh-frame Gaussians to UR5 body frames; preserve originals."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def main():
    root = Path(__file__).resolve().parents[1]
    model = root / 'models/mjcf/ur5'
    folder = root / 'models/3dgs/ur5_single_red_pick_place'
    manifest = json.loads((folder/'ur5_merged/merge_manifest.json').read_text())
    xml = ET.parse(model/'ur5.xml').getroot()
    meshes = {m.get('name'): model/m.get('file') for m in xml.findall('./asset/mesh')}
    bodies = {b.get('name'): b for b in xml.iter('body')}
    outputs, report = {}, {}
    for body, entry in manifest['links'].items():
        data = PlyData.read(str(folder/'ur5_merged'/f'{body}.ply'))['vertex'].data.copy()
        offset, parts = 0, []
        for src in entry['sources']:
            part = Path(src['path']).parents[2].name
            g = next(g for g in bodies[body].findall('geom') if g.get('mesh') == part)
            vertices = np.array([[float(x) for x in l.split()[1:4]] for l in meshes[part].read_text().splitlines() if l.startswith('v ')])
            n = src['points']
            v = data[offset:offset+n]
            xyz = np.column_stack([v[k] for k in ('x','y','z')])
            tree = cKDTree(vertices)
            choices = [np.eye(3), np.diag([1.,-1.,1.])]
            errors = [float(np.median(tree.query(xyz@a.T)[0])) for a in choices]
            a = choices[int(np.argmin(errors))]
            q = np.fromstring(g.get('quat','1 0 0 0'), sep=' ')
            r = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            t = np.fromstring(g.get('pos','0 0 0'), sep=' ')
            transform = r@a
            result = xyz@transform.T+t
            for j,k in enumerate(('x','y','z')): v[k] = result[:,j]
            qs = np.column_stack([v[f'rot_{j}'] for j in range(4)])
            matrices = transform@Rotation.from_quat(qs[:,[1,2,3,0]]).as_matrix()
            # Principal-axis sign does not change covariance.
            if np.linalg.det(transform)<0: matrices[:,:,2] *= -1
            qs = Rotation.from_matrix(matrices).as_quat()[:,[3,0,1,2]]
            for j in range(4): v[f'rot_{j}'] = qs[:,j]
            if all(k in v.dtype.names for k in ('nx','ny','nz')):
                normals = np.column_stack([v[k] for k in ('nx','ny','nz')])@transform.T
                for j,k in enumerate(('nx','ny','nz')): v[k] = normals[:,j]
            # Current DISCOVERSE loader uses DC only; discard stale directional SH.
            for k in v.dtype.names:
                if k.startswith('f_rest_'): v[k] = 0
            parts.append(dict(part=part, points=n, errors_m=errors, reflect_y=bool(a[1,1]<0), matrix=transform.tolist(), translation=t.tolist()))
            offset += n
        assert offset == len(data)
        outputs[body], report[body] = data, parts
    dest = folder/'ur5_body_aligned'
    dest.mkdir(exist_ok=True)
    for body,data in outputs.items():
        PlyData([PlyElement.describe(data,'vertex')],text=False).write(str(dest/f'{body}.ply'))
    (dest/'alignment_manifest.json').write_text(json.dumps(report,indent=2))
    for body,parts in report.items():
        print(body,[(p['part'],p['reflect_y'],[round(e,5) for e in p['errors_m']]) for p in parts])

if __name__ == '__main__':
    main()
