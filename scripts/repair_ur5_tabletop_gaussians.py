"""Add a measured, opaque tabletop backing to the UR5 background asset.

This is a geometric appearance repair, not a learned texture reconstruction.
Run with the discoverse Python environment; --verify requires its CUDA renderer.
The input PLY and all of its original records are retained unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData, PlyElement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "models/3dgs/ur5_single_red_pick_place"
SOURCE = ASSETS / "background_aligned_plane1.ply"
DESTINATION = ASSETS / "background_tabletop_repaired_v1.ply"
SH_C0 = 0.28209479177387814


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_patch(vertices):
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")])
    opacity = 1 / (1 + np.exp(-vertices["opacity"]))
    rgb = np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) * SH_C0 + .5
    # Asset-local table: center (0.60, 0, 0), size 0.80 x 0.90 m.
    # Task applies (-0.01, 0, 0.497), placing it on the physical tabletop.
    dx, dy = abs(xyz[:, 0] - .60), abs(xyz[:, 1])
    ring = ((dx < .395) & (dy < .445) & ((dx > .34) | (dy > .39))
            & (abs(xyz[:, 2]) < .008) & (opacity > .5)
            & np.all((rgb > .25) & (rgb < .95), axis=1))
    if ring.sum() < 100:
        raise ValueError("Insufficient intact tabletop rim samples for color estimation")
    color = np.median(rgb[ring], axis=0)
    # Overlapping flat splats cover the interior; leave the scanned edge intact.
    spacing = .004
    xs = np.arange(.216, .984 + 1e-6, spacing)
    ys = np.arange(-.434, .434 + 1e-6, spacing)
    xx, yy = np.meshgrid(xs, ys)
    patch = np.zeros(xx.size, dtype=vertices.dtype)
    patch["x"], patch["y"], patch["z"] = xx.ravel(), yy.ravel(), -.0015
    patch["rot_0"] = 1
    for i, sigma in enumerate((.0034, .0034, .00035)):
        patch[f"scale_{i}"] = np.log(sigma)
    patch["opacity"] = np.log(.95 / .05)
    for i in range(3):
        patch[f"f_dc_{i}"] = (color[i] - .5) / SH_C0
    return patch, {
        "color_rgb_0_1": color.tolist(), "color_sample_count": int(ring.sum()),
        "local_table_center": [.60, 0, 0], "table_size_m": [.80, .90],
        "spacing_m": spacing, "sigma_m": [.0034, .0034, .00035],
        "surface_center_z_m": -.0015, "single_splat_opacity": .95,
        "note": "Uniform color estimated from intact rim. No recovered fine texture; original vertices unchanged.",
    }


def verify(source, destination, output):
    import cv2
    from PIL import Image, ImageDraw
    import torch
    from discoverse.gaussian_renderer.util_gau import Camera, load_ply
    from discoverse.gaussian_renderer.renderer_cuda import CUDARenderer
    from diff_gaussian_rasterization import GaussianRasterizer

    output.mkdir(parents=True, exist_ok=True)
    camera = Camera(600, 800)
    camera.update_fovy(np.deg2rad(58))
    renderer = CUDARenderer(800, 600)
    data = [("BEFORE", load_ply(source)), ("AFTER", load_ply(destination))]
    views = {
        "top": ([.60, 0, 1.25], [.60, 0, 0], [0, 1, 0]),
        "front": ([.60, -1.15, .90], [.60, 0, -.02], [0, 0, 1]),
        "side": ([1.6, -.65, .65], [.60, 0, -.02], [0, 0, 1]),
        "close": ([.56, -.15, .12], [.60, 0, 0], [0, 0, 1]),
    }
    metrics = {}
    for name, (eye, target, up) in views.items():
        camera.position = np.array(eye, np.float32)
        camera.target = np.array(target, np.float32)
        camera.up = np.array(up, np.float32)
        panels = []
        metrics[name] = {}
        for label, gaussians in data:
            renderer.update_gaussian_data({"background": gaussians})
            renderer.update_camera_pose(camera)
            g = renderer.gaussians
            with torch.no_grad():
                color, _, depth, alpha = GaussianRasterizer(renderer.raster_settings)(
                    means3D=g.xyz, means2D=None, shs=g.sh, colors_precomp=None,
                    opacities=g.opacity, scales=g.scale, rotations=g.rot, cov3D_precomp=None)
            rgb = color.permute(1, 2, 0).cpu().numpy()
            a = alpha.squeeze().cpu().numpy()
            if not np.isfinite(rgb).all() or not np.isfinite(a).all():
                raise ValueError("Nonfinite render")
            corners = np.array([[.25, -.39, 0, 1], [.95, -.39, 0, 1],
                                [.95, .39, 0, 1], [.25, .39, 0, 1]])
            clip = corners @ renderer.raster_settings.projmatrix.cpu().numpy()
            mask = np.zeros(a.shape, np.uint8)
            # This renderer's flipped view convention yields negative clip W
            # for visible points. Require a consistent sign, not positive W.
            if np.all(abs(clip[:, 3]) > 1e-6) and (np.all(clip[:, 3] > 0) or np.all(clip[:, 3] < 0)):
                pixels = (clip[:, :2] / clip[:, 3, None] + 1) * [400, 300] - .5
                cv2.fillConvexPoly(mask, np.round(pixels).astype(np.int32), 1)
                mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
                val = a[mask > 0]
                metrics[name][label] = {"mean_alpha": float(val.mean()),
                                        "minimum_alpha": float(val.min()),
                                        "fraction_below_0.99": float((val < .99).mean())}
            yy, xx = np.indices(a.shape)
            checker = np.where(((xx // 24 + yy // 24) % 2)[..., None],
                               [.65, .15, .65], [.15, .65, .65])
            im = Image.fromarray((np.clip(rgb + (1-a[..., None])*checker, 0, 1)*255).astype(np.uint8))
            im.save(output / f"{name}_{label.lower()}.png")
            panel = Image.new("RGB", (800, 630))
            panel.paste(im, (0, 30))
            ImageDraw.Draw(panel).text((10, 10), label, fill="white")
            panels.append(panel)
        both = Image.new("RGB", (1600, 630))
        for i, panel in enumerate(panels):
            both.paste(panel, (800*i, 0))
        both.save(output / f"{name}_comparison.png")
        print(name, metrics[name], flush=True)
    # Test the added surface alone: background behind it must not hide holes.
    from discoverse.gaussian_renderer.util_gau import GaussianData
    repaired = data[1][1]
    original_count = len(data[0][1].xyz)
    patch = GaussianData(*(getattr(repaired, k)[original_count:] for k in
                           ("xyz", "rot", "scale", "opacity", "sh")))
    camera.position = np.array(views["top"][0], np.float32)
    camera.target = np.array(views["top"][1], np.float32)
    camera.up = np.array(views["top"][2], np.float32)
    renderer.update_gaussian_data({"patch": patch})
    renderer.update_camera_pose(camera)
    g = renderer.gaussians
    with torch.no_grad():
        _, _, _, alpha = GaussianRasterizer(renderer.raster_settings)(
            means3D=g.xyz, means2D=None, shs=g.sh, colors_precomp=None,
            opacities=g.opacity, scales=g.scale, rotations=g.rot, cov3D_precomp=None)
    clip = corners @ renderer.raster_settings.projmatrix.cpu().numpy()
    pixels = (clip[:, :2]/clip[:, 3, None]+1)*[400, 300]-.5
    mask = np.zeros((600, 800), np.uint8)
    cv2.fillConvexPoly(mask, np.round(pixels).astype(np.int32), 1)
    val = alpha.squeeze().cpu().numpy()[mask > 0]
    metrics["patch_alone_top"] = {"minimum_alpha": float(val.min()), "mean_alpha": float(val.mean())}
    assert val.min() > .99, "Added surface still has holes"
    (output / "verification.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    before = sha256(SOURCE)
    ply = PlyData.read(SOURCE)
    original = ply["vertex"].data
    patch, params = make_patch(original)
    merged = np.concatenate([original, patch])
    if DESTINATION.exists():
        existing = PlyData.read(DESTINATION)["vertex"].data
        if existing.dtype != merged.dtype or existing.tobytes() != merged.tobytes():
            raise FileExistsError(f"Refusing to overwrite differing asset: {DESTINATION}")
    else:
        PlyData([PlyElement.describe(merged, "vertex")], text=False).write(DESTINATION)
    saved = PlyData.read(DESTINATION)["vertex"].data
    assert saved[:len(original)].tobytes() == original.tobytes()
    assert sha256(SOURCE) == before
    report = {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": before,
              "output": str(DESTINATION.relative_to(ROOT)), "output_sha256": sha256(DESTINATION),
              "original_count": len(original), "added_count": len(patch), "parameters": params}
    diagnostic = ROOT / "data/point_cloud/tabletop_repair_v1"
    diagnostic.mkdir(parents=True, exist_ok=True)
    if args.verify:
        report["verification"] = verify(SOURCE, DESTINATION, diagnostic)
    (diagnostic / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "verification"}, indent=2))


if __name__ == "__main__":
    main()
