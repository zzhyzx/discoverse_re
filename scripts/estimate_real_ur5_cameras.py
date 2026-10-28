"""Offline, prior-assisted INITIAL camera estimate. Never connects to hardware.

Uses manually approximated table corners and colored-object silhouettes, NOT
calibration targets. Do not use the output as a robot safety/calibration result.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def rays(uv, p):
    # RealSense rs2_deproject_pixel_to_point inverse-Brown branch (10 iterations).
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    xo = (uv[:, 0] - p["cx"]) / p["fx"]
    yo = (uv[:, 1] - p["cy"]) / p["fy"]
    x, y = xo.copy(), yo.copy()
    k1, k2, p1, p2, k3 = p["distortion_coefficients"]
    assert p["distortion_model"] == "distortion.inverse_brown_conrady"
    for _ in range(10):
        r2 = x*x + y*y
        ic = 1 / (1 + ((k3*r2+k2)*r2+k1)*r2)
        xq, yq = x/ic, y/ic
        dx = 2*p1*xq*yq + p2*(r2+2*xq*xq)
        dy = 2*p2*xq*yq + p1*(r2+2*yq*yq)
        x, y = (xo-dx)*ic, (yo-dy)*ic
    return np.c_[x, y, np.ones(len(x))]


def transform(r, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(r).as_matrix()
    T[:3, 3] = t
    return T


def blobs(path):
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(path)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    masks = {"red": ((h < 12) | (h > 170)) & (s > 110) & (v > 45),
             "blue": (h > 95) & (h < 125) & (s > 110) & (v > 40),
             "yellow": (h > 17) & (h < 38) & (s > 100) & (v > 70)}
    result = {}
    for name, mask in masks.items():
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in cs if cv2.contourArea(c) > 60]
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        x, y, w, hh = cv2.boundingRect(c)
        if x < 3 or y < 3 or x+w > 637 or y+hh > 477:
            continue
        m = cv2.moments(c)
        result[name] = np.array([m["m10"]/m["m00"], m["m01"]/m["m00"]])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intrinsics", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT/"data/real_ur5_pick_cube_v2_analysis")
    ap.add_argument("--output", type=Path, default=ROOT/"data/real_ur5_camera_initial_v1")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(a.intrinsics.read_text(encoding="utf-8-sig"))
    profiles = {}
    for role, model in (("front", "D436"), ("wrist", "D405")):
        cam = next(c for c in source["cameras"] if model in c["name"])
        profiles[role] = next(p for p in cam["profiles"] if p["format"] == "format.rgb8")
        assert (profiles[role]["width"], profiles[role]["height"]) == (640, 480)
    (a.output/"intrinsics_source.json").write_text(json.dumps(source, indent=2))
    # Table axes: x left->right in front image, y near->far, z up.
    # Rounded/partly occluded corners are explicitly manual approximations.
    corners = np.array([[29, 440], [626, 435], [473, 199], [145, 204]], float)
    xyz = np.array([[-.45,-.40,0],[.45,-.40,0],[.45,.40,0],[-.45,.40,0]])
    uv = np.ascontiguousarray(rays(corners, profiles["front"])[:, :2])
    ok, r, t = cv2.solvePnP(xyz, uv, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("Table PnP failed")
    C_T_table = transform(r.ravel(), t.ravel())
    table_T_front = np.linalg.inv(C_T_table)
    if table_T_front[2, 3] <= 0:
        raise RuntimeError("Camera below table")
    predicted = (xyz @ C_T_table[:3,:3].T + C_T_table[:3,3])
    predicted = predicted[:,:2] / predicted[:,2,None]
    f = np.array([profiles["front"]["fx"], profiles["front"]["fy"]])
    table_error = np.linalg.norm((predicted-uv)*f, axis=1)
    # Centroid plane is an approximation, not a measured 3D cube landmark.
    rows = []
    for folder in sorted((a.data/"episodes").iterdir()):
        if not folder.is_dir():
            continue
        traj = np.load(folder/"trajectory.npz")
        initial = blobs(folder/"front_00000.jpg")
        for fp in sorted(folder.glob("front_*.jpg")):
            frame = int(fp.stem.split("_")[1])
            front, wrist = blobs(fp), blobs(folder/f"wrist_{frame:05d}.jpg")
            for color in front.keys() & wrist.keys() & initial.keys():
                if np.linalg.norm(front[color]-initial[color]) > 4:
                    continue
                ray = table_T_front[:3,:3] @ rays([front[color]], profiles["front"])[0]
                origin = table_T_front[:3,3]
                d = (.025-origin[2])/ray[2]
                point = origin+d*ray
                if d <= 0 or abs(point[0]) > .45 or abs(point[1]) > .4:
                    continue
                rows.append(dict(episode=int(folder.name), frame=frame, color=color,
                                 xyz_table=point.tolist(), uv_front=front[color].tolist(),
                                 uv_wrist=wrist[color].tolist(),
                                 T_base_tool=traj["T_base_tool0_nominal"][frame].tolist()))
    if len(rows) < 30:
        raise RuntimeError(f"Insufficient color correspondences: {len(rows)}")
    points = np.array([v["xyz_table"] for v in rows])
    fk = np.array([v["T_base_tool"] for v in rows])
    measured = rays([v["uv_wrist"] for v in rows], profiles["wrist"])[:,:2]
    train = np.array([v["episode"] < 40 for v in rows])
    fw = np.array([profiles["wrist"]["fx"], profiles["wrist"]["fy"]])

    def unpack(x):
        bt = transform([0,0,x[0]], [x[1], x[2], .054])
        wc = transform(x[3:6], x[6:9])
        return bt, wc

    def project(x):
        bt, wc = unpack(x)
        pb = points @ bt[:3,:3].T + bt[:3,3]
        pt = np.einsum("nji,nj->ni", fk[:,:3,:3], pb-fk[:,:3,3])
        pc = (pt-wc[:3,3]) @ wc[:3,:3]
        return pc[:,:2]/np.maximum(pc[:,2,None], .02), pc[:,2]

    def residual(x):
        p, z = project(x)
        # Weak but necessary installation prior: translation gauge is poorly observable.
        prior = (x[6:9]-np.array([.05,0,.10]))/.05
        return np.r_[((p-measured)*fw)[train].ravel(),
                     np.maximum(.02-z[train],0)*5000, prior*5]

    rng = np.random.default_rng(42)
    best = None
    for i in range(16):
        x = np.r_[rng.uniform(-np.pi,np.pi), -.4, -.4,
                  Rotation.random(random_state=rng).as_rotvec(), .05,0,.10]
        fit = least_squares(residual, x, loss="soft_l1", f_scale=5, max_nfev=350)
        if best is None or fit.cost < best.cost:
            best = fit
        print(f"initialization {i+1}/16 cost={fit.cost:.1f}", flush=True)
    bt, wc = unpack(best.x)
    p, z = project(best.x)
    err = np.linalg.norm((p-measured)*fw, axis=1)
    def stats(mask):
        e = err[mask]
        return dict(n=len(e), median_px=float(np.median(e)),
                    p95_px=float(np.percentile(e,95)), max_px=float(e.max()))
    sv = np.linalg.svd(best.jac, compute_uv=False)
    report = dict(
        status="INITIAL_PRIOR_ASSISTED_NOT_CALIBRATED",
        conventions="T_parent_child maps child coordinates to parent; camera OpenCV right/down/forward",
        table_axes="x image right, y away from front camera, z up; origin tabletop center",
        assumptions=["Original videos are native 640x480 without cropping/resizing",
                     "User-confirmed video/robot synchronization",
                     "Table 0.9m across x, 0.8m along y; table z in UR base is 0.054m",
                     "Color silhouette centroids approximate object centers at table z=0.025m",
                     "Wrist translation prior [0.05,0,0.10]m in tool0, sigma 0.05m; NOT measured",
                     "Manual approximate rounded/occluded table corners; no independent corner validation"],
        T_table_front_camera=table_T_front.tolist(),
        T_base_table=bt.tolist(),
        T_base_front_camera=(bt@table_T_front).tolist(),
        T_tool0_wrist_camera=wc.tolist(),
        fit=stats(train), heldout=stats(~train),
        table_corner_raw_pixels=corners.tolist(),
        table_corner_equivalent_pixel_errors=table_error.tolist(),
        positive_depth_fraction=float(np.mean(z>0)),
        optimizer_success=bool(best.success),
        jacobian_singular_values_with_prior=sv.tolist(),
        automatic_scene_update=False,
        warning="Pixel errors are undistorted focal-scaled errors, NOT independent physical calibration accuracy. Translation is prior-dependent.")
    (a.output/"estimate.json").write_text(json.dumps(report,indent=2))
    (a.output/"correspondences.json").write_text(json.dumps(rows,indent=2))
    # Plot in undistorted normalized-pixel canvas so inverse distortion is handled consistently.
    for role in ("front","wrist"):
        pr = profiles[role]
        xx, yy = np.meshgrid(np.arange(640),np.arange(480))
        desired = np.c_[xx.ravel(),yy.ravel()].astype(float)
        raw = desired.copy()
        # Numerically invert SDK deprojection for a hole-free diagnostic remap.
        for _ in range(12):
            undistorted = rays(raw,pr)[:,:2]*[pr["fx"],pr["fy"]]+[pr["cx"],pr["cy"]]
            raw += desired-undistorted
        original = cv2.imread(str(a.data/f"episodes/000/{role}_00000.jpg"))
        canvas = cv2.remap(original,raw[:,0].reshape(480,640).astype(np.float32),
                           raw[:,1].reshape(480,640).astype(np.float32),cv2.INTER_LINEAR)
        if role == "front":
            obs = uv*f+[pr["cx"],pr["cy"]]
            pred = predicted*f+[pr["cx"],pr["cy"]]
        else:
            sel = np.array([v["episode"]==0 and v["frame"]==0 for v in rows])
            obs = measured[sel]*fw+[pr["cx"],pr["cy"]]
            pred = p[sel]*fw+[pr["cx"],pr["cy"]]
        for o, v in zip(obs,pred):
            if np.isfinite(v).all() and np.max(np.abs(v)) < 10000:
                oi, vi = tuple(np.rint(o).astype(int)), tuple(np.rint(v).astype(int))
                cv2.circle(canvas,oi,5,(0,255,0),1)
                cv2.drawMarker(canvas,vi,(0,0,255),cv2.MARKER_CROSS,12,2)
                cv2.line(canvas,oi,vi,(0,255,255),1)
        cv2.putText(canvas,"INITIAL: green observed / red predicted",(8,20),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
        cv2.imwrite(str(a.output/f"{role}_overlay.jpg"),canvas)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
