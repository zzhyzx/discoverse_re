import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots.ur5.ur5_ik import UR5IK
from discoverse.robots_env.ur5_base import UR5Base, UR5Cfg
from discoverse.task_base.airbot_task_base import PyavImageEncoder
from discoverse.utils import step_func


TASK_INSTRUCTION = "Pick up the red block and hold it steadily above the table"
CAPTURE_HZ = 30
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0
GRIPPER_JOINT_RANGE = 0.8
# Heights below are clearances above the measured tabletop, not world Z.
ROBOT_MOUNT_Z = 0.443
TRANSFER_HEIGHT = 0.28
PLACE_COMMAND_HEIGHT = 0.050
TRANSFER_WAYPOINT_SPACING = 0.04
RETREAT_WAYPOINT_SPACING = 0.025
NORMAL_JOINT_SPEED = 0.60
CARRY_JOINT_SPEED = 0.35
RETREAT_JOINT_SPEED = 0.45
MIN_CARRY_BLOCK_Z = 0.08
RELEASE_SETTLE_TIME = 0.25
RELEASE_OPEN_TIME = 0.20


class EpisodeFailure(RuntimeError):
    pass


class SimNode(UR5Base):
    def updateControl(self, action):
        super().updateControl(action)
        if not getattr(self.config, "arm_gravity_compensation", False):
            return
        # Task-local gravity feedforward; no object/finger anti-gravity.
        if not hasattr(self, "_gravity_data"):
            model = self.mj_model
            if np.any(model.body_gravcomp):
                raise ValueError("Refusing double gravity compensation")
            for i in range(6):
                jid = int(model.actuator_trnid[i, 0])
                if (model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT
                        or model.jnt_qposadr[jid] != i or model.jnt_dofadr[jid] != i
                        or not np.allclose(model.actuator_gear[i], [1, 0, 0, 0, 0, 0])
                        or model.actuator_gainprm[i, 0] <= 0
                        or not np.isclose(model.actuator_biasprm[i, 1], -model.actuator_gainprm[i, 0])):
                    raise ValueError("Expected six unit-gear position actuators")
            self._gravity_data = mujoco.MjData(model)
        scratch = self._gravity_data
        scratch.qpos[:] = self.mj_data.qpos
        scratch.qvel[:] = 0
        mujoco.mj_forward(self.mj_model, scratch)
        self.arm_gravity_torque = scratch.qfrc_bias[:6].copy()
        kp = self.mj_model.actuator_gainprm[:6, 0]
        # kp*(q_target + g/kp - q) - kv*qd = PD + g.
        # Original actuator torque limits remain active. No Coriolis/contact cancellation.
        self.mj_data.ctrl[:6] = np.clip(
            np.asarray(action)[:6] + self.arm_gravity_torque / kp,
            self.mj_model.actuator_ctrlrange[:6, 0], self.mj_model.actuator_ctrlrange[:6, 1])

    def getRgbImg(self, cam_id):
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            return super().getRgbImg(cam_id)
        if cam_id == -1:
            self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
        elif cam_id >= 0:
            self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
        else:
            return None
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = getattr(
            self.config, "render_shadows", False)
        return self.renderer.render()

    def post_load_mjcf(self):
        super().post_load_mjcf()
        base_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if base_id < 0 or self.mj_model.body_parentid[base_id] != 0:
            raise ValueError("Expected base_link directly under worldbody")
        # Task-local absolute height in meters; never accumulate offsets.
        self.mj_model.body_pos[base_id, 2] = ROBOT_MOUNT_Z
        wrist_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_view")
        tool_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "tool0")
        if wrist_id < 0 or tool_id < 0:
            raise ValueError("Missing wrist_view/tool0")
        # Approximate mount: local +X side offset, +Z along tool extension.
        # Optical axis +tool-Z, MuJoCo viewing direction -camera-Z.
        self.mj_model.cam_bodyid[wrist_id] = tool_id
        self.mj_model.cam_pos[wrist_id] = [0.05, 0.0, 0.10]
        self.mj_model.cam_quat[wrist_id] = [0.0, 1.0, 0.0, 0.0]
        if getattr(self.config, "initial_camera_binding", True):
            self.bind_initial_cameras(base_id, wrist_id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        table_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        self.table_z = float(self.mj_data.geom_xpos[table_id, 2] + self.mj_model.geom_size[table_id, 2])
        self.table_center = self.mj_data.geom_xpos[table_id].copy()

    def bind_initial_cameras(self, base_link_id, wrist_id):
        """Prior-assisted estimate; preserve the existing table/3DGS world frame.

        Cameras use centered pinhole FOV approximation for both rendering engines.
        This is not calibrated rendering or a real robot deployment transform.
        """
        path = Path(DISCOVERSE_ROOT_DIR) / "models/mjcf/ur5/camera_initial_v3.json"
        payload = json.loads(path.read_text())
        estimate = payload["estimate"]
        mujoco.mj_forward(self.mj_model, self.mj_data)

        def body_transform(name):
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Missing camera reference body {name}")
            t = np.eye(4)
            t[:3, :3] = self.mj_data.xmat[bid].reshape(3, 3)
            t[:3, 3] = self.mj_data.xpos[bid]
            return t

        def quat(rotation):
            return Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]

        # Estimated table X is across its 90cm edge, Y points toward robot.
        # Existing scene table X is its 80cm edge pointing away from robot.
        world_table = body_transform("table")
        tid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
        world_table[:3, 3] += world_table[:3, :3] @ [0, 0, self.mj_model.geom_size[tid, 2]]
        world_table[:3, :3] = world_table[:3, :3] @ np.array([[0,-1,0],[1,0,0],[0,0,1]])
        link_base = np.linalg.inv(body_transform("base_link")) @ body_transform("base")
        world_base = world_table @ np.linalg.inv(np.asarray(estimate["T_base_table"]))
        reconstructed_world_base = world_base.copy()
        # Task-only layout override; replay callers retain the reconstructed pose.
        if getattr(self.config, "align_base_table_centerline", False):
            world_base[1, 3] = world_table[1, 3]
        world_link = world_base @ np.linalg.inv(link_base)
        self.mj_model.body_pos[base_link_id] = world_link[:3, 3]
        self.mj_model.body_quat[base_link_id] = quat(world_link[:3, :3])
        # Move the visual mounting stand with the base (height remains measured).
        pid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "robot_pedestal")
        if pid >= 0:
            self.mj_model.geom_pos[pid, :2] = world_link[:2, 3]
            self.mj_model.geom_quat[pid] = quat(world_link[:3, :3])
        cv_to_mj = np.diag([1., -1., -1.])
        front_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "primary_view")
        front = world_table @ np.asarray(estimate["T_table_front_camera"])
        wrist = np.asarray(estimate["T_tool0_wrist_camera"]).copy()
        # User-confirmed mounting side: opposite upstream wrist/forearm.
        # In recorded reference posture these lie in tool0 +Y, not +X.
        # Fixed tool-local mount (never recompute from moving arm each frame).
        # Keep estimated axial distance and optical orientation; only change side.
        wrist[:2, 3] = [0.0, -0.05]
        for cid, t, model_name in ((front_id, front, "D436"), (wrist_id, wrist, "D405")):
            self.mj_model.cam_pos[cid] = t[:3, 3]
            self.mj_model.cam_quat[cid] = quat(t[:3, :3] @ cv_to_mj)
            camera = next(c for c in payload["intrinsics"]["cameras"] if model_name in c["name"])
            p = camera["profiles"][0]
            self.mj_model.cam_fovy[cid] = np.degrees(2*np.arctan(p["height"]/(2*p["fy"])))
        # Old decorative camera rigs are no longer at the estimated optical poses.
        rig = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "front_camera_rig")
        for gid in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if self.mj_model.geom_bodyid[gid] == rig or name.startswith(("front_tripod", "wrist_camera")):
                self.mj_model.geom_rgba[gid, 3] = 0
        self.config.camera_binding_metadata = {
            "source": str(path), "status": estimate["status"],
            "intrinsic_mode": "centered_fovy_approximation_no_distortion",
            "T_world_ur_base": world_base.tolist(),
            "T_world_ur_base_reconstructed": reconstructed_world_base.tolist(),
            "base_table_centerline_override": getattr(self.config, "align_base_table_centerline", False),
            "T_world_estimated_table": world_table.tolist(),
            "T_tool0_wrist_camera_effective": wrist.tolist(),
            "wrist_mount_override": "user-confirmed outward side: tool0 -Y 0.05m; axial distance/rotation unchanged; not calibrated",
        }
        print("Initial camera binding v3: D436 front / D405 tool0; NOT calibrated; FOV approximation")
        print("Simulation UR base XYZ (m):", np.round(world_base[:3, 3], 6).tolist(),
              "table surface center:", np.round(world_table[:3, 3], 6).tolist())

    def __init__(self, config, rng, randomize=True):
        self.rng = rng
        self.randomize = randomize
        super().__init__(config)
        if getattr(config, "request_gaussian", False):
            if not self.config.use_gaussian_renderer:
                raise RuntimeError("3DGS requested but renderer unavailable; refusing silent mesh fallback")
            # Translate the static background only. Dynamic PLYs are body-local.
            import torch
            offset = np.asarray(config.gs_background_offset, dtype=np.float32)
            gs = self.gs_renderer
            count = gs.gaussians_size["background"]
            gs.gaussians_all["background"].xyz += offset
            with torch.no_grad():
                xyz = gs.renderer.gaussians.xyz
                xyz[:count] += torch.as_tensor(offset, device=xyz.device, dtype=xyz.dtype)
            gs.renderer.need_rerender = True
            for name in self.config.obj_list + self.config.rb_link_list:
                if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
                    raise ValueError(f"Missing 3DGS binding body: {name}")
        self.block_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "red_block"
        )
        self.target_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_zone"
        )
        block_jid = self.mj_model.body_jntadr[self.block_id]
        self.block_qpos_adr = self.mj_model.jnt_qposadr[block_jid]
        self.block_qvel_adr = self.mj_model.jnt_dofadr[block_jid]
        gripper_jid = mujoco.mj_name2id(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "gripper_right_driver_joint",
        )
        self.gripper_qpos_adr = self.mj_model.jnt_qposadr[gripper_jid]
        self.gripper_qvel_adr = self.mj_model.jnt_dofadr[gripper_jid]
        self.home_q = self.mj_model.key_qpos[self.home_key_id, :6].copy()
        if getattr(config, "use_real_training_start", False):
            reference_path = Path(DISCOVERSE_ROOT_DIR) / "policies/smolvla/configs/ur5_real_deploy.json"
            reference = json.loads(reference_path.read_text())["start_reference_deg"]
            self.home_q = np.deg2rad(reference)
            if self.home_q.shape != (6,) or not np.isfinite(self.home_q).all():
                raise ValueError("Invalid real training start")
            for i, angle in enumerate(self.home_q):
                jid = int(self.mj_model.dof_jntid[i])
                if self.mj_model.jnt_limited[jid] and not self.mj_model.jnt_range[jid, 0] <= angle <= self.mj_model.jnt_range[jid, 1]:
                    raise ValueError("Real start outside simulation joint limits")
            config.start_reference_rad = self.home_q.tolist()
            print("Training start degrees (no angle wrapping):", reference)
        self.home_tcp_rotation = None
        self.target_control = np.zeros(7)
        self.sampling_cells = []

    def sample_block_pose(self):
        mode = getattr(self.config, "sampling", "local")
        if mode == "local":
            xy = self.table_center[:2] + self.rng.uniform([-0.03, -0.06], [0.03, 0.06])
            yaw = self.rng.uniform(-15, 15)
            cell = None
            bounds = None
        else:
            # Current measured table is world-axis aligned; do not silently use this on a rotated table.
            tid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
            if not np.allclose(self.mj_data.geom_xmat[tid].reshape(3, 3), np.eye(3), atol=1e-6):
                raise ValueError("Tabletop sampler requires the current axis-aligned table")
            half = self.mj_model.geom_size[tid, :2]
            low = self.table_center[:2] - half + self.config.edge_margin
            high = self.table_center[:2] + half - self.config.edge_margin
            # +world-X is the far side in this measured scene.
            high[0] -= self.config.far_edge_inset
            # Keep lateral extremes out of the default collection workspace.
            side_margin = getattr(self.config, "side_margin", self.config.edge_margin)
            low[1] = self.table_center[1] - half[1] + max(side_margin, self.config.edge_margin)
            high[1] = self.table_center[1] + half[1] - max(side_margin, self.config.edge_margin)
            if np.any(high <= low):
                raise ValueError("Empty tabletop sampling region")
            if not self.sampling_cells:
                self.sampling_cells = self.rng.permutation(36).tolist()
            cell = int(self.sampling_cells.pop())
            grid = np.array([cell // 6, cell % 6])
            xy = low + (grid + self.rng.random(2)) / 6 * (high - low)
            yaw = self.rng.uniform(-45, 45)
            bounds = [low.tolist(), high.tolist()]
        self.config.sampling_metadata = {"mode": mode, "proposal_grid_cell": cell,
            "grid_shape": [6, 6] if mode == "tabletop" else None,
            "world_xy_bounds_m": bounds, "sampled_xy_m": xy.tolist(), "yaw_deg": float(yaw),
            "filter": "IK endpoint and sampled joint-path table collision rejection; not a safety guarantee"}
        return xy, np.deg2rad(yaw)

    def resetState(self):
        super().resetState()
        if getattr(self.config, "use_real_training_start", False):
            self.mj_data.qpos[:6] = self.home_q
            self.mj_data.ctrl[:6] = self.home_q
        if self.randomize:
            adr = self.block_qpos_adr
            xy, yaw = self.sample_block_pose()
            self.mj_data.qpos[adr:adr + 2] = xy
            quat = Rotation.from_euler("z", yaw).as_quat()
            self.mj_data.qpos[adr + 3 : adr + 7] = quat[[3, 0, 1, 2]]
        self.mj_data.qvel[:] = 0
        self.target_control[:6] = self.mj_data.qpos[:6]
        self.target_control[6] = GRIPPER_OPEN
        mujoco.mj_forward(self.mj_model, self.mj_data)
        if self.home_tcp_rotation is None:
            self.home_tcp_rotation = (
                self.mj_data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
            )

    def block_pos(self):
        return self.mj_data.xpos[self.block_id].copy()

    def target_pos(self):
        return self.mj_data.xpos[self.target_id].copy()

    def normalized_state(self):
        grip = np.clip(
            self.mj_data.qpos[self.gripper_qpos_adr] / GRIPPER_JOINT_RANGE, 0, 1
        )
        return np.r_[self.mj_data.qpos[:6], grip].astype(np.float32)

    def gripper_gap(self):
        """Separation of opposed pad1 box faces; not a real gripper calibration."""
        ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
               for name in ("gripper_left_pad1", "gripper_right_pad1")]
        if min(ids) < 0:
            raise ValueError("Missing gripper pad collision boxes")
        delta = self.mj_data.geom_xpos[ids[1]] - self.mj_data.geom_xpos[ids[0]]
        distance = np.linalg.norm(delta)
        direction = delta / max(distance, 1e-12)
        extents = [np.abs(self.mj_data.geom_xmat[g].reshape(3, 3).T @ direction)
                   @ self.mj_model.geom_size[g] for g in ids]
        return float(max(0, distance - sum(extents)))

    def normalized_velocity(self):
        grip = self.mj_data.qvel[self.gripper_qvel_adr] / GRIPPER_JOINT_RANGE
        return np.r_[self.mj_data.qvel[:6], grip].astype(np.float32)

    def success_info(self):
        if getattr(self.config, "lift_only", False):
            gids = np.flatnonzero(self.mj_model.geom_bodyid == self.block_id)
            bottoms = []
            for gid in gids:
                if self.mj_model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
                    raise ValueError("Lift clearance requires a box block geometry")
                rotation = self.mj_data.geom_xmat[gid].reshape(3, 3)
                extent_z = np.abs(rotation[2]) @ self.mj_model.geom_size[gid]
                bottoms.append(float(self.mj_data.geom_xpos[gid, 2] - extent_z))
            clearance = min(bottoms) - self.table_z
            speed = float(np.linalg.norm(self.mj_data.qvel[self.block_qvel_adr:self.block_qvel_adr + 3]))
            angular_speed = float(np.linalg.norm(self.mj_data.qvel[self.block_qvel_adr + 3:self.block_qvel_adr + 6]))
            return {"clearance_m": clearance, "lifted": clearance >= self.config.success_height,
                    "stationary": speed <= 0.03 and angular_speed <= 0.3,
                    "speed": speed, "angular_speed": angular_speed}
        block = self.block_pos()
        target = self.target_pos()
        xy_error = np.abs(block[:2] - target[:2])
        speed = np.linalg.norm(
            self.mj_data.qvel[self.block_qvel_adr : self.block_qvel_adr + 3]
        )
        return {
            "inside": bool(np.all(xy_error <= 0.035)),
            "on_table": abs(float(block[2]) - (self.table_z + 0.025)) <= 0.015,
            "stationary": speed <= 0.03,
            "open": self.mj_data.qpos[self.gripper_qpos_adr] < 0.15,
            "xy_error": xy_error,
            "speed": float(speed),
        }

    def check_success(self):
        s = self.success_info()
        if getattr(self.config, "lift_only", False):
            return s["lifted"] and s["stationary"]
        return s["inside"] and s["on_table"] and s["stationary"] and s["open"]


class Expert:
    NAMES = [
        "home/open",
        "above block",
        "pregrasp",
        "grasp height",
        "close",
        "lift to safe height",
        "cartesian transfer",
        "place height",
        "open",
        "vertical retreat",
    ]

    def __init__(self, env):
        self.env = env
        self.ik_data = mujoco.MjData(env.mj_model)
        self.ik = UR5IK(env.mj_model, self.ik_data)
        self.state = 0
        self.state_start = 0.0
        self.wait_until = 0.0
        self.target_tcp = None
        self.grasp_xy = None
        self.transfer_waypoints = []
        self.transfer_index = 0
        self.retreat_waypoints = []
        self.retreat_index = 0
        self.release_phase = None
        self.finished = False

    def reset(self):
        self.state = 0
        self.grasp_xy = self.env.block_pos()[:2].copy()
        # Align finger closing direction to a cube face normal, not base/home yaw.
        # Camera reconstruction can rotate the base; the cube's world yaw is independent.
        model, data = self.env.mj_model, self.env.mj_data
        pad_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                   for name in ("gripper_left_pad", "gripper_right_pad")]
        if min(pad_ids) < 0:
            raise EpisodeFailure("Missing finger pads for grasp orientation")
        closing = data.xpos[pad_ids[1]] - data.xpos[pad_ids[0]]
        if np.linalg.norm(closing[:2]) < 1e-6:
            raise EpisodeFailure("Cannot determine horizontal finger closing direction")
        current_yaw = np.arctan2(closing[1], closing[0])
        block_rotation = data.xmat[self.env.block_id].reshape(3, 3)
        block_yaw = np.arctan2(block_rotation[1, 0], block_rotation[0, 0])
        # Square cube: four equivalent face-normal directions; choose least wrist rotation.
        differences = [(block_yaw + k*np.pi/2 - current_yaw + np.pi) % (2*np.pi) - np.pi
                       for k in range(4)]
        correction = min(differences, key=abs)
        current_rotation = data.site_xmat[self.env.ee_site_id].reshape(3, 3).copy()
        self.grasp_rotation = Rotation.from_euler("z", correction).as_matrix() @ current_rotation
        print(f"  grasp yaw correction: {np.degrees(correction):.2f} deg (cube-aligned)")
        self.transfer_waypoints = []
        self.transfer_index = 0
        self.retreat_waypoints = []
        self.retreat_index = 0
        self.release_phase = None
        self.finished = False
        self.enter()

    def approach_path(self):
        start = self.env.mj_data.site_xpos[self.env.ee_site_id].copy()
        height = max(float(start[2]), self.env.table_z + 0.24)
        count = max(1, int(np.ceil(np.linalg.norm(self.grasp_xy - start[:2]) / 0.06)))
        return [np.r_[start[:2], height]] + [
            np.r_[start[:2] + alpha * (self.grasp_xy - start[:2]), height]
            for alpha in np.linspace(1 / count, 1, count)
        ] + [np.r_[self.grasp_xy, self.env.table_z + 0.18]]

    def solve(self, pos, reference=None):
        pos = np.asarray(pos, dtype=float)
        reference = self.env.mj_data.qpos[:6].copy() if reference is None else np.asarray(reference)
        q = self.ik.properIK(
            pos, self.grasp_rotation, reference
        )
        # Choose an equivalent solution near feedback, but never wrap the saved state.
        for i in range(6):
            jid = int(self.env.mj_model.dof_jntid[i])
            choices = q[i] + 2 * np.pi * np.arange(-2, 3)
            if self.env.mj_model.jnt_limited[jid]:
                lo, hi = self.env.mj_model.jnt_range[jid]
                choices = choices[(choices >= lo) & (choices <= hi)]
            if not len(choices):
                raise EpisodeFailure("IK has no equivalent solution inside joint limits")
            q[i] = choices[np.argmin(np.abs(choices - reference[i]))]
        self.ik_data.qpos[:] = self.env.mj_data.qpos
        self.ik_data.qvel[:] = 0
        self.ik_data.qpos[:6] = q
        mujoco.mj_forward(self.env.mj_model, self.ik_data)
        got_pos = self.ik_data.site_xpos[self.env.ee_site_id]
        got_rot = self.ik_data.site_xmat[self.env.ee_site_id].reshape(3, 3)
        pos_err = np.linalg.norm(got_pos - pos)
        rot_err = np.linalg.norm(
            Rotation.from_matrix(
                self.grasp_rotation @ got_rot.T
            ).as_rotvec()
        )
        if not np.all(np.isfinite(q)) or pos_err > 0.015 or rot_err > 0.15:
            raise EpisodeFailure(
                f"IK failed: position_error={pos_err:.4f}, rotation_error={rot_err:.4f}"
            )
        return q

    def table_collision(self, data):
        model = self.env.mj_model
        shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")
        table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        def moving_arm(body):
            while body > 0:
                if body == shoulder:
                    return True
                body = int(model.body_parentid[body])
            return False
        for contact in data.contact[:data.ncon]:
            if contact.dist >= -0.0005:
                continue
            for arm, obstacle in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, obstacle)
                if moving_arm(int(model.geom_bodyid[arm])) and (model.geom_bodyid[obstacle] == table or name in ("floor", "robot_pedestal")):
                    return True
        return False

    def preflight(self):
        """Kinematic rejection only; dynamic success still required before saving."""
        reference = self.env.mj_data.qpos[:6].copy()
        path = self.approach_path() + [np.r_[self.grasp_xy, self.env.table_z + h]
                                      for h in (0.085, 0.033, TRANSFER_HEIGHT)]
        for position in path:
            target = self.solve(position, reference)
            steps = max(2, int(np.ceil(np.max(np.abs(target - reference)) / 0.04)))
            for alpha in np.linspace(0, 1, steps + 1):
                self.ik_data.qpos[:] = self.env.mj_data.qpos
                self.ik_data.qpos[:6] = reference + alpha * (target - reference)
                mujoco.mj_forward(self.env.mj_model, self.ik_data)
                if self.table_collision(self.ik_data):
                    raise EpisodeFailure("preflight arm/gripper collision with table/floor/pedestal")
            reference = target

    def pose(self, pos):
        self.target_tcp = np.asarray(pos, dtype=float)
        self.env.target_control[:6] = self.solve(self.target_tcp)

    def begin_transfer(self, target):
        start_xy = np.asarray(self.grasp_xy, dtype=float)
        target_xy = np.asarray(target[:2], dtype=float)
        distance = np.linalg.norm(target_xy - start_xy)
        count = max(1, int(np.ceil(distance / TRANSFER_WAYPOINT_SPACING)))
        self.transfer_waypoints = [
            np.r_[start_xy + alpha * (target_xy - start_xy), self.env.table_z + TRANSFER_HEIGHT]
            for alpha in np.linspace(1.0 / count, 1.0, count)
        ]
        self.transfer_index = 0
        self.pose(self.transfer_waypoints[0])

    def begin_vertical_retreat(self):
        start = self.env.mj_data.site_xpos[self.env.ee_site_id].copy()
        start_z = max(self.env.table_z + PLACE_COMMAND_HEIGHT, float(start[2]))
        distance = max(0.0, self.env.table_z + TRANSFER_HEIGHT - start_z)
        count = max(1, int(np.ceil(distance / RETREAT_WAYPOINT_SPACING)))
        self.retreat_waypoints = [
            np.array(
                [start[0], start[1], start_z + alpha * distance],
                dtype=float,
            )
            for alpha in np.linspace(1.0 / count, 1.0, count)

        ]
        self.retreat_index = 0
        self.pose(self.retreat_waypoints[0])

    def enter(self):
        now = float(self.env.mj_data.time)
        self.state_start = now
        self.target_tcp = None
        target = self.env.target_pos()
        block = self.env.block_pos()

        if self.state == 0:
            self.env.target_control[:6] = self.env.home_q
            self.env.target_control[6] = GRIPPER_OPEN
        elif self.state == 1:
            if getattr(self.env.config, "sampling", "local") == "tabletop":
                self.approach_waypoints = self.approach_path()
                self.approach_index = 0
                self.pose(self.approach_waypoints[0])
            else:
                self.pose([*self.grasp_xy, self.env.table_z + 0.18])
        elif self.state == 2:
            self.pose([*self.grasp_xy, self.env.table_z + 0.085])
        elif self.state == 3:
            self.pose([*self.grasp_xy, block[2] + 0.008])
        elif self.state == 4:
            self.env.target_control[6] = GRIPPER_CLOSED
            self.wait_until = now + 0.8
        elif self.state == 5:
            self.pose([*self.grasp_xy, self.env.table_z + TRANSFER_HEIGHT])
        elif self.state == 6:
            if getattr(self.env.config, "lift_only", False):
                self.finished = True
                print("  -> hold lifted block (gripper stays closed)")
                return
            self.begin_transfer(target)
        elif self.state == 7:
            self.pose([target[0], target[1], self.env.table_z + PLACE_COMMAND_HEIGHT])
        elif self.state == 8:
            self.release_phase = "settle"
            self.env.target_control[6] = GRIPPER_CLOSED
            self.wait_until = now + RELEASE_SETTLE_TIME
        elif self.state == 9:
            self.begin_vertical_retreat()
        else:
            self.finished = True
            return
        print(f"  -> state {self.state}: {self.NAMES[self.state]}")

    @property
    def joint_speed(self):
        if self.state == 9:
            return RETREAT_JOINT_SPEED
        return CARRY_JOINT_SPEED if self.state in (5, 6, 7) else NORMAL_JOINT_SPEED

    def arm_done(self):
        if self.target_tcp is not None:
            rotation = self.env.mj_data.site_xmat[self.env.ee_site_id].reshape(3, 3)
            rotation_error = Rotation.from_matrix(self.grasp_rotation @ rotation.T).magnitude()
            if rotation_error > np.deg2rad(3):
                return False
        joint_err = np.max(
            np.abs(self.env.mj_data.qpos[:6] - self.env.target_control[:6])
        )
        joint_speed = np.max(np.abs(self.env.mj_data.qvel[:6]))
        tcp_err = 0.0
        if self.target_tcp is not None:
            tcp_err = np.linalg.norm(
                self.env.mj_data.site_xpos[self.env.ee_site_id] - self.target_tcp
            )
        if self.state == 7:
            return joint_err < 0.030 and tcp_err < 0.020 and joint_speed < 0.15
        tcp_tolerance = 0.040 if self.state == 9 else (0.035 if self.state == 6 else 0.020)
        return joint_err < 0.045 and tcp_err < tcp_tolerance

    def update(self):
        if self.finished:
            return
        if self.state == 6 and self.env.block_pos()[2] < self.env.table_z + MIN_CARRY_BLOCK_Z:
            raise EpisodeFailure(
                f"block dropped or touched table during transfer: z={self.env.block_pos()[2]:.4f}"
            )
        now = float(self.env.mj_data.time)
        if now - self.state_start > getattr(self.env.config, "expert_state_timeout_s", (12.0 if getattr(self.env.config, "sampling", "local") == "tabletop" else 5.0)):
            joint_err = np.max(np.abs(self.env.mj_data.qpos[:6] - self.env.target_control[:6]))
            tcp_err = 0.0 if self.target_tcp is None else np.linalg.norm(
                self.env.mj_data.site_xpos[self.env.ee_site_id] - self.target_tcp
            )
            raise EpisodeFailure(f"state {self.state} ({self.NAMES[self.state]}) timed out: joint_error={joint_err:.4f}, tcp_error={tcp_err:.4f}")
        if self.state == 8 and self.release_phase == "settle":
            if now < self.wait_until:
                return
            self.release_phase = "opening"
            self.env.target_control[6] = GRIPPER_OPEN
            self.wait_until = now + RELEASE_OPEN_TIME
            print("  -> release: gripper opening")
            return
        done = now >= self.wait_until if self.state in (4, 8) else self.arm_done()
        if not done:
            return
        if self.state == 1 and getattr(self.env.config, "sampling", "local") == "tabletop" and self.approach_index + 1 < len(self.approach_waypoints):
            self.approach_index += 1
            self.state_start = now
            self.pose(self.approach_waypoints[self.approach_index])
            print(f"  -> approach waypoint {self.approach_index + 1}/{len(self.approach_waypoints)}")
            return
        if self.state == 6 and self.transfer_index + 1 < len(self.transfer_waypoints):
            self.transfer_index += 1
            self.state_start = now
            self.pose(self.transfer_waypoints[self.transfer_index])
            print(
                f"  -> transfer waypoint {self.transfer_index + 1}/"
                f"{len(self.transfer_waypoints)}"
            )
            return
        if self.state == 9 and self.retreat_index + 1 < len(self.retreat_waypoints):
            self.retreat_index += 1
            self.state_start = now
            self.pose(self.retreat_waypoints[self.retreat_index])
            print(
                f"  -> retreat waypoint {self.retreat_index + 1}/"
                f"{len(self.retreat_waypoints)}"
            )
            return
        print(f"  <- state {self.state}: done")
        if self.state == 5 and self.env.block_pos()[2] < self.env.table_z + 0.070:
            raise EpisodeFailure("grasp verification failed: block was not lifted")
        self.state += 1
        self.enter()


def make_config(auto, gaussian=False, background_offset=(-0.01, 0.0, 0.497)):
    cfg = UR5Cfg()
    cfg.mjcf_file_path = "mjcf/ur5/scene_single_red_block.xml"
    cfg.timestep = 1 / 240
    cfg.decimation = 8
    cfg.sync = not auto
    cfg.headless = auto
    cfg.render_set = {"fps": CAPTURE_HZ, "width": 640, "height": 480}
    cfg.obs_rgb_cam_id = [0, 1]


    cfg.use_gaussian_renderer = gaussian
    cfg.request_gaussian = gaussian
    cfg.gs_background_offset = list(background_offset)
    cfg.gs_background_sh_degree = 3

    cfg.obj_list = ["red_block"]

    cfg.rb_link_list = [
        "base_link_inertia",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]

    asset_dir = "ur5_single_red_pick_place"

    cfg.gs_model_dict = {
        # Full scene, calibrated to the 0.80 x 0.90 m tabletop; filmed robot removed.
        # Asset-local tabletop z=0; gs_background_offset places it at world z=0.497.
        "background": f"{asset_dir}/background_20261028_ready.ply",
        "red_block": f"{asset_dir}/red_block.ply",
    }

    for body_name in cfg.rb_link_list:
        cfg.gs_model_dict[body_name] = (
            f"{asset_dir}/ur5_body_aligned_color_v1/{body_name}.ply"
        )

    # Body-local assets: MuJoCo supplies mounting rotations and moving poses.
    gripper_parts = ["base_mount", "base"] + [
        f"{side}_{part}"
        for side in ("left", "right")
        for part in (
            "driver", "coupler", "spring_link", "follower", "pad", "silicone_pad"
        )
    ]
    for part in gripper_parts:
        body_name = f"gripper_{part}"
        cfg.rb_link_list.append(body_name)
        cfg.gs_model_dict[body_name] = (
            f"{asset_dir}/robotiq_2f85_3dgs/{part}.ply"
        )

    if gaussian:
        from discoverse import DISCOVERSE_ASSETS_DIR
        missing = [str(Path(DISCOVERSE_ASSETS_DIR) / "3dgs" / value)
                   for value in cfg.gs_model_dict.values()
                   if not (Path(DISCOVERSE_ASSETS_DIR) / "3dgs" / value).is_file()]
        if missing:
            raise FileNotFoundError("Missing 3DGS assets:\n" + "\n".join(missing))
    return cfg

def new_buffers():
    return {
        "time": [],
        "qpos": [],
        "qvel": [],
        "ee_pos": [],
        "ee_quat": [],
        "block_pos": [],
        "target_pos": [],
        "act": [],
        "block_clearance_m": [],
        "gripper_gap_m": [],
        "lift_condition": [],
    }


def record(env, obs, raw_action, buffers, encoders):
    if buffers["time"] and not np.isclose(obs["time"] - buffers["time"][-1], 1 / CAPTURE_HZ, atol=1e-8):
        raise EpisodeFailure("State/action sampling is not 30 Hz")
    if env.config.use_gaussian_renderer and not env.show_gaussian_img:
        raise EpisodeFailure("Renderer switched during collection; mixed-renderer episode rejected")
    for camera_id, encoder in enumerate(encoders):
        encoder.encode(obs["img"][camera_id], obs["time"])
    buffers["time"].append(float(obs["time"]))
    buffers["qpos"].append(env.normalized_state().tolist())
    buffers["qvel"].append(env.normalized_velocity().tolist())
    buffers["ee_pos"].append(list(obs["ep"]))
    buffers["ee_quat"].append(list(obs["eq"]))
    buffers["block_pos"].append(env.block_pos().tolist())
    buffers["target_pos"].append(env.target_pos().tolist())
    buffers["block_clearance_m"].append(env.success_info().get("clearance_m"))
    buffers["gripper_gap_m"].append(env.gripper_gap())
    buffers["lift_condition"].append(bool(env.check_success()))
    policy_action = np.asarray(raw_action, dtype=np.float32).copy()
    policy_action[6] = np.clip(policy_action[6] / 255.0, 0, 1)
    buffers["act"].append(policy_action.tolist())


def save_episode(path, buffers, seed, attempt, config, success=True, failure=None, diagnostic=False):
    payload = {
        "time": buffers["time"],
        "obs": {
            "qpos": buffers["qpos"],
            "qvel": buffers["qvel"],
            "ee_pos": buffers["ee_pos"],
            "ee_quat": buffers["ee_quat"],
            "block_pos": buffers["block_pos"],
            "target_pos": buffers["target_pos"],
            "block_clearance_m": buffers["block_clearance_m"],
            "gripper_gap_m": buffers["gripper_gap_m"],
            "lift_condition": buffers["lift_condition"],
        },
        "act": buffers["act"],
        "language_instruction": TASK_INSTRUCTION,
        "metadata": {
            "robot": "ur5_robotiq_2f85",
            "camera_ids": [0, 1],
            "camera_names": ["primary_view", "wrist_view"],
            "camera_calibrated": False,
            "video_encoding": {"fps": CAPTURE_HZ, "crf": 18, "preset": "veryfast"},
            "camera_feature_mapping": {"cam_0": "observation.images.d436", "cam_1": "observation.images.d405"},
            "camera_roles": {"cam_0": "front", "cam_1": "wrist"},
            "state_semantics": "actual joint radians + clipped driver_joint/0.8",
            "action_semantics": "absolute joint target radians + gripper command/255",
            "sample_alignment": "observation at t; action applied over [t,t+1/30)",
            "start_reference_rad": getattr(config, "start_reference_rad", None),
            "sampling": getattr(config, "sampling_metadata", {"mode": "fixed"}),
            "success_criterion": {"block_bottom_clearance_m": config.success_height, "continuous_hold_s": config.hold_seconds,
                                  "max_linear_speed_m_s": 0.03, "max_angular_speed_rad_s": 0.3},
            "gripper_calibration": {"status": "direction_only_not_physically_calibrated",
                                    "simulation_feedback": "driver_joint/0.8", "real_feedback": "POS/255",
                                    "physical_opening_equivalence_verified": False},
            "render_shadows": getattr(config, "render_shadows", False),
            "camera_binding": getattr(config, "camera_binding_metadata", None),
            "renderer": "3dgs" if config.use_gaussian_renderer else "mujoco",
            "background_offset_m": config.gs_background_offset if config.use_gaussian_renderer else None,
            "background_asset": config.gs_model_dict.get("background") if config.use_gaussian_renderer else None,
            "background_sh_degree": getattr(config, "gs_background_sh_degree", 0) if config.use_gaussian_renderer else None,
            "scene": "measured_v1",
            "robot_mount_z": ROBOT_MOUNT_Z,
            "fps": CAPTURE_HZ,
            "physics_timestep_s": config.timestep,
            "physics_steps_per_action": config.decimation,
            "arm_gravity_compensation": getattr(config, "arm_gravity_compensation", False),
            "gravity_compensation_method": "position_actuator_g_over_kp_arm6" if getattr(config, "arm_gravity_compensation", False) else "off",
            "state_dim": 7,
            "action_dim": 7,
            "gripper_convention": "normalized: 0=open, 1=closed",
            "seed": seed,
            "attempt": attempt,
            "success": bool(success),
            "failure_reason": failure,
            "diagnostic_only": bool(diagnostic),
            "training_eligible": bool(success and not diagnostic),
        },
    }
    with open(os.path.join(path, "obs_action.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0)
    parser.add_argument("--data_set_size", type=int, default=1)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--arm-gravity-compensation", action="store_true",
                        help="Gravity feedforward on six arm actuators; original torque limits retained")
    parser.add_argument("--sampling", choices=("local", "tabletop"), default="tabletop")
    parser.add_argument("--edge-margin", type=float, default=0.12, help="Block-center inset from table edges, metres")
    parser.add_argument("--far-edge-inset", type=float, default=0.14, help="Additional exclusion on far +X edge, metres")
    parser.add_argument("--side-margin", type=float, default=0.27, help="Minimum block-center inset from both Y edges, metres")
    parser.add_argument("--success-height", type=float, default=0.05, help="Block bottom clearance above table, metres")
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, help="Separate dataset directory; existing samples are backed up")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_randomization", action="store_true")
    parser.add_argument("--max_attempts", type=int, default=0)
    parser.add_argument("--camera_preview", action="store_true",
                        help="Save both static camera images and calibrated=false parameters; no task motion")
    parser.add_argument("--preview", action="store_true",
                        help="Interactive static viewer; Ctrl+G switches renderers; no data collection")
    parser.add_argument("--gaussian", action="store_true", help="Enable body-bound 3DGS rendering")
    parser.add_argument("--render-shadows", action="store_true",
                        help="Enable expensive MuJoCo shadows (off by default for dual cameras)")
    parser.add_argument("--legacy-cameras", action="store_true",
                        help="Use previous approximate camera/base poses instead of initial reconstruction")
    parser.add_argument("--reconstructed-base", action="store_true",
                        help="Keep reconstructed base lateral offset instead of aligning with table centerline")
    parser.add_argument("--background-offset", nargs=3, type=float, default=[-0.01, 0.0, 0.497],
                        metavar=("X", "Y", "Z"), help="Static background translation in metres; provisional alignment")
    args = parser.parse_args()
    if args.preview and (args.auto or args.camera_preview):
        parser.error("--preview opens a window; use without --auto or --camera_preview")
    if not np.isfinite([args.edge_margin, args.far_edge_inset]).all() or args.edge_margin < 0.04 or args.far_edge_inset < 0:
        parser.error("Require finite edge-margin >= 0.04m and far-edge-inset >= 0")
    if not np.isfinite([args.success_height, args.hold_seconds]).all() or not 0 < args.success_height <= 0.20 or args.hold_seconds <= 0:
        parser.error("Require 0 < success-height <= 0.20m and finite positive hold-seconds")
    if not np.isfinite(args.background_offset).all():
        parser.error("Background offset must be finite")

    config = make_config(args.auto or args.camera_preview, args.gaussian, args.background_offset)
    config.initial_camera_binding = not args.legacy_cameras
    config.align_base_table_centerline = not args.reconstructed_base
    config.render_shadows = args.render_shadows
    config.lift_only = True
    config.use_real_training_start = True
    config.success_height = args.success_height
    config.hold_seconds = args.hold_seconds
    config.arm_gravity_compensation = args.arm_gravity_compensation
    config.sampling = args.sampling
    if not np.isfinite(args.side_margin) or args.side_margin < 0.04:
        parser.error("side-margin must be finite and at least 0.04m")
    config.edge_margin = args.edge_margin
    config.side_margin = args.side_margin
    config.far_edge_inset = args.far_edge_inset
    env = SimNode(
        config,
        np.random.default_rng(args.seed),
        randomize=not args.no_randomization,
    )
    if not np.isclose(env.delta_t, 1 / CAPTURE_HZ):
        raise ValueError("Control and capture clocks must both be 30 Hz")
    names = [mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in (0, 1)]
    if names != ["primary_view", "wrist_view"]:
        raise ValueError(f"Camera IDs no longer match front/wrist: {names}")
    if args.preview:
        import glfw
        env.reset()
        env.cam_id = 0
        print("Preview only: no physics stepping or data collection. "
              "Ctrl+G: MuJoCo/3DGS; [ / ]: cameras; Esc: free view; close window to exit.", flush=True)
        try:
            while env.running:
                env.render()
        except KeyboardInterrupt:
            pass
        finally:
            env.renderer.close()
            if env.window is not None:
                glfw.destroy_window(env.window)
                env.window = None
            glfw.terminate()
        return
    if args.camera_preview:
        import cv2
        env.reset()
        folder = Path(DISCOVERSE_ROOT_DIR) / "data/camera_previews" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder.mkdir(parents=True, exist_ok=False)
        report = {"calibrated": False, "camera_geometry_collision_enabled": False, "cameras": {}}
        report["camera_binding"] = getattr(env.config, "camera_binding_metadata", None)
        report["renderer"] = "3dgs" if args.gaussian else "mujoco"
        report["background_offset_m"] = args.background_offset if args.gaussian else None
        if args.gaussian:
            env.update_gs_scene()
        frames = []
        # Direct rendering: no mj_step, no episode creation, no robot connection.
        with mujoco.Renderer(env.mj_model, height=480, width=640) as renderer:
            for name in ("primary_view", "wrist_view"):
                cid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                renderer.update_scene(env.mj_data, camera=cid)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = args.render_shadows
                rgb = renderer.render().copy()
                if args.gaussian:
                    gs_rgb = env.getRgbImg(cid).copy()
                    comparison = np.concatenate([rgb, gs_rgb], axis=1)
                    if not cv2.imwrite(str(folder / f"{name}_mesh_vs_gs.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("Could not save renderer comparison")
                    rgb = gs_rgb
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(folder / f"{name}.png"), bgr):
                    raise RuntimeError(f"Could not save {name}")
                frames.append(bgr)
                report["cameras"][name] = {
                    "id": int(cid), "parent_body": mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, int(env.mj_model.cam_bodyid[cid])),
                    "local_position": env.mj_model.cam_pos[cid].tolist(),
                    "local_quat_wxyz": env.mj_model.cam_quat[cid].tolist(),
                    "world_position": env.mj_data.cam_xpos[cid].tolist(),
                    "fovy_deg": float(env.mj_model.cam_fovy[cid]), "resolution": [640, 480],
                }
        if not cv2.imwrite(str(folder / "both.png"), np.concatenate(frames, axis=1)):
            raise RuntimeError("Could not save combined preview")
        with (folder / "cameras.json").open("x") as stream:
            json.dump(report, stream, indent=2)
        print(f"Camera previews: {folder}")
        return
    expert = Expert(env)
    save_root = os.path.join(
        DISCOVERSE_ROOT_DIR, "data", "ur5_grasp_lift_gs_30hz_v1" if args.gaussian else "ur5_grasp_lift_sim_30hz_v1"
    )
    if args.output_dir is not None:
        save_root = str(args.output_dir.resolve())
    elif args.sampling == "tabletop":
        save_root += "_tabletop"
    os.makedirs(save_root, exist_ok=True)

    index = args.data_idx
    end = index + args.data_set_size
    attempts = 0
    raw_action = np.zeros(7)
    try:
        while env.running and index < end:
            if args.max_attempts and attempts >= args.max_attempts:
                print(f"Reached max_attempts={args.max_attempts}")
                break
            attempts += 1
            obs = env.reset()
            try:
                expert.reset()
                if args.sampling == "tabletop":
                    expert.preflight()
            except EpisodeFailure as exc:
                print(f"REJECTED attempt {attempts}: {exc}; block={env.block_pos()}")
                with open(os.path.join(save_root, "attempts.jsonl"), "a") as stream:
                    stream.write(json.dumps({"attempt": attempts, "seed": args.seed, "sampling": getattr(config, "sampling_metadata", None), "success": False, "reason": str(exc)}) + "\n")
                continue
            raw_action[:] = env.target_control
            episode_start = float(env.mj_data.time)
            settle_start = None
            buffers = new_buffers()
            save_path = os.path.join(save_root, f"{index:03d}")
            # Replace the whole sample, never mix old metadata with new videos.
            episode_path = Path(save_path)
            if episode_path.is_symlink():
                raise ValueError(f"Refusing to replace a symlink: {episode_path}")
            if episode_path.exists() and (not episode_path.is_dir() or any(episode_path.iterdir())):
                # Keep backups outside the dataset so converters do not count them.
                backup_root = Path(save_root).parent / (Path(save_root).name + "_replaced")
                backup_root.mkdir(parents=True, exist_ok=True)
                backup_path = backup_root / (episode_path.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                if backup_path.exists():
                    raise FileExistsError(f"Backup destination already exists: {backup_path}")
                episode_path.rename(backup_path)
                print(f"Replacing sample {index:03d}; previous files backed up to: {backup_path}")
            os.makedirs(save_path, exist_ok=True)
            encoders = [PyavImageEncoder(640, 480, save_path, i, fps=CAPTURE_HZ, crf=18,
                                        preset="veryfast") for i in (0, 1)]
            print(
                f"\n=== attempt {attempts} | sample {index:03d} | "
                f"block={np.round(env.block_pos(), 3)} ==="
            )

            success = False
            failure = None
            try:
                while env.running:
                    expert.update()
                    joint_step = expert.joint_speed * env.delta_t
                    for i in range(6):
                        raw_action[i] = step_func(
                            raw_action[i],
                            env.target_control[i],
                            joint_step,
                        )
                    raw_action[6] = env.target_control[6]
                    record(env, obs, raw_action, buffers, encoders)
                    obs, _, _, _, _ = env.step(raw_action)
                    if args.sampling == "tabletop" and expert.table_collision(env.mj_data):
                        raise EpisodeFailure("dynamic arm/gripper contact with table/floor/pedestal")

                    if env.mj_data.time - episode_start > 30:
                        raise EpisodeFailure("episode timed out")
                    if expert.finished:
                        if env.check_success():
                            if settle_start is None:
                                settle_start = float(env.mj_data.time)
                            elif env.mj_data.time - settle_start >= config.hold_seconds + env.delta_t:
                                success = True
                                break
                        else:
                            settle_start = None
                            if env.mj_data.time - expert.state_start > config.hold_seconds + 5:
                                raise EpisodeFailure(f"lift/hold verification failed: {env.success_info()}")
            except EpisodeFailure as exc:
                failure = str(exc)
            finally:
                for encoder in encoders:
                    encoder.close()

            if success:
                save_episode(save_path, buffers, args.seed, attempts, env.config)
                print(f"SUCCESS {index + 1 - args.data_idx}/{args.data_set_size}: {save_path}")
                index += 1
            else:
                for encoder in encoders:
                    encoder.remove_av_file()
                print(f"FAILED attempt {attempts}: {failure or 'simulation stopped'}")
            with open(os.path.join(save_root, "attempts.jsonl"), "a") as stream:
                stream.write(json.dumps({"attempt": attempts, "seed": args.seed, "sampling": getattr(config, "sampling_metadata", None), "success": success, "reason": failure}) + "\n")
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print(f"\nCollected {index - args.data_idx}/{args.data_set_size} episodes")
    print(f"Data directory: {save_root}")


if __name__ == "__main__":
    main()
