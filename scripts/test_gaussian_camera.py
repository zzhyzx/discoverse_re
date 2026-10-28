"""GPU regressions for near clipping, camera projection and render caching.

Run in the discoverse environment after rebuilding the local CUDA extension:
    MUJOCO_GL=egl python scripts/test_gaussian_camera.py
No robot, external assets or dataset writes are needed.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer

from discoverse.gaussian_renderer.gsRenderer import GSRenderer
from discoverse.gaussian_renderer.renderer_cuda import CUDARenderer
from discoverse.gaussian_renderer.util_gau import Camera, GaussianData


def synthetic_renderer(distance=0.3, x=0.04):
    # Exercise the real renderer without loading the large scene assets.
    gs = GSRenderer.__new__(GSRenderer)
    gs.camera = Camera(80, 120)
    gs.renderer = CUDARenderer(120, 80)
    gs.scale_modifier = 1.0
    gs.camera_tran = np.full(3, np.nan)
    gs.camera_quat = np.full(4, np.nan)
    data = GaussianData(
        np.array([[x, 0.0, -distance]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        np.full((1, 3), 0.003, dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        (np.array([[1.0, 0.0, 0.0]], dtype=np.float32) - 0.5) / 0.28209479177387814,
    )
    gs.update_activated_renderer_state({"background": data})
    gs.set_camera_pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    return gs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GaussianCameraTests(unittest.TestCase):
    def assert_projection_consistent(self, gs):
        settings = gs.renderer.raster_settings
        view = settings.viewmatrix.cpu().numpy().T
        expected = (gs.camera.get_project_matrix() @ view).T
        np.testing.assert_allclose(settings.projmatrix.cpu().numpy(), expected, atol=1e-6)
        self.assertEqual(settings.znear, gs.camera.znear)

    def test_near_plane_and_mark_visible_agree(self):
        for distance, visible in ((-0.1, False), (0.005, False), (0.01, False),
                                  (0.02, True), (0.05, True), (0.10, True),
                                  (0.19, True), (0.21, True), (0.30, True)):
            with self.subTest(distance=distance):
                gs = synthetic_renderer(distance, x=0.0)
                image = gs.render()
                marked = GaussianRasterizer(gs.renderer.raster_settings).markVisible(
                    gs.renderer.gaussians.xyz
                ).item()
                self.assertEqual(marked, visible)
                self.assertEqual(bool(image.max() > 0), visible)
                self.assertTrue(np.isfinite(gs.render(render_depth=True)).all())

    def test_runtime_near_plane_change_invalidates_cache(self):
        gs = synthetic_renderer(0.1, x=0.0)
        self.assertGreater(gs.render().max(), 0)
        gs.set_camera_clip_planes(0.2)
        self.assertEqual(gs.render().max(), 0)
        gs.set_camera_clip_planes(0.01)
        self.assertGreater(gs.render().max(), 0)
        self.assert_projection_consistent(gs)

    def test_fixed_pose_fov_refreshes_projection_and_pixels(self):
        gs = synthetic_renderer()
        old = gs.render().copy()
        projection = gs.renderer.raster_settings.projmatrix.clone()
        gs.set_camera_fovy(0.65)
        new = gs.render()
        self.assertFalse(torch.equal(projection, gs.renderer.raster_settings.projmatrix))
        self.assertFalse(np.array_equal(old, new))
        self.assert_projection_consistent(gs)

    def test_fixed_pose_resize_refreshes_projection_rgb_and_depth(self):
        gs = synthetic_renderer()
        old = gs.render()
        projection = gs.renderer.raster_settings.projmatrix.clone()
        gs.set_camera_resolution(96, 192)
        image = gs.render()
        self.assertEqual(image.shape, (96, 192, 3))
        self.assertEqual(gs.render(render_depth=True).shape, (96, 192, 1))
        self.assertIsNot(old, image)
        self.assertFalse(torch.equal(projection, gs.renderer.raster_settings.projmatrix))
        self.assert_projection_consistent(gs)

    def test_unchanged_camera_reuses_rgb_and_depth(self):
        gs = synthetic_renderer()
        image = gs.render()
        depth = gs.render(render_depth=True)
        gs.set_camera_pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
        gs.set_camera_fovy(gs.camera.fovy)
        gs.set_camera_resolution(gs.camera.h, gs.camera.w)
        self.assertIs(gs.render(), image)
        self.assertIs(gs.render(render_depth=True), depth)

    def test_two_camera_roundtrip(self):
        gs = synthetic_renderer()
        original = gs.render().copy()
        fovy = gs.camera.fovy
        gs.set_camera_fovy(0.8)
        gs.set_camera_pose(np.array([0.02, 0.0, 0.0]), np.array([0., 0., 0., 1.]))
        self.assertFalse(np.array_equal(original, gs.render()))
        gs.set_camera_fovy(fovy)
        gs.set_camera_pose(np.zeros(3), np.array([0., 0., 0., 1.]))
        np.testing.assert_array_equal(gs.render(), original)

    def test_failed_render_keeps_cache_dirty(self):
        gs = synthetic_renderer()
        gs.render()
        gs.set_camera_fovy(0.8)
        with patch("discoverse.gaussian_renderer.renderer_cuda.GausssianRasterizer_3d",
                   side_effect=RuntimeError("test render failure")):
            with self.assertRaisesRegex(RuntimeError, "test render failure"):
                gs.render()
        self.assertTrue(gs.renderer.need_rerender)
        self.assertGreater(gs.render().max(), 0)
        self.assertFalse(gs.renderer.need_rerender)

    def test_simulator_same_camera_fov_and_free_camera_depth(self):
        from discoverse.envs.simulator import SimulatorBase
        env = SimulatorBase.__new__(SimulatorBase)
        env.config = SimpleNamespace(use_gaussian_renderer=True)
        env.show_gaussian_img = True
        env.gs_renderer = synthetic_renderer()
        env.last_cam_id = 0
        env.mj_model = SimpleNamespace(cam_fovy=np.array([60.0]),
            vis=SimpleNamespace(global_=SimpleNamespace(fovy=55.0)))
        env.getCameraPose = lambda _: (np.zeros(3), np.array([1., 0., 0., 0.]))
        original = env.getRgbImg(0).copy()
        env.mj_model.cam_fovy[0] = 35.0
        self.assertFalse(np.array_equal(original, env.getRgbImg(0)))
        env.renderer = SimpleNamespace(update_scene=lambda *args: None)
        env.mj_data = env.free_camera = env.options = None
        env.getRgbImg(-1)
        projection = env.gs_renderer.renderer.raster_settings.projmatrix.clone()
        env.last_cam_id = 0  # Force the formerly inconsistent free-depth FOV path.
        env.getDepthImg(-1)
        self.assertAlmostEqual(env.gs_renderer.camera.fovy, np.deg2rad(55))
        self.assertTrue(torch.equal(projection, env.gs_renderer.renderer.raster_settings.projmatrix))

    def test_invalid_camera_parameters(self):
        camera = Camera(80, 120)
        for near in (0, -1, float("nan"), float("inf"), 100):
            with self.subTest(near=near), self.assertRaises(ValueError):
                camera.update_clip_planes(near)
        for fov in (0, -1, np.pi, float("nan")):
            with self.subTest(fov=fov), self.assertRaises(ValueError):
                camera.update_fovy(fov)
        with self.assertRaises(ValueError):
            camera.update_resolution(0, 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
