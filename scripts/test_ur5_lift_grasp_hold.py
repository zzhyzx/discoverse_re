"""Simulation-only regression checks for the old evaluator's opt-in latch."""
from types import SimpleNamespace
import unittest
from ur5_sim_grasp_hold import GraspHold, BilateralPadContact


class Tests(unittest.TestCase):
    def confirm(self):
        latch = GraspHold()
        for step in range(4):
            self.assertEqual(latch.command(1., .5, True, step / 30), 255.)
        self.assertTrue(latch.confirmed)
        return latch

    def test_confirmed_grasp_ignores_open_predictions(self):
        latch = self.confirm()
        for step in range(4, 40):
            self.assertEqual(latch.command(0., .5, True, step / 30), 255.)
        self.assertEqual(latch.suppressed_frames, 36)

    def test_empty_close_can_still_reopen(self):
        latch = GraspHold()
        for step in range(10):
            latch.command(1., 1., False, step / 30)
        self.assertFalse(latch.confirmed)
        self.assertEqual(latch.command(0., 1., False, 1.), 0.)

    def test_contact_without_close_does_not_latch(self):
        latch = GraspHold()
        for step in range(10):
            self.assertEqual(latch.command(0., .5, True, step / 30), 0.)
        self.assertFalse(latch.confirmed)

    def test_moving_fingers_do_not_confirm(self):
        latch = GraspHold()
        for step in range(10):
            latch.command(1., step * .01, True, step / 30)
        self.assertFalse(latch.confirmed)

    def test_contact_gap_resets_confirmation(self):
        latch = GraspHold()
        for step, contact in enumerate([True, True, True, False, True, True]):
            latch.command(1., .5, contact, step / 30)
        self.assertFalse(latch.confirmed)
        latch.command(1., .5, True, 6 / 30)
        self.assertTrue(latch.confirmed)

    def test_contact_loss_does_not_reopen_or_fake_success(self):
        latch = self.confirm()
        self.assertEqual(latch.command(0., .9, False, 1.), 255.)
        self.assertTrue(latch.confirmed)  # Historical confirmation, not current success.

    def test_new_episode_has_no_latch(self):
        self.confirm()
        latch = GraspHold()
        self.assertEqual(latch.command(0., .5, True, 0.), 0.)
        self.assertFalse(latch.confirmed)

    def test_only_both_pad_contacts_with_cube_count(self):
        detector = BilateralPadContact.__new__(BilateralPadContact)
        detector.cube = 3
        detector.pad_geoms = {10: 0, 11: 1}
        contact = lambda a, b, d=0.: SimpleNamespace(geom1=a, geom2=b, dist=d)
        for contacts, expected in [
            ([contact(3, 10)], False),
            ([contact(3, 10), contact(11, 3)], True),
            ([contact(3, 10), contact(11, 8)], False),
            ([contact(3, 10), contact(11, 3, .01)], False),
        ]:
            self.assertEqual(detector(SimpleNamespace(contact=contacts, ncon=len(contacts))), expected)


if __name__ == '__main__':
    unittest.main()
