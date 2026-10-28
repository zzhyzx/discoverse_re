"""Offline camera timing regressions; no cameras or robot connections."""
import unittest
from unittest.mock import Mock, patch
import warnings
import ur5_camera_wire as wire

class TimingTests(unittest.TestCase):
    def request(self, age=.02, skew=.01, waited=.03, elapsed=.05, previous=None):
        client = wire.CameraClient.__new__(wire.CameraClient)
        client.sock = Mock()
        client.token = 'offline-test'
        client.previous = previous or {}
        meta = dict(max_receipt_age_s=age, receipt_skew_s=skew,
                    server_wait_seconds=waited,
                    cameras={role:dict(frame_number=2) for role in wire.ROLES})
        with patch.object(wire,'read_packet',return_value=(meta,{})), patch.object(
                wire.time,'monotonic',side_effect=[100.,100.+elapsed]):
            return client.get(.25,.08)

    def test_normal_uses_full_round_trip(self):
        meta,_=self.request()
        self.assertAlmostEqual(meta['conservative_receipt_age_s'],.07)
        self.assertAlmostEqual(meta['local_observed_monotonic'],99.98)

    def test_cross_clock_disagreement_remains_conservative(self):
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            meta,_=self.request(waited=.06)
        self.assertTrue(messages)
        self.assertAlmostEqual(meta['conservative_receipt_age_s'],.07)
        self.assertAlmostEqual(meta['local_observed_monotonic'],99.98)

    def test_server_wait_cannot_hide_stale_image(self):
        with self.assertRaises(wire.CameraTimingError):
            self.request(age=.20,elapsed=.10,waited=.09)

    def test_skew_rejected(self):
        with self.assertRaises(wire.CameraTimingError): self.request(skew=.09)

    def test_invalid_values_rejected(self):
        for key in ('age','skew','waited','elapsed'):
            for value in (float('nan'),float('inf'),-.01):
                with self.subTest(key=key,value=value), self.assertRaises(wire.CameraTimingError):
                    self.request(**{key:value})

    def test_repeated_frames_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'did not advance'):
            self.request(previous={role:2 for role in wire.ROLES})

if __name__=='__main__': unittest.main()
