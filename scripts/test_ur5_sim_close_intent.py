"""CPU-only regressions for remembered simulation close events. No robot I/O."""
import math
from types import SimpleNamespace
import unittest
import numpy as np
from ur5_sim_close_intent import SimCloseIntent
from ur5_control_profile import SimCommandAdapter,load_profile
from ur5_deploy_core import Limiter,GripperDecision


def action(q=0.,grip=.8):
    return np.r_[np.full(6,q),grip]


class IntentTests(unittest.TestCase):
    def test_one_spike_and_repeated_same_action_do_not_confirm(self):
        c=SimCloseIntent()
        for now in [0.,.03,.1,.19]:
            self.assertEqual(c.select(action(),now,(1,0))[6],0.)
            self.assertFalse(c.pending)
        c.select(action(grip=0),.3,(1,1))
        c.select(action(),.34,(1,2))
        self.assertFalse(c.pending)

    def test_two_consistent_votes_survive_later_open_prediction(self):
        c=SimCloseIntent()
        c.select(action(.02),0.,(1,45))
        out=c.select(action(.022),.067,(1,47))
        self.assertTrue(c.pending)
        np.testing.assert_allclose(out,action(.022,1.))
        out=c.select(action(-.1,0.),.12,(1,48))
        np.testing.assert_allclose(out,action(.022,1.))
        self.assertEqual(len([e for e in c.events if e['event']=='confirmed']),1)

    def test_different_target_or_old_evidence_cannot_confirm(self):
        c=SimCloseIntent()
        c.select(action(),0.,(1,0))
        c.select(action(.04),.03,(1,1))
        self.assertFalse(c.pending)
        c.select(action(.04),.3,(1,2))
        self.assertFalse(c.pending)

    def test_pending_expiry_drops_request_without_rearming_same_action(self):
        c=SimCloseIntent()
        c.select(action(),0.,(1,0));c.select(action(),.04,(1,1))
        out=c.select(action(),2.1,(1,1))
        self.assertFalse(c.pending);self.assertEqual(out[6],0.)
        c.select(action(),2.2,(1,1));self.assertFalse(c.pending)
        self.assertTrue(any(e['event']=='expired' for e in c.events))

    def test_nonfinite_rejected(self):
        c=SimCloseIntent()
        with self.assertRaises(ValueError):c.select(action(grip=float('nan')),0,(1,0))


class AdapterTests(unittest.TestCase):
    def adapter(self,q=0.,enabled=True):
        a=SimCommandAdapter.__new__(SimCommandAdapter)
        a.state=np.r_[np.full(6,q),0.]
        a.env=SimpleNamespace(normalized_state=lambda:a.state.copy())
        a.profile=load_profile();a.hold_after_grasp=True
        a.close_intent=SimCloseIntent() if enabled else None
        a.grasp_confirmed=False;a.release_suppressed=False
        a.busy=False;a.started=0.;a.closed=False;a.stable=0;a.previous=0.
        a.limiter=Limiter(a.state[:6],.06,.10)
        a.decision=GripperDecision(False,.2);a.has_contact=lambda:True
        return a

    def test_unaligned_close_waits_moves_to_goal_then_dispatches(self):
        a=self.adapter();dt=1/30
        a.step(action(.025),0,dt,action_key=(1,0))
        a.step(action(.026),dt,dt,action_key=(1,1))
        self.assertTrue(a.pending_close);self.assertFalse(a.closed)
        for step in range(2,61):
            previous=a.limiter.q.copy()
            cmd,aligned=a.step(action(-.2,0.),step*dt,dt,action_key=(1,2))
            self.assertLessEqual(float(abs(cmd[:6]-previous).max()),.06*dt+1e-9)
            if a.closed:
                self.assertTrue(aligned)
                self.assertLessEqual(float(abs(a.state[:6]-.026).max()),math.radians(.5))
                self.assertTrue(a.busy);self.assertFalse(a.pending_close)
                self.assertTrue(any(e['event']=='dispatched' for e in a.intent_events))
                break
            a.state[:6]=cmd[:6]
        else:self.fail('remembered close was not dispatched')

    def test_unreachable_goal_expires_without_closure(self):
        a=self.adapter();dt=1/30
        a.step(action(.12),0,dt,action_key=(1,0))
        a.step(action(.12),dt,dt,action_key=(1,1))
        for i in range(2,64):a.step(action(0,0),i*dt,dt,action_key=(1,2))
        self.assertFalse(a.closed);self.assertFalse(a.pending_close)
        self.assertIsNone(a.decision.candidate)

    def test_aligned_single_spike_does_not_close_in_buffer_mode(self):
        a=self.adapter()
        for i in range(20):a.step(action(),i/30,1/30,action_key=(1,0))
        self.assertFalse(a.closed);self.assertFalse(a.pending_close)

    def test_legacy_default_still_closes_after_aligned_dwell(self):
        a=self.adapter(enabled=False)
        for i in range(8):a.step(action(),i/30,1/30)
        self.assertTrue(a.closed)


if __name__=='__main__':unittest.main()
