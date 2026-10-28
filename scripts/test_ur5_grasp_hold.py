"""No hardware: test simulation grasp-confirmation latch."""
import unittest
from types import SimpleNamespace
import numpy as np
from ur5_control_profile import SimCommandAdapter,load_profile
from ur5_deploy_core import Limiter,GripperDecision
class Tests(unittest.TestCase):
 def adapter(self,enabled=True,contact=True):
  a=SimCommandAdapter.__new__(SimCommandAdapter)
  a.env=SimpleNamespace(normalized_state=lambda:np.r_[np.zeros(6),.5])
  a.profile=load_profile();a.hold_after_grasp=enabled;a.grasp_confirmed=False;a.release_suppressed=False
  a.busy=True;a.started=0.;a.closed=True;a.stable=0;a.previous=.5
  a.limiter=Limiter(np.zeros(6),.06,.10);a.decision=GripperDecision(True,.2)
  a.has_contact=lambda:contact
  return a
 def test_contact_then_open_prediction_stays_closed(self):
  a=self.adapter()
  for now in [.3,.34,.38]:a.step(np.r_[np.zeros(6),0.],now,1/30)
  self.assertTrue(a.grasp_confirmed);self.assertFalse(a.busy)
  for now in [.5,.8,1.]:
   command,_=a.step(np.r_[np.zeros(6),0.],now,1/30)
   self.assertEqual(command[6],255);self.assertTrue(a.release_suppressed)
  self.assertIsNone(a.decision.candidate)
 def test_no_contact_does_not_confirm(self):
  a=self.adapter(contact=False)
  for now in [.3,.4,.5]:a.step(np.r_[np.zeros(6),0.],now,1/30)
  self.assertFalse(a.grasp_confirmed)
  with self.assertRaisesRegex(RuntimeError,'feedback timeout'):
   a.step(np.r_[np.zeros(6),0.],9.,1/30)
 def test_default_release_error_preserved(self):
  a=self.adapter(enabled=False)
  for now in [.3,.34,.38]:a.step(np.r_[np.zeros(6),0.],now,1/30)
  with self.assertRaisesRegex(RuntimeError,'Release disabled'):
   a.step(np.r_[np.zeros(6),0.],.8,1/30)
if __name__=='__main__':unittest.main()
