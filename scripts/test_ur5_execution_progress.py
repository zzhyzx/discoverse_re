"""Offline tests: no RTDE, cameras, sockets or model imports."""
import ast
import math
from pathlib import Path
import threading
import types
import unittest
import numpy as np
from ur5_control_profile import guarded_command
from ur5_deploy_core import (ExecutionProgress, GripperDecision, Limiter, Plan,
                             interpolate, remaining_plan, validate_actions, takeover_reason)

class ProgressTests(unittest.TestCase):
    def test_lag_and_pause_do_not_advance(self):
        p = ExecutionProgress(30, 10, .01)
        z = np.zeros(6)
        p.advance(z+.1, z, z, .008)
        p.advance(z, z, z, .008, paused=True)
        self.assertEqual(p.steps, 0)
        for _ in range(42): p.advance(z, z, z, .008)
        self.assertTrue(p.can_replace(False, False))
        self.assertFalse(p.can_replace(True, False))
        self.assertFalse(p.can_replace(False, True))


    def test_per_axis_limiter_bounds_and_reversal(self):
        speed=np.array([.15,.15,.2,.2,.03,.15])
        acceleration=np.array([.17,.15,.21,.21,.05,.17])
        limiter=Limiter(np.zeros(6),speed,acceleration)
        previous_q=limiter.q.copy()
        previous_v=limiter.velocity.copy()
        for n in range(600):
            q=limiter.step(np.ones(6) if n<300 else -np.ones(6),1/30)
            self.assertTrue(np.all(abs(q-previous_q)*30 <= speed+1e-10))
            self.assertTrue(np.all(abs(limiter.velocity-previous_v)*30 <= acceleration+1e-10))
            previous_q=q.copy()
            previous_v=limiter.velocity.copy()

    def test_early_takeover_without_minimum_steps(self):
        p=ExecutionProgress(30,10,.01)
        p.elapsed=2/30
        self.assertEqual(takeover_reason(p,.6,0.,0.,.1,1.65,False,False,False,True),'stalled_progress')
        self.assertEqual(takeover_reason(p,1.3,0.,0.,1.2,1.65,False,False,True,True),'expiry_margin')
        self.assertIsNone(takeover_reason(p,.6,0.,0.,.1,1.65,False,True,False,True))
        self.assertIsNone(takeover_reason(p,.6,0.,0.,.1,1.65,False,False,False,False))
        self.assertIsNone(takeover_reason(p,.2,0.,0.,-.5,1.65,False,False,False,True))

    def run_loop(self, expiry=2, fault=False, blocked=False):
        # Execute the actual production loop with a deterministic in-memory robot.
        path = Path(__file__).resolve().parents[1]/'policies/smolvla/deploy_real_ur5.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'run_servo')
        clock = types.SimpleNamespace(t=0.)
        events, requests = [], []
        shared = types.SimpleNamespace(lock=threading.Lock(), stop=threading.Event(),
                                       gripper_request=None, gripper_busy=False)
        q = np.zeros(6)
        def sample():
            now = clock.t
            if shared.gripper_request is not None:
                requests.append(now)
                shared.gripper_request = None
                clock.busy_until = now+.064
            shared.gripper_busy = now < getattr(clock, 'busy_until', -1)
            observed = now if expiry > .1 else 0.
            shared.plan = Plan(int(now/.08)+1, observed, now,
                               np.c_[np.full((30,6),.04 if blocked else 0.), np.zeros(30) if blocked else np.ones(30)],
                               np.zeros(7), {})
            shared.camera = ({'local_observed_monotonic': now},)
            shared.gripper = (now, {'POS': 0})
            return q.copy(), np.zeros(6), now
        def check():
            if fault and clock.t > .1: raise RuntimeError('injected worker failure')
        shared.check = check
        sample()
        receive = types.SimpleNamespace(getRuntimeState=lambda:2, getRobotStatus=lambda:2,
                                        getSpeedScaling=lambda:1., getTargetSpeedFraction=lambda:1.)
        feedback = types.SimpleNamespace(sample=sample, receive=receive)
        stopped=[]
        def servo(command, *args):
            q[:] = command
            return True
        control = types.SimpleNamespace(
            initPeriod=lambda:None, isConnected=lambda:True, kickWatchdog=lambda:True,
            isJointsWithinSafetyLimits=lambda q:True, servoJ=servo,
            waitPeriod=lambda tick:setattr(clock,'t',clock.t+.008),
            servoStop=lambda a:stopped.append(True) or True, stopJ=lambda a:None)
        recorder = types.SimpleNamespace(event=lambda name,**values:events.append((clock.t,name,values)))
        cfg = dict(max_command_speed_rad_s=.03, max_command_acceleration_rad_s2=.05,
                   control_hz=125, action_hz=30, minimum_execution_steps=10,
                   progress_tolerance_deg=.5, duration_seconds=2. if blocked else .9, max_control_gap_s=.08,
                   max_camera_age_s=.25, max_gripper_status_age_s=.5,
                   next_chunk_grace_s=.05, max_observation_age_s=expiry,
                   max_observation_joint_drift_deg=3, execution_steps=30,
                   max_prediction_delta_deg=12, max_tracking_error_deg=3,
                   allow_release=False, lookahead_s=.1, servo_gain=200)
        ns=dict(globals(), time=types.SimpleNamespace(monotonic=lambda:clock.t))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        error=None
        try: ns['run_servo'](control,feedback,shared,recorder,cfg,-np.ones(6),np.ones(6),True)
        except RuntimeError as exc:error=str(exc)
        return events,requests,stopped,error

    def test_new_predictions_do_not_preempt_and_gripper_is_executed(self):
        events, requests, stopped, error=self.run_loop()
        self.assertIsNone(error)
        transitions=[(t,v) for t,n,v in events if n=='chunk_transition']
        self.assertGreater(len(transitions),1)
        self.assertGreaterEqual(transitions[1][1]['previous_executed_steps'],10)
        self.assertGreater(transitions[1][0],.5)
        self.assertEqual(len(requests),1)
        self.assertTrue(stopped)
        paused=[v['executed_steps'] for t,n,v in events if n=='servo' and v['gripper_busy']]
        self.assertTrue(paused)
        self.assertEqual(max(paused),min(paused))

    def test_production_loop_early_takeover_when_blocked(self):
        events,requests,stopped,error=self.run_loop(expiry=1.65,blocked=True)
        self.assertIsNone(error)
        early=[v for _,name,v in events if name=='chunk_transition' and v['reason']=='stalled_progress']
        self.assertTrue(early)
        self.assertTrue(any(v['previous_executed_steps']<10 for v in early))
        self.assertTrue(stopped)
        self.assertEqual(requests,[])
        commands=np.asarray([v['executed_q'] for _,name,v in events if name=='servo'])
        velocity=np.diff(np.vstack([np.zeros(6),commands]),axis=0)/.008
        acceleration=np.diff(np.vstack([np.zeros(6),velocity]),axis=0)/.008
        self.assertLessEqual(np.max(abs(velocity)),.03+1e-9)
        self.assertLessEqual(np.max(abs(acceleration)),.05+1e-9)

    def test_expired_prediction_still_stops(self):
        _,requests,stopped,error=self.run_loop(expiry=.1)
        self.assertEqual(error,'Active prediction expired')
        self.assertEqual(requests,[])
        self.assertTrue(stopped)

    def test_worker_failure_still_stops(self):
        _,_,stopped,error=self.run_loop(fault=True)
        self.assertEqual(error,'injected worker failure')
        self.assertTrue(stopped)

if __name__=='__main__': unittest.main()
