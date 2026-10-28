"""Shared low-speed command contract. No hardware imports."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from ur5_deploy_core import Limiter, GripperDecision

DEFAULT=Path(__file__).resolve().parents[1]/'configs/control/ur5_low_speed_v1.json'
def load_profile(path=DEFAULT):
    p=json.loads(Path(path).read_text())
    if (p['id']!='ur5_low_speed_v1' or p['max_command_speed_rad_s']!=.03
        or p['max_command_acceleration_rad_s2']!=.05 or p['action_hz']!=30
        or p['minimum_execution_steps']!=10 or p['execution_steps']!=30
        or p['progress_tolerance_deg']!=.5 or p['gripper_dwell_s']!=.2
        or p['control_hz']!=125 or p['episode_timeout_s']!=180
        or p['close_threshold']!=.65 or p['open_threshold']!=.35 or p['allow_release'] is not False):
        raise ValueError('Changed v1 control contract; create a new version explicitly')
    canonical=json.dumps(p,sort_keys=True,separators=(',',':')).encode()
    return dict(p,sha256=hashlib.sha256(canonical).hexdigest())

def guarded_command(limiter,decision,desired,actual,now,dt,tolerance,busy=False,enabled=True,waiting=False):
    """Common arm/gripper ordering; caller handles device feedback and stopping."""
    aligned=max(np.max(abs(actual-desired[:6])),np.max(abs(limiter.q-desired[:6])))<=tolerance
    request=None
    if enabled and not waiting and not busy and aligned:
        request=decision.update(float(desired[6]),now)
    elif enabled and not busy:
        decision.candidate=decision.since=None
    paused=busy or waiting or request is not None
    target=actual if paused else desired[:6]
    command=limiter.step(target,dt)
    return command,request,aligned

class SimCommandAdapter:
    """Same command kernel; physics feedback adapter is not a UR controller."""
    def __init__(self,env,profile,hold_after_grasp=False,cache_close_intent=False):
        self.env=env;self.profile=profile
        from ur5_sim_close_intent import SimCloseIntent
        self.close_intent=SimCloseIntent() if cache_close_intent else None
        self.effective_desired=None;self.intent_events=[]
        self.hold_after_grasp=hold_after_grasp;self.grasp_confirmed=False;self.release_suppressed=False
        self.limiter=Limiter(env.normalized_state()[:6],profile['max_command_speed_rad_s'],
                             profile['max_command_acceleration_rad_s2'])
        self.decision=GripperDecision(False,profile['gripper_dwell_s'])
        self.busy=False;self.started=0.;self.closed=False;self.stable=0
        self.previous=float(env.normalized_state()[6])
        import mujoco
        self.pads={mujoco.mj_name2id(env.mj_model,mujoco.mjtObj.mjOBJ_BODY,'gripper_'+side+'_pad') for side in ('left','right')}
        self.cube=mujoco.mj_name2id(env.mj_model,mujoco.mjtObj.mjOBJ_GEOM,'red_block_geom')
        if -1 in self.pads or self.cube<0:raise ValueError('Missing contact geometry')

    def has_contact(self):
        touched=set()
        for c in self.env.mj_data.contact[:self.env.mj_data.ncon]:
            if c.dist>0.001 or self.cube not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==self.cube else c.geom1
            body=int(self.env.mj_model.geom_bodyid[other])
            while body:
                if body in self.pads:touched.add(body)
                body=int(self.env.mj_model.body_parentid[body])
        return touched==self.pads

    @property
    def pending_close(self):
        intent=getattr(self,'close_intent',None)
        return intent is not None and intent.pending

    def step(self,desired,now,dt,action_key=None):
        feedback=float(self.env.normalized_state()[6])
        if self.busy:
            # MuJoCo has no URCap OBJ register: require closed/open motion settled.
            settled=abs(feedback-self.previous)<.002 and (self.has_contact() if self.closed else feedback<.05)
            self.stable=self.stable+1 if settled else 0
            if self.stable>=3 and now-self.started>=.2:
                self.busy=False
                if self.closed:self.grasp_confirmed=True
            if self.busy and now-self.started>8:raise RuntimeError('Simulated gripper feedback timeout')
        self.previous=feedback
        desired=np.asarray(desired,dtype=float).copy()
        intent=getattr(self,'close_intent',None)
        self.intent_events=[]
        if intent is not None and not self.closed and not self.busy:
            was_pending=intent.pending
            desired=intent.select(desired,now,action_key)
            if was_pending and not intent.pending:
                self.decision.candidate=self.decision.since=None
        self.release_suppressed=bool(self.hold_after_grasp and self.grasp_confirmed and desired[6]<.35)
        if self.hold_after_grasp and self.grasp_confirmed:
            desired[6]=1.
            self.decision.candidate=self.decision.since=None
        self.effective_desired=desired.copy()
        q,request,aligned=guarded_command(self.limiter,self.decision,np.asarray(desired),
            self.env.normalized_state()[:6],now,dt,math.radians(self.profile['progress_tolerance_deg']),self.busy)
        if request is not None:
            if not request and not self.profile['allow_release']:raise RuntimeError('Release disabled')
            if request and intent is not None:
                intent.dispatched(now)
            self.closed=request;self.busy=True;self.started=now;self.stable=0
        if intent is not None:self.intent_events=intent.drain_events()
        return np.r_[q,255. if self.closed else 0.],aligned
