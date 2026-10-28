"""Simulation-only close-event buffer; no hardware interfaces or object-pose oracle."""
from collections import deque
import math
import numpy as np


class SimCloseIntent:
    """Confirm distinct action samples, then retain their joint goal until dispatch.

    Evidence ages in simulation time. Repeated reads of a paused action do not
    count as new evidence. The caller still enforces limits, alignment, dwell,
    contact confirmation and collision checks before/after dispatch.
    """
    def __init__(self, threshold=.65, votes=2, window_s=.2,
                 goal_tolerance_deg=1., timeout_s=2.):
        self.threshold=threshold;self.votes=votes;self.window_s=window_s
        self.goal_tolerance=math.radians(goal_tolerance_deg);self.timeout_s=timeout_s
        self.history=deque();self.last_key=None
        self.goal=None;self.confirmed_at=None;self.deadline=None
        self.events=[]

    @property
    def pending(self):
        return self.goal is not None

    def configuration(self):
        return dict(close_threshold=self.threshold,distinct_action_votes=self.votes,
                    evidence_window_sim_s=self.window_s,
                    target_consistency_deg=math.degrees(self.goal_tolerance),
                    pending_timeout_sim_s=self.timeout_s,
                    uses_object_pose=False)

    def _clear(self):
        self.goal=None;self.confirmed_at=None;self.deadline=None;self.history.clear()

    def select(self, desired, now, action_key):
        desired=np.asarray(desired,dtype=float).copy()
        if desired.shape!=(7,) or not np.isfinite(desired).all() or not math.isfinite(now):
            raise ValueError('Invalid simulated close-intent input')
        if action_key is None:
            raise ValueError('Close-intent buffering requires a distinct action key')
        key=tuple(action_key)
        expired=False
        if self.pending and now>self.deadline+1e-9:
            self.events.append(dict(event='expired',time_s=float(now),goal_q_rad=self.goal.tolist()))
            self._clear();self.last_key=key;expired=True
        if not self.pending and not expired:
            while self.history and now-self.history[0][0]>self.window_s+1e-9:
                self.history.popleft()
            if key!=self.last_key:
                self.last_key=key
                if desired[6]>=self.threshold:
                    # Evidence must refer to essentially the same grasp location.
                    self.history=deque(h for h in self.history
                        if np.max(abs(h[1]-desired[:6]))<=self.goal_tolerance)
                    self.history.append((now,desired[:6].copy(),float(desired[6]),key))
                    self.events.append(dict(event='evidence',time_s=float(now),
                        action_key=list(key),raw_gripper=float(desired[6])))
                    if len(self.history)>=self.votes:
                        self.goal=desired[:6].copy();self.confirmed_at=now
                        self.deadline=now+self.timeout_s
                        self.events.append(dict(event='confirmed',time_s=float(now),
                            goal_q_rad=self.goal.tolist(),deadline_s=float(self.deadline),
                            evidence=[dict(time_s=float(h[0]),raw_gripper=h[2],action_key=list(h[3]))
                                      for h in self.history]))
                        self.history.clear()
        if self.pending:
            return np.r_[self.goal,1.]
        # In this opt-in mode only confirmed evidence may initiate closing.
        desired[6]=0.
        return desired

    def dispatched(self, now):
        if not self.pending:
            raise RuntimeError('Closing without a pending simulated close event')
        self.events.append(dict(event='dispatched',time_s=float(now),goal_q_rad=self.goal.tolist()))
        self._clear()

    def drain_events(self):
        events=self.events;self.events=[]
        return events
