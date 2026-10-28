"""Pure deployment guards and time-based setpoint generation. No hardware imports."""
from dataclasses import dataclass
import math
import numpy as np

def six(values):
    result = np.asarray(values, dtype=float)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError('Expected six finite joint radians')
    return result

@dataclass
class Plan:
    sequence: int
    observed: float
    completed: float
    actions: np.ndarray
    state: np.ndarray
    camera_numbers: dict

def remaining_plan(plan, now, fps, maximum):
    """Start a new prediction at its prefix; latency is not executed motion.

    The caller enforces observation freshness and state drift before adoption.
    A sequence may be adopted only once; do not repeat a finished prefix.
    """
    age = now-plan.observed
    if not math.isfinite(age) or age < 0:
        raise RuntimeError('Invalid prediction observation time')
    skipped = 0
    actions = plan.actions[:maximum].copy()
    if not len(actions):
        raise RuntimeError('Prediction horizon exhausted')
    result = Plan(plan.sequence, plan.observed, plan.completed, actions,
                  plan.state, plan.camera_numbers)
    return result, skipped, 0.0

class TargetDeltaError(RuntimeError):
    pass

def validate_actions(actions, actual, lower, upper, max_delta):
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 1 or not np.isfinite(actions).all():
        raise ValueError('Expected finite N x 7 action chunk')
    joints = actions[:, :6]
    if np.any(joints < six(lower)) or np.any(joints > six(upper)):
        raise RuntimeError('Predicted absolute joints outside training/controller envelope; no angle wrapping')
    if np.any(actions[:, 6] < -.25) or np.any(actions[:, 6] > 1.25):
        raise RuntimeError('Unexpected gripper action range')
    delta = np.abs(joints-six(actual))
    step, joint = np.unravel_index(np.argmax(delta), delta.shape)
    if delta[step,joint] > max_delta:
        raise TargetDeltaError('Predicted target too far from current actual joints: '
            f'step={step+1}, joint={joint+1}, actual={math.degrees(actual[joint]):.3f}deg, '
            f'target={math.degrees(joints[step,joint]):.3f}deg, '
            f'delta={math.degrees(delta[step,joint]):.3f}deg, limit={math.degrees(max_delta):.3f}deg')

class Limiter:
    def __init__(self, q, speed, acceleration):
        self.q = six(q).copy()
        self.velocity = np.zeros(6)
        self.speed, self.acceleration = np.asarray(speed, dtype=float), np.asarray(acceleration, dtype=float)
        if (self.speed.shape not in ((), (6,)) or self.acceleration.shape not in ((), (6,))
                or not np.isfinite(self.speed).all() or not np.isfinite(self.acceleration).all()
                or np.any(self.speed <= 0) or np.any(self.speed > .2)
                or np.any(self.acceleration <= 0) or np.any(self.acceleration > .5)):
            raise ValueError('Deployment speed/acceleration outside bounded configuration')

    def step(self, target, dt):
        if not 0 < dt <= .05:
            raise RuntimeError('Invalid setpoint interval')
        distance = six(target)-self.q
        braking_speed = np.sqrt(2*self.acceleration*np.abs(distance))
        desired = np.sign(distance)*np.minimum(self.speed, braking_speed)
        velocity = np.clip(desired, self.velocity-self.acceleration*dt,
                           self.velocity+self.acceleration*dt)
        velocity = np.clip(velocity, -self.speed, self.speed)
        self.q = self.q + velocity*dt
        self.velocity = velocity
        return self.q.copy()

class GripperDecision:
    def __init__(self, initially_closed, dwell=.2):
        self.closed = initially_closed
        self.dwell = dwell
        self.candidate = None
        self.since = None

    def update(self, raw, now):
        if not math.isfinite(raw):
            raise ValueError('Nonfinite gripper prediction')
        wanted = True if raw >= .65 else False if raw <= .35 else self.closed
        if wanted == self.closed:
            self.candidate = self.since = None
            return None
        if wanted != self.candidate:
            self.candidate, self.since = wanted, now
            return None
        if now-self.since >= self.dwell:
            self.closed = wanted
            self.candidate = self.since = None
            return wanted
        return None

def interpolate(actions, initial, elapsed, fps=30):
    """Reach action[0] after 1/fps; then piecewise linear absolute targets."""
    position = elapsed*fps
    index = int(position)
    if index >= len(actions):
        return np.asarray(actions[-1])
    previous = np.r_[six(initial), actions[0, 6]] if index == 0 else actions[index-1]
    return previous + (actions[index]-previous)*(position-index)


class ExecutionProgress:
    """Feedback-gated action time; observation expiry remains wall-clock based."""
    def __init__(self, fps, minimum_steps, tolerance):
        self.fps = fps
        self.minimum_steps = minimum_steps
        self.tolerance = tolerance
        self.elapsed = 0.0

    @property
    def steps(self):
        return self.elapsed*self.fps

    def can_replace(self, busy, pending):
        return self.steps >= self.minimum_steps and not busy and not pending

    def advance(self, desired, actual, commanded, dt, paused=False):
        lag = max(np.max(np.abs(six(desired)-six(actual))),
                  np.max(np.abs(six(desired)-six(commanded))))
        if not paused and lag <= self.tolerance:
            self.elapsed += dt
        return float(lag)


def takeover_reason(progress, now, observed, adopted, last_progress, max_age,
                    aligned, busy, pending, has_newer):
    """Scheduling only. Caller MUST recheck freshness, drift and target bounds.
    Never replaces a chunk during gripper motion. Pending debounce may be
    cancelled by urgent replacement; issued gripper commands remain latched.
    """
    if not has_newer or busy:
        return None
    if aligned and progress.can_replace(False, pending):
        return 'minimum_steps'
    if progress.steps >= 30 and not pending:
        return 'chunk_finished'
    if now-adopted < .25:
        return None
    if max_age-(now-observed) <= .45:
        return 'expiry_margin'
    if not pending and now-last_progress >= .30:
        return 'stalled_progress'
    return None
