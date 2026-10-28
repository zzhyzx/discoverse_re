"""Optional MuJoCo grasp latch for lift-only evaluation; no hardware control."""
import math


class GraspHold:
    """Keep a confirmed grasp closed until a new episode creates a new latch.

    Confirmation uses feedback from the preceding command, not the current
    prediction. Empty closing, one-sided contact and unsettled motion do not
    confirm a grasp. Losing contact after confirmation does not command opening.
    """

    def __init__(self, confirm_frames=3, settled_delta=.002):
        self.confirm_frames = confirm_frames
        self.settled_delta = settled_delta
        self.confirmed = False
        self.confirmed_at = None
        self.stable_frames = 0
        self.previous_feedback = None
        self.previous_closed = False
        self.release_suppressed = False
        self.suppressed_frames = 0

    def command(self, raw, feedback, bilateral_contact, now):
        if not all(math.isfinite(x) for x in (raw, feedback, now)):
            raise ValueError('Non-finite simulated grasp feedback or prediction')
        settled = (self.previous_feedback is not None
                   and abs(feedback - self.previous_feedback) < self.settled_delta)
        if not self.confirmed:
            eligible = self.previous_closed and bilateral_contact and settled
            self.stable_frames = self.stable_frames + 1 if eligible else 0
            if self.stable_frames >= self.confirm_frames:
                self.confirmed = True
                self.confirmed_at = float(now)
        self.previous_feedback = feedback
        requested_closed = raw >= .5
        self.release_suppressed = self.confirmed and not requested_closed
        self.suppressed_frames += int(self.release_suppressed)
        self.previous_closed = self.confirmed or requested_closed
        return 255. if self.previous_closed else 0.


class BilateralPadContact:
    """Simulation contact feedback, not a deployable RGB grasp detector."""

    def __init__(self, model):
        import mujoco
        self.cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'red_block_geom')
        pads = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                'gripper_' + side + '_pad') for side in ('left', 'right')]
        if self.cube < 0 or min(pads) < 0 or len(set(pads)) != 2:
            raise ValueError('Missing or invalid block / finger-pad contact geometry')
        self.pad_geoms = {}
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            while body:
                if body in pads:
                    self.pad_geoms[geom] = pads.index(body)
                    break
                body = int(model.body_parentid[body])

    def __call__(self, data):
        touched = set()
        for contact in data.contact[:data.ncon]:
            if contact.dist > .001 or self.cube not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.cube else contact.geom1
            if other in self.pad_geoms:
                touched.add(self.pad_geoms[other])
        return len(touched) == 2
