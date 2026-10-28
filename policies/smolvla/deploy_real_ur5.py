"""UR5 SmolVLA deployment with Windows RGB bridge. Default live GET-only.
--execute enables servoJ after confirmation; --control-gripper is separate.
No automatic positioning, power/brake release, fault recovery or return.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'submodules/lerobot/src'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
from ur5_camera_wire import CameraClient, CameraTimingError, ROLES
from ur5_deploy_core import GripperDecision, Limiter, Plan, interpolate, six, validate_actions, remaining_plan, TargetDeltaError, ExecutionProgress, takeover_reason
from ur5_control_profile import load_profile, guarded_command
CAMERAS = ('observation.images.d436', 'observation.images.d405')

def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()

def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)

def configuration(path):
    cfg = json.loads(path.read_text())
    if cfg.get('control_profile'):
        profile=load_profile(resolve(cfg['control_profile']))
        for key in ('max_command_speed_rad_s','max_command_acceleration_rad_s2','action_hz',
                    'control_hz','execution_steps','minimum_execution_steps','progress_tolerance_deg','allow_release'):
            cfg[key]=profile[key]
        cfg['control_contract']=profile
    cfg.setdefault('minimum_execution_steps', 10)
    cfg.setdefault('progress_tolerance_deg', .5)
    if type(cfg['minimum_execution_steps']) is not int or not 1 <= cfg['minimum_execution_steps'] <= cfg['execution_steps']:
        raise ValueError('minimum_execution_steps must be 1..execution_steps')
    if not math.isfinite(cfg['progress_tolerance_deg']) or not 0 < cfg['progress_tolerance_deg'] <= 1:
        raise ValueError('progress_tolerance_deg must be within (0, 1]')
    positive = ('start_tolerance_deg', 'training_envelope_margin_deg', 'max_prediction_delta_deg', 'max_observation_joint_drift_deg',
        'max_tracking_error_deg', 'max_actual_speed_rad_s', 'max_command_speed_rad_s',
        'max_command_acceleration_rad_s2', 'watchdog_min_hz', 'max_control_gap_s',
        'next_chunk_grace_s', 'max_feedback_age_s', 'max_camera_age_s', 'max_camera_receipt_skew_s',
        'max_observation_age_s', 'max_inference_seconds', 'max_gripper_status_age_s', 'duration_seconds')
    for key in positive:
        if isinstance(cfg[key], bool) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError('Invalid positive setting: ' + key)
    if cfg['control_hz'] != 125 or cfg['action_hz'] != 30:
        raise ValueError('Requires CB3 125Hz control and 30Hz actions')
    if type(cfg['execution_steps']) is not int or not 1 <= cfg['execution_steps'] <= 50:
        raise ValueError('execution_steps must be 1..50')
    if cfg['max_inference_seconds']+cfg['max_camera_age_s']+.1 >= 50/cfg['action_hz']:
        raise ValueError('Inference budget leaves insufficient prediction horizon')
    if not .03 <= cfg['lookahead_s'] <= .2 or not 100 <= cfg['servo_gain'] <= 2000:
        raise ValueError('Invalid servo lookahead/gain')
    if cfg['max_control_gap_s'] >= 1/cfg['watchdog_min_hz']:
        raise ValueError('Host gap limit must be below controller watchdog interval')
    if not 0 < cfg['max_command_speed_rad_s'] <= .2 or not 0 < cfg['max_command_acceleration_rad_s2'] <= .5:
        raise ValueError('Command speed/acceleration exceed bounded entry point')
    for key in ('gripper_speed_raw', 'gripper_force_raw'):
        if type(cfg[key]) is not int or not 0 <= cfg[key] <= 255:
            raise ValueError('Invalid gripper raw setting')
    if not 0 <= cfg['gripper_feedback_open_raw'] < cfg['gripper_feedback_closed_raw'] <= 255:
        raise ValueError('Invalid gripper feedback mapping')
    if type(cfg['allow_release']) is not bool or type(cfg['record_video']) is not bool:
        raise ValueError('allow_release and record_video must be JSON booleans')
    six(cfg['start_reference_deg'])
    return cfg

class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.camera = self.gripper = self.plan = None
        self.gripper_request = None
        self.gripper_busy = False

    def fail(self, exc):
        with self.lock:
            if self.error is None: self.error = str(exc) or type(exc).__name__
        self.stop.set()

    def check(self):
        if self.error: raise RuntimeError(self.error)

class Recorder:
    """Bounded disk queue, separate from servo thread."""
    def __init__(self, folder, shared, video):
        self.folder, self.shared, self.video = folder, shared, video
        self.queue = queue.Queue(maxsize=512)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def event(self, event, **values):
        try: self.queue.put_nowait(('event', dict(event=event, monotonic=time.monotonic(), **values)))
        except queue.Full: raise RuntimeError('Log queue overflow')

    def frames(self, metadata, images):
        if self.video:
            try: self.queue.put_nowait(('frames', (metadata, images)))
            except queue.Full: raise RuntimeError('Video/log queue overflow')
        else:
            self.event('camera', **metadata)

    def run(self):
        streams, containers, last_pts = {}, {}, {}
        first = None
        try:
            with (self.folder/'events.jsonl').open('x', buffering=1) as log:
                while True:
                    item = self.queue.get()
                    if item is None: break
                    kind, payload = item
                    if kind == 'event':
                        log.write(json.dumps(payload, allow_nan=False)+'\n')
                        continue
                    import av
                    from fractions import Fraction
                    metadata, images = payload
                    observed = metadata['local_observed_monotonic']
                    first = observed if first is None else first
                    for role in ROLES:
                        if role not in containers:
                            containers[role] = av.open(str(self.folder/f'{role}.mp4'), 'w')
                            stream = containers[role].add_stream('libx264rgb', rate=30)
                            stream.width, stream.height, stream.pix_fmt = 640, 480, 'rgb24'
                            stream.options = {'crf': '0', 'preset': 'veryfast', 'g': '30'}
                            streams[role] = stream
                        pts = max(last_pts.get(role, -1)+1, round((observed-first)*30))
                        frame = av.VideoFrame.from_ndarray(images[role], format='rgb24')
                        frame.pts, frame.time_base = pts, Fraction(1,30)
                        for packet in streams[role].encode(frame): containers[role].mux(packet)
                        last_pts[role] = pts
                    log.write(json.dumps(dict(event='camera', monotonic=time.monotonic(), **metadata))+'\n')
        except Exception as exc: self.shared.fail('Recorder: ' + str(exc))
        finally:
            for role, container in containers.items():
                try:
                    for packet in streams[role].encode(): container.mux(packet)
                    container.close()
                except Exception as exc: self.shared.fail('Video finalize: ' + str(exc))

    def close(self):
        if self.thread.is_alive():
            try: self.queue.put(None, timeout=1)
            except queue.Full: self.shared.fail('Could not finalize recording')
            self.thread.join(5)
            if self.thread.is_alive(): self.shared.fail('Recorder did not finalize within 5 seconds')

class Feedback:
    def __init__(self, receive, cfg):
        self.receive, self.cfg = receive, cfg
        self.lock = threading.Lock()
        self.last_stamp = None
        self.last_fresh = time.monotonic()

    def sample(self):
        with self.lock:
            r, now = self.receive, time.monotonic()
            if not r.isConnected(): raise RuntimeError('RTDE receive disconnected')
            stamp = float(r.getTimestamp())
            if not math.isfinite(stamp) or (self.last_stamp is not None and stamp < self.last_stamp):
                raise RuntimeError('Invalid RTDE timestamp')
            if self.last_stamp is None or stamp > self.last_stamp:
                self.last_stamp, self.last_fresh = stamp, now
            if now-self.last_fresh > self.cfg['max_feedback_age_s']: raise RuntimeError('Stale RTDE feedback')
            if r.getRobotMode() != 7 or r.getSafetyMode() not in (1, 2):
                raise RuntimeError('Robot/safety state unsuitable; inspect pendant')
            q, qd = six(r.getActualQ()), six(r.getActualQd())
            if np.max(np.abs(qd)) > self.cfg['max_actual_speed_rad_s']:
                raise RuntimeError('Measured joint speed exceeds configured limit')
            return q, qd, stamp

    def idle(self):
        if self.receive.getRuntimeState() != 1 or self.receive.getRobotStatus() & 6:
            raise RuntimeError('Stop other program/freedrive before deployment')

    def stationary(self, expected):
        beginning = time.monotonic()
        while time.monotonic()-beginning < .5:
            q, qd, _ = self.sample()
            if np.max(np.abs(q-six(expected))) > math.radians(.15) or np.max(np.abs(qd)) > .005:
                raise RuntimeError('Robot moved since preview; restart deployment')
            time.sleep(.02)

def load_model(cfg):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    checkpoint, statistics = resolve(cfg['checkpoint']), resolve(cfg['train_stats'])
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in WSL')
    stats = {k: {name: np.asarray(value) for name, value in data.items()}
             for k, data in json.loads(statistics.read_text()).items()}
    for key in ('observation.state', 'action'):
        for name in ('mean', 'std'):
            value = stats[key][name]
            if value.shape != (7,) or not np.isfinite(value).all() or (name == 'std' and np.any(value <= 0)):
                raise ValueError('Invalid saved normalization: ' + key)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if config.type != 'smolvla' or config.n_obs_steps != 1 or config.n_action_steps != 10 or config.chunk_size != 50:
        raise ValueError('Expected verified SmolVLA 1 observation / 50 chunk / 10 action config')
    if set(config.input_features) != {'observation.state', *CAMERAS} or tuple(config.output_features['action'].shape) != (7,):
        raise ValueError('Input/output feature mismatch')
    if tuple(config.input_features['observation.state'].shape) != (7,) or any(
            tuple(config.input_features[key].shape) != (3,480,640) for key in CAMERAS):
        raise ValueError('Expected 7 state and two 640x480 cameras')
    config.device, config.push_to_hub = 'cuda', False
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, dataset_stats=stats).to('cuda').eval()
    for name, value in policy.named_buffers():
        if name.startswith(('normalize', 'unnormalize')) and not torch.isfinite(value).all():
            raise RuntimeError('Normalization not loaded: ' + name)
    margin = math.radians(cfg['training_envelope_margin_deg'])
    limits = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
    lower = np.maximum(stats['action']['min'][:6]-margin, -limits)
    upper = np.minimum(stats['action']['max'][:6]+margin, limits)
    return policy, lower, upper

def predict(policy, state, images, task):
    import torch
    batch = {'observation.state': torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to('cuda'), 'task': [task]}
    for key, role in zip(CAMERAS, ROLES):
        batch[key] = torch.from_numpy(images[role].copy()).permute(2,0,1).unsqueeze(0).to('cuda', dtype=torch.float32)/255
    policy.reset()
    with torch.inference_mode(): actions = policy.predict_action_chunk(batch)[0].cpu().numpy()
    if actions.shape != (50,7) or not np.isfinite(actions).all(): raise RuntimeError('Invalid model output')
    return actions

def camera_worker(client, shared, recorder, cfg):
    rejected_since = None
    try:
        while not shared.stop.is_set():
            try:
                metadata, raw = client.get(cfg['max_camera_age_s'], cfg['max_camera_receipt_skew_s'])
            except CameraTimingError as exc:
                now = time.monotonic()
                rejected_since = now if rejected_since is None else rejected_since
                recorder.event('camera_frame_rejected', reason=str(exc))
                # Never publish invalid images. Servo and observation() still
                # enforce the age of the last VALID frame on every access.
                if now-rejected_since > cfg['max_camera_age_s']: raise
                continue
            rejected_since = None
            images = {role: np.frombuffer(raw[role], np.uint8).reshape(480,640,3).copy() for role in ROLES}
            with shared.lock: shared.camera = (metadata, images)
            recorder.frames(metadata, images)
            shared.stop.wait(.02)
    except Exception as exc: shared.fail('Camera: ' + str(exc))
    finally: client.close()

def gripper_worker(gripper, shared, recorder, cfg, enabled):
    pending = None
    sent = False
    holding = False
    try:
        while not shared.stop.is_set():
            started = time.monotonic()
            status = gripper.status()
            if shared.stop.is_set(): break
            if status['ACT'] != 1 or status['STA'] != 3 or status['FLT'] != 0:
                raise RuntimeError('Gripper activation/fault changed')
            if status['OBJ'] == 1: raise RuntimeError('Gripper opening contact')
            if holding and status['OBJ'] != 2:
                raise RuntimeError('Previously confirmed closing contact lost or changed')
            if holding and (status['PRE'] != 255 or status['GTO'] != 1):
                raise RuntimeError('Holding command changed by another controller')
            if pending is None and status['GTO'] == 1 and status['OBJ'] == 0:
                raise RuntimeError('Unrequested gripper movement; stop competing controller')
            with shared.lock:
                shared.gripper = (time.monotonic(), status)
                request = shared.gripper_request
                shared.gripper_request = None
            recorder.event('gripper_status', registers=status, read_seconds=time.monotonic()-started)
            if pending:
                target, issued = pending['target'], pending['issued']
                if time.monotonic()-issued > 8: raise RuntimeError('Gripper command timeout')
                if status['PRE'] == target:
                    if status['GTO'] != 1: raise RuntimeError('Gripper command overridden')
                    if status['OBJ'] == 3 and target == 255:
                        raise RuntimeError('Closed endpoint without contact; do not continue lifting')
                    pending['moved'] = pending['moved'] or abs(status['POS']-pending['initial_pos']) > 2 or status['OBJ'] == 0
                    terminal = (target == 255 and status['OBJ'] == 2 and pending['moved']) or (target == 0 and status['OBJ'] == 3)
                    stable = terminal and pending['previous_pos'] is not None and abs(status['POS']-pending['previous_pos']) <= 1
                    pending['stable'] = pending['stable']+1 if stable else 0
                    pending['previous_pos'] = status['POS']
                    if pending['stable'] >= 3:
                        holding = target == 255
                        with shared.lock: shared.gripper_busy = False
                        recorder.event('gripper_terminal', closing_contact=holding, registers=status)
                        pending = None
                elif time.monotonic()-issued > 2:
                    raise RuntimeError('Gripper PRE did not acknowledge target')
            if request is not None:
                if not enabled or pending: raise RuntimeError('Unexpected concurrent gripper request')
                if request is False and not cfg['allow_release']:
                    raise RuntimeError('Release disabled for this grasp/lift checkpoint')
                target = 255 if request else 0
                if shared.stop.is_set(): break
                sent = True
                gripper.set(POS=target, SPE=cfg['gripper_speed_raw'], FOR=cfg['gripper_force_raw'], GTO=1)
                holding = False
                pending = dict(target=target,issued=time.monotonic(),initial_pos=status['POS'],
                               moved=False,previous_pos=None,stable=0)
                recorder.event('gripper_command', closed=request, raw_target=target)
            shared.stop.wait(.02)
    except Exception as exc: shared.fail('Gripper: ' + str(exc))
    finally:
        if enabled and sent:
            try:
                gripper.set(GTO=0)
                print('Gripper GTO=0 sent; no automatic reopening. Inspect held object.', flush=True)
            except Exception as exc: print('GRIPPER STOP UNCONFIRMED:', exc, flush=True)
        gripper.close()

def observation(shared, feedback, cfg):
    shared.check()
    now = time.monotonic()
    with shared.lock: camera, grip = shared.camera, shared.gripper
    if camera is None or grip is None: raise RuntimeError('Live observation not ready')
    metadata, images = camera
    if now-metadata['local_observed_monotonic'] > cfg['max_camera_age_s']:
        raise RuntimeError('Latest camera observation stale')
    if now-grip[0] > cfg['max_gripper_status_age_s']:
        raise RuntimeError('Latest gripper status stale')
    q, qd, stamp = feedback.sample()
    raw = grip[1]['POS']
    value = (raw-cfg['gripper_feedback_open_raw'])/(cfg['gripper_feedback_closed_raw']-cfg['gripper_feedback_open_raw'])
    if not 0 <= value <= 1: raise RuntimeError('Gripper POS outside configured feedback mapping')
    return np.r_[q,value].astype(np.float32), images, metadata, stamp

def inference_worker(policy, shared, feedback, recorder, cfg, lower, upper):
    sequence = 0
    try:
        # CUDA library handles can be thread-local: warm the actual inference
        # worker before publishing any executable prediction.
        started = time.monotonic()
        state, images, _, _ = observation(shared, feedback, cfg)
        predict(policy, state, images, cfg['task'])
        recorder.event('inference_worker_warmup', seconds=time.monotonic()-started)
        while not shared.stop.is_set():
            started = time.monotonic()
            state, images, metadata, stamp = observation(shared, feedback, cfg)
            actions = predict(policy, state, images, cfg['task'])
            completed = time.monotonic()
            sequence += 1
            plan = Plan(sequence, metadata['local_observed_monotonic'], completed,
                        actions.copy(), state.copy(),
                        {role: metadata['cameras'][role]['frame_number'] for role in ROLES})
            remaining, skipped, _ = remaining_plan(plan, completed, cfg['action_hz'], cfg['execution_steps'])
            try:
                validate_actions(remaining.actions, state[:6], lower, upper, math.radians(cfg['max_prediction_delta_deg']))
            except TargetDeltaError as exc:
                recorder.event('prediction_rejected', sequence=sequence, reason=str(exc),
                    state=state.tolist(), actions=remaining.actions.tolist())
                if cfg.get('execute', False): raise
                print('PREVIEW WARNING (would reject execution):', exc, flush=True)
            recorder.event('prediction', sequence=sequence, observed_monotonic=plan.observed,
                state=state.tolist(), rtde_timestamp=stamp, actions=actions.tolist(),
                camera_numbers=plan.camera_numbers, inference_seconds=completed-started,
                skipped_steps=skipped, remaining_steps=len(remaining.actions),
                target_delta_deg=np.degrees(remaining.actions[0,:6]-state[:6]).tolist())
            if completed-started > cfg['max_inference_seconds']:
                raise RuntimeError(f'Inference exceeds configured deadline: {completed-started:.3f}s')
            with shared.lock: shared.plan = plan
            # Predict continuously in this worker while the servo executes.
            # Small minimum cadence also prevents a fast/mock model busy-loop.
            shared.stop.wait(max(.001, .1-(time.monotonic()-started)))
    except Exception as exc: shared.fail('Inference: ' + str(exc))

def bridge_client(args, cfg):
    session = json.loads(resolve(cfg['camera_session']).read_text())
    for role in ROLES:
        if session[role+'_serial'] != cfg[role+'_serial']:
            raise ValueError('Camera serial mapping differs from deployment config')
    hosts = [args.camera_host] if args.camera_host != 'auto' else ['127.0.0.1']
    if args.camera_host == 'auto':
        route = subprocess.check_output(['ip','route','show','default'], text=True).split()
        if 'via' in route: hosts.append(route[route.index('via')+1])
    failures = []
    for host in dict.fromkeys(hosts):
        try:
            client = CameraClient(host, session)
            print('Windows camera host:', host, flush=True)
            return client
        except OSError as exc: failures.append(f'{host}: {exc}')
    raise ConnectionError('Camera bridge unreachable. Start Windows bridge/check firewall or --camera-host. '+ '; '.join(failures))

def return_to_start(receive, actual, reference, status, cfg, recorder):
    """Confirmed low-speed moveJ before camera/inference workers and servo startup."""
    from ur5_move_to_pose import idle, validate_move
    from ur5_large_ab import Feedback as MoveFeedback, move, TOL
    difference = np.max(np.abs(actual-reference))
    if difference <= TOL:
        print('Already at training start (within 0.15deg); no homing motion.', flush=True)
        return
    # Existing move helper rejects tiny 0.15..0.3 degree moves. Those already
    # satisfy a stricter tolerance than the deployment start check.
    if difference < math.radians(.3):
        print('Within 0.3deg of training start; homing skipped.', flush=True)
        return
    validate_move(actual.tolist(), reference.tolist())
    if status['POS'] > 20 or status['OBJ'] == 2:
        raise RuntimeError('Homing requires open EMPTY gripper; remove object manually. No automatic opening.')
    print('Return-to-start current degrees:',np.round(np.degrees(actual),3).tolist(),flush=True)
    print('Return-to-start target degrees:',np.round(np.degrees(reference),3).tolist(),flush=True)
    print('Low-speed moveJ: 0.03rad/s, 0.05rad/s^2. No path collision checking. '
          'Confirm EMPTY tool, correct payload and clear FULL return path.',flush=True)
    if input('Type HOME to return to training start (anything else cancels): ').strip().upper()!='HOME':
        raise RuntimeError('Homing cancelled; model will not execute')
    f=MoveFeedback(receive)
    f.stationary(actual.tolist());idle(receive)
    from rtde_control import RTDEControlInterface
    control=None
    try:
        control=RTDEControlInterface(cfg['robot_ip'],125)
        recorder.event('homing_start',source=actual.tolist(),target=reference.tolist())
        move(control,f,actual.tolist(),reference.tolist(),
             lambda event,**values: recorder.event('homing_'+event,**values))
        f.stationary(reference.tolist())
        recorder.event('homing_complete')
    finally:
        if control is not None:
            try: control.stopJ(.5)
            finally:
                try: control.stopScript()
                finally: control.disconnect()
    deadline=time.monotonic()+3
    while receive.getRuntimeState()!=1 or receive.getRobotStatus() & 6:
        if time.monotonic()>deadline:
            raise RuntimeError('Homing program did not stop; model execution cancelled')
        time.sleep(.02)

def run_servo(control, feedback, shared, recorder, cfg, lower, upper, enabled_gripper):
    print(f"Execution: minimum {cfg['minimum_execution_steps']} steps for normal replanning; fresh early takeover on stall/expiry; "
          f"progress tolerance {cfg['progress_tolerance_deg']}deg; gripper pauses action time.",
          flush=True)
    q, _, _ = feedback.sample()
    limiter = Limiter(q, cfg['max_command_speed_rad_s'], cfg['max_command_acceleration_rad_s2'])
    period = 1/cfg['control_hz']
    start, last, used = time.monotonic(), time.monotonic(), 0
    active, initial = None, q.copy()
    progress = ExecutionProgress(cfg['action_hz'], cfg['minimum_execution_steps'],
                                 math.radians(cfg['progress_tolerance_deg']))
    finished_at = None
    adopted_at = last_progress_at = start
    with shared.lock: status = shared.gripper[1]
    decision = GripperDecision(status['POS'] >= 128)
    try:
        while time.monotonic()-start < cfg['duration_seconds']:
            tick = control.initPeriod()
            now = time.monotonic()
            shared.check()
            if shared.stop.is_set(): raise RuntimeError('Stop requested')
            if now-last > cfg['max_control_gap_s']: raise RuntimeError('Servo loop missed timing deadline')
            last = now
            actual, qd, stamp = feedback.sample()
            r = feedback.receive
            if not control.isConnected() or r.getRuntimeState() != 2 or not r.getRobotStatus() & 2:
                raise RuntimeError('Control program disconnected/stopped/paused')
            factors = (float(r.getSpeedScaling()), float(r.getTargetSpeedFraction()))
            if any(not math.isfinite(v) or not 0 <= v <= 1.001 for v in factors):
                raise RuntimeError('Invalid speed scaling')
            if now-start > 2 and factors[0]*factors[1] < .1:
                raise RuntimeError('Effective speed scaling below 10%; inspect pendant')
            with shared.lock:
                newest, camera, grip, busy = shared.plan, shared.camera, shared.gripper, shared.gripper_busy
            if camera is None or now-camera[0]['local_observed_monotonic'] > cfg['max_camera_age_s']:
                raise RuntimeError('Camera stream stale during servo')
            if grip is None or now-grip[0] > cfg['max_gripper_status_age_s']:
                raise RuntimeError('Gripper feedback stale during servo')
            waiting = False
            at_setpoint = active is None or max(
                np.max(np.abs(actual-interpolate(active.actions, initial, progress.elapsed, cfg['action_hz'])[:6])),
                np.max(np.abs(limiter.q-interpolate(active.actions, initial, progress.elapsed, cfg['action_hz'])[:6]))
            ) <= progress.tolerance
            transition = 'initial' if active is None else takeover_reason(
                progress, now, active.observed, adopted_at, last_progress_at,
                cfg['max_observation_age_s'], at_setpoint, busy,
                decision.candidate is not None, newest is not None and newest.sequence > used)
            if transition or (active is not None and progress.steps >= len(active.actions)
                              and not busy and decision.candidate is None):
                if newest is None or newest.sequence <= used:
                    if finished_at is None: finished_at = now
                    if active is None or now-finished_at > cfg['next_chunk_grace_s']:
                        raise RuntimeError('No fresh next action chunk; stopping instead of repeating')
                    waiting = True
                else:
                    if now-newest.observed > cfg['max_observation_age_s']: raise RuntimeError('Next prediction observation too old')
                    drift = np.max(np.abs(actual-newest.state[:6]))
                    if drift > math.radians(cfg['max_observation_joint_drift_deg']):
                        raise RuntimeError(f'Robot moved too far since prediction observation: {math.degrees(drift):.3f}deg')
                    newest, skipped, _ = remaining_plan(newest, now, cfg['action_hz'], cfg['execution_steps'])
                    validate_actions(newest.actions, actual, lower, upper, math.radians(cfg['max_prediction_delta_deg']))
                    if not control.kickWatchdog(): raise RuntimeError('Controller watchdog kick failed')
                    # Supplementary endpoint limits; does not check swept-path collisions.
                    if not control.isJointsWithinSafetyLimits(newest.actions[-1,:6].tolist()):
                        raise RuntimeError('Controller rejected chunk endpoint')
                    recorder.event('chunk_transition', previous_sequence=used,
                        previous_executed_steps=progress.steps, next_sequence=newest.sequence, reason=transition)
                    active, used = newest, newest.sequence
                    progress.elapsed = 0.0
                    adopted_at = last_progress_at = now
                    finished_at = None
                    decision.candidate = decision.since = None
                    initial = limiter.q.copy()
                    recorder.event('chunk_started', sequence=used, skipped_steps=skipped,
                        execution_steps=len(active.actions), observation_age_s=now-active.observed)
            if now-active.observed > cfg['max_observation_age_s']:
                raise RuntimeError('Active prediction expired')
            if np.max(np.abs(actual-limiter.q)) > math.radians(cfg['max_tracking_error_deg']):
                raise RuntimeError('Joint tracking error exceeded')
            desired = interpolate(active.actions, initial, progress.elapsed, cfg['action_hz'])
            command, request, aligned = guarded_command(
                limiter, decision, desired, actual, now, period, progress.tolerance,
                busy=busy, enabled=enabled_gripper, waiting=waiting)
            if request is not None:
                if not request and not cfg['allow_release']:
                    raise RuntimeError('Model requested release; disabled for this checkpoint')
                with shared.lock:
                    if shared.gripper_request is not None or shared.gripper_busy:
                        raise RuntimeError('Gripper command overlap')
                    shared.gripper_request = request
                    shared.gripper_busy = True
                busy = True
            if np.any(command < lower) or np.any(command > upper): raise RuntimeError('Limited command outside envelope')
            if time.monotonic()-now > cfg['max_control_gap_s']:
                raise RuntimeError('Control tick blocked before servo command')
            if not control.kickWatchdog(): raise RuntimeError('Controller watchdog kick failed')
            if not control.servoJ(command.tolist(), 0., 0., period, cfg['lookahead_s'], cfg['servo_gain']):
                raise RuntimeError('servoJ rejected')
            before_progress = progress.elapsed
            progress_lag = progress.advance(desired[:6], actual, command, period,
                paused=busy or waiting or decision.candidate is not None)
            if progress.elapsed > before_progress: last_progress_at = now
            recorder.event('servo', sequence=used, actual_q=actual.tolist(), actual_qd=qd.tolist(),
                raw_target=desired.tolist(), executed_q=command.tolist(), rtde_timestamp=stamp,
                effective_scaling=factors[0]*factors[1], gripper_busy=busy, waiting_for_chunk=waiting,
                executed_steps=progress.steps, target_lag_deg=math.degrees(progress_lag),
                gripper_pending=decision.candidate is not None)
            control.waitPeriod(tick)
    finally:
        shared.stop.set()
        print('Stopping arm; no automatic return.', flush=True)
        try:
            if not control.servoStop(.5): raise RuntimeError('servoStop rejected')
        except Exception as exc:
            print('servoStop failed:', exc, flush=True)
            try: control.stopJ(.5)
            except Exception as second: print('ARM STOP UNCONFIRMED:', second, flush=True)

def model_check(policy, cfg):
    """Real held-out observation; no camera, RTDE or gripper connection."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset('local/real_ur5_pick_cube_v21', root=ROOT/'data/real_ur5_pick_cube_v21', video_backend='pyav')
    row = dataset[int(dataset.episode_data_index['from'][5])]
    images = {role: np.round(row[key].permute(1,2,0).numpy()*255).astype(np.uint8)
              for role,key in zip(ROLES,CAMERAS)}
    timings = []
    for index in range(22):
        started = time.monotonic()
        actions = predict(policy, row['observation.state'].numpy(), images, cfg['task'])
        elapsed = time.monotonic()-started
        if index >= 2: timings.append(elapsed)
        print('Model check',index+1,'shape',actions.shape,'seconds',round(elapsed,3),
              'first joints deg',np.round(np.degrees(actions[0,:6]),3).tolist(),flush=True)
    print('20 measured predictions (2 warmups excluded): median/p95/max seconds',
          np.round([np.median(timings), np.percentile(timings,95), max(timings)],3).tolist(), flush=True)
    passed = max(timings) <= cfg['max_inference_seconds']
    print('Offline inference budget:', 'PASS' if passed else 'FAIL',
          '; live camera/servo timing still requires validation.', flush=True)
    return 0 if passed else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'policies/smolvla/configs/ur5_real_deploy.json')
    parser.add_argument('--camera-host', default='auto')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--control-gripper', action='store_true')
    parser.add_argument('--start-pose', type=Path, help='Optional saved q_rad snapshot; no automatic move')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--check-model', action='store_true', help='Offline observation; no hardware access')
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    if args.control_gripper and not args.execute: parser.error('--control-gripper requires --execute')
    if args.check_model and (args.execute or args.control_gripper): parser.error('--check-model cannot execute')
    cfg = configuration(args.config.resolve())
    cfg['execute'] = args.execute
    print(f"Async prediction: up to {cfg['execution_steps']} steps at {cfg['action_hz']}Hz; "
          f"inference budget {cfg['max_inference_seconds']}s; feedback-gated prefix; "
          f"target delta limit {cfg['max_prediction_delta_deg']}deg.", flush=True)
    if args.duration is not None:
        if not math.isfinite(args.duration) or not 0 < args.duration <= 120:
            parser.error('--duration must be 0..120 seconds')
        cfg['duration_seconds'] = args.duration
    policy, lower, upper = load_model(cfg)
    if args.check_model: return model_check(policy,cfg)
    if args.start_pose:
        document = json.loads(args.start_pose.read_text())
        snapshot = document.get('snapshot',document)
        if snapshot.get('ip') != cfg['robot_ip']: raise ValueError('Start snapshot IP mismatch')
        reference = six(snapshot['q_rad'])
        if np.max(np.abs(reference-np.radians(cfg['start_reference_deg']))) > math.radians(cfg['start_tolerance_deg']):
            raise ValueError('Saved start differs from training start; no automatic angle wrapping')
    else: reference = np.radians(cfg['start_reference_deg'])
    if cfg['record_video'] and not args.no_video:
        import av
        av.codec.Codec('libx264rgb','w')
    folder = ROOT/'data/real_ur5/smolvla_deploy'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'run_config.json',dict(config=cfg,execute=args.execute,
        control_gripper=args.control_gripper, model_statistics_sha256=hashlib.sha256(resolve(cfg['train_stats']).read_bytes()).hexdigest(),
        start_reference_deg=np.degrees(reference).tolist(),
        camera_pairing='Windows host receipt-time only; no hardware synchronization',
        joints='Absolute rad, unchanged branches', collision_checking=False))
    print('Logs:',folder,flush=True)
    shared = Shared()
    recorder = Recorder(folder,shared,cfg['record_video'] and not args.no_video)
    receive = control = client = gripper = None
    workers = []
    result = 1
    try:
        from rtde_receive import RTDEReceiveInterface
        from robotiq_gripper_test import Gripper, ready
        receive = RTDEReceiveInterface(cfg['robot_ip'],125)
        feedback = Feedback(receive,cfg)
        actual,_,_ = feedback.sample()
        feedback.idle(); feedback.stationary(actual)
        gripper = Gripper(cfg['robot_ip'],cfg['gripper_port'])
        status = gripper.status(); ready(status)
        if args.execute:
            return_to_start(receive,actual,reference,status,cfg,recorder)
            actual,_,_=feedback.sample()
            feedback.idle(); feedback.stationary(actual)
            status=gripper.status();ready(status)
        with shared.lock: shared.gripper = (time.monotonic(),status)
        # Open after homing: an idle camera socket times out during a long move.
        client = bridge_client(args,cfg)
        camera_thread = threading.Thread(target=camera_worker,args=(client,shared,recorder,cfg),daemon=True)
        grip_thread = threading.Thread(target=gripper_worker,args=(gripper,shared,recorder,cfg,args.control_gripper),daemon=True)
        workers = [camera_thread,grip_thread]
        for worker in workers: worker.start()
        client = gripper = None
        deadline = time.monotonic()+3
        while shared.camera is None and time.monotonic()<deadline:
            shared.check(); time.sleep(.02)
        state,images,_,_ = observation(shared,feedback,cfg)
        predict(policy,state,images,cfg['task'])
        print('Current degrees:',np.round(np.degrees(actual),3).tolist(),flush=True)
        print('Training start degrees:',np.round(np.degrees(reference),3).tolist(),flush=True)
        print('Gripper state=POS/255 by default; confirm collector mapping. Homing requires HOME confirmation.',flush=True)
        if args.execute:
            if np.max(np.abs(actual-reference)) > math.radians(cfg['start_tolerance_deg']):
                raise RuntimeError('Not at training start; position manually and restart')
            if args.control_gripper and status['POS'] > 20:
                raise RuntimeError('Start with open empty gripper; no automatic opening')
            print('Confirm correct tool/payload, clear full path, operator at E-stop, no competing controller.',flush=True)
            print('Bounded servoJ speed',cfg['max_command_speed_rad_s'],'acceleration',cfg['max_command_acceleration_rad_s2'],
                  '; model gripper enabled:',args.control_gripper,'; no collision detection.',flush=True)
            if input('Type MOVE to execute (anything else cancels): ').strip().upper() != 'MOVE':
                raise RuntimeError('Cancelled')
            shared.check(); feedback.stationary(actual); feedback.idle()
        if args.execute:
            # Initialize before launching inference: native RTDE startup may
            # block Python scheduling and age a previously generated plan.
            from rtde_control import RTDEControlInterface
            from ur5_large_ab import wait_program, Feedback as ExistingFeedback
            startup = time.monotonic()
            control = RTDEControlInterface(cfg['robot_ip'],125)
            wait_program(control,ExistingFeedback(receive),actual.tolist())
            if not control.setWatchdog(cfg['watchdog_min_hz']): raise RuntimeError('Controller watchdog setup failed')
            recorder.event('control_ready', seconds=time.monotonic()-startup)
            print('Control ready; waiting for a new prediction before motion.', flush=True)
        policy.reset()
        inference = threading.Thread(target=inference_worker,args=(policy,shared,feedback,recorder,cfg,lower,upper),daemon=True)
        workers.append(inference); inference.start()
        deadline = time.monotonic()+5
        while shared.plan is None and time.monotonic()<deadline:
            shared.check()
            if control is not None:
                q, qd, _ = feedback.sample()
                if np.max(np.abs(q-actual)) > math.radians(.15) or np.max(np.abs(qd)) > .005:
                    raise RuntimeError('Robot moved while waiting for initial prediction')
                if not control.isConnected() or receive.getRuntimeState()!=2 or not receive.getRobotStatus() & 2:
                    raise RuntimeError('Control program stopped during prediction startup')
                if not control.kickWatchdog(): raise RuntimeError('Controller watchdog kick failed during startup')
            time.sleep(.01)
        shared.check()
        if shared.plan is None: raise RuntimeError('No initial prediction')
        if args.execute:
            run_servo(control,feedback,shared,recorder,cfg,lower,upper,args.control_gripper)
        else:
            print('LIVE PREDICTION ONLY: receive + gripper GET, no control interface or SET.',flush=True)
            started,seen = time.monotonic(),0
            while time.monotonic()-started < cfg['duration_seconds']:
                shared.check()
                with shared.lock: plan=shared.plan
                if plan.sequence != seen:
                    seen=plan.sequence
                    remaining, skipped, _ = remaining_plan(plan,time.monotonic(),cfg['action_hz'],cfg['execution_steps'])
                    print('prediction',seen,'skipped',skipped,'remaining',len(remaining.actions),
                          'first usable target deg',np.round(np.degrees(remaining.actions[0,:6]),3).tolist(),
                          'gripper',round(float(remaining.actions[0,6]),3),flush=True)
                shared.stop.wait(.05)
        shared.check()
        recorder.event('complete')
        result = 0
    except BaseException as exc:
        shared.fail(exc)
        try: recorder.event('aborted',reason=str(exc))
        except Exception: pass
        print('ABORTED:',exc,flush=True)
    finally:
        shared.stop.set()
        if control is not None:
            try: control.stopJ(.5)
            except Exception as exc: print('Final stopJ unconfirmed:',exc,flush=True)
            try: control.stopScript()
            except Exception as exc: print('stopScript failed:',exc,flush=True)
            try: control.disconnect()
            except Exception: pass
        for worker in workers:
            worker.join(2)
            if worker.is_alive(): shared.fail('Worker did not shut down: '+worker.name)
        if client is not None: client.close()
        if gripper is not None: gripper.close()
        if receive is not None: receive.disconnect()
        recorder.close()
        write_json(folder/'result.json',dict(completed=result==0 and shared.error is None,
            reason=shared.error,execute=args.execute,control_gripper=args.control_gripper,automatic_return=False))
    return result if shared.error is None else 1

if __name__ == '__main__':
    try: raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        print('Deployment startup failed:',exc,flush=True)
        raise SystemExit(1)
