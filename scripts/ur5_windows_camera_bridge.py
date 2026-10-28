"""Windows RealSense RGB bridge for WSL. Camera access only; no robot interfaces."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import threading
import time
from ur5_camera_wire import FRAME_BYTES, ROLES, send_packet

ROOT = Path(__file__).resolve().parents[1]

class Frames:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {}
        self.error = None

    def put(self, role, image, info):
        if len(image) != FRAME_BYTES:
            raise ValueError('Unexpected RealSense RGB buffer size')
        with self.lock:
            self.latest[role] = (image, info)

    def packet(self):
        with self.lock:
            if self.error:
                return {'error': self.error}, ()
            if set(self.latest) != set(ROLES):
                return {'error': 'Waiting for both cameras'}, ()
            items = [self.latest[role] for role in ROLES]
        now = time.monotonic()
        receipts = [item[1]['host_receipt_monotonic'] for item in items]
        metadata = dict(version=1, shape=[480, 640, 3], format='RGB8',
            max_receipt_age_s=now-min(receipts), receipt_skew_s=max(receipts)-min(receipts),
            cameras={role: items[i][1] for i, role in enumerate(ROLES)})
        return metadata, [item[0] for item in items]

class BridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(2)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                line = self.rfile.readline(4097)
                if not line:
                    return
                if len(line) > 4096 or not line.endswith(b'\n'):
                    return
                request = json.loads(line)
                if not secrets.compare_digest(str(request.get('token', '')), self.server.token):
                    send_packet(self.connection, {'error': 'Authentication failed'})
                    return
                if request.get('operation') != 'frames':
                    return
                deadline = time.monotonic()+.3
                after = request.get('after', {})
                requested = time.monotonic()
                while True:
                    snapshot_started = time.monotonic()
                    metadata, frames = self.server.frames.packet()
                    if metadata.get('error'):
                        if metadata['error'] != 'Waiting for both cameras':
                            break
                    elif all(metadata['cameras'][role]['frame_number'] > int(after.get(role,-1)) for role in ROLES):
                        break
                    if time.monotonic() >= deadline:
                        metadata, frames = {'error': 'No new dual-camera frames within 0.3s'}, ()
                        break
                    time.sleep(.003)
                metadata['server_wait_seconds'] = snapshot_started-requested
                send_packet(self.connection, metadata, frames)
        except (OSError, ValueError):
            return

def camera_loop(rs, role, serial, frames, stop):
    pipeline = rs.pipeline()
    try:
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = pipeline.start(config)
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1)
        previous = -1
        previous_timestamp = None
        while not stop.is_set():
            # Device startup can take longer than the runtime freshness limit.
            timeout_ms = 10000 if previous < 0 else 500
            try:
                packet = pipeline.wait_for_frames(timeout_ms)
            except RuntimeError as exc:
                phase = 'startup/first frame' if previous < 0 else 'streaming'
                raise RuntimeError(f'{phase}: {exc}') from exc
            frame = packet.get_color_frame()
            if not frame:
                continue
            receipt = time.monotonic()
            number = frame.get_frame_number()
            if number <= previous:
                raise RuntimeError('Non-advancing camera frame')
            previous = number
            timestamp = frame.get_timestamp()
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise RuntimeError('Non-advancing sensor timestamp')
            previous_timestamp = timestamp
            if frame.get_stride_in_bytes() != 640*3:
                raise RuntimeError('Unexpected RGB row stride')
            frames.put(role, bytes(frame.get_data()), dict(serial=serial,
                frame_number=number, sensor_timestamp_ms=timestamp,
                timestamp_domain=str(frame.get_frame_timestamp_domain()),
                host_receipt_monotonic=receipt))
    except Exception as exc:
        with frames.lock:
            frames.error = f'{role}: {exc}'
        stop.set()
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--front-serial', default='405622072928')
    parser.add_argument('--wrist-serial', default='262422272105')
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--session', type=Path, default=ROOT/'data/real_ur5/camera_bridge/session.json')
    parser.add_argument('--list', action='store_true', help='List devices; no streams')
    parser.add_argument('--check-deps', action='store_true', help='No device access')
    args = parser.parse_args()
    if args.check_deps:
        available = importlib.util.find_spec('pyrealsense2') is not None
        print('pyrealsense2 available:', available)
        return 0 if available else 1
    import pyrealsense2 as rs
    context = rs.context()
    devices = {device.get_info(rs.camera_info.serial_number):
               device.get_info(rs.camera_info.name) for device in context.query_devices()}
    print('RealSense devices:', devices, flush=True)
    if args.list:
        return 0
    if args.front_serial == args.wrist_serial or any(s not in devices for s in (args.front_serial, args.wrist_serial)):
        raise ValueError('Two distinct configured camera serials are required; use --list')
    frames, stop = Frames(), threading.Event()
    token = secrets.token_urlsafe(32)
    with BridgeServer((args.bind, args.port), Handler) as server:
        server.frames, server.token = frames, token
        args.session.parent.mkdir(parents=True, exist_ok=True)
        descriptor = dict(version=1, port=server.server_address[1], token=token,
            front_serial=args.front_serial, wrist_serial=args.wrist_serial,
            width=640, height=480, fps=30, pid=os.getpid())
        temporary = args.session.with_suffix('.tmp')
        temporary.write_text(json.dumps(descriptor, indent=2))
        os.replace(temporary, args.session)
        workers = [threading.Thread(target=camera_loop,
            args=(rs, role, serial, frames, stop), daemon=True)
            for role, serial in zip(ROLES, (args.front_serial, args.wrist_serial))]
        for worker in workers:
            worker.start()
        network = threading.Thread(target=server.serve_forever, daemon=True)
        network.start()
        try:
            deadline = time.monotonic()+20
            while not stop.is_set() and len(frames.latest) != 2 and time.monotonic()<deadline:
                stop.wait(.02)
            if len(frames.latest) != 2 or frames.error:
                raise RuntimeError(frames.error or 'Cameras did not start within 20 seconds')
            print('Camera bridge ready; session:', args.session, flush=True)
            print('RGB8 640x480 30Hz. Receipt-time pairing; not hardware synchronized.', flush=True)
            while not stop.wait(.5):
                pass
            if frames.error:
                raise RuntimeError(frames.error)
        finally:
            stop.set()
            server.shutdown()
            for worker in workers:
                worker.join(1)
            if args.session.exists():
                current = json.loads(args.session.read_text())
                if current.get('token') == token:
                    args.session.unlink()
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print('Camera bridge stopped:', exc, flush=True)
        raise SystemExit(1)
