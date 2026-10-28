"""Bounded, authenticated raw-RGB protocol. No pickle or robot commands."""
import json
import math
import warnings
import socket
import struct
import time

ROLES = ('front', 'wrist')
FRAME_BYTES = 640 * 480 * 3

class CameraTimingError(RuntimeError):
    """Complete packet received, but unsuitable for control; stream remains framed."""

def read_exact(sock, count):
    parts = []
    while count:
        part = sock.recv(min(count, 1024 * 1024))
        if not part:
            raise ConnectionError('Camera bridge disconnected')
        parts.append(part)
        count -= len(part)
    return b''.join(parts)

def send_packet(sock, metadata, frames=()):
    header = json.dumps(metadata, allow_nan=False).encode('utf8')
    if len(header) > 16384:
        raise ValueError('Oversized camera metadata')
    sock.sendall(struct.pack('!I', len(header)) + header + b''.join(frames))

def read_packet(sock):
    count, = struct.unpack('!I', read_exact(sock, 4))
    if not 1 <= count <= 16384:
        raise ValueError('Invalid camera metadata size')
    metadata = json.loads(read_exact(sock, count))
    if metadata.get('error'):
        raise RuntimeError('Camera bridge: ' + str(metadata['error']))
    if metadata.get('version') != 1 or metadata.get('shape') != [480, 640, 3]:
        raise ValueError('Camera protocol/shape mismatch')
    frames = {role: read_exact(sock, FRAME_BYTES) for role in ROLES}
    return metadata, frames

class CameraClient:
    def __init__(self, host, session, timeout=.4):
        self.sock = socket.create_connection((host, int(session['port'])), timeout=timeout)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.token = session['token']
        self.previous = {}

    def get(self, max_age, max_skew):
        started = time.monotonic()
        request = json.dumps({'token': self.token, 'operation': 'frames', 'after': self.previous}).encode() + b'\n'
        self.sock.sendall(request)
        metadata, frames = read_packet(self.sock)
        age = float(metadata['max_receipt_age_s'])
        skew = float(metadata['receipt_skew_s'])
        elapsed = time.monotonic() - started
        for role in ROLES:
            number = int(metadata['cameras'][role]['frame_number'])
            if number <= self.previous.get(role, -1):
                raise RuntimeError(f'Camera frame did not advance: {role}')
            self.previous[role] = number
        # Windows and WSL measure durations on different clocks. Do not subtract
        # a Windows duration from the local round trip or advance the observation
        # timestamp using it. Counting the WHOLE round trip overestimates age,
        # including time waiting for a fresh frame, rather than accepting stale data.
        waited = float(metadata.get('server_wait_seconds', 0))
        if not all(math.isfinite(v) and v >= 0 for v in (age, skew, elapsed, waited)):
            raise CameraTimingError(
                f'Invalid camera timing values: server_age={age!r}s skew={skew!r}s '
                f'request_total={elapsed!r}s server_wait={waited!r}s')
        metadata['server_wait_exceeds_roundtrip_s'] = max(0., waited-elapsed)
        if waited > elapsed and not getattr(self, '_timing_warning_emitted', False):
            warnings.warn(
                f'Camera clock discrepancy: server_wait={waited:.6f}s '
                f'request_total={elapsed:.6f}s difference={waited-elapsed:.6f}s; '
                'using full round trip for conservative image age.', RuntimeWarning)
            self._timing_warning_emitted = True
        transport = elapsed
        metadata['conservative_receipt_age_s'] = age + transport
        metadata['timing_policy'] = 'server_receipt_age_plus_full_local_roundtrip'
        if age > max_age or skew > max_skew or age + transport > max_age:
            raise CameraTimingError(f'Camera receipt age/skew exceeded: {age+transport:.3f}s / {skew:.3f}s; '
                f'server_age={age:.3f}s request_total={elapsed:.3f}s server_wait={waited:.3f}s '
                f'transport_bound={transport:.3f}s')
        # At receipt, elapsed + age has passed since this conservative timestamp.
        # This remains receipt-time freshness, NOT synchronized sensor exposure time.
        metadata['local_observed_monotonic'] = started - age
        metadata['transfer_seconds'] = elapsed
        return metadata, frames

    def close(self):
        self.sock.close()
