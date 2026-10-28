# UR5 real-data SmolVLA deployment on Windows + WSL

Windows owns the two USB RealSense cameras. WSL owns the existing CUDA model,
RTDE receive/control and Robotiq TCP connection. No USB forwarding is required.
The old motion/gripper scripts and trained weights are unchanged.

Entry: `policies/smolvla/deploy_real_ur5.py`.
Settings: `policies/smolvla/configs/ur5_real_deploy.json`.
Windows camera service: `scripts/ur5_windows_camera_bridge.py`.

## 1. Windows camera environment (once)

The inspected Windows base Python is 3.14 and lacks `pyrealsense2`. In an
Anaconda PowerShell Prompt, create a separate camera environment:

```powershell
conda create -n ur5_camera python=3.11 pip -y
conda activate ur5_camera
python -m pip install pyrealsense2
Set-Location '\\wsl$\Ubuntu-Recover\home\boluo\DISCOVERSE'
python scripts/ur5_windows_camera_bridge.py --list
```

This does not modify the WSL `discoverse` environment. The camera service
itself needs no PyTorch, LeRobot or OpenCV.

Defaults, taken from the supplied intrinsics:
- external D436: `405622072928`
- wrist D405: `262422272105`

If different devices are connected, change the serials in the deployment JSON
and supply matching `--front-serial` / `--wrist-serial` to the Windows service.

## 2. Start Windows cameras

Keep this PowerShell window running:

```powershell
conda activate ur5_camera
Set-Location '\\wsl$\Ubuntu-Recover\home\boluo\DISCOVERSE'
python scripts/ur5_windows_camera_bridge.py
```

Wait for `Camera bridge ready`. Close RealSense Viewer/other camera applications
if they have the devices open. The service requests RGB8 640x480 at 30Hz.
It binds TCP port 8765 and uses a fresh per-session token saved at
`data/real_ur5/camera_bridge/session.json`; keep this token file local/private.
For WSL NAT, allow this Python application's inbound connection only on the
appropriate private/WSL network. Do not disable the firewall globally.

The WSL client tries localhost first and then the default-route Windows host.
If necessary, explicitly supply `--camera-host <Windows IPv4>`.
Two cameras are paired by host receipt times; this is not hardware exposure
synchronization. Sensor frame numbers/timestamps and receipt skew are logged.
Raw pixels cross the bridge without JPEG compression or RGB/BGR conversion.

## 3. WSL live prediction (no arm/gripper commands)

In a WSL terminal:

```bash
conda activate discoverse
cd /home/boluo/DISCOVERSE
python policies/smolvla/deploy_real_ur5.py --duration 20
```

This loads checkpoint `010000` and its own `train_stats.json`. It connects
RTDE receive and uses gripper GET only; it does not create RTDE control.
The robot must be initialized, stationary, and not running another program
or freedrive. The gripper must already be activated.

Robot IP defaults to `192.168.1.101`, gripper port to `63352`. Change the JSON
when the network differs. Camera access and robot Ethernet access are separate.

The trained state is actual six joint radians + normalized gripper feedback.
Recorded feedback is consistent with `POS/255`; this is the default mapping.
Confirm it against the original collector before moving. The JSON supports
different raw open/closed endpoints, but those must reproduce training values,
not introduce a new physical opening-width calibration.

## 4. Actual execution

The median training start, from the 40 training episodes, is:

```text
[44.651057, -74.312014, 76.460799, -91.346656, -89.610081, -3.046336] degrees
```

Place the robot at a verified compatible start using the pendant or an existing
verified saved-pose route. These angles alone do not establish a collision-free
route from another posture. The deployment entry never automatically repositions.
Do not reuse an old A/B pose with wrist1 near +260 degrees: this checkpoint's
training branch is near -91 degrees. Do not wrap the measured angles automatically.

Arm only, keeping model gripper commands disabled:

```bash
python policies/smolvla/deploy_real_ur5.py --execute --duration 5
```

Arm and model gripper for grasp/lift:

```bash
python policies/smolvla/deploy_real_ur5.py --execute --control-gripper --duration 20
```

Execution requires typing `MOVE` after preview. Start with an open empty gripper
for model gripper control. Ensure the tool/payload, swept path and operator/E-stop
arrangements are appropriate. There is no swept-path collision detector.

Optional `--start-pose /absolute/path/Start.json` accepts existing `snapshot.q_rad`
files for this robot IP; they must still be near the training start. Optional
`--no-video` disables encoding while retaining camera/feedback/prediction logs.

## Timing and bounded control

- Absolute six joint targets in radians; no deltas, IK or angle wrapping.
- Predict 50 actions, use the first 10 at the recorded 30Hz time scale.
- Separate camera, inference, gripper and recording workers; CB3 servoJ at 125Hz.
- Piecewise-linear action-time interpolation and explicit command velocity/
  acceleration limits. servoJ speed/acceleration arguments are not relied on.
- Initial command limits: 0.03rad/s and 0.05rad/s², matching the earlier low-speed
  tests. These may slow the learned task; a successful full grasp is not promised.
- Prediction envelope from training action min/max plus 3 degrees, max target
  offset from actual 8 degrees, max tracking error 3 degrees.
- Cameras/feedback/predictions have freshness limits. Inference deadline 0.3s.
  A missing next chunk allows up to 0.05s decelerating hold, then stops; old chunks
  are never looped indefinitely. Active prediction age is also limited.
- Controller watchdog minimum 10Hz and host control-gap limit 0.08s. Controller
  watchdog setup/kicks must succeed; paused/stopped programs abort.
- Gripper thresholds 0.35/0.65, sustained 0.2s; raw SPE=32, FOR=16.
  Arm progression pauses during closure until stable closing contact is reported.
  Endpoint without contact aborts. Contact status is a proxy, not a grip guarantee.
- Release is disabled by default because these held-out demonstrations do not
  include release. A sustained model release request aborts instead of dropping.

The warmed local offline forward took about 0.255s in implementation validation.
The inference worker also warms its own CUDA handles before publishing a plan.
Live execution performance and WSL scheduling are not yet verified on the robot.
If a deadline is exceeded, inspect recorded timings rather than disabling guards.

## Stop and output

Press Ctrl+C in the WSL deployment terminal for cleanup: servoStop/stopJ are
attempted, then the control script is stopped. No automatic return or gripper
reopening. When model gripper commands were sent, GTO=0 is attempted on shutdown;
inspect the held object because holding behavior is not guaranteed by this stop.
Physical E-stop remains the fallback if a software/network stop is ineffective.
Stop the WSL entry before stopping the Windows camera service.

Outputs get unique directories:

```text
data/real_ur5/smolvla_deploy/<timestamp>/
  run_config.json
  events.jsonl
  front.mp4
  wrist.mp4
  result.json
```

Videos are lossless RGB encoded via PyAV/libx264rgb, not system ffmpeg. Logs store
camera IDs/timestamps, actual state, every 50-action raw prediction, the 10-action
chunk used, limited execution targets, gripper registers and termination reason.
Video timestamps use the estimated local frame receipt timeline at a 30Hz grid;
use JSON timestamps for precise diagnostics.

## Offline implementation checks

These checks do not connect hardware:

```bash
python policies/smolvla/deploy_real_ur5.py --check-model
python scripts/tests/test_ur5_deployment.py
```

`--check-model` uses the first observation of held-out episode 5 from the local
converted real dataset. Tests use synthetic TCP RGB frames and fake RTDE/gripper
objects, including default no-control behavior, stale data, watchdog failure,
velocity/acceleration limits and video finalization. They are not a hardware
acceptance test or a closed-loop task success measurement.

Implementation validation on 2026-10-09: 21 fake-device tests passed, checkpoint
010000 produced finite 50x7 actions with training statistics, and a native Windows
synthetic bridge transmitted three exact RGB pairs to WSL in about 0.09s via the
default-route host. Localhost was unavailable in the current NAT setup; automatic
fallback worked. No real camera or robot was accessed by these checks.
# 推理较慢时的异步执行（2026-10-12）

当前入口保留全部 50 步预测，`execution_steps=30` 表示每次最多采用 30 步，动作时间间隔保持 1/30 秒。推理线程连续更新；控制线程收到新批次即切换，不等旧批次耗尽。按观测年龄裁掉已经过去的动作，因而 0.813 秒延迟后通常只剩约 26 步，而不是再从第 0 步执行一整秒。跨批次沿用速度/加速度限制器，重新检查相对实际关节的目标偏差。

`max_inference_seconds=1.05`；`max_observation_age_s=1.65` 限制预测依据的年龄。实时相机新鲜度、关节速度、跟踪误差、看门狗限制没有放宽。50 步对应的原始预测时间范围仍是 1.667 秒，耗尽且没有新批次则停止。这是时间裁剪与限速衔接，不是经过训练的延迟补偿或轨迹连续性保证；抓取效果需要现场验证。推理长时间接近 1 秒时仍可能耗尽预测范围，不能据超时预算推断一定连续运行。

`--check-model` 现在运行 2 次预热与 20 次测量，并报告 median/p95/max。先以 `--duration 20 --no-video` 完成实时无运动预览。旧文中关于固定 10 步/0.3 秒的说明以本节为准。

# 当前批次衔接修订（取代下文“按时间裁剪”说明）

预测目标相对观测/实际关节的上限现为 12 度（原为 8 度）。每批最多采用前 30 步，收到新批次即重新以限速器当前位置衔接，绝不因为推理花费时间就跳过尚未执行的接近动作。采纳时实际关节相对该批观测变化超过 3 度则停止；预测观测年龄限制仍为 1.65 秒，相机新鲜度与原有速度/加速度、跟踪误差、看门狗均不变。没有新批次时不会重放已执行的批次。

无运动预览遇到单纯目标偏差超限会打印 PREVIEW WARNING 并写入 prediction_rejected，继续收集预测；这不代表执行合格。执行模式仍拒绝超限，错误包含动作序号、关节序号、实际/目标角度、差值与上限。轨迹包络、非有限数值、相机/推理异常依然中止预览。保持 30Hz 目标序列但实际轨迹受限速影响，必须现场验证抓取效果。

