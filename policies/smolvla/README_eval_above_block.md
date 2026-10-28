# Above-block diagnostic evaluation

`eval_sim_ur5_lift.py --start-mode above-block` initializes the open gripper
directly over the red block, using the collection IK reference sequence and
cube-aligned orientation. Default TCP height is 0.18 m above the tabletop
(about 0.155 m above block center), controlled by `--above-height`.

Initialization holds the pose for 0.5 simulated seconds with the existing
gravity-compensated controller. It checks position error, joint speed,
gripper opening, block displacement and the existing obstacle collision
predicate, then refreshes both cameras and starts policy time at zero.
Initialization errors are recorded as failed trials, never silently resampled.

The policy controls all subsequent actions; no expert stage signals or object
coordinates are fed into the observation. The known object pose is used ONLY
to prepare the initial position and orientation. This evaluates the partial
task (descend, close, lift, hold), not autonomous acquisition from home.

```bash
cd /home/boluo/DISCOVERSE
MUJOCO_GL=egl python policies/smolvla/eval_sim_ur5_lift.py \
  --checkpoint policies/smolvla/checkpoints/sim_ur5_padding_fix_from023000_20261021/checkpoints/024000/pretrained_model \
  --output data/smolvla_above_block_repeat \
  --episodes 5 --seed 20261021 --action-steps 10 \
  --max-command-rate 0.9 --start-mode above-block --above-height 0.18
```

Output must be a NEW directory. Default `--start-mode home` retains the full-task
test. Weights are not changed. Simulation pauses for inference.
For recovery tests, add `--initial-offset-xy DX DY` in WORLD meters or
`--initial-yaw-deg ANGLE` around WORLD Z relative to cube alignment. These
modify only the initial arm pose; the object and subsequent policy control
are unchanged. Bounds: +/-0.1 m per axis and +/-45 degrees. Nonzero offsets
require above-block mode. Initial pose perturbations are recorded in protocol
and initialization metadata. Position and orientation can be varied separately.
Use `--block-yaw-deg 0 --initial-yaw-deg 0` to fix the cube faces along
the table axes and avoid adding an initial gripper rotation. The scene RNG
still consumes the original yaw sample, preserving the same XY layouts
across conditions. Omitting `--block-yaw-deg` retains random cube yaw.
Each episode contains initialization.json, observation/action logs, prediction
trace and two MP4s. Diagnostic episodes are excluded from training eligibility.

## 2026-10-22 result

Same repaired 024000 checkpoint, five object layouts/yaws, per-episode policy
seeds, action stride 10, slew limit 0.9 rad/s, 20-second timeout and unchanged
success criterion:

| Start | Success | Collision stops |
|---|---:|---:|
| Home | 0/5 | 1/5 |
| Above block | 5/5 | 0/5 |

Above-block durations: 6.80, 5.73, 10.13, 9.03, 4.80 simulated seconds.
Peak block-bottom clearances: 26.05, 23.55, 25.31, 23.67, 23.91 cm.
All five had a recorded stable-lift span of 1.033 seconds. All ten videos
decoded and matched log lengths. Initial TCP errors were below 0.1 mm.

Results: data/smolvla_above_block_024000_rate09_20261022.
Baseline: data/smolvla_stride10_fixedseed_20261022.

This supports focusing investigation on approach, orientation alignment and
the transition into descent. Position, orientation and motion history were
changed together by initialization, so this does not isolate a single cause.
Five trials do not establish generalization or full-task success. Future
diagnostics can introduce lateral/yaw offsets and start from real expert
transition snapshots to identify the recovery region.

## Larger translation-only test

For example, start 8 cm from the cube along world +X:

```bash
MUJOCO_GL=egl python policies/smolvla/eval_sim_ur5_lift.py \
  --checkpoint policies/smolvla/checkpoints/sim_ur5_padding_fix_from023000_20261021/checkpoints/024000/pretrained_model \
  --output data/smolvla_translation_repeat_x_plus8 \
  --episodes 5 --seed 20261021 --action-steps 10 --max-command-rate 0.9 \
  --start-mode above-block --above-height 0.18 \
  --block-yaw-deg 0 --initial-yaw-deg 0 --initial-offset-xy 0.08 0
```

The 2026-10-22 translation suite uses 6, 8 and 10 cm in each of four world
axis directions plus a new zero-yaw aligned baseline, five layouts each.
Results: `data/smolvla_translation_large_20261022`.
Run `python scripts/analyze_ur5_translation_eval.py data/smolvla_translation_large_20261022`
to verify videos and summarize pre-close XY correction. Distances are relative
to the original block center, not a block displaced by contact. A reduction
of at least 50% and at least 3 cm is an explicit diagnostic threshold, not a
general grasp tolerance or evidence that vision alone caused the correction.

The audit also reports correction up to the first 5 cm lift (or the end of
the trial if no lift occurs), including reopening/retry cycles. This avoids
classifying a successful retry as having no lateral recovery. Neither XY
metric alone verifies a suitable grasp height, orientation, or stable grasp.

To resume an interrupted condition in a NEW output directory, retain the
same settings and total `--episodes 5`, and add `--episode-start N`, where N
is the first incomplete zero-based episode. Earlier scene resets are replayed
without policy rollout to preserve layout sampling. For example, start 4
completes only episode 004. The audit merges condition directories by offset
and rejects duplicate completed episodes. Incomplete old episode folders are
retained but are not included in the completed-episode summaries.
