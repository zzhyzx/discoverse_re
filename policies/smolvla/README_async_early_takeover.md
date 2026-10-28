# Async early takeover v1
Shared rule: scripts/ur5_deploy_core.py takeover_reason.
Deployment run_servo already has a continuous inference worker; adoption now uses shared rule.
Low-speed simulation now uses one background inference worker and latest-result mailbox.
Only one worker touches the model. Observation state/images are copied before submission.

Normal: >=10 executed action steps, aligned, no pending/moving gripper.
Early: active plan held >=0.25s AND no progress for 0.30s (no pending debounce),
or <=0.45s remaining before observation expires. Requires a newer prediction.
Never adopts while the gripper is moving. Urgent adoption cancels unissued debounce;
it does not reopen or reverse an already issued command.
Freshness, state drift, finite targets, joint envelopes, watchdog and stop checks remain.
Limiter q/velocity preserved across adoption; interpolation restarts at current commanded q.
Speed .03rad/s; acceleration .05rad/s²; observation age1.65s; prediction delta12deg unchanged.
Scheduling is versioned separately from the v1 training-data motion contract; no retraining performed.

Simulation is paced at most30Hz; rendering may slow it further. Freshness uses wall time,
not simulation time. This is an offline physics test, not125Hz hardware certification.
External camera freshness and real gripper status still checked by deployment workers.
No real robot connection or movement was performed for this change.

Test command (from project root, discoverse env):
MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python policies/smolvla/eval_sim_ur5_low_speed.py --checkpoint policies/smolvla/checkpoints/sim_real_reference_low_speed_v1/checkpoints/010000/pretrained_model --reference-data data/ur5_real_reference_low_speed_v1 --output data/ur5_low_speed_async_010000_full_v1 --episodes 10 --seconds 180

Episodes save two videos, obs_action.json, prediction_trace.json, transitions.json.
summary.json records task result and takeover counts; early takeover is NOT grasp success.
