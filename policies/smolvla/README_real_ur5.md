# UR5 pure-real SmolVLA baseline

Run from `/home/boluo/DISCOVERSE` in the `discoverse` environment.
No robot connections, simulator, downloads, uploads, or real-arm commands.

```bash
conda activate discoverse
python policies/smolvla/train_real_ur5.py
```

Defaults: original local smolvla_base, raw D436+D405 RGB, 40 training episodes
(12,862 frames), 10 held-out episodes (2,956 frames), batch 4, 10,000 updates,
checkpoint/validation every 1,000 updates, terminal log every 50 updates.
Output gets a new timestamped directory under policies/smolvla/checkpoints.
An existing output directory is refused for a fresh run.

The dataset's action is absolute joint radians [q1..q6] plus gripper 0=open,
1=closed. State is actual joint radians plus recorded gripper feedback, NOT
the gripper command. TCP pose is excluded from policy inputs. Both cameras
keep their named keys and raw preprocessing; route-A rectification is NOT used.
Prediction chunk is 50 frames. n_action_steps=10 is an inference setting only;
it is not a verified real-robot deployment controller or safety guarantee.

Training normalization is explicitly aggregated from the 40 training episodes
only. The full dataset is loaded for correct episode indexing in this old
LeRobot checkout, but a frame Subset restricts all optimization to training
episodes. Validation reuses training statistics and uses 40 spread frames
per held-out episode (400 total), not every held-out frame. Validation flow
loss uses fixed RNG seeds; it is not grasp success rate.

```bash
# Short test, separate timestamped output; two updates and one validation batch
python policies/smolvla/train_real_ur5.py --smoke --batch-size 4

# Customize total updates and batch size
python policies/smolvla/train_real_ur5.py --steps 10000 --batch-size 4

# Resume YOUR actual run, replacing the placeholder path below
python policies/smolvla/train_real_ur5.py \
  --resume /absolute/path/to/run/checkpoints/last --steps 10000
```

`--steps` is the total target, not additional steps. Resume restores weights,
optimizer, scheduler, RNG and saved settings; shuffled data iteration restarts,
so it is not bit-identical to an uninterrupted run. Extending total steps does
not restart warmup or redesign the saved learning-rate schedule. Resume only
trusted checkpoints. Use this wrapper rather than the upstream train.py so
the train-only statistics and episode-index safeguards remain active.

Artifacts: metrics.jsonl (each update), real_training.json (split and semantics),
train_stats.json, checkpoints/NNNNNN/pretrained_model and training_state.
Only --smoke creates smoke_pass.json. These short-test weights are not trained
task policies; start the real experiment from smolvla_base, not a smoke run.

Model backbone is loaded from the existing local Hugging Face cache. If moved,
provide --backbone /complete/local/SmolVLM2-directory on a fresh run. Offline
missing files fail rather than download. The existing Transformers pad-token
configuration warning has not been removed by modifying model assets.

No source data or upstream LeRobot files are changed by this entry point.
