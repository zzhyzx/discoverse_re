# SmolVLA action padding repair (2026-10-21)

The local policy loss read `actions_id_pad`, while the LeRobot dataset and training
entry pass `action_is_pad`. Episode-end repeated actions were therefore included
in the loss. The repair uses the actual boolean `(batch, chunk)` mask, excludes
padded timesteps and divides by valid timestep count times action dimension.
The existing 32-dimensional action objective is otherwise unchanged.

Regression tests:

```bash
python scripts/tests/test_smolvla_padding.py
```

Tests cover zero loss/gradient on padding, invariance to padding length, unchanged
unmasked loss, and invalid-mask rejection. A real dataset boundary check confirmed
50 valid actions at episode 0 start and 1 valid plus 49 padded actions at its end.

## Isolated repair trial

```bash
python policies/smolvla/train_sim_ur5.py \
  --resume policies/smolvla/checkpoints/sim_ur5_lift_20261020_234246/checkpoints/023000 \
  --steps 24000 \
  --fork-output policies/smolvla/checkpoints/sim_ur5_padding_fix_from023000_20261021
```

`--steps` is the total update count (1000 additional updates here).
`--fork-output` requires a NEW directory, retains optimizer/scheduler state and
restores RNG as normal resume does. The shuffled data iterator restarts. Original
run files and checkpoints are not overwritten. Manifest records `resumed_from`
and `loss_revision=action_is_pad_valid_timestep_mean_v1`.

The patch changes training/validation loss, not inference. An old checkpoint
does not acquire repaired weights simply by loading the new code. Old versus new
loss values have different definitions and must not be compared as improvement.
A short resumed run is only a diagnostic: it does not undo all earlier training.
Any outcome combines the repair with 1000 extra updates; causal attribution would
require an equal-update unpatched control or matched training from initialization.

All testing here uses local simulation and saved data; no real robot is connected.
