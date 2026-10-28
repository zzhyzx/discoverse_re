# Current source snapshot (2026-10-29)

This snapshot continues the private repository initial snapshot history. Models, datasets, generated reports and camera bridge session credentials are excluded. Checkpoint paths in configurations require separate local model files.

## Submodule changes

The recorded submodule revisions use the previous snapshot baselines. Apply these patches after fetching those revisions; they include the current local committed and uncommitted source changes. Do not apply the older local_changes/policies_act patch in addition.

```bash
git submodule update --init --recursive
git -C policies/act apply --check "$PWD/local_changes/snapshot_20261029/policies_act.patch"
git -C policies/act apply "$PWD/local_changes/snapshot_20261029/policies_act.patch"
git -C submodules/lerobot apply --check "$PWD/local_changes/snapshot_20261029/submodules_lerobot.patch"
git -C submodules/lerobot apply "$PWD/local_changes/snapshot_20261029/submodules_lerobot.patch"
git -C submodules/diff-gaussian-rasterization apply --check "$PWD/local_changes/snapshot_20261029/submodules_diff-gaussian-rasterization.patch"
git -C submodules/diff-gaussian-rasterization apply "$PWD/local_changes/snapshot_20261029/submodules_diff-gaussian-rasterization.patch"
```

Patch baselines:
- `policies/act`: `d8430bf91a55b0b501e2ed9eea398fd6a93575d0`
- `submodules/lerobot`: `cf86b9300dc83fdad408cfe4787b7b09b55f12cf`
- `submodules/diff-gaussian-rasterization`: `8829d14f814fccdaf840b7b0f3021a616583c0a1`
