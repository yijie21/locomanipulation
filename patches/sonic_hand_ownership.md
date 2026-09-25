# Preserve hand commands when returning from POSE to the planner

`sonic_hand_ownership.patch` makes hand getters follow the selected input mode.
The old manager continued asking the POSE endpoint for hands while the lower-body
planner was initializing. That endpoint had already received its safety reset,
so it returned a default Dex3 pose instead of the primed closed Dex1 command.
A bottle could be lifted successfully and then dropped during the mode change.

The caller must publish a planner frame containing both hand targets before
switching to planner mode. `StreamedArm.stop(preserve_reference=True)` does this.
The patch does not retain stale hands indefinitely or bypass emergency stops;
existing planner timeout and control-reset behavior remains in place.

> **Do not apply this file by hand.** It is the second (pose) / third (hand) patch of the series in
> [`docs/setup.md` step 3](../docs/setup.md), which pins the upstream commit, applies all three in order and
> refuses a half-patched tree. The block below records what the patch is for and how it was first applied.

Apply after the POSE ownership patch and rebuild with the same policy calibration:

```bash
(cd third_party/GR00T-WholeBodyControl && git apply ../../patches/sonic_hand_ownership.patch)
source scripts/env_deploy.sh
cmake --build "$SONIC_DEPLOY_ROOT/build" -j2
strings "$SONIC_DEPLOY_ROOT/target/release/g1_deploy_onnx_ref" | rg PLANNER_HAND_OWNERSHIP_V1
```

The bottle `whole_body` entry requires both capability markers before starting
resources. An alternate `SONIC_DEPLOY_BIN` must include both patches.
See [POSE ownership and calibration](sonic_pose_ownership.md) and
[installation](../docs/setup.md).
