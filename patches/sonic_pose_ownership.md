# POSE owns the complete joint reference

`sonic_pose_ownership.patch` clears the planner's upper-body override when the
decoded POSE becomes the active motion reference. Previously the override survived the mode
change until another planner message cleared it. A direct handoff could therefore
keep the arms frozen despite receiving a full pose stream.

The manager retains the override while waiting for the first valid pose, so a
mode flag cannot briefly expose the planner's unmodified arms. The caller must
prime the pose buffer before switching. This patch does not blend
encoder outputs or guarantee physical tracking across modes. Planner mode and the
existing emergency-stop behavior are unchanged.

> **Do not apply this file by hand.** It is the second (pose) / third (hand) patch of the series in
> [`docs/setup.md` step 3](../docs/setup.md), which pins the upstream commit, applies all three in order and
> refuses a half-patched tree. The block below records what the patch is for and how it was first applied.

Apply from the repository root and rebuild:

```bash
(cd third_party/GR00T-WholeBodyControl && git apply ../../patches/sonic_pose_ownership.patch)
source scripts/env_deploy.sh
cmake --build "$SONIC_DEPLOY_ROOT/build" -j2
strings "$SONIC_DEPLOY_ROOT/target/release/g1_deploy_onnx_ref" | rg POSE_UPPER_OWNERSHIP_V2
```

Both visual and carry whole-body entries check the selected binary for this capability and
rejects an old build. An alternate `SONIC_DEPLOY_BIN` must contain the same patch.
See [setup](../docs/setup.md) for build dependencies and the separate DDS-domain patch.

Keep the model's default-angle convention when rebuilding. The local SIMPLE
`auto engage` branch changed shoulder-roll offsets to zero; the deployed production
binary uses left `+0.2` and right `-0.2` radians. If using that modified source, apply
`patches/sonic_policy_defaults.patch` before building. Stock source already using
those offsets needs no defaults patch. This is a controller calibration difference,
not a grasp gain; changing it invalidates a behavior comparison.
