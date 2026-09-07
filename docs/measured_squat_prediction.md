# Measured squat prediction experiment

`configs/collection/ready_grasp_measured_prediction.json` retains the timely
coupled controller and changes its prediction model only. It is experimental;
no default or trained checkpoint is replaced.

The stationary squat command does not imply purely vertical pelvis motion.
Across the frozen regression batches on 2026-09-06, successful squat
handovers moved backward by approximately 7–8 cm. The median requested minus
delivered descent was 14.8 mm, and the median backward/vertical displacement
ratio was 0.781. The previous coupled predictor ignored this retreat and
predicted the entire requested descent. This can invalidate reachability and
squat-height choices even if the later local servo uses live observations.

The candidate predicts:

```
delivered_drop = max(0, requested_drop - 0.015 m)
backward_displacement = 0.8 * delivered_drop
```

The backward direction follows the current horizontal body heading. Both
initial planning and cached-plan revalidation use the same projection.
The execution command remains stationary. Live grasp geometry, comfort,
collision and carry checks remain authoritative. The estimate does not
command an extra retreat or bypass a failed execution check.

Calibration comes from regression cases, with no independent holdout results
used. This is an empirical estimate for this Sonic controller, scene and
height range, not a universal humanoid model. It does not predict torso pitch,
roll, uncertainty or transient braking motion. New tasks, controller versions
or different squat ranges require measuring those errors again.

Zero-valued prediction parameters preserve previous configurations.
Evidence: `outputs/experiments/policy_goal_20260906/squat_motion_audit.json`
and the exact-source batch snapshots in the same experiment directory.
