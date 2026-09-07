# World-level bottle entry experiment

Opt-in configuration: `configs/collection/ready_grasp_level_bottle.json`.
The retained coupled, timing and continuous configurations remain available.
This experiment has not passed expert acceptance and is not the pilot default.

A full-state replay of `candidate_01_timely/local_far` under
`outputs/experiments/policy_goal_20260906/` identified wrist-shell contact
0.67 seconds before closing. The open fingers were not yet touching the
bottle. The palm entry axis pointed down by about 19 degrees. Accepting a
grip-center region alone did not guarantee clearance between the bottle neck
and the wrist housing.

The new configuration chooses a horizontal world entry and jaw axis while
preserving the template's horizontal bearing. The planning and measured
alignment paths use the same rotation helper. This is a nominal preference;
the actual jaw geometry, arm joint comfort and collision checks still gate
execution. It keeps the grip point 2 cm below the object center and preserves
the existing capture region.

`TransportGuard` can now allow target contact only on finger geometry.
The level-bottle configuration enables this option, so a grasp allowance
cannot exempt wrist/forearm/shoulder contact with the bottle. Furniture checks
and collision margins are unchanged. The option defaults off for legacy
configurations.

The first level-entry batch failed 0/8 at a pre-existing planner/executor
mismatch: planning accepted orientation residual up to 0.30 rad while local
execution rejected over 0.15 rad. A recorded failed target re-solved to
2.271 mm position error and 0.150909 rad orientation error. The level-bottle
executor now shares the planner's 20 mm / 0.30 rad IK residual bounds.
These are template residuals, distinct from measured jaw tilt or actual
capture clearance. Existing configurations retain their previous local bound.

The third fixed-case batch tests this correction. No learned-policy success
is established by the geometry tests or expert videos.

Related validation improvements:

- `--stand-retries 0` disables reset retries for fixed-case acceptance;
  default 1 preserves the previous collection behavior. All attempts remain
  in the manifest. Primary success must not be replaced by eventual success.
- The timing auditor now includes attempts that failed before closing.
- `StateTrace.start` flushes the prior unsaved attempt before a new trace,
  matching the video recorder and preserving failure evidence across resets.
- Third-person video time follows simulation publish sequence at fixed
  12.5 fps. Use shared sequence numbers when aligning control wall-time
  events to video; compression preserving FPS is not proof of wall-clock
  playback equivalence.
