# Continuous arrival and capture experiment

Explicit configuration: `configs/collection/ready_grasp_continuous.json`.
The prior coupled and timing configurations remain available and the pilot
default is not changed by this experiment.

The three requested changes are implemented together:

1. Navigation commands decelerate over the last 25 cm relative to the nominal
   stop depth, from at most 0.30 toward 0.12 m/s. Inside 55 cm of the object,
   the right arm starts a bounded high pregrasp approach while walking and
   continues during braking. Its reference advances at most 1.5 mm per tick;
   the initial wrist attitude is retained. Current and 0.1-second projected
   base translation are screened before sending the arm reference. The base
   still needs speed below 0.06 m/s for 0.25 seconds before crouching.
   During squat startup, arm progress can lead measured descent by a bounded
   20% of the transport path, with each increment collision-screened.
2. On reaching the capture region, the controller anchors the measured palm
   relative to the object and re-solves arm IK against the actual base pose.
   The reference advances at most 0.8 mm/tick with 15 mm lead and bounded
   tracking compensation. Actual wrist rotation is compensated at the grasp
   center, with at most 6 mm per-axis interior preference, so a palm position
   held in world coordinates does not mask a rotating finger offset.
   It follows at most 10 mm of object translation;
   a displaced object is not chased. This replaces holding fixed joints in
   the opt-in branch.
3. Three consecutive safe low-relative-speed observations permit entry to
   gradual closure. Relative velocity is measured in the actual jaw frame,
   consistent with capture geometry, so coordinated wrist rotation and palm
   translation that preserve the grasp center do not cause a false pause.
   The same object-relative reference is carried into the
   existing servo. Closing advances only on a fresh observation with valid
   geometry and relative speed below 45 mm/s; otherwise the arm still
   compensates while closing progress pauses. Losing this condition for
   more than 0.8 seconds during closure aborts. Before closure, at most two
   returns to the same short alignment are allowed if the object stayed in
   place; this never stands or repositions the base. The 2-second full closing ramp, bilateral
   finger contact before lift and complete carry verification remain.

These are sampled kinematic guards, not a whole-body dynamics guarantee.
The configuration does not relax the capture region, accelerate the video,
change dataset FPS, remove failed attempts, or start VLA training. A moving
reference does not itself prove that physical motion or VLA prediction will
follow within one 0.4-second execution interval.

Relevant tests: `test_capture_feedback`, `test_measured_grasp`,
`test_coupled_grasp_policy`, plus the existing recovery, transition and guard
tests (88 relevant tests in the final pre-collection check).

Initial diagnostic artifacts: `outputs/diag/continuous_grasp_20260906/`;
the run was interrupted after exposing the wrist-offset compensation bug.
The failing rotation regression reproduced a 15 mm grasp-center error before
the correction. That run's partial records are retained and not mixed into
the final eight-case gallery.

The intermediate `outputs/diag/continuous_grasp_v2_20260906/` run is also
retained: it exposed the inconsistent coordinate frame in the closing speed
gate and was interrupted to correct it. A dedicated regression reproduced
false closing inhibition with a stationary grasp center before that fix.

Final batch: `outputs/diag/continuous_grasp_final_20260906/`. Its `source/`
contains the frozen controller files; the manifest retains every outcome.
The final README and video gallery report actual results, including failures.

Final result: 3/8 formal cases passed the complete grasp/carry acceptance.
`local_left`, `local_right`, and `local_near` succeeded. `angle` was inhibited
by the relative-speed gate despite no recorded closure-geometry violation;
`near_offset` lost closing conditions; `height` exceeded the carry displacement
threshold; `local_far` and `base_correction` failed recovery. All eight videos
are retained. This experiment is not promoted to the default configuration
and no VLA training was started. Active compensation is present in the trace,
but zero pauses, better robustness and VLA learnability are not established.
