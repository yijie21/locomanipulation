# Collection transition timing

The retained default is `configs/collection/ready_grasp_coupled.json`.
The opt-in experiment is `configs/collection/ready_grasp_coupled_timing.json`.
Both preserve the coordinated squat trajectory and nominal grasp point 2 cm
below the bottle center. This work does not accelerate videos or resample
demonstrations to simulate faster execution.

## Changes and checks

`timely_capture` accepts the existing measured capture region immediately,
instead of chasing the preferred 15 mm region for up to five seconds. A
0.25-second stability check remains. Relative speed must be below 35 mm/s
to start counting and to close; the count survives small fluctuations below
45 mm/s. Invalid geometry, collision, faster motion or a sampling gap above
0.1 seconds resets it. The old configuration retains its previous gate.

For this experiment, Dex1 lift requires the complete closing ramp and
three consecutive observations of contact on both gripper fingers. A wrist
or one-finger touch is insufficient. The first experimental batch exposed
that the previous contact latch could start lifting before full closure,
which is especially unsuitable for the wider capture region. Two-finger
contact is still not a force/friction guarantee: the full lift and carry
verification remains authoritative.

`brake_preplan` can overlap candidate IK with braking, followed by current
geometry, floor and terminal collision checks. A stale/invalid candidate
falls back to normal planning. It is disabled in the new configuration:
the first batch did not establish a braking-time improvement. Planning is
synchronous during braking, not an asynchronous navigation planner.

The remaining physical base stabilization, the squat actuator's response
delay, the closing ramp and intermediate lift/stand checks are not removed.
This is not a claim of zero pauses or of solved VLA closed-loop stalling.

## Training boundaries

After retreat and before the final two-second hold, collection writes
`execution.metrics.training_tail.end_seq`. Raw recording and success/failure
verification continue unchanged. The converter's `--trim-verification-tail`
joins that sequence against each episode's `priv_seq` sidecar and retains
one contiguous prefix. The exact same row selection selects video frames.
It refuses missing/out-of-range boundaries, failed episodes, invalid joins,
or combination with `--drop-idle`. It never removes intermediate pauses.
The pilot now selects the retained coupled configuration by default and
enables this terminal trimming for newly collected data.

At 25 Hz, 40 steps span 1.6 seconds; a 0.4-second refresh spans about 10
steps. A long continuous action can span multiple refreshes. Expert prefix
audits cannot establish what a trained VLA will predict or execute.

## Reproduction and evidence

- First experiment: `outputs/diag/coupled_timing_20260906/`; 4/8 formal
  attempts succeeded. It enabled both early capture and braking preplanning,
  without the new bilateral/full-closure lift check. It failed acceptance.
- Revised experiment: `outputs/diag/coupled_timing_v2_20260906/`; frozen
  configuration and controller copies are in `source/`. All attempts,
  including failures, are retained. Initialization warmup is separate from
  the eight formal cases and is excluded from review videos.
  Result: 6/8 formal attempts succeeded; `angle` slipped during carry and
  `base_correction` exceeded local IK limits. It has not passed acceptance
  and does not replace the retained default. The six successful records
  passed all 32 dataset checks after terminal trimming; 81 relevant unit
  tests passed. No training or learned-policy evaluation was started.
- `tools/audit_collection_timing.py ROOT` produces `timing_audit.json` with
  stage durations, first-safe-to-close time, actual squat delivery delay and
  0.4-second windows of expert command changes. Its command thresholds are
  diagnostic heuristics, not a physical-motion or learned-policy guarantee.
- `test_measured_grasp` exercises the actual local-control entry: a safe
  18 mm offset must close within 0.4 seconds without chasing IK precision.
  Tests also cover hysteresis, unsafe geometry/contact, stale candidate
  rejection, and contiguous training boundary selection.
- The first batch's four successes were converted into `conversion_check/`
  solely to verify conversion: 32 dataset checks passed, including video
  alignment, grasp causality and action/state timing. No model was trained.

Final revised-batch results are recorded in the experiment's `README.md`.
