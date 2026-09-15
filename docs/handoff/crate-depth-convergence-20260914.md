# Crate stance depth convergence — 2026-09-14

Scope: the user asked for a depth-axis convergence criterion on the crate visual expert's
walk-in stance, after the 2026-09-14 collection batch showed that the achieved stance, not the
basket placement, separated successes from failures. Usage contract: [pilot guide](../crate_visual_collection.md).
Baseline commit `fcf23d9`; no deploy binary, policy weights, scene or gate thresholds changed.

## Why

`outputs/crate_collect10_20260914` (seeds 41–50, 7/10): placement jitter is ±2 cm / ±5° and
does not separate outcomes. Two of the three failures stood 0.46 and 0.51 m from the crate
centre against a nominal `visual.stand_off` of 0.39; successes stood 0.37–0.44. This matches
the 2026-09-11 finding on the wall scene (29 rounds, zero overlap at ≈0.27 m to the rim).
The visual approach had a lateral stance tolerance but no depth tolerance: it braked at
`stand_off + walk_brake_margin`, waited for a quiet base and handed over wherever the gait had
coasted to.

## Implementation

1. `VisualApproach._stance_converged` (`tasks/crate/visual_approach.py`): once the quiet
   interval is satisfied, the mean visible depth over that interval is compared with
   `stand_off`. Farther than `visual.stand_depth_tolerance` (0.04 m) re-walks with the braking
   allowance halved, at most `visual.stand_depth_retries` (2) times; a near stance is never
   backed away from; retries remain bounded by `walk_timeout`. Every quiet stance is recorded
   in `stand_attempts` (navigation record and attempt result) with `accepted`.
2. `visual.walk_speed_min` (0.35) replaces the hard-coded 0.10 speed floor of the walk
   command. Found while validating 1: the first re-walk commanded 0.10 for 20 s and the base
   moved at 0.01–0.03 m/s (round 06 of batch 2). Across 940 samples of two batches, a 0.10
   command delivered a median 0.011 m/s 0.4 s later; 0.35 delivered 0.48 m/s. The planner's
   speed field is a step (`backends/sonic/gait_speeds.py`), so the old floor was a dead
   command and the old "deceleration" below ≈0.3 was already a coast.
3. Config validation and the generated field reference; the execution test stub carries
   `stand_attempts`; new approach tests cover re-walk, retry bound, near stance, the floor and
   validation.

## Batches (seeds 41–50, same scene copy, instance 1 / GPU 0, all attempts retained)

| Batch | Code | Result | Stance depth (visible crate centre, m) | Notes |
|---|---|---|---|---|
| `outputs/crate_collect10_20260914` | `fcf23d9` | 7/10 | 0.35–0.47 | failures: 2 far stances, 1 single-hand loss |
| `outputs/crate_depth_converge10_20260914` | criterion, 0.10 floor | 7/10 | 0.35–0.47 | criterion fired once (06: 0.475 → 0.437 → 0.449, then `reach_stalled` at 36 s: the re-walk never moved); new failures `basket_tilt` 15.2° (02) and `evaluation_missing_contact` (05) are contact failures at ordinary stances |
| `outputs/crate_depth_converge10b_20260914` | criterion + `walk_speed_min` | 10/10 | 0.23–0.32 | criterion never fired; every stance accepted first time |

Batch 3's result comes from the floor change, not from re-walking: holding 0.35 until the
0.46 m stop lengthened the coast from ≈9 cm to ≈22 cm, so stances now land ≈0.25 m from the
crate centre, 14 cm nearer than before and 14 cm nearer than `stand_off` says. Ten rounds
cannot resolve a success rate (the same seeds gave 7, 7, 10 with different failure mixes);
what the batches do establish is that the stance distribution moved and that the far
stances of batches 1–2 did not recur.

## Limits and next steps

- `visual.stand_off` (0.39) no longer describes the achieved stance (≈0.25). The depth
  criterion is therefore inert on this configuration: it only fires when the coast is under
  3 cm. Either lower `stand_off` to the measured stance or raise `walk_brake_margin` so the
  stop lands on `stand_off`; that is a calibration decision for the owner, not made here.
- Near stances (0.23 m) passed 10/10 here and 0.148–0.265 m succeeded on the wall scene, but
  hip/table clearance at the 30 cm table was not separately measured.
- The re-walk path is exercised only by CPU tests and by batch 2 (where it could not move).
  A stance farther than the tolerance with the new floor has not been observed live.
- Previews (≤500,000 bytes, full duration) with `verification.json` are under each batch's
  `previews/`; originals under `videos_00/`.
