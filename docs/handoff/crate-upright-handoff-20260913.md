# Crate contact, handoff and loaded standing — 2026-09-13

Source: `20528e8` (contact, handoff and standing) plus `50805b8` (physical clock),
on `fix/crate-upright-handoff-20260913`, based on `7557102`.
This work addresses the previous ten-video batch's flying basket (02), early
admission failure (06), handoff shake and completion while still bent.
All outputs are preserved under `outputs/crate_upright_handoff_20260913/`.

## What is established

**02 is a collision-detection failure, before grasping.** In the original
`assisted_delivery` batch the basket flew during walking. A new three-round probe
captured the complete state immediately before the same seed's impulse.
At simulation time 4.355 s, one 5 ms step changed basket speed from
`8.55e-11` to `15.896895` m/s. The table/basket-floor contact included a fabricated
`-0.6701806` m penetration, alongside correct contacts of about `-5.6e-6` m.

Replaying the saved model and complete step state reproduced the velocity exactly.
Zeroing controls, applied forces or warm-start acceleration did not remove it.
Reducing timestep or softening contacts only reduced the resulting impulse.
A two-box model with the captured basket pose reproduces the failure without any
robot or controller. Equal-size convex meshes select a different collision
algorithm and eliminate the phantom contact in that replay.
This agrees with [MuJoCo issue 3476](https://github.com/google-deepmind/mujoco/issues/3476),
which reports the same class of box/box failure in 3.10/3.11. Local runtime: 3.11.0.

Evidence: `ejection_event.json`, `ejection_model.mjb`, `ejection_full_state.npz`,
`minimal_ejection_replay.json`, `physics_probe/` and `tests/test_convex_boxes.py`.
The automated regression checks actual contact depths, velocity and drift over
200 physical steps, plus mass/inertia and idempotent scene conversion.

The final runner converts the **support table's collider**, preserving the basket's
existing finger-contact geometry. It uses eight vertices with the same box boundary;
visual meshes, basket mass and the input scene remain intact. An earlier probe
converted basket colliders instead; that is not the final implementation.

**06 ended because reach admission failed; the MP4 was not truncated.** Its old
result was `unsupported_assisted_reach`. Replaying the recorded final pose gave a
23.38 mm support COM shift; the unchanged admission limit is 15 mm. Moving the
support pose 30 mm toward the basket made the same geometric problem admissible
(COM shift 3.18 mm). This is geometric evidence, not proof of live stepping.
The captured input is `episode06_last.json`.

The expert now checks support before POSE handoff. If unsupported, navigation can
use its remaining braking allowance for a closer visible approach, stopping no
closer than the configured `stand_off`. The allowance shrinks on each retry, and
the existing approach deadline remains. Admission hand/foot/COM errors are recorded.
Unreachable placements can still fail; they are not relabeled successful.

**A controller mode change remains.** Planner-to-POSE takeover now has no VR
intermediate stage. The Python sender preserves the consumed joint/root reference.
Deploy patch V2 keeps the planner override until a decoded POSE actually owns the
active motion, eliminating the interval where releasing it exposes the planner's
unmodified arms. The entry rejects a binary without `POSE_UPPER_OWNERSHIP_V2`.
This reference-ownership repair does not prove zero physical shake or blend the
neural controller's outputs; measured handoff motion is reported separately.

## Loaded standing and completion

After closure and attachment probing, the same bounded whole-body reference moves
the paired grasp toward its upright ready-pose centre. Foot support, hand separation,
waist/hip coordination, joint speed and acceleration remain in the shared solve.
The torso orientation target recovers continuously during `stand`.

Completion requires measured torso lean within 12 degrees, bilateral attachment
within 30 mm, nonempty grippers, sufficient basket rise and tilt within 15 degrees.
These conditions must hold for a complete two-second window whose load-position
radius is at most 25 mm. Small stance sway is allowed; sustained load drift is not.
Commanded posture alone cannot satisfy completion. Failed standing and unstable
holding have bounded, distinct terminal reasons.

A preliminary matched-calibration run (`matched_deploy_probe/`) physically stood
up with the basket in all three attempts, rising 27.2–30.1 cm. All three failed its
then-current instantaneous velocity gate, so its recorded result remains **0/3**.
The final window-based criterion replaces that gate; the earlier failures are not
retroactively changed. CPU consumer tests cover complete attached standing,
rejection of a bent measured posture, bounded sway and persistent drift.

## Deployment calibration and intermediate trials

The production deploy binary used shoulder-roll default offsets `+0.2 / -0.2` rad.
The local third-party SIMPLE `auto engage` source had changed them to `0 / 0`.
Rebuilding that source initially introduced an unintended calibration difference.
ELF constant reads are saved in `compiled_default_angles.json` and
`matched_compiled_default_angles.json`. `sonic_policy_defaults.patch` restores the
production convention for that modified source; stock source already using those
values needs no defaults patch.

`upright_probe_v2/`, `upright_tracking/`, `table_contact_probe/` and
`bridge_comparison/` used the mismatched build and are not clean handoff comparisons.
Their failures are retained. `upright_probe/` failed at startup because the private
build lacked policy links; no episode was collected. `original_deploy_comparison/`
retains a separate old-source/old-binary single-seed attempt. The private build
commands and logs remain beside `deploy/`; production source/binary were not
modified during these comparisons.

## Clock correction and earlier ten-attempt result

`upright_delivery/` is the ten-attempt run of source `20528e8`, seeds 41–50,
30 cm table, original basket, GPU 0 / instance 1 and the V2 deploy patch with matched
default offsets. Source/config/scene hashes, binary/checkpoint receipts, observations,
raw videos and physics traces are retained. It completed **2/10**: 02 and 04.
Four attempts stalled during reach, one failed stable holding, and three fell
before the body handoff. Episode 06 reached the reach phase after walking/settling;
it did not repeat the previous support-admission abort. Successes lifted the basket
29.6–31.5 cm with measured final torso lean 2.6–3.2 degrees.

This run exposed a clock mismatch under shared-machine load: the expert advanced
by wall time while physics slowed. For example, 05 ran for 18.63 wall seconds but
its overview contained only 6.24 physical seconds. The expert could exhaust a reach
deadline before the robot executed the motion. GPU/CPU contention was observed
during the late episodes; it does not by itself establish the cause of each fall.

`50805b8` uses simulation time for assisted navigation, settling, trajectory rates,
phase deadlines and the final hold window. Camera freshness and the simulator
progress watchdog retain wall time. A measured quiet upright stance now admits
initialization before recording; failure receives a distinct zero-frame result.
The start-distance check still rejects drift outside 1.2–1.5 m.
Consumer tests compare identical physical inputs under normal and four-times-slower
wall clocks, and separately prove that stale camera data is still rejected.

The `physics_clock_probe/` three-attempt run reached actual loaded standing in all
three attempts, with final torso lean 3.2–4.1 degrees and basket rise 28.2–30.0 cm.
Its recorded result is **1/3**: one unstable hold, one tilt over 15 degrees, one
completed hold. This is a diagnostic run, not a reliability estimate.

The upright criterion refers to the torso, not neutral values of every joint.
The earlier two successes retained about 19 degrees of pelvis pitch compensated
by about -16 degrees of waist pitch, with knees around 13–15 degrees. This satisfies
the requested nearly upright upper body but does not demonstrate an entirely
neutral loaded stance.

## Handoff measurement

`measure_upright.py` reconstructs wrist/tool positions and velocities from the
recorded physics traces using actual model FK and Jacobians. The handoff window
runs from the last settling observation through the first second of body control.
The positional diagnostic subtracts a centred nine-sample mean (about 0.18 physical
seconds), excludes its four boundary samples, and takes each wrist's 95th percentile.

For the four shared early episodes 00, 01, 03 and 04 that actually delivered the
bend in both old `assisted_delivery` and new `upright_delivery`, the median of the
worse wrist's residual was **21.7 → 1.4 mm**. Wrist-speed P95 was **1.044 → 0.102 m/s**;
base angular-speed P95 was **1.210 → 0.191 rad/s**. Raw per-episode results are in
`baseline_upright_metrics.json` and `upright_delivery/upright_metrics.json`.
These are descriptive comparisons of four asynchronous attempts, not a controlled
isolation of the deploy patch or proof of zero shake. The subset excludes the old
02 ejection and new late attempts that did not deliver the bend; the full ten-attempt
outcomes remain reported above. The physical-clock batch is measured separately.

## Final validation and delivery

`clock_delivery/` runs source `50805b8`, the same ten seeds and unchanged scene/profile
with the matched V2 deploy binary. GPU 1 was not used or stopped by this task.
The final additional source change extends the existing deploy-capability rejection
to the separate whole-body carry entry; it does not change the validated visual motion.
The full command is:

```bash
LOCOMANI_ROOT=/workspace/code/locomanipulation-crate-upright-handoff-20260913 \
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 \
SONIC_DEPLOY_BIN=/workspace/code/locomanipulation/outputs/crate_upright_handoff_20260913/deploy/target/release/g1_deploy_onnx_ref \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u \
  -m locomani.pipeline.run_crate_pilot --expert visual --rounds 10 --seed 41 \
  --gpu 0 --instance 1 --config configs/tasks/crate_visual_low_reach.json \
  --scene /workspace/code/locomanipulation/outputs/crate_table30_review_20260913/scene_table30 \
  --out /workspace/code/locomanipulation/outputs/crate_upright_handoff_20260913/clock_delivery
```

The result is **6/10**. All ten attempts started 1.218–1.465 m from the table edge,
with a measured quiet initial stance. No initialization failures or external falls
occurred in this batch. Basket speed stayed below 0.253 m/s in the recorded physics
traces; the old sudden ejection was not observed.

| Episode | Result | Basket rise at end | Torso lean at end |
|---|---|---|---|
| 00 | completed | 30.7 cm | 3.65° |
| 01 | completed | 30.0 cm | 3.46° |
| 02 | completed | 30.0 cm | 4.19° |
| 03 | completed | 31.2 cm | 3.77° |
| 04 | basket_tilt | Failed during lift | 46.48° |
| 05 | reach_stalled | Failed during reaching | 60.77° |
| 06 | approach_timeout | No grasp | 4.75° |
| 07 | completed | 28.6 cm | 3.93° |
| 08 | approach_timeout | No grasp | 0.42° |
| 09 | completed | 30.0 cm | 3.91° |

06 now records the full 30-second physical approach. Two support checks still exceeded
the COM admission limit (18.2 and 19.8 mm). It then failed to converge laterally:
the final target lateral error was 18.6 cm. 08 also timed out approaching. Neither
attempt is a successful demonstration, and extending the video does not fix its
navigation. This remains a concrete next motion problem, alongside 04's grasp tilt
and 05's tracking/capture stall.

The eight attempts that entered body control have median worse-wrist handoff
high-pass P95 **1.06 mm**, with a worst episode of **2.61 mm**. Median wrist-speed
P95 is 0.084 m/s, and median base angular-speed P95 is 0.119 rad/s. These measurements
describe this batch; they do not eliminate the planner/POSE mode transition.
Successful attempts settle for about 1.8–1.9 physical seconds before reaching.
The remaining pause is measured braking/stance admission, not a removed idle clip.

`clock_delivery/clock_metrics.json` measures joint-reference differences against
the recorded simulation timestamps. Active motion stays within **0.7 rad/s** and
**2.0 rad/s²**, with no changed reference at zero elapsed time. Immediate terminal
freezes in failed 04/05 have discrete accelerations 8.35/9.08 rad/s²; the smoothness
bound applies to active trajectory generation, not emergency termination.
The successful final hold windows have load-position radii 22.4–24.7 mm.

`clock_previews/` contains all ten full-duration previews and a result CSV. They
are 388,043–498,530 bytes each, 13.0–30.0 seconds, at original playback speed.
Every MP4 passed complete decode, actual byte-size and duration checks (within one
10 FPS output frame). All first/last frames were visually inspected, along with
02's bending frame; successful endings show an upright torso and the retained load.
`crate_upright_10_previews.zip` is **4,404,772 bytes**, includes failures with their
labels, and passed archive integrity and per-member size checks. Source recordings
remain untouched. `video_checks.json` and `package_receipt.json` contain hashes.

The validated executable is installed at the default deploy path, with both source
patches applied. `production_install.json` records its SHA-256
`4add30143c97a0214c1937802d25f65350a6ca9c7a15c5ac4a9053e392e4713f`, identical to the
private executable used for the run. The original binary is preserved as
`production_binary_before_v2`; pre-existing third-party Python changes are retained.
The default executable passes the capability check. New installations must apply
the tracked patches and rebuild as described in [the deploy guide](../../patches/sonic_pose_ownership.md).

`make check` and the source commit's full pre-commit checks pass. This remains an
experimental video/control pilot; formal VLA action export and learning are not
validated. Failed attempts remain in the manifest and video package.
