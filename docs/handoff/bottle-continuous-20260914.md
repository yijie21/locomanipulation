# Bottle continuous reach — fixed-sample implementation and verification

Read when comparing the new `whole_body` candidate with the existing single-arm
profile, investigating action labels, or extending supported reach. Usage remains
in [single-arm collection](../collection.md); this is a dated experiment record.
Branch: `fix/bottle-continuous-20260914`, based on `e302d11`.

## Delivery and decision

Added an opt-in whole-body bottle strategy and completed a fixed ten-case comparison.
**Baseline 3/10; candidate 3/10. Do not promote the candidate to the default.**
The three candidate successes lift 14.9–15.3 cm, recover to torso lean 0.46–3.20°,
back away and pass the original terminal hold. Recorded movement takes longer.
The task still has navigation, visibility and wrist-clearance failures.
No VLA training or closed-loop learning improvement is established.

The crate investigation/fix was separately pushed as `e302d11`; see
[crate reproduction](crate-reproduction-20260914.md). Continuous crate code was
already on main. The guide omitted a 4 cm inset and alternate entry points could
select the wall expert. The collaborator's exact command, revision and failures
remain unavailable, so neither finding proves the cause of their 0/50.

## Implementation

- `policy.py` selects `whole_body`; `WholeBodyConfig` owns limits and supports
  nested typed overrides. Default `bottle_visible.json` remains unchanged.
- The bottle strategy prepares the arm while braking, then drives waist, hip and
  hands through a supported reference. All three waist axes participate.
  Position and rotation feedback compensate measured tracking error; limits
  apply to reference speed/acceleration, feet and COM. These are geometric
  constraints, not a proof of dynamic stability.
- Camera-admitted simulation geometry guides reach. Fresh ego/right-wrist RGB,
  frustum and ray tests admit a target; unseen new truth cannot update its cache.
  This is assisted planning, not an RGB pose estimator. The current one-second
  cache can cause false-negative admission/expiry; two final failures hit it.
- Hand descent, insertion and gradual closure share the reference. Closure begins
  inside the existing capture region at low relative speed, with sustained
  bilateral contact required before lifting. Upright recovery overlaps the lift
  after real object clearance. The original backoff and hold criteria remain.
- Shared supported geometry moved to `primitives/whole_body_ik.py`; crate keeps a
  compatibility import and identical default hand weights. No cross-task import
  was added to bottle. Reachable/unreachable old/new solves compare bit-for-bit.
- Fixed case banks now include reset yaw, consume cases in order across batches,
  validate scene identity and reject insufficient coverage. No success-driven
  replacement cases are introduced into an A/B comparison.

## Action-chain defects found and repaired

1. The upstream exporter can fall back to old planner hand values for joint-only
   POSE messages. In probe 13, the physical fingers closed and lifted a bottle
   while `teleop.right_hand_joints` still said open. The default launcher now
   uses our exporter wrapper, which records `last_*_hand_action` from the same
   C++ state packet as the consumed token. Missing/invalid commands fail rather
   than fabricating labels. Receipts preserve those commands for row comparison.
2. POSE-to-planner return primes a full planner frame with the held upper-body
   and hand targets. `sonic_hand_ownership.patch` makes hand ownership follow the
   selected mode instead of lower-body planner readiness; existing timeouts and
   emergency stops remain. The bottle candidate requires both deployment markers.
3. Logging itself caused a real control gap. Earlier probes copied ~50 future
   poses per control tick and serialized tens of MB before loaded transport.
   Probe 14 logged `Planner timeout (1003ms)` and the consumed closed hand briefly
   reverted to a default Dex3 pose. Moving serialization to round cleanup and
   logging preview endpoints/identity removed this path; the final successful
   episodes preserve closed hand commands through handoff/backoff/hold.

The timeout in item 3 is directly observed. The independent contribution of the
mode-getter change was not isolated in a separate randomized experiment; it must
not be presented as the sole proven cause of previous dropped bottles.

## Protocol and artifacts

All artifacts are under `outputs/bottle_continuous_20260914/` and remain untracked.
The scene is the existing `outputs/scene/sim`: table 0.62 m, bottle center 0.70 m.
This run does not test a new lower table. The candidate grip point is 1 cm below
bottle center rather than the baseline's 2 cm; object geometry/placement is unchanged.

Baseline was collected before bottle changes at `0f9e0f2`, with the previous
production-equivalent binary in `outputs/crate_upright_handoff_20260913/deploy`.
The final candidate uses a private rebuild in this run's `deploy/`; decoder,
encoder and observation-config identities match. Build commands are retained in
`deploy/build_commands.json`, and exact binaries/models are in deploy receipts.
Candidate snapshots include all changed source digests; subsequent source edits
before release were formatting or audit/documentation corrections.

The original seed-31000 target/stand/spawn/appearance samples were recovered as
`cases_fixed10.json`; its `object_yaw` values are the actual baseline reset yaws
(old reset yaw was not seeded). Every candidate consumes these same ten cases.
Generated new banks have deterministic yaw without consuming the existing sampling
RNG. The scene-hash check prevents accidental cross-scene comparisons.
Fixed inputs do not make the asynchronous physical stack bit-deterministic.

The automatic runner uses `learn_200.json`, `through=collect`, `successes=10`,
`batch_rounds=10`, `max_batches=1`, `seed0=31000`, GPU 0 / instance 0. Baseline
selects `bottle_visible.json`, final candidate selects `bottle_whole_body.json`
and `--cases outputs/bottle_continuous_20260914/cases_fixed10.json`.
Insufficient success quota makes the runner exit nonzero; it is not a lost run.
Full launch commands, task/config snapshots, scenes and model receipts are saved.

| Artifact | Meaning |
|---|---|
| `baseline10/`, `candidate10/` | Full manifests, videos, physical traces, config/source/model identities |
| `probe2/` through `probe15/` | Preserved development failures and partial lifts; not pooled with final ten |
| `baseline_actions_final.json`, `candidate_actions_final.json` | Current schema/consumed-command and timing audit |
| `baseline_gate.json`, `candidate_gate.json`, `candidate_gate.log` | Unmodified behavior gate and diagnostics |
| `baseline_dataset/`, `candidate_dataset/` | Three successful episodes each, terminal verification tail trimmed |
| `token_replay/result.json` | Live protocol-v4 consumption probe, exact checkpoint checks |
| `crate_ik_equivalence.json` | Exact old/new default shared-IK comparison |
| `previews/candidate_case01.mp4`, `previews/verification.json` | Full-duration preview, 369,601 bytes, decode/size/duration checks |

Case IDs are zero-based: baseline successes `31000:1,4,5`; candidate `31000:1,3,4`.
Candidate failures: three initialization/standing/arrival failures, two visibility
expiries and two wrist-to-bottle clearance failures. Nine attempts produced raw
recordings; one initialization failure produced none and remains in the manifest.

## Measurements and limits

Metrics below summarize the three successful episodes in each run. Raw action/state
records are 50 Hz; converted data is 25 Hz. Joint-step quantiles use the 29 body
joints only, excluding hand fields with different units. They are speed/jump
proxies, not jerk bounds or evidence that all mode changes disappeared.

| Metric, median across successes | Baseline | Candidate |
|---|---:|---:|
| Successes / all attempts | 3/10 | 3/10 |
| Recorded frames / 50 Hz | 857 / 17.14 s | 1116 / 22.32 s |
| Per-episode body joint-step p99 | 0.0304 rad | 0.0247 rad |
| Per-episode WBC joint-command-step p99 | 0.0804 rad | 0.0702 rad |
| Longest identical-token run | 0.42 s | 0.38 s |
| Longest quiet interval before final verification | 0.36 s | 0.26 s |
| Control-code closure duration | 2.03 s | 1.34 s |

Quiet means all body joint displacements over 0.2 s imply <0.08 rad/s and right
jaw motion <0.001 m/s; final verification is excluded by its sequence boundary.
Jitter can prevent this detector from calling a visually stationary robot quiet.
These numbers therefore do not prove zero perceived hesitation or VLA learnability.
Case 1's quiet interval increased 0.20→0.44 s; the improvement is not universal.
Success subsets differ and the sample is small; no statistical gain is claimed.

The unchanged bottle gate is **FAIL (7/10 checks)**: new/removed stages and failure
categories, plus the shortened closure outside its ±30% equivalence tolerance.
Yield and final approach depth pass. Intentional strategy changes explain some
gate failures, but the result is not relabeled PASS and thresholds were not loosened.

A previous diagnostic incorrectly read decimated `samples` intervals as controller
frequency. The sent-reference records corrected this: representative probes were
median 20–40 ms, p95 40–60 ms, not 120–160 ms. The attempted shared stability-gap
relaxation was reverted; it is not part of this delivery.

## Verification and next work

- Related CPU tests: 74 pass before final integration; final commit-hook checks
  are recorded with the release, rather than rerunning the same full suite manually.
- Final raw action audit: 9/9 recordings have exact token/hand roundtrip and
  consumed-hand label matches, including failed attempts. Old baseline hand
  receipts are absent: its label-source verification is **UNVERIFIED**, even
  though its shape/wire-format checks pass.
- Live replay: 150 recorded actions around closure, `NO_ENCODER=1`, protocol v4;
  decoder and observation-config hashes match. After 12 startup rows, 138/138
  consumed token and hand observations match the recent sent stream; no fall.
  Fresh-state short replay verifies the route, not full-episode reproduction.
- Dataset conversion: baseline 3 episodes / 1134 frames; candidate 3 / 1540,
  both 25 Hz with only the marked terminal verification tail removed. Both full
  dataset audits pass 32 checks. Candidate image/table lag diagnostics are zero
  on six wrist measures, two left-wrist correlations are weak; token/body median
  lag is +3 frames (120 ms). These are not exposure-time guarantees.
- Crate smoke with the corrected 30 cm / 4 cm inset scene and private rebuild,
  seed 41: **0/1**, bilateral contact and 18.84 cm rise, rejected at 15.06° basket
  tilt. Reference seed 41 succeeded at 7.49°, with slightly different measured
  initial stance despite identical sampled placement. A single asynchronous
  comparison cannot isolate a binary regression. The crate gate is FAIL (2/3
  reported checks); visual manifests lack wall geometry/stage fields, and 1 vs
  10 rounds is not a compatible equivalence test. Keep physical equivalence
  UNVERIFIED. The user previously excluded crate success as a blocker.
- The private deployment is retained separately; the machine's original default
  binary remains available and unchanged. To reproduce this candidate locally,
  set `SONIC_DEPLOY_BIN` to
  `outputs/bottle_continuous_20260914/deploy/target/release/g1_deploy_onnx_ref`
  **using an absolute path**. On a fresh machine apply the documented patches
  and rebuild; runtime binaries are not stored in Git.

Next priorities are camera-expiry diagnosis, more robust arrival placement and
wrist approach geometry, then reducing the candidate's longer descent/insertion.
Repeat a larger fixed-case comparison after those changes. Keep the candidate
opt-in until reliability and movement quality justify promotion. VLA training and
closed-loop evaluation are a separate required test before claiming better learning.
