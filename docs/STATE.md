# STATE — 2026-09-15

## Delivery

- [Guideline completion](handoff/agent-guideline-completion-20260914.md): domain navigation, current/historical guide separation, local skills audit and measured forward-tests; first delivery owns verification/completion rules.
- Structural maintenance completed at source commits `bac6492` and `ec7ca1d`,
  against baseline `2e9af84`. Full CPU checks and independent Standards/Spec reviews pass.
- Short agent entry, shared runtime/configuration seams, task/backend separation,
  typed results, portable test assets and schema/provenance checks are implemented.
- [Delivery, checks and measured limits](handoff/maintenance-completion.md).

- [Crate formal export](handoff/crate-formal-export-20260914.md): `--record` on any instance, converter unchanged, `check_dataset --run-dir` judges the lift in world z; validation 7/10, 39/39 audit; production 219/260 recorded, 209/10 train/val episodes audited 39/39, 10k-step model at `outputs/crate_vla200_record_20260914/model/checkpoint-10000`; bimanual closed-loop evaluator `eval_crate_closed_loop.py` added, validation 9/10, `outputs/crate_eval100_20260915` **71/100** (20 close-without-lift, 5 never close, 4 reset). GPU 1 / instance 2 released. Crate RGB expert video pilot: final batch 7/10 successes, three hold timeouts, all ten videos retained. [Implementation and evidence](handoff/crate-visual-pilot-20260912.md). [Stance depth convergence](handoff/crate-depth-convergence-20260914.md): depth tolerance + re-walk and a stepping speed floor (the 0.10 floor never moved the robot); seeds 41–50 went 7/10 → 7/10 → 10/10, the 10/10 from stances landing 14 cm nearer, not from re-walks; `stand_off` needs recalibration. GPU 0 / instance 1 released; GPU 1 untouched.
- [Earlier walking extension](handoff/crate-walk-stability-20260912.md): 3/10; starts 1.278–1.479 m. Seven reaches pass the shake diagnostic; navigation/hold failures remain.

- Isaac asset portability repaired: tracked room source, installed table meshes/texture
  fallback, strict source validation and cache invalidation. Isaac 4.5 fresh-install
  render passes. [Cause, update commands and evidence](handoff/isaac-asset-portability-20260912.md).

- Carry / wall waist pitch enabled and measured on v1.1; deep-grasp reliability
  remains limited. [Implementation and evidence](handoff/crate-carry-waist-20260913.md).

- [Carry motion](handoff/crate-motion-flow-20260913.md): 11/30 vs 0/30. [55 cm validation](handoff/crate-table55-20260913.md): 6/10 vs 5/10; posture/contact limits remain.
- [30 cm validation](handoff/crate-table30-20260913.md): 0/10; three crouches delivered, but pelvis retreats 23–29 cm. Experimental profile; default flow retained.
- [30 cm hip hinge](handoff/crate-low-reach-20260913.md): privileged pilot 6/10; torso 55–58°, shallow knees, all waist axes free.
- [30 cm RGB low reach](handoff/crate-visual-low-reach-20260913.md): 4/10 fixed batch, final check 1/3; faster rim capture and smoother references. Tracking/physics failures and VLA export remain.

- [Earlier assisted coordination](handoff/crate-assisted-coordination-20260913.md): 4/10. [Loaded standing and handoff](handoff/crate-upright-handoff-20260913.md): 6/10; collision fix, physical motion clock, torso 3.5–4.2° at successful endings. Navigation/contact failures remain.

## Runtime evidence and remaining uncertainty

- Final bottle A/B: coupled 2/30 -> 3/30; teleop 10/10 -> 9/10. Both gates FAIL
  on newly observed failure categories. Coupled stage-time coverage is insufficient.
- Same-revision A/A: coupled 2/30 -> 2/30; teleop 10/10 -> 9/10. Both gates also
  FAIL on new categories. Physical equivalence remains UNVERIFIED; no causal
  attribution is established and no failure is relabeled PASS.
- Real policy-server replay PASS: three fixed observations/seeds yield exactly
  equal action arrays; checkpoint identities verified. Crate pilot flow PASS, 2/2.
- User excludes crate success/reproducibility as a blocker; scene redesign is
  separate. No physical control or gate thresholds were tuned in this maintenance.
- New [bottle fixed-sample work](handoff/bottle-continuous-20260914.md) retains failure diagnostics; prior comparisons remain in the [delivery report](handoff/maintenance-completion.md).

## Shared resources
- [Natural bottle expert](handoff/bottle-natural-expert-20260915.md): natural preparation/stance, wrist ROI, continuous POSE/closure/handback integrated on main from `fix/bottle-natural-20260915`; `make check` passed, diagnostic 3/4, fresh collection 10/11 accepted. Ten full-duration previews under `outputs/bottle_natural_20260915/final_collect/previews`; actual jitter and recovery contact remain optimization work. GPU 0 / instance 0 released; GPU 1 untouched. Current follow-up delivery checkpoint11: central grasp/contact recovery/tracking/stopping integrated; fresh30700/30800 collection10/15 in `outputs/bottle_contact_20260915/final_collect_v3`, ten compliant previews/ZIP4552364bytes audited. Candidate12 full6D hold2/4 rejected; production restored11, candidate preserved. Naturalness/high reliability remain unmet; final main `make check` passed. GPU0/instance0 released; GPU1 untouched.
- [Bottle filtered walking](handoff/bottle-direct-20260914.md): final 6/10 vs 3/10; raw action labels pass; rebound and image alignment remain unresolved. [Hanging-arm start](handoff/bottle-hanging-start-20260914.md): the continuous expert now prepares only the waist, so the recorded episode starts with hanging arms and raises the right arm during the walk (10/10 traces without the READY raise-then-drop); same 10 cases 4/10 → 6/10; the lift then keeps the hand's torso-frame offset instead of a world point (no retract, no folded elbow), 7/10; a sampled 81-round run then showed 46/52 failures were guard predictions (held bottle vs wrist link in the lift, finger at the bottle at handoff), guard semantics changed (8/20 → 15/20), failure path now hands control back to the planner (reset-miss clusters gone), lift 0.06 m / 5 s; slips persisted until the finger pads were hardened; hardened pads are now the `install_scene.sh` default (15/20 sampled, 7/10 fixed bank, 0 slips), the fixed bank is re-stamped as `cases_fixed10_hardpads.json`, reproduction table in docs/collection.md. Full chain `outputs/bottle_vla200b_20260914`: 200/272 collected, 190/10 episodes, 10k steps (loss 0.087), closed loop **17/100** (60 rounds never command a close); bisection: the close is a joint function of both images, failed rounds stop at the expert's park (0.40 m brake) and skip the 10 cm whole-body advance; `visible_brake_depth` 0.40→0.31 gives 17/20 vs 15/20 on the same seeds, hip/table 2 vs 3 (`outputs/bottle_brake031_sampled20_20260915`); crate-style walking preparation: v1 2/20 (finger length vs table), v2 0/20 (hand chased the bottle), v3 **14/20** with the wanted motion (palm 0.76 m, reach 3.4 s slide-in) but hip/table 5 vs 2 from the saturated waist roll; roll-free envelope untested (`outputs/bottle_walkprep3_sampled20_20260915`). Shake analysis: the tracked reference is smooth in every phase, the body is not (walk = SONIC gait, manipulation = tracking + contact); three smoothing variants (`pose_lowpass_hz` 4, `tracking_gain` 0.3, `reach_seconds` 3) went 7/6/6 of 20, only the gain cut body jitter, none adoptable (`outputs/bottle_smooth_*_20260915`); the table-edge/capsule plan (`visible_brake_edge` 0.27, `raise_trigger_px` 36, roll-free walk, distance lean) 12/20: roll removal cut hip/table 5→2, the edge rule and pixel trigger were inoperative at this spawn band (`outputs/bottle_edge_sampled20_20260915`). GPU 0 / instance 0 released.

- [Bottle continuous candidate](handoff/bottle-continuous-20260914.md): 3/10 → 3/10; consumed-action and dataset audits pass, opt-in retained. GPU 0 / instance 0 released; GPU 1 untouched.
- [Review follow-up fixes](handoff/review-fixes-20260914.md): CPU, live wall offset and Isaac 4.5 inset checks pass; instance 1 released, GPU 1 untouched.
- [Continuous assisted motion](handoff/crate-continuous-motion-20260914.md): camera/control separation, future POSE window, overlapping reach/recovery and training end marker. GPU batch 6/10; failed attempts retained. GPU 0 / instance 1 released; GPU 1 untouched.
- Earlier outputs, datasets, recordings and weights are preserved; see the linked reports.

## Evidence and preserved constraints

- Historical conclusions and retractions remain unverified; use the
  [evidence rules](reference/standards.md#evidence) before relying on them.
- [Earlier GPU handoff](handoff/maintenance-gpu-regression-20260912.md) retains
  the first delivery's comparisons and protocol corrections.
- The original 1473-line [STATE archive](archive/state_20260912.md) is verbatim
  from `4cc52ba`; its RUNNING claims do not describe current resource availability.
- The historical record attributes NO DAgger and parked crate stand tuning to the
  user; this maintenance task does not change either policy.
