# STATE — 2026-09-13

## Delivery

- Structural maintenance completed at source commits `bac6492` and `ec7ca1d`,
  against baseline `2e9af84`. Full CPU checks and independent Standards/Spec reviews pass.
- Short agent entry, shared runtime/configuration seams, task/backend separation,
  typed results, portable test assets and schema/provenance checks are implemented.
- [Delivery, checks and measured limits](handoff/maintenance-completion.md).
  [Change map](reference/change-map.md) is the entry for new code changes.

- Crate RGB expert video pilot completed: final batch 7/10 successes, three hold
  timeouts, all ten videos retained. [Implementation and evidence](handoff/crate-visual-pilot-20260912.md).
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
- Future bottle behavior work needs a discriminating protocol for these failures.
  Raw evidence and repeated comparisons: [delivery report](handoff/maintenance-completion.md).

## Shared resources

- Upright-carry validation complete; owned instance 1 released. GPU 1 untouched.
- Latest evidence: `outputs/crate_upright_handoff_20260913/`; all earlier motion evidence retained.
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
- Resource/artifact rules: [AGENTS](../AGENTS.md). [Maintenance status](ROADMAP.md).
