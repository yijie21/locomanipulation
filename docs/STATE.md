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
- Walking/stability extension: corrected full-flow batch 3/10 success; initial table-edge
  distances 1.278–1.479 m. Seven reaching episodes pass the pre-contact shake diagnostic;
  navigation and hold failures remain. [Results and small videos](handoff/crate-walk-stability-20260912.md).

- Isaac asset portability repaired: tracked room source, installed table meshes/texture
  fallback, strict source validation and cache invalidation. Isaac 4.5 fresh-install
  render passes. [Cause, update commands and evidence](handoff/isaac-asset-portability-20260912.md).

- Carry / wall waist pitch enabled and measured on v1.1; deep-grasp reliability
  remains limited. [Implementation and evidence](handoff/crate-carry-waist-20260913.md).

- Carry continuous motion: 11/30 vs 0/30, shorter handoff; navigation failures remain.
  [Motion comparison](handoff/crate-motion-flow-20260913.md). [55 cm validation](handoff/crate-table55-20260913.md): 6/10 vs 5/10; less wrist bend, other posture/contact issues remain.

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

- Table-height validation complete: GPU 0 / instance 1 released; GPU 1 untouched.
- Current evidence: `outputs/crate_table55_20260913/`; previous motion evidence and originals retained.
- Previous outputs: `outputs/crate_visual_expert_20260912/`,
  `outputs/crate_walk_stability_20260912/`, `outputs/isaac_asset_portability_20260912/`.
  Datasets, recordings and weights are preserved.

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
