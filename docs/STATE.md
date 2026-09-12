# STATE — 2026-09-12

## Delivery

- Structural maintenance completed at source commits `bac6492` and `ec7ca1d`,
  against baseline `2e9af84`. Full CPU checks and independent Standards/Spec reviews pass.
- Short agent entry, shared runtime/configuration seams, task/backend separation,
  typed results, portable test assets and schema/provenance checks are implemented.
- [Delivery, checks and measured limits](handoff/maintenance-completion.md).
  [Change map](reference/change-map.md) is the entry for new code changes.

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

- ACTIVE: crate visual expert implementation and 10-video pilot on branch
  `feat/crate-visual-expert-20260912`; GPU 0 / instance 1, admitted through the stack lock.
  Outputs: `outputs/crate_visual_expert_20260912/`. Recheck live state before reuse.

- Owned runs ended 2026-09-12 13:31 UTC. GPU 0 was at 2 MiB / 0%; instance 0/1
  locks were free. GPU 1 has an unrelated job. Check live state before reuse.
- No maintenance GPU task remains running. Runtime artifacts and weights are preserved.

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
