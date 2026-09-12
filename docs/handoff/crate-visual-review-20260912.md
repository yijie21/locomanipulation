# Crate visual expert review — 2026-09-12

Independent read-only Standards and Spec reviews used the `code-review` skill.
Initial diff: `git diff 79df9ef...2f89d1b`. Correction diff: `2f89d1b...3ac447a`.
The final follow-up inspected the exception-reporting correction to `expert.py`
and its tests after `3ac447a`. Reviewers did not change files or GPU processes.
Scope and measured delivery: [video evidence](crate-visual-pilot-20260912.md).

## Standards

Initial report: three documented violations and one optional heuristic concern.

- **Invalid configuration reaches execution.** `pregrasp_above=0` was accepted,
  then produced NaN rotations during alignment. This violated “Validate before
  starting resources.” The correction requires a positive value and tests the
  invalid input through `CrateTaskConfig`; the reviewer independently reproduced
  rejection of the formerly failing value.
- **Completed-result contract bypassed.** The entry returned and serialized a
  dictionary without `RoundResult`. It now returns `RoundResult`; the recording
  boundary consumes `accepted()` and preserves pilot diagnostics. Tests distinguish
  task success from produced training data and exercise failed setup recording.
- **Evidence claim lacked a concrete record.** The first-grasp claim had no linked
  run, command, identities or metrics. The evidence report now links the preserved
  video, manifests, source/config/scene hashes and deployment receipts, and identifies
  precisely which revisions each batch validates. The reviewer inspected those files.
- **Possible Feature Envy (judgement call).** Shared IK and camera FK depend on
  `ArmIK`'s MuJoCo model/state internals. This coupling remains acknowledged. A future
  primitive interface can own tool Jacobians and camera poses if those consumers
  expand; a general kinematics refactor is outside this video-pilot delivery.

Follow-up: all three documented violations resolved; no remaining material
Standards findings. Source rules: [architecture and standards](../reference/standards.md).

## Spec

Approved source: [expert design](crate-visual-expert-design.md), revision `79df9ef`.
Initial report: four current implementation findings, separate from explicitly
deferred training/workspace/recovery work.

- **Loaded motion lacked shared progress.** A lagging hand did not pause probe,
  lift or hold references. Loaded phases now wait for bilateral tracking, with
  observed attachment checks during lift/hold. A one-hand-lag regression covers it.
- **Hold completed on elapsed time despite slip/tracking failure.** Completion
  now requires uninterrupted observed tracking, attachment, rise and gripper
  conditions. Observation loss or instability resets the hold interval.
- **Closure jumped after brief occlusion.** Wall-clock phase duration made up
  motion that had been skipped. Closure now integrates visible control time and
  does not catch up on reacquisition; a regression verifies the resumed increment.
- **Setup/finalization exceptions could disappear from manifests.** The attempt
  journal now records labels/seeds before setup, and the wrapper preserves failed
  setup, cleanup and evaluation outcomes. Tests also prevent overwriting a prior attempt.

Follow-up identified one additional exception-path issue: a cleanup/evaluation
error could erase the marker for an external fall intervention. `control_reason`
is now captured before cleanup and retained independently of the final error;
both error paths have regression coverage through the actual `run_attempt` entry.
The reviewer confirmed the correction and four passing recording tests. This
last diff changes exception reporting only; normal control commands are unchanged.

No remaining findings from the final Spec follow-up. Review found no unrelated
scope expansion or online hidden-object-state control path. It did not certify
student learnability or completion of the deferred design roadmap.

Totals: Standards — 3 documented findings resolved, 1 optional coupling concern;
Spec — 5 findings resolved. No unresolved material finding remains on either axis.
