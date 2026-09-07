# Collection policy experiments (2026-09-06)

These JSON files select opt-in policies. A newer filename is not evidence of a better policy. The default collection script has not been switched by this experiment.

- `ready_grasp_coupled.json` and `ready_grasp_coupled_timing.json` predate the unattended goal; they remain available.
- `ready_grasp_measured_prediction.json`: best completed eight-mode regression at its recorded source revision, 8/8; first independent perturbation test only 17/24. Not yet accepted for the planned VLA comparison.
- `ready_grasp_reference_handover.json`: preserves prior arm command compensation at closure. Two diagnostic failures succeeded on rerun; this alone is not independent acceptance.
- `ready_grasp_stable_reference.json`: simplified candidate, 7/8 primary; height failed coupled IK. It combines calibrated squat prediction, closure-reference handover and actual-feasibility region stopping; the conditional transport IK lead is disabled. It is not promoted.
- `ready_grasp_stable_bounded_reference.json`: final regression6/8; adds only conditional25mm infeasible-target retry to the simplified candidate. Height succeeds but near wrist clearance and carry drift fail. Not promoted.
- `ready_grasp_capture_continuity.json`: intermediate profile with conditional transport IK retry. Its full regression was 5/8; not promoted.
- `ready_grasp_level_bottle.json`, `ready_grasp_tool_point.json`, `ready_grasp_region_stop.json`, `ready_grasp_bounded_transport.json`, `ready_grasp_lower_body.json`, and `ready_grasp_reference_region.json` are retained experimental intermediate profiles, not recommendations.

The `coupled_hand_lead` option now bounds a retry target only when the full transport target fails the existing IK limits; executed joint increments remain bounded in both cases. Earlier runs used an unconditional target lead bound. Reproducing an earlier run therefore requires its saved source revision, not only its JSON.

Each experiment keeps its configuration, source snapshot, hashes, failures, videos and available state traces under `outputs/experiments/policy_goal_20260906/`. See that directory's `REPORT.md`, `progress.md` and `attempt_ledger.jsonl` for outcomes and limitations. A successful expert episode does not establish VLA learnability.
