# Rules for the TODO 25.9.1 loop (version 1, 2026-10-14)

This is the only human knowledge the loop reads (plan §8, D3). It is distilled from the project's notes. Every rule says where it is enforced: **code** (the gate or the runner refuses) or **prompt** (the Reviewer and Engineer are told). A new rule is a new version of this file, committed, and counted as one human intervention.

## The task and its limits (code)

1. The scene, the box, the layout ranges and the success rule are frozen (plan §3; `accept_gate.FROZEN`). Making the task easier is not an improvement.
2. A start state is never removed from any pool. A start judged infeasible still counts as a failure.
3. A candidate is accepted only by `accept_gate.py` on the acceptance pool. The thresholds are in plan §6.4. Nothing else counts as evidence that a change helps.
4. A field a candidate sets must be read on ScaleBFM's path (`fields_read`). A setting that never reaches the code is the repository's most common failure: a config applied only in part, with the symptom showing up much later.
5. Gate verdicts are the only acceptance-pool output a role may read. Per-round files under `gate/` are off limits.

## Evidence (prompt; code where marked)

6. Read the per-round records (`handoff.py`, the 50 Hz state files), not the failure strings. A failure string names the stage where the code gave up, not the cause. Numbers inside it are whatever that check happened to measure.
7. Use per-round statistics: the value at the deciding moment, or the round's min or max. Then aggregate across rounds. Per-frame aggregates are weighted by dwell time, so stalled rounds dominate them. `envelope.py` is per-frame: read its ratios as "during the reach", not "per round".
8. Only closed-loop re-runs from a start state are evidence. Replaying logged commands, re-solving IK offline, or re-scoring stored states does not show what the closed loop would do. Closed-loop programs compensate for the controller; a replay does not.
9. Ten rounds cannot measure this line: the same configuration has given 6/10 and 2/10. Diagnostic forks are for direction only. Never claim a gain from them.
10. Subsets picked because they succeeded fall back towards the mean when re-run. Do not tune on the rounds that currently succeed and report those same rounds.
11. Before saying information is absent ("the trace cannot tell X"), spend minutes looking for it. Probes are cheap.
12. When a measured constant looks decisive, first find every caller of that constant. Several "decisive" constants in this repository had no callers.

## What earlier rounds of this line found (prompt)

13. The warehouse expert was tuned with SONIC, which delivers about half the commanded waist yaw and roll and less of the crouch (`envelope/sonic_v1_1.json`). ScaleBFM follows the reference more fully and crouches and leans further (plan §2). This is a hypothesis; phase 1 tests it.
14. Gates inside the executor that are tighter than the criterion the round is judged by make whichever controller tracks worse look worse. Relaxing such a gate to the criterion can change results more than changing the controller.
15. Pinching a 1 cm box rim with Dex1 jaws needs the jaw centre within 1–2 cm. ScaleBFM's wrist attitude at the palm-down grasp is 10–30° off, and the jaw centre sits 15 cm ahead of the wrist, so open loop the jaws miss by 2–9 cm. The jaw-centre + attitude servo (`servo: pad`) fixes the miss. Even with it, one jaw often pinches lower than the other and the box hangs tilted.
16. With SONIC, `reach_stalled` happened in the descend: both hands drift the same way across the rim (median 2.9 cm), growing during the 8 cm descend, while successful closes sit at 1.2 cm. Lateral correction alone traded stalls for slips. Every failure is terminal: the expert has no re-centre or re-seat.
17. ScaleBFM's feet slide. Stance width grows by about 17 cm over 30 s, so long waits in a crouch change the stance.
18. The joints of the robot's body are not contiguous in `qpos` (a gripper slide joint sits between the arms). Index joints by name.
19. When a list of fixes keeps growing, the layer is probably wrong. Say so instead of adding item N+1.

## Operations (code where marked)

20. ScaleBFM runs on GPU 1, at most 4 processes with `OMP_NUM_THREADS=2` each. Never run ScaleBFM while a SONIC stack is up (code: the driver checks).
21. Long jobs run under `nohup setsid` with a log. Never edit a bash script while it is running.
22. Keep artifacts under `outputs/scalebfm_warehouse_rsi_20261014/`. Total budget 10 GB: keep every failed round, and 10% of successful rounds.
23. Stop and write an escalation instead of acting when a step needs any of: the scene, the box, the success rule, the controller checkpoint or its training, the pose source, the real robot, the frozen rules, or the budget.
