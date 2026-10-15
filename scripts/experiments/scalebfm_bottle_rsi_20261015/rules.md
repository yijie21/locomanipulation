# Rules for the TODO 26.3 bottle loop (version 1, 2026-10-15)

This is the only human knowledge the loop reads (plan §8). Rules 6–12 and 18–22 are the warehouse loop's, word for word; rules 13–17 are what earlier work on this line found. Every rule says where it is enforced: **code** (the gate or the runner refuses) or **prompt**.

## The task and its limits (code)

1. The start states, the bottle, the table and the success rule are frozen (plan §3; `accept_gate.FROZEN`: `simple.min_rise_m`, `simple.disturb_limit_m`, `recovery_timeout`; the round's own checks in `bfm_expert_bottle.py` are not editable). Making the task easier is not an improvement.
2. A start state is never removed from any pool. A start judged infeasible still counts as a failure.
3. A candidate is accepted only by `accept_gate.py` on the acceptance pool. The thresholds are in plan §6. Nothing else counts as evidence that a change helps.
4. A field a candidate sets must be read on ScaleBFM's path (`fields_read`). A setting that never reaches the code is the repository's most common failure: a config applied only in part, with the symptom showing up much later.
5. Gate verdicts are the only acceptance-pool output a role may read. Per-round files under `gate/` are off limits.

## Evidence (prompt; code where marked)

6. Read the per-round records (the 50 Hz state files, the JSON's `events` and `stats`), not the failure strings. A failure string names the stage where the code gave up, not the cause. Numbers inside it are whatever that check happened to measure.
7. Use per-round statistics: the value at the deciding moment, or the round's min or max. Then aggregate across rounds. Per-frame aggregates are weighted by dwell time, so stalled rounds dominate them.
8. Only closed-loop re-runs from a start state are evidence. Replaying logged commands, re-solving IK offline, or re-scoring stored states does not show what the closed loop would do. Closed-loop programs compensate for the controller; a replay does not.
9. Ten rounds cannot measure this line: the same configuration has given 6/10 and 2/10. Diagnostic forks are for direction only. Never claim a gain from them.
10. Subsets picked because they succeeded fall back towards the mean when re-run. Do not tune on the rounds that currently succeed and report those same rounds.
11. Before saying information is absent ("the trace cannot tell X"), spend minutes looking for it. Probes are cheap.
12. When a measured constant looks decisive, first find every caller of that constant. Several "decisive" constants in this repository had no callers.

## What earlier work on this line found (prompt)

13. On the 45-start L2 nominal pool, SONIC's production expert finished 40 and this port on ScaleBFM (mode 4) 39. ScaleBFM's six failures were all "simple lift did not raise the bottle", counted as drops (6 against SONIC's 2); SONIC's were lost pad contact 3, misalignment 1, out of reach 1. These are failure strings (rule 6), not causes.
14. The port's mode was chosen on the 10 L1 starts: mode 4 10/10, mode 6 10/10, mode 2 8/10, mode 7 7/10 (SONIC 6/10). Replaying SONIC's measured keypoints with ScaleBFM mode 6 lifted all six of SONIC's L1 successes; SONIC's failed grasps failed under every mode.
15. The Dex1 jaw is a PD on its setpoint (this round: kp 1.5, kd 0.1, clip 20 N): the grip force grows with how far the closed setpoint lies beyond the contact, not with time.
16. ScaleBFM's feet slide. Stance width grows by about 17 cm over 30 s, so long waits in a crouch change the stance.
17. The joints of the robot's body are not contiguous in `qpos` (a gripper slide joint sits between the arms). Index joints by name.

## Operations (code where marked)

18. When a list of fixes keeps growing, the layer is probably wrong. Say so instead of adding item N+1.
19. ScaleBFM runs on the GPU the prompt names, at most 4 processes with `OMP_NUM_THREADS=2` each when you run it yourself. Never run ScaleBFM while a SONIC stack is up (code: the driver checks).
20. Long jobs run under `nohup setsid` with a log. Never edit a bash script while it is running.
21. Keep artifacts under `outputs/scalebfm_bottle_rsi_20261015/`.
22. Stop and write an escalation instead of acting when a step needs any of: the scene, the bottle, the success rule, the controller checkpoint or its training, the pose source, the real robot, the frozen rules, or the budget.
