# Cheaper SIMPLE rounds without changing outcomes (TODO 52, 2026-11-25)

Two mechanisms, both approved by the user. Evidence, tables and scripts: `outputs/fast_rounds_20261125/notes.md` and
`scripts/experiments/fast_rounds_20261125/`. Branch `fast-rounds` (worktree `locomanipulation-fast`), not committed.

## 1. Lazy render with warm-up renders is the default in every round

An OpenOvenTeleop episode costs 85 s with the head camera rendered every step and 36 s lazily (the cameras render only when the
skill looks). The loop's diagnostic rounds already ran lazily; the gate, the sealed set and filming rendered every step.

Why the lazy look differed: one render of a frozen scene is brighter than the frame a running simulation shows (the renderer's
auto-exposure adapts over many frames: mean brightness 53.8 on the first render, 51.3 after 12 sub-framed renders, 51.0 in the
continuous run). Mean absolute pixel difference from the continuous image: 3.4 (door), 6.7 (faucet, a specular table), 2.9
(oven) on the first render, 1.5, 4.1 and 1.1 after 48 renders; two continuous episodes of the same start differ by 0.9-1.3
(render noise, not repeatable run to run). On one faucet start the unwarmed look lost the object for SAM 3 (0 instances against 9).

Fix: `lazy_warm` (SimpleConfig, default 48): `eyes` renders the same state 1 + 48 times before it reads (about 2 s per look;
server.py `lazy_render` message takes `warm` and `subframes`). Outcomes on the loop's 60 diagnostic starts (15 per task), round-1
package, `{"feed_torque": true}`, ScaleBFM: every step 46 and 47 of 60, lazy (warm 48) 51 and 49, lazy unfixed 50; per-start
agreement every-vs-lazy 0.85, within every-step repeats 0.92, within lazy repeats 0.83; pooled 77.5 % against 83.3 %, Fisher
p 0.33; 73-88 s per start against 28 s. Equivalent within what 60 starts can show (a 20-point difference would be seen; a
5-point one would not), so it is now the default for every loop round, the gate and the sealed set included.

How to run: `SimpleConfig.lazy_render` defaults to true, `lazy_warm` to 48. The explicit every-step mode, for official benchmark
and exam reports: `--set lazy_render=false`, or `"render": "every_step"` in the family's `bench` block (scripts/rsi/backend_simple.py
`render_sets`; the render mode enters the bed signature through `render_tag`, so the two never share a pool; probe and gate now
have one signature). The esc bed's `bed_env_gate` (families/simple_q.json) is untouched.

## 2. Replay from a saved state (memo)

`src/locomani/bench/simple/memo.py` ports TODO 46.1's memo: the key of a top-level Motor primitive (walk_to, turn, sidestep,
posture, reach, adopt_hand, grip, release, squeeze, hold) chains the start's label, every earlier primitive and its arguments;
after a miss the state is stored, on a hit it is restored and the stepping skipped while the skill's Python runs unchanged.
Stored: server (`Episode.snap` / `restore`, server.py: MuJoCo data, env / task / robot counters, the judge, the Isaac scene
synced) and client (mirror Sim, ScaleBFM history, Motor, Belief and odometry, clock), by `statecopy.py`, only the attributes a
primitive changed. Perception is stored with the state (`memo_eyes`): renders are not repeatable, so without it no two runs of
a start ever agree on a goal to the last bit and no key would hit.

Correctness: 4 tasks x 3 starts, a run restored from another setting's states against the same run from scratch with perception
frozen, step by step: bit-identical (0.0 difference in every joint and base reading over the 1 948 steps stepped after 5 884 restored ones, every row field equal; a second run of the base setting with live perception, all states and looks restored, gave the base run's row). A
bug found by the speed test and fixed: restoring rewound the server's snapshot counter (ids were reused).

Speed: a sweep of 5 values of the move speed over 15 starts of CloseDoorTeleop (75 runs): 1 776 s without, 1 175 s with the memo
(1.5x; the other four values of a start take 12.7 s against 23.7 s; the first value pays about 4 s for taking snapshots).
The reset (2-15 s) is not skipped. Success identical in all 75 runs.

Limits: the store lives in one process, so the saving needs the variants of a start to run in one bed process (as
`sweep_speed.py` does); the loop's units run one setting per process, so `bench.memo = true` (adds `--set memo=true` outside the
gate and the sealed set; off by default) saves nothing there until the store is persisted. Not persisted, not shared between
servers. A state is restored only for the same episode id, stance and the Motor / bed parameters in the label.

## 3. Depth service leak (2026-11-24)

The Isaac server and the stereo depth service run in sessions of their own (`start_new_session`), so a session that was killed
(SIGTERM, SIGKILL, a restart of the terminal) left them running: four stereo servers kept 13 GB on GPU 1. `Session.run_round`
now turns SIGTERM / SIGHUP / SIGINT into an exit through its `finally` (the services are stopped by pid) and the services get
`PR_SET_PDEATHSIG(SIGTERM)` (`die_with_parent`), which covers SIGKILL. Tests: `tests/test_simple_session.py::NoLeak`.
