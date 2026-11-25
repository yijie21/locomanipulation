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

## 4. The memo on disk (TODO 53, 2026-11-25)

The loop runs one setting per process, so section 2's in-memory store saved nothing there. The states are now also written to
disk and read by the next process. Evidence and tables: `outputs/memo_disk_20261125/notes.md`; scripts:
`scripts/experiments/memo_disk_20261125/`. Branch `memo-disk` (worktree `locomanipulation-memo`), not committed.

**What is on disk.** One file per stored item under `<store>/<signature>/<ab>/<name>` (`src/locomani/bench/simple/memostore.py`): a
state after a primitive (the client's changed attributes, the primitive's return value, trace, calls and row growth, and the server's
state as bytes from the new `snap_dump` message; `snap_load` puts bytes back as a new saved state) and every perception output of
a state. A state is about 0.77 MB compressed (the MuJoCo data of both sides, mostly), a stored look about 1.2 MB. A CloseDoor start
stores 2-27 states, an oven 3-33, a faucet or a trash can about 53. `statecopy.dumps / loads` turn a snapshot into bytes: classes by name,
MuJoCo data by its numeric fields, shared handles (the MuJoCo views a list holds, the renderer's prims) are not written and the live
ones stay. A callable is refused unless it is one of the few set once at setup (the server's `STATIC_FN`; a plain function of the
client) or `Motor.ref`, which is written as the pose it holds when that pose is constant over the policy's look-ahead (a finished
primitive's is); a state that cannot be written stays in memory only and is counted (`unpersistable`; none in 8 starts x 5
runs). The server no longer keeps `ep._last_frame` (a 0.77 MB image, set at setup, not state).

**Signature and walls.** The folder is `sha1(memo_sig, code hash, switches, render mode)`: `memo_sig` is the loop's bed signature
(adapter files, switches, render tag), the code hash covers `bench/options.py`, `bench/simple/*.py`, `backends/scalebfm/*.py` and
`perception/*.py` (`run.code_signature`), so any change to those files, a switch or the render mode reaches no old state. The chain key
inside (section 2) already carries the episode, stance, parameters and the package's `camera_view.py`. The loop places the store at
`<scratch>/simple_<family>_memo` (`backend_simple.memo_store_dir`: the scratch area, TMPDIR or `/workspace/tmp`), refuses a scratch
area inside the repository, the origin checkout or the hidden tree, and the store path is not written into rows. The sandbox audit is
unchanged and clean. The gate and the sealed set never get the memo flags (`memo_sets(fam, full=True)` is empty), so they neither read
nor write it. `bench.memo_gb` (default 20) caps the store, oldest first (a hit refreshes a file), pruned when a process opens it and every 64
writes; writes are temporary file plus rename; a truncated or damaged file is a miss (counted as `corrupt`, and removed). Each row
carries `memo.disk`: hits, misses, writes, bytes read and written, corrupt, pruned, state_hits, eyes_hits, persisted, unpersistable.

**Correctness across processes.** Process A (base setting, memo on, perception recorded) then a fresh process B (fresh server, empty
memory): D, D2 from scratch with A's perception, C with the memo restoring from A's files, L the base setting again with live
perception, 12 starts, 3 per task: C against D, bit-identical in all 12 (0.0 difference in every joint and base reading over the
2 351 steps stepped after 5 828 restored from disk, every row field equal); L (everything from disk) reproduced A's rows; D2 equals D.
No corrupt file in any read.

**Gain in the loop's shape** (CloseDoorTeleop, 15 starts, one `python -m locomani.bench.simple run` process per setting, two sequences
side by side on slots 1 and 2, memo off against on): (a) a sweep of 5 values of `art_speed`: 1 830 s without, 1 263 s with (1.45x;
the first value pays 384 s against 365 s for writing the states, every later value 213-225 s against 363-368 s); (b) the incumbent's
round then a draft that changes only the arc (`art_sweep` 80 to 70): the draft 226 s against 373 s (1.65x), the incumbent's writing
380 s against 368 s. The floor of a call is the services' start and the reset (about 60 s), so a longer round gains more. A repeat of a
value with everything stored: 138 s, and its 15 rows equal the first run's in every field, which two runs without the memo never do
(0 of 15; the renderer is not repeatable).

**On by default.** `bench.memo: true` in `scripts/rsi/families/simple_art.json` (diagnostic, probe and draft rounds). Its probe
signature now differs from its gate signature (the memo rounds are their own pool, as before). Rules `rules_simple_art.md` version 5.

**Open.** (1) The family's `bench.root` is the worktree pinned at 4bdbbe14, older than TODO 52 and 53: its `SimpleConfig` rejects `--set memo` (and `lazy_warm`) as unknown fields (checked), so the loop must move the pin to a commit that holds this branch before the family's `bench.memo: true` can run; until then keep the pin or set `memo` to false. (2) With the memo a round sees the perception that the first round of that state stored; it equals no every-run
render, so its numbers are comparable to other memo rounds, not bit for bit to non-memo ones (success agreed in 74 of 75 runs here).
(3) Any edit of the code files above is a new store (whitespace included); a skill that changes `camera_view.py` shares nothing.
(4) The state stores the MuJoCo data twice on the server (`u` and `mj` hold the same one): 20 % could be saved.
