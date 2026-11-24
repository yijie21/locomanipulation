# Runbook: the BiGym 2.0 family on the bench adapter, `bigym2_reach_drawer` (2026-11-24)

All commands run from the repository root (the sandbox for anything but a stub). `PY` is the sim python; the simulator runs on CPU, one at a time. The bed is the bench adapter
`locomani.bench.bigym2` of the bench worktree (`bench.root` in `families/bigym2_reach_drawer.json`); the loop calls `python -m locomani.bench.bigym2 run <task> ...`
through the generic backend `backend_bench.py`, never an evaluation episode: BiGym's evaluation seeds [620000, 630000) are never reachable (the adapter and its server refuse them).

    cd /workspace/code/rsi-scene/bigym2_r1
    export PY=<BiGym2 venv python or the sim python> PYTHONPATH=src TMPDIR=/workspace/tmp
    F=bigym2_reach_drawer
    O=outputs/scene_rsi_20261016/$F  H=outputs/scene_rsi_20261016_hidden/$F

The loop's working package is `$O/package` (created from the adapter's seed `src/locomani/bench/bigym2/seed` on first use; every backend write is refused
inside the seed). `put_skills` rewrites it, so run the loop from a sandbox. Simulator room: one MuJoCo simulator at a time.

## (a) Draw the pools (no bed needed)
    mkdir -p $O $H
    $PY scripts/rsi/family.py draw $F diag   $O/pool_diag      # 20 starts: 10 of the practice pool P per task
    $PY scripts/rsi/family.py draw $F accept $H/pool_accept    # 40 starts of the pool A (hidden tree)
    $PY scripts/rsi/family.py draw $F sealed $H/pool_sealed    # 60 starts of the pool S (hidden tree), only after the inner loop ends
The pools are seed pools: the episode id is `<task>/<P|A|S>/<n>` and the label `<task>_s<n>`; there are no practice files. `scripts/rsi/sandbox.py /workspace/code/rsi-scene/<name> $F`
builds the sandbox and draws diag and accept inside it.

## (b) One stub iteration (no model call; the diagnostic rounds still run on the real bed)
    nohup setsid $PY scripts/rsi/loop.py run $F --stub --max-iters 1 > $O/stub.log 2>&1 < /dev/null &
A round takes 1 to 8 s of wall time, so the 20 diagnostic starts of the incumbent take a few minutes. The stub gate is a dry run (static checks only).
What the loop gives the adapter (arguments, no environment variables): `--skills` the package, `--set pools=P run_dir=... lock_file=...` for a diagnostic round or a probe (`pools=PAS` for the gate and the sealed run),
and the family's `bench.options` as `--bench-set`; PYTHONPATH=src.

## (c) One real iteration (model calls, gate)
    nohup setsid $PY scripts/rsi/loop.py run $F --max-iters 1 > $O/run.log 2>&1 < /dev/null &
Needs the role models' credentials as for the other families and the sandbox checkout. The gate runs the incumbent's and the candidate's rounds on the same bed with the acceptance pool (`where="gate"`).
A role's own check: `PYTHONPATH=src $PY scripts/rsi/try.py $F --skills <package> [--set '<JSON>'] --labels a,b --out <dir>`; a `set` key with the prefix `stance_` is refused
(there is no stance on this backend), every other key is a parameter. A second call with the same `--out` runs nothing.

## (d) See the ledger
    $PY scripts/rsi/loop.py status $F
    tail -n 20 $O/loop/ledger.jsonl | cut -c1-300          # the iteration's steps
    cat $H/gate/ledger.jsonl | cut -c1-300                  # gate calls (hidden tree: the loop's user only, not a role)
Iteration files: `$O/loop/iter_NN/` (cards, prompts, hypotheses, manager.json, candidate_K, gate.json, gate.log).
Stopping a running loop: SIGTERM to its process group (`kill -- -<pgid>`); a runner call stops its own children and frees its lock.
