# Runbook: the HumanoidArena family on the bench adapter, `humanoidarena_pp_box` (2026-11-24; adapted from prompts_lml_bench/RUNBOOK.md)

All commands run from the repository root (the sandbox for anything but a stub). `PY` is the sim python of the adapter. The bed is
the bench adapter `locomani.bench.humanoidarena` of the bench worktree (`bench.root` in `families/humanoidarena_pp_box.json`);
the loop calls `python -m locomani.bench.humanoidarena run pp_box ...` through the generic backend `backend_bench.py`, never an
evaluation episode (the 60 derived evaluation seeds of the task are never reachable).

    cd /workspace/code/rsi-scene/humanoidarena_r1
    export PY=<the adapter's sim python> PYTHONPATH=src
    F=humanoidarena_pp_box
    O=outputs/scene_rsi_20261016/$F  H=outputs/scene_rsi_20261016_hidden/$F

The loop's working package is `$O/package` (created from the adapter's seed `src/locomani/bench/humanoidarena/seed` on first use;
every backend write is refused inside the seed). `put_skills` rewrites it, so run the loop from a sandbox. Simulator room: one
Isaac server at a time, on GPU 1; an episode takes one to three minutes.

## (a) Draw the pools (no bed needed)
    mkdir -p $O $H
    $PY scripts/rsi/family.py draw $F diag   $O/pool_diag      # the pool P: 10 starts of pp_box
    $PY scripts/rsi/family.py draw $F accept $H/pool_accept    # 20 starts of the pool A (hidden tree)
    $PY scripts/rsi/family.py draw $F sealed $H/pool_sealed    # 30 starts of the pool S (hidden tree), only after the inner loop ends
These are seed pools: the episode id is `pp_box/<P|A|S>/<n>` and the label `pp_box_s<n>`; there are no practice files.
`scripts/rsi/sandbox.py /workspace/code/rsi-scene/<name> $F` builds the sandbox and draws diag and accept inside it.

## (b) One stub iteration (no model call; the diagnostic rounds still run on the real bed)
    nohup setsid $PY scripts/rsi/loop.py run $F --stub --max-iters 1 > $O/stub.log 2>&1 < /dev/null &
A round takes one to three minutes, one at a time, so the 10 diagnostic starts of the incumbent take well under an hour. The stub
gate is a dry run (static checks only).
What the loop gives the adapter (arguments, no environment variables): `--skills` the package, `--set pools=P` for a diagnostic round or
a probe and `pools=PAS` for the gate and the sealed check, `run_dir=...`, `lock_file=...`; the family's `bench.options` as
`--bench-set`; PYTHONPATH=src.

## (c) One real iteration (model calls, GPU, gate)
    nohup setsid $PY scripts/rsi/loop.py run $F --max-iters 1 > $O/run.log 2>&1 < /dev/null &
Needs the role models' credentials and the sandbox checkout. The gate runs the incumbent's and the candidate's rounds on the same
bed with the acceptance pool (`where="gate"`).
A role's own check: `PYTHONPATH=src $PY scripts/rsi/try.py $F --skills <package> [--set '<JSON>'] --labels a,b --out <dir>`;
every `set` key is a parameter of `K`; a key with the prefix `stance_` is refused. A second call with the same `--out` runs nothing.

## (d) See the ledger
    $PY scripts/rsi/loop.py status $F
    tail -n 20 $O/loop/ledger.jsonl | cut -c1-300          # the iteration's steps
    cat $H/gate/ledger.jsonl | cut -c1-300                  # gate calls (hidden tree: the loop's user only, not a role)
Iteration files: `$O/loop/iter_NN/` (cards, prompts, hypotheses, manager.json, candidate_K, gate.json, gate.log).
Stopping a running loop: SIGTERM to its process group (`kill -- -<pgid>`); a runner call stops its own children and frees the server.
