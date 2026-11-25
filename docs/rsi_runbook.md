# Runbook: running the RSI loop on a benchmark family (SIMPLE example; moved from the prompt directories, TODO 54 phase 2, one backend in phase 3)

All commands run from the repository root. `PY` is the sim python; CPU only is fine for steps (a), (b) and (d).

    cd /workspace/code/locomanipulation
    export PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python PYTHONPATH=src CUDA_VISIBLE_DEVICES=""
    F=simple_art
    O=outputs/scene_rsi_20261016/$F  H=outputs/scene_rsi_20261016_hidden/$F

Every benchmark family runs on the one benchmark backend, `scripts/rsi/backend_bench.py` (its `ADAPTERS` table: simple, locomanilab,
bigym2, humanoidarena; shared code in `bench_common.py`, `bench_pools.py`, `bench_rows.py`, the adapter's rows in `bench_<adapter>.py`).
The old standalone `simple` / `lml` backends, the esc bed (`bed_simple.py`, `official_runner`, `probe_bed`) and the families `simple_fake`,
`lml_q`, `lml_fridge`, `lml_drawer` are gone (TODO 54 phase 3); `simple_q` is a bench family now.

Run the loop from a sandbox for anything but a stub (`put_skills` rewrites the loop's working package; the seed is never written:
the working copy is `$O/package`, created from the seed on first use, and every backend write is refused inside the seed).

## (a) Draw the pools (no bed needed)
    mkdir -p $O $H
    $PY scripts/rsi/backend.py draw $F diag   $O/pool_diag
    $PY scripts/rsi/backend.py draw $F accept $H/pool_accept
    $PY scripts/rsi/backend.py draw $F sealed $H/pool_sealed
Or `python3 scripts/rsi/sandbox.py /workspace/code/rsi-scene/<name> $F`, which builds the sandbox (git archive HEAD, then the family's
`sandbox.overlay` from the working tree; a benchmark sandbox holds the core modules and no `scripts/rsi/scene/`) and draws diag and accept
inside it. Sealed: `$PY scripts/rsi/sealed.py $F`.

## (b) One stub iteration (no model call)
    $PY scripts/rsi/loop.py run $F --stub --max-iters 1
The stub gate is a dry run (static checks only). `RSI_STUB_REAL_GATE=1` runs the real gate too.

## (c) One real iteration (model calls, GPUs)
    nohup setsid $PY scripts/rsi/loop.py run $F --max-iters 1 > $O/run.log 2>&1 &
Needs the role models' credentials and a sandbox checkout. Every round, the gate's and the sealed set's included, runs through the
adapter's runner (step (e)).

## (d) See the ledger
    $PY scripts/rsi/loop.py status $F
    tail -n 20 $O/loop/ledger.jsonl | cut -c1-300          # the iteration's steps
    cat $H/gate/ledger.jsonl | cut -c1-300                  # gate calls (hidden tree: the loop's user only, not a role)
Iteration files: `$O/loop/iter_NN/` (cards, prompts, hypotheses, manager.json, candidate_K, gate.json, gate.log).

## (e) simple_art on the bench adapter (TODO 49.6a, 2026-11-24)
The family file's `bench` block (root of the bench worktree, `options` = BenchOptions overrides, `config` = SimpleConfig `--set`
overrides) makes every round of every step run `python -m locomani.bench.simple run <Task> --ids F --setting J --out F --skills <package>
--bench-set k=v --set kinds=Q --set practice_dir=.. --set run_dir=.. --set lazy_render=true --set lazy_warm=48` (every round, the gate and the sealed set included; `bench.render = "every_step"` gives `lazy_render=false`, the mode for official benchmark and exam reports; `bench.memo = true` (the family's default since TODO 53) adds `--set memo=true --set memo_dir=<scratch>/simple_<family>_memo --set memo_sig=<signature> --set memo_gb=20` outside the gate and the sealed set: saved states of the diagnostic, probe and draft rounds on disk, shared between the processes)
in the bench root with PYTHONPATH=src, no SIMPLE_* variable. The seed is `src/locomani/bench/simple/seed`
(skill.py, pick.py, camera_view.py, __init__.py), the guard is `bench/simple/guard.py`, and the pool and gate signatures carry the
resolved switches (`BenchOptions.signature()`) and the adapter's files.

## What the roles read (TODO 54 phase 2)
Roles read no rules file. `scripts/rsi/rules_core.md` holds the rules every family shares under stable ids (`[win]`, `[gate]`,
`[check-half]` ...), each marked code or prompt. `scripts/rsi/rules_build.py` assembles, per role, the core rules it acts on plus
the family appendix and inserts them in the prompt (`scripts/rsi/prompts/<role>.md`): the appendix is generated (tasks and limits
from the tasks file, editable files and refused modules from the adapter's guard.py, `bed_fixed`, pools and caps, the backend's
failure classes) and followed by the family's hand-written notes (`rules_notes/<benchmark>.md`, the family file's `rules` key);
the scene line's appendix is `rules_scene.md`. Print one: `python scripts/rsi/rules_build.py <family> <role>`.
A new rule goes into the core (every family) or into one notes file (one benchmark); the driver cites rules by id.
