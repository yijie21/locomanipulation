# Runbook: the SIMPLE family `simple_q` (2026-11-19)

All commands run from the repository root. `PY` is the sim python; CPU only is fine for steps (a), (b) and (d).

    cd /workspace/code/locomanipulation
    export PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python PYTHONPATH=src CUDA_VISIBLE_DEVICES=""
    F=simple_q
    O=outputs/scene_rsi_20261016/$F  H=outputs/scene_rsi_20261016_hidden/$F

Run the loop from a sandbox for anything but a stub (`put_skills` rewrites the loop's working package; the seed in the bed tree is
never written: the working copy is `$O/package`, created from the seed on first use, and every backend write is refused inside the seed).

## (a) Draw the pools (no bed needed; reads the practice files of the bed tree)
    mkdir -p $O $H
    $PY scripts/rsi/family.py draw $F diag   $O/pool_diag
    $PY scripts/rsi/family.py draw $F accept $H/pool_accept
    $PY scripts/rsi/family.py draw $F sealed $H/pool_sealed
Or `$PY scripts/rsi/sandbox.py /workspace/code/rsi-scene/<name> $F`, which builds the sandbox (git archive HEAD, then the family's
`sandbox.overlay` from the working tree) and draws diag and accept inside it. Sealed: `$PY scripts/rsi/sealed.py` as for the scene families.
Needs: the practice files `outputs/escalation_20261116/practice2/<Task>.jsonl` (150 lines per task, `src.level` 0/1/2).

## (b) One stub iteration on the real mirror bed (no model call)
    $PY scripts/rsi/loop.py run $F --stub --max-iters 1
Diagnostic rounds run `bed_simple.py batch` in the bed tree (`bed_root`), one bed process per call, GPUs from `--gpus` (default as the loop).
The stub gate is a dry run (static checks only). `RSI_STUB_REAL_GATE=1` runs the real gate too, which needs step (c)'s official runner.
Environment the loop sets for the bed: SIMPLE_SKILLS, SIMPLE_RUN_DIR, SIMPLE_PRACTICE, SIMPLE_LOOP=1, PYTHONPATH=src, MUJOCO_GL=egl.
WAITS ON THE BED SIDE: `bed_simple.py` must honour SIMPLE_SKILLS (package directory), SIMPLE_RUN_DIR (scratch and output root),
SIMPLE_PRACTICE (practice directory) and SIMPLE_LOOP (loop mode), and write the `comply` field `{ok, why}` in each record.

## (c) One real iteration (model calls, GPUs, official gate)
    nohup setsid $PY scripts/rsi/loop.py run $F --max-iters 1 > $O/run.log 2>&1 &
Needs the role models' credentials as for the scene families, a sandbox checkout, and for the gate the official environment:
`official_runner` in simple_q.json is `scripts/experiments/escalation_20261116/run_official_q.sh`, called in the bed tree as
`<script> <Task> --ids <file> --setting <json> [--stance <json>] --out <rows.jsonl>`; it must bring up what the task needs
(official server, depth server, SAM 3, look pass, grasps), write one bed record per id to `--out`, and shut down again.
The gate runs the incumbent's and the candidate's rounds there (`where="gate"`), at most three servers at once.
WAITS ON THE BED SIDE: that script's real behaviour (the file exists in the bed tree; check it follows the command line above), the
`comply` field, and the caps (`min_net`, `p_max`, `sentinel_flips`) once the official bed's run-to-run noise is known.

## (d) See the ledger
    $PY scripts/rsi/loop.py status $F
    tail -n 20 $O/loop/ledger.jsonl | cut -c1-300          # the iteration's steps
    cat $H/gate/ledger.jsonl | cut -c1-300                  # gate calls (hidden tree: the loop's user only, not a role)
Iteration files: `$O/loop/iter_NN/` (cards, prompts, hypotheses, manager.json, candidate_K, gate.json, gate.log).

## (e) simple_art on the bench adapter (TODO 49.6a, 2026-11-24)
The family file's `bench` block (root of the bench worktree, `options` = BenchOptions overrides, `config` = SimpleConfig `--set`
overrides) makes every round of every step run `python -m locomani.bench.simple run <Task> --ids F --setting J --out F --skills <package>
--bench-set k=v --set kinds=Q --set practice_dir=.. --set run_dir=.. --set lazy_render=true --set lazy_warm=48` (every round, the gate and the sealed set included; `bench.render = "every_step"` gives `lazy_render=false`, the mode for official benchmark and exam reports; `bench.memo = true` (the family's default since TODO 53) adds `--set memo=true --set memo_dir=<scratch>/simple_<family>_memo --set memo_sig=<signature> --set memo_gb=20` outside the gate and the sealed set: saved states of the diagnostic, probe and draft rounds on disk, shared between the processes)
in the bench root with PYTHONPATH=src, no SIMPLE_* variable. (b) and (c) above describe the esc bed, which a family without the block
still uses; with the block there is no mirror bed (`probe_bed` must be "official"). The seed is `src/locomani/bench/simple/seed`
(skill.py, pick.py, camera_view.py, __init__.py), the guard is `bench/simple/guard.py`, and the pool and gate signatures carry the
resolved switches (`BenchOptions.signature()`) and the adapter's files.
