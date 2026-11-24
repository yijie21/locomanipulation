# Runbook: the LocoManiLab family `lml_q` (2026-11-20)

All commands run from the repository root (the sandbox for anything but a stub). `PY` is the sim python; CPU only is fine
for steps (a) and (d). The bed is the other worktree's `scripts/experiments/escalation_20261116` (`bed_root` and `bed_dir` in
`families/lml_q.json`); the loop calls its runner `run_lml_q.sh`, never an evaluation episode (loop mode takes only `Q` ids).

    cd /workspace/code/rsi-scene/lml_q_r1
    export PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python PYTHONPATH=src CUDA_VISIBLE_DEVICES="" TMPDIR=/workspace/tmp
    F=lml_q
    O=outputs/scene_rsi_20261016/$F  H=outputs/scene_rsi_20261016_hidden/$F

The loop's working package is `$O/package` (created from the seed `.../seed_pkg_lml` on first use; every backend write is refused
inside the seed). `put_skills` rewrites it, so run the loop from a sandbox. Simulator room: two instances at a time (the two
slot locks of the bed tree), GPU 0 only.

## (a) Draw the pools (no bed needed; reads the practice files `practice_dir` of the family)
    mkdir -p $O $H
    $PY scripts/rsi/family.py draw $F diag   $O/pool_diag      # 600 starts: the first 10 lines of each level, per task
    $PY scripts/rsi/family.py draw $F accept $H/pool_accept    # 240 starts (hidden tree)
    $PY scripts/rsi/family.py draw $F sealed $H/pool_sealed    # 420 starts (hidden tree), only after the inner loop ends
`scripts/rsi/sandbox.py /workspace/code/rsi-scene/<name> $F` builds the sandbox and draws diag and accept inside it. Labels are
`<Task>_s<n>` (n = the line of the practice file); the episode id is `<Task>/Q/<n>`.

## (b) One stub iteration (no model call; the diagnostic rounds still run on the real bed)
    nohup setsid $PY scripts/rsi/loop.py run $F --stub --max-iters 1 > $O/stub.log 2>&1 < /dev/null &
A round takes about 17 s of wall time on the bed, two at a time, plus a look and grasp pass the first time a start's cloud is made,
so the 600 diagnostic starts of the incumbent take well over an hour. The stub gate is a dry run (static checks only).
Environment the loop gives the bed: LML_SKILLS (the package), LML_PRACTICE, LML_RUN_DIR, LML_SLOTS, LML_LOOP=1, PYTHONPATH=src.

## (c) One real iteration (model calls, GPUs, gate)
    nohup setsid $PY scripts/rsi/loop.py run $F --max-iters 1 > $O/run.log 2>&1 < /dev/null &
Needs the role models' credentials as for the other families and the sandbox checkout. The gate runs the incumbent's and the
candidate's rounds on the same bed with the full practice files (`where="gate"`).
A role's own check: `PYTHONPATH=src $PY scripts/rsi/try.py $F --skills <package> [--set '<JSON>'] --labels a,b --out <dir>`;
a `set` key with the prefix `stance_` (`stance_d`, `stance_t`, `stance_yaw`, `stance_here`, `stance_d2`, `stance_t2`,
`stance_stay`) reaches the bed as the stance, every other key as a parameter. A second call with the same `--out` runs nothing.

## (d) See the ledger
    $PY scripts/rsi/loop.py status $F
    tail -n 20 $O/loop/ledger.jsonl | cut -c1-300          # the iteration's steps
    cat $H/gate/ledger.jsonl | cut -c1-300                  # gate calls (hidden tree: the loop's user only, not a role)
Iteration files: `$O/loop/iter_NN/` (cards, prompts, hypotheses, manager.json, candidate_K, gate.json, gate.log).
Stopping a running loop: SIGTERM to its process group (`kill -- -<pgid>`); a runner call stops its own children and frees its slot.
