#!/bin/bash
# Capture the tree the visible200 eval ACTUALLY runs against, at its start and at its end.
#
# Why this exists: run_learn_pilot writes config_snapshot.json once, at the pilot's first stage.
# For this run that was the 07:02 collect, so it records git_sha 56617ae and the source digests of
# that moment -- while the eval starts ~16:00 against a tree that has moved several commits and
# carries ~31 uncommitted files from other sessions. Attributing the eval's number to the 07:02
# snapshot is exactly the failure mode ledger bottle §17.7 is about: a number whose conditions
# cannot be established afterwards. Two captures, so a mid-eval tree change is visible too.
set -u
cd /workspace/code/locomanipulation
E=outputs/experiments/visible200_20260910
D=$E/eval_provenance
mkdir -p "$D"
snap() {  # $1 = label
    local d="$D/$1"; mkdir -p "$d"
    date -u +%FT%TZ > "$d/captured_utc"
    git rev-parse HEAD > "$d/git_sha"
    git status --porcelain > "$d/worktree_status"
    git diff HEAD > "$d/worktree.patch"
    PYTHONPATH=src python3 -c "
import json
from locomani.pipeline.provenance import source_digests
json.dump(source_digests(), open('$d/source_digests.json', 'w'), indent=1)
" 2>/dev/null || echo "source_digests failed" > "$d/source_digests.error"
}
until grep -q "eval start" "$E/runner_train_eval.log" 2>/dev/null; do sleep 30; done
snap at_eval_start
until grep -q "eval exit" "$E/runner_train_eval.log" 2>/dev/null; do sleep 60; done
snap at_eval_end
if cmp -s "$D/at_eval_start/source_digests.json" "$D/at_eval_end/source_digests.json"; then
    echo "tree unchanged during the eval" > "$D/verdict"
else
    echo "TREE CHANGED DURING THE EVAL -- diff the two source_digests.json before trusting the number" > "$D/verdict"
fi
cat "$D/verdict"
