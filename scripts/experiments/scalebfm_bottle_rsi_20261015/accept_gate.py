"""TODO 26.3 (user 2026-10-15) acceptance gate for the bottle loop: scalebfm_warehouse_rsi_20261014/accept_gate.py (the
warehouse plan §6.4, frozen) with the bottle's round, frozen fields and outcome rules (docs/handoff/scalebfm-bottle-rsi-
20261015.md §3, §6). Code decides; no model reads the per-round results.

Steps (each writes into <gate>/<n>_<candidate hash>/ and stops at the first rejection): static checks; a 5-round
sentinel of the incumbent; candidate and incumbent on the first 30 starts (reject when net wins <= 0); the other 30;
accept iff b - c >= 3, one-sided exact McNemar p <= 0.05, falls <= incumbent + 1, drops <= incumbent + 1 and every set
field was read on the candidate's path.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/scalebfm_bottle_rsi_20261015/accept_gate.py \
        --candidate cand.yaml [--procs 4] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE)]

from bottle_round import load_spec, spec_hash  # noqa: E402

HIDDEN = REPO / "outputs" / "scalebfm_bottle_rsi_20261015_hidden"  # outside the loop's tree: no role reads it (§4)
POOL = HIDDEN / "pool_accept" / "states"
GATE = HIDDEN / "gate"
SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
STAGE1 = 30
SENTINELS = 5
MIN_NET = 3
ALPHA = 0.05
# §3: the task definition -- layout and success rule -- is frozen
FROZEN = {
    "simple.min_rise_m",
    "simple.disturb_limit_m",
    "recovery_timeout",
}  # fmt: skip -- plan §3: the success rule's knobs (rise, knock-over limit, the round's time budget)


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(at least b of the b + c discordant pairs favour the candidate | no difference)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(b, n + 1)) / 2**n


def outcome(npz: Path) -> dict:
    """The round's verdict plus l2_measure's bottle rules: fell (pelvis < 0.40 m); dropped (lifted >= 3 cm, not >= 3 cm up
    at the end)."""
    f = json.loads(npz.with_suffix(".json").read_text())
    z = np.load(npz)
    rise = z["bottle"][:, 2] - z["bottle"][0, 2]
    return dict(
        ok=bool(f["ok"]),
        why=f.get("why", ""),
        end=round(float(f["events"]["end"]), 2),
        fell=bool(np.min(z["pelvis"][:, 2]) < 0.40),
        dropped=bool(np.max(rise) >= 0.03 and float(rise[-1]) < 0.03),
        fields_read=f.get("fields_read", []),
    )


def run(spec_path: Path, labels: list[str], out: Path, procs: int, log: Path, pool: Path) -> None:
    """bottle_round.py on the given starts, `procs` shards on GATE_GPU."""
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=os.environ.get("GATE_GPU", "1"))
    env.setdefault("TMPDIR", "/workspace/tmp")
    env["NUMPY_MADVISE_HUGEPAGE"] = "0"
    todo = [x for x in labels if not (out / f"{x.replace('sonic', 'bfm', 1)}.json").exists()]
    if not todo:
        return
    n = min(procs, len(todo))
    ps = []
    with log.open("a") as fh:
        for i in range(n):
            cmd = [str(SIM_PY), str(HERE / "bottle_round.py"), str(pool), "--spec", str(spec_path), "--out", str(out),
                   "--only", ",".join(todo), "--shard", f"{i}/{n}"]  # fmt: skip
            ps.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT))
        for p in ps:
            p.wait()
    missing = [x for x in todo if not (out / f"{x.replace('sonic', 'bfm', 1)}.json").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} rounds missing after the run (see {log})")


def tally(cand: dict, inc: dict) -> dict:
    keys = sorted(set(cand) & set(inc))
    b = sum(cand[k]["ok"] and not inc[k]["ok"] for k in keys)
    c = sum(inc[k]["ok"] and not cand[k]["ok"] for k in keys)
    return dict(
        n=len(keys),
        candidate_ok=sum(cand[k]["ok"] for k in keys),
        incumbent_ok=sum(inc[k]["ok"] for k in keys),
        b=b,
        c=c,
        p_one_sided=round(mcnemar_one_sided(b, c), 5),
        candidate_fell=sum(cand[k]["fell"] for k in keys),
        incumbent_fell=sum(inc[k]["fell"] for k in keys),
        candidate_dropped=sum(cand[k]["dropped"] for k in keys),
        incumbent_dropped=sum(inc[k]["dropped"] for k in keys),
        candidate_why={w: sum(cand[k]["why"] == w for k in keys if not cand[k]["ok"]) for w in sorted({cand[k]["why"] for k in keys})},
    )  # fmt: skip


def load_dir(d: Path) -> dict:
    return {p.stem.replace("bfm", "sonic", 1): outcome(p) for p in sorted(d.glob("*.npz"))}


def main(argv=None) -> int:
    global GATE, POOL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="static checks only")
    ap.add_argument(
        "--pool", type=Path, default=POOL, help="start states (tests only: the default is the acceptance pool)"
    )
    ap.add_argument("--gate-dir", type=Path, default=GATE, help="gate state and rounds (tests only)")
    a = ap.parse_args(argv)
    GATE, POOL = a.gate_dir, a.pool
    GATE.mkdir(parents=True, exist_ok=True)
    ledger = GATE / "ledger.jsonl"
    history = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    inc_state = GATE / "incumbent.json"
    if not inc_state.exists():
        inc_state.write_text(
            json.dumps(dict(spec=str(HERE / "incumbent.yaml"), rounds=str(GATE / "incumbent_0")), indent=1)
        )
    inc = json.loads(inc_state.read_text())
    labels = sorted(p.stem for p in POOL.glob("*.npz"))
    spec = load_spec(a.candidate)
    h = spec_hash(spec)
    verdict = dict(
        candidate=str(a.candidate), spec=spec, hash=h, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    verdict["call"] = len([x for x in history if not x.get("dry_run")]) + 1

    def finish(accepted: bool, reason: str, **extra) -> int:
        verdict.update(accepted=accepted, reason=reason, dry_run=a.dry_run, **extra)
        verdict["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with ledger.open("a") as fh:
            fh.write(json.dumps(verdict) + "\n")
        print(json.dumps({k: v for k, v in verdict.items() if k != "spec"}, indent=1))
        return 0 if accepted else 1

    # 1. static
    bad = sorted(set(spec["set"]) & FROZEN)
    if bad:
        return finish(False, f"sets frozen task fields {bad} (§3)")
    if any(x["hash"] == h and not x["accepted"] and not x.get("dry_run") for x in history):
        return finish(False, "this exact spec was rejected before")
    if h == spec_hash(load_spec(inc["spec"])):
        return finish(False, "the candidate is the incumbent")
    if len(labels) < (2 * STAGE1 if POOL == HIDDEN / "pool_accept" / "states" else STAGE1 + 1):
        return finish(False, f"acceptance pool has {len(labels)} starts, needs {2 * STAGE1}")
    if a.dry_run:
        return finish(False, "dry run: static checks passed")
    work = GATE / f"{verdict['call']:02d}_{h}"
    work.mkdir(exist_ok=True)
    log = work / "run.log"
    inc_dir = Path(inc["rounds"])
    # the incumbent's own rounds, once per incumbent
    run(Path(inc["spec"]), labels, inc_dir, a.procs, log, POOL)
    inc_out = load_dir(inc_dir)
    # 2. sentinel
    sent = work / "sentinel"
    run(Path(inc["spec"]), labels[:SENTINELS], sent, a.procs, log, POOL)
    again = load_dir(sent)
    drift = [k for k in again if (again[k]["ok"], again[k]["end"]) != (inc_out[k]["ok"], inc_out[k]["end"])]
    if drift:
        return finish(
            False, f"sentinel: incumbent rounds no longer reproduce ({drift}); the pairing is void -- fix first"
        )
    # 3. stage 1
    cand_dir = work / "candidate"
    (work / "candidate.yaml").write_text(a.candidate.read_text())
    run(work / "candidate.yaml", labels[:STAGE1], cand_dir, a.procs, log, POOL)
    t1 = tally(load_dir(cand_dir), inc_out)
    if t1["b"] - t1["c"] <= 0:
        return finish(False, f"stage 1: net wins {t1['b'] - t1['c']} <= 0 on {t1['n']} pairs", stage1=t1)
    # 4. stage 2
    run(work / "candidate.yaml", labels, cand_dir, a.procs, log, POOL)
    cand_out = load_dir(cand_dir)
    t = tally(cand_out, inc_out)
    # 5. verdict
    # a field must reach the code on this path in at least one round: some are read only on a branch (a retry count
    # only after a failure), so "every round" would refuse them; "no round" is the half-applied setting the rule is for
    unread = sorted(k for k in spec["set"] if not any(k in r["fields_read"] for r in cand_out.values()))
    fails = []
    if t["b"] - t["c"] < MIN_NET:
        fails.append(f"net wins {t['b'] - t['c']} < {MIN_NET}")
    if t["p_one_sided"] > ALPHA:
        fails.append(f"one-sided McNemar p {t['p_one_sided']} > {ALPHA}")
    if t["candidate_fell"] > t["incumbent_fell"] + 1:
        fails.append(f"falls {t['candidate_fell']} > {t['incumbent_fell']} + 1")
    if t["candidate_dropped"] > t["incumbent_dropped"] + 1:
        fails.append(f"drops {t['candidate_dropped']} > {t['incumbent_dropped']} + 1")
    if unread:
        fails.append(f"fields set but never read on this path: {unread}")
    if fails:
        return finish(False, "; ".join(fails), stage1=t1, result=t)
    inc_state.write_text(
        json.dumps(dict(spec=str(work / "candidate.yaml"), rounds=str(cand_dir), since=verdict["call"]), indent=1)
    )
    return finish(True, "accepted", stage1=t1, result=t)


if __name__ == "__main__":
    raise SystemExit(main())
