"""TODO 25.9.1.6.3 (docs/handoff/scalebfm-warehouse-rsi-20261014.md §17.1): score each phase-4 arm's Reviewer predictions.
Arm A's were scored live by its Prober (iter_NN/probe.json). The flat arm B had no Prober, so after it stopped this
re-runs each of its proposals as an intervention on the 50 diagnostic starts, against the incumbent the proposal was
made for, and writes iter_NN/shadow/probe.json -- nothing the arm ever read. A proposal that needs extension code
nobody wrote (its hook is not registered in the sandbox) cannot run and is listed as unscored. `table` prints both arms.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/phase4_score.py shadow B   # one pool, GPUs 0,1, 12 processes
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/phase4_score.py table
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SANDBOX = Path("/workspace/code/rsi-phase4")
REL = Path("scripts/experiments/scalebfm_warehouse_rsi_20261014")
LOOP = Path("outputs/scalebfm_warehouse_rsi_20261014/loop")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop import start_label  # noqa: E402


def sandbox_loop(arm: str):
    """The arm's own loop module (its REPO, LOOP and extensions are the sandbox's); `arm` is A, B, A2, ..."""
    s = SANDBOX / f"arm-{arm.lower()}"
    spec = importlib.util.spec_from_file_location(f"loop_{arm}", s / REL / "loop.py")
    argv, sys.argv = sys.argv, ["loop.py"]
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def void_iters(arm: str) -> set[int]:
    led = SANDBOX / f"arm-{arm.lower()}" / LOOP / "ledger.jsonl"
    return {
        e["iter"] for e in map(json.loads, led.read_text().splitlines()) if e.get("step") == "review" and e.get("void")
    }


def shadow(arm: str, gpu: str, procs: int) -> None:
    """All pending iterations' runnable proposals in one run_batch pool (across the listed GPUs), then each scored."""
    loop = sandbox_loop(arm)
    loop.ARM, loop.GPU, loop.PROCS = arm[0].upper(), gpu, procs
    hooks = set(loop.extensions.HOOKS)
    skip = void_iters(arm)
    todo, jobs = [], []
    for it in sorted(loop.STATE.glob("iter_*")):
        n = int(it.name[-2:])
        out = it / "shadow"
        if n in skip or not (it / "hypotheses.json").exists() or (it / "probe.json").exists():
            continue
        if (out / "probe.json").exists():
            continue
        inc = json.loads((it / "reviewer_context.json").read_text())["incumbent"]
        hyps = json.loads((it / "hypotheses.json").read_text())["hypotheses"]
        runnable = [h for h in hyps if set(h["delta"].get("extensions", {})) <= hooks]
        unscored = [dict(id=h["id"], error="needs extension code that was never written: " +
                         ", ".join(sorted(set(h["delta"].get("extensions", {})) - hooks))) for h in hyps if h not in runnable]  # fmt: skip
        out.mkdir(exist_ok=True)
        jobs += loop.probe_jobs(runnable, inc, out)
        todo.append((it, out, runnable, unscored, loop.LOOP / "diag" / f"incumbent_{inc['hash']}"))
    if jobs:
        loop.run_batch(jobs, loop.STATE / "shadow_prober.log", gpu)
    for it, out, runnable, unscored, base in todo:
        res = loop.probe_results(runnable, base, out) + unscored
        loop.jdump(out / "probe.json", res)
        print(
            it.name,
            [(r["id"], r.get("ok_before"), r.get("ok_after"), r.get("error", "")[:40]) for r in res],
            flush=True,
        )


def table() -> None:
    rows = []
    for arm in sorted(d.name.removeprefix("arm-").upper() for d in SANDBOX.glob("arm-*")):
        s = SANDBOX / f"arm-{arm.lower()}" / LOOP
        skip = void_iters(arm)
        for it in sorted(s.glob("iter_*")):
            n = int(it.name[-2:])
            pj = it / "probe.json" if (it / "probe.json").exists() else it / "shadow" / "probe.json"
            if n in skip or not pj.exists():
                continue
            hyps = {h["id"]: h for h in json.loads((it / "hypotheses.json").read_text())["hypotheses"]}
            for r in json.loads(pj.read_text()):
                h = hyps[r["id"]]
                delta = json.dumps(h["delta"], ensure_ascii=False)[:70]
                if "error" in r:
                    rows.append(
                        f"| {arm} | {n} | {r['id']} | {delta} | {len(h['predicted_fixed'])} | – | – | {r['error'][:60]} |"
                    )
                    continue
                # hits recomputed from the stored flips: Reviewers name starts sonic_/bfm_/bare (start_label)
                pf = {start_label(x) for x in h["predicted_fixed"]}
                hit = len(pf & set(r["fixed"]))
                rows.append(
                    f"| {arm} | {n} | {r['id']} | {delta} | {len(pf)} | {r['ok_before']} → {r['ok_after']} "
                    f"(+{len(r['fixed'])} / −{len(r['broken'])}) | {hit}/{len(r['fixed'])} | |"
                )
    print("| arm | iter | id | delta | predicted fixed | diagnostic pool | hits / observed fixed | note |")
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("shadow")
    r.add_argument("arm", help="sandbox suffix: B, B2, ...")
    r.add_argument("--gpu", default="0,1", help="GPUs for the pool, comma-separated")
    r.add_argument("--procs", type=int, default=12, help="ScaleBFM processes across those GPUs (about 4.7 GB RAM each)")
    sub.add_parser("table")
    a = ap.parse_args(argv)
    shadow(a.arm, a.gpu, a.procs) if a.cmd == "shadow" else table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
