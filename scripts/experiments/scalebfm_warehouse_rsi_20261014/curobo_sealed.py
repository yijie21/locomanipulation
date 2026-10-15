"""TODO 26.5.6 (docs/handoff/scalebfm-warehouse-rsi-20261014.md §21): the sealed set for the all-CuRobo cold start -- fixed
configurations only. `run start|final` plays the sandbox's start (incumbent.yaml) or its last accepted incumbent on the
80 sealed starts with the sandbox's own code (batch.py there: 12 processes on GPUs 0 and 1; the route services must be
up and TF_CUROBO_PORTS set). `judge` scores them against SONIC (production, segmented) by controller-compare §2's
warehouse rules, as phase4_sealed.py does, and pairs them with each other and with the earlier sealed results.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    TF_CUROBO_PORTS=5795,5796 $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/curobo_sealed.py run start
    TF_CUROBO_PORTS=5795,5796 $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/curobo_sealed.py run final
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/curobo_sealed.py judge
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "experiments" / "controller_compare_20261013"), str(REPO / "src")]

from l2_measure import Torso, compare, mcnemar_exact  # noqa: E402

REL = HERE.relative_to(REPO)
POOL = REPO / "outputs" / "scalebfm_warehouse_rsi_20261014_hidden" / "pool_sealed" / "states"
OUT = REPO / "outputs" / "scalebfm_warehouse_curobo_20261015" / "sealed"
SANDBOX = Path("/workspace/code/rsi-curobo/arm-c")
SCENE = "warehouse_box_45x32x33_mars"
# earlier sealed results on the same 80 starts: phase 2's final, its CuRobo-route rerun, phase-4 arm A (no route)
EARLIER = {
    "phase2_final": REPO / "outputs" / "scalebfm_warehouse_rsi_20261014" / "sealed" / "final",
    "phase2_final_route": REPO / "outputs" / "scalebfm_warehouse_rsi_20261014" / "route" / "eval_sealed_route",
    "phase4_A": REPO / "outputs" / "scalebfm_warehouse_rsi_20261014" / "sealed" / "phase4_A",
}


def sandbox_batch():
    """The sandbox's batch.py (its REPO, venv and bfm_round), loaded under its own name."""
    spec = importlib.util.spec_from_file_location("sandbox_batch", SANDBOX / REL / "batch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


def spec_of(which: str) -> Path:
    if which == "start":
        return SANDBOX / REL / "incumbent.yaml"
    gate = SANDBOX / "outputs" / "scalebfm_warehouse_rsi_20261014_hidden" / "gate" / "incumbent.json"
    return Path(json.loads(gate.read_text())["spec"])


def run(which: str, procs: int, gpus: str) -> None:
    name = f"curobo_{which}"
    spec = spec_of(which)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.yaml").write_text(spec.read_text())
    batch = sandbox_batch()
    missing = batch.run_jobs([batch.Job(spec, OUT / name)], POOL, procs, OUT / f"{name}.log", gpus)
    print(name, len(list((OUT / name).glob("bfm_*.json"))), "rounds; missing", missing, flush=True)


def judge() -> dict:
    torso = Torso(SCENE)
    dirs = {"curobo_start": OUT / "curobo_start", "curobo_final": OUT / "curobo_final", **EARLIER}
    rep = {n: compare(POOL, d, "box", torso) for n, d in dirs.items() if len(list(d.glob("bfm_*.npz"))) == 80}
    lines = []
    for name, r in rep.items():
        ok = r["ok"]
        lines.append(
            f"{name}: SONIC {ok['sonic']} ScaleBFM {ok['scalebfm']} of {r['n']} (only SONIC {ok['only_sonic']}, only "
            f"ScaleBFM {ok['only_scalebfm']}, exact McNemar p {ok['p']}); falls {r['fell']['scalebfm']} vs "
            f"{r['fell']['sonic']}; drops {r['dropped']['scalebfm']} vs {r['dropped']['sonic']}; ego p95 "
            f"{r['ego_p95']['scalebfm']} vs {r['ego_p95']['sonic']}; failures {r['why_counts']['scalebfm']}"
        )
    pairs = {}
    for a_name in ("curobo_final",):
        for b_name in ("curobo_start", *EARLIER):
            if a_name not in rep or b_name not in rep:
                continue
            a, b = rep[a_name]["rounds"], rep[b_name]["rounds"]
            keys = sorted(set(a) & set(b))
            fixed = sum(a[k]["scalebfm"]["ok"] and not b[k]["scalebfm"]["ok"] for k in keys)
            broken = sum(b[k]["scalebfm"]["ok"] and not a[k]["scalebfm"]["ok"] for k in keys)
            pairs[f"{a_name} vs {b_name}"] = dict(fixed=fixed, broken=broken, p=round(mcnemar_exact(fixed, broken), 6))
            lines.append(
                f"{a_name} vs {b_name}: only {a_name} {fixed}, only {b_name} {broken}, p {pairs[f'{a_name} vs {b_name}']['p']}"
            )
    out = dict(
        report={k: {x: y for x, y in v.items() if x != "rounds"} for k, v in rep.items()}, pairs=pairs, lines=lines
    )
    (OUT / "curobo.json").write_text(json.dumps(out, indent=1, default=str))
    (OUT / "curobo.md").write_text("\n".join(f"- {x}" for x in lines) + "\n")
    print("\n".join(lines))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("which", choices=("start", "final"))
    r.add_argument("--procs", type=int, default=12)
    r.add_argument("--gpus", default="0,1")
    sub.add_parser("judge")
    a = ap.parse_args(argv)
    run(a.which, a.procs, a.gpus) if a.cmd == "run" else judge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
