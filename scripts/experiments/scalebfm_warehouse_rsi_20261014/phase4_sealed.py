"""TODO 25.9.1.6.4 (docs/handoff/scalebfm-warehouse-rsi-20261014.md §17): the sealed set for the phase-4 arms -- its fifth
use, fixed configurations only. `run` plays an arm's last incumbent on the 80 sealed starts with the arm's own sandbox
code (the code its gate accepted it with); `judge` scores every configuration on the sealed set against SONIC by
controller-compare §2's warehouse rules (as sealed.py does) and pairs each arm with start, phase1_best and final.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/phase4_sealed.py run A --gpu 1
    $PY scripts/experiments/scalebfm_warehouse_rsi_20261014/phase4_sealed.py judge
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "experiments" / "controller_compare_20261013"), str(REPO / "src")]

from l2_measure import Torso, compare, mcnemar_exact  # noqa: E402

POOL = REPO / "outputs" / "scalebfm_warehouse_rsi_20261014_hidden" / "pool_sealed" / "states"
OUT = REPO / "outputs" / "scalebfm_warehouse_rsi_20261014" / "sealed"
SANDBOX = Path("/workspace/code/rsi-phase4")
SCENE = "warehouse_box_45x32x33_mars"
CONFIGS = ("start", "phase1_best", "final", "phase4_A", "phase4_B")


def run(arm: str, gpu: str, procs: int) -> None:
    s = SANDBOX / f"arm-{arm.lower()}"
    spec = Path(
        json.loads((s / "outputs" / "scalebfm_warehouse_rsi_20261014_hidden" / "gate" / "incumbent.json").read_text())[
            "spec"
        ]
    )
    name = f"phase4_{arm}"
    (OUT / f"{name}.yaml").write_text(spec.read_text())
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", TMPDIR="/workspace/tmp", NUMPY_MADVISE_HUGEPAGE="0")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    rnd = s / "scripts" / "experiments" / "scalebfm_warehouse_rsi_20261014" / "bfm_round.py"
    with (OUT / f"{name}.log").open("a") as fh:
        ps = [subprocess.Popen([str(s / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"), str(rnd),
                                str(POOL), "--spec", str(spec), "--out", str(OUT / name), "--shard", f"{i}/{procs}"],
                               cwd=s, env=env, stdout=fh, stderr=subprocess.STDOUT) for i in range(procs)]  # fmt: skip
        for p in ps:
            p.wait()
    print(name, len(list((OUT / name).glob("*.json"))), "rounds", flush=True)


def judge() -> dict:
    torso = Torso(SCENE)
    rep = {n: compare(POOL, OUT / n, "box", torso) for n in CONFIGS if len(list((OUT / n).glob("*.npz"))) == 80}
    lines = []
    for name, r in rep.items():
        ok, g = r["ok"], r
        lines.append(
            f"{name}: SONIC {ok['sonic']} ScaleBFM {ok['scalebfm']} of {r['n']} (only SONIC {ok['only_sonic']}, only "
            f"ScaleBFM {ok['only_scalebfm']}, exact McNemar p {ok['p']}); falls {g['fell']['scalebfm']} vs "
            f"{g['fell']['sonic']} ({'pass' if g['fell']['scalebfm'] <= g['fell']['sonic'] + 1 else 'FAIL'}); drops "
            f"{g['dropped']['scalebfm']} vs {g['dropped']['sonic']} "
            f"({'pass' if g['dropped']['scalebfm'] <= g['dropped']['sonic'] + 1 else 'FAIL'}); ego p95 "
            f"{g['ego_p95']['scalebfm']} vs {g['ego_p95']['sonic']} "
            f"({'pass' if g['ego_p95']['scalebfm'] <= 1.2 * g['ego_p95']['sonic'] else 'FAIL'}); measured end state "
            f"{r['measured']['scalebfm']}; failures {r['why_counts']['scalebfm']}"
        )
    pairs = {}
    for arm in ("phase4_A", "phase4_B"):
        for ref in ("start", "phase1_best", "final", "phase4_A"):
            if arm not in rep or ref not in rep or arm == ref:
                continue
            a, b = rep[arm]["rounds"], rep[ref]["rounds"]
            keys = sorted(set(a) & set(b))
            fixed = sum(a[k]["scalebfm"]["ok"] and not b[k]["scalebfm"]["ok"] for k in keys)
            broken = sum(b[k]["scalebfm"]["ok"] and not a[k]["scalebfm"]["ok"] for k in keys)
            pairs[f"{arm} vs {ref}"] = dict(fixed=fixed, broken=broken, p=round(mcnemar_exact(fixed, broken), 6))
            lines.append(
                f"{arm} vs {ref}: only {arm} {fixed}, only {ref} {broken}, exact McNemar p {pairs[f'{arm} vs {ref}']['p']}"
            )
    out = dict(
        report={k: {x: y for x, y in v.items() if x != "rounds"} for k, v in rep.items()}, pairs=pairs, lines=lines
    )
    (OUT / "phase4.json").write_text(json.dumps(out, indent=1, default=str))
    (OUT / "phase4.md").write_text("\n".join(f"- {x}" for x in lines) + "\n")
    print("\n".join(lines))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("arm", choices=("A", "B"))
    r.add_argument("--gpu", default="1")
    r.add_argument("--procs", type=int, default=6)
    sub.add_parser("judge")
    a = ap.parse_args(argv)
    run(a.arm, a.gpu, a.procs) if a.cmd == "run" else judge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
