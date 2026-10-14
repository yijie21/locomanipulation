"""TODO 25.9.1 phase 3 (docs/handoff/scalebfm-warehouse-rsi-20261014.md §9, §11): the sealed set, untouched until the loop
stopped. SONIC's 80 rounds are its own result and ScaleBFM's start states; ScaleBFM runs the loop's final incumbent and
the starting incumbent (25.7's port) from them. Judged by l2_measure's per-round rules and controller-compare §2 for the
warehouse scene: point estimate >= SONIC (exact McNemar reported), falls <= SONIC + 1, drops <= SONIC + 1, ego p95 <=
SONIC x 1.2 -- plus final vs start (the loop's gain on starts it never saw).

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/scalebfm_warehouse_rsi_20261014/sealed.py run
    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/scalebfm_warehouse_rsi_20261014/sealed.py judge
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "experiments" / "controller_compare_20261013"), str(REPO / "src")]

from l2_measure import Torso, compare  # noqa: E402

HIDDEN = REPO / "outputs" / "scalebfm_warehouse_rsi_20261014_hidden"
POOL = HIDDEN / "pool_sealed" / "states"
OUT = REPO / "outputs" / "scalebfm_warehouse_rsi_20261014" / "sealed"
SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
SCENE = "warehouse_box_45x32x33_mars"


def specs() -> dict:
    final = json.loads((HIDDEN / "gate" / "incumbent.json").read_text())["spec"]
    return {"final": Path(final), "start": HERE / "incumbent.yaml"}


def run() -> None:
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", TMPDIR="/workspace/tmp", NUMPY_MADVISE_HUGEPAGE="0")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in specs().items():
        (OUT / f"{name}.yaml").write_text(spec.read_text())
        ps = []
        with (OUT / f"{name}.log").open("a") as fh:
            for i in range(8):
                cmd = [str(SIM_PY), str(HERE / "bfm_round.py"), str(POOL), "--spec", str(spec), "--out", str(OUT / name),
                       "--shard", f"{i}/8"]  # fmt: skip
                ps.append(
                    subprocess.Popen(
                        cmd,
                        cwd=REPO,
                        env={**env, "CUDA_VISIBLE_DEVICES": str(i % 2)},
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                    )
                )
            for p in ps:
                p.wait()
        print(name, len(list((OUT / name).glob("*.json"))), "rounds", flush=True)


def judge() -> dict:
    torso = Torso(SCENE)
    rep = {name: compare(POOL, OUT / name, "box", torso) for name in specs()}
    lines = []
    for name, r in rep.items():
        ok, g = r["ok"], r
        lines.append(
            f"{name}: n {r['n']} SONIC {ok['sonic']} ScaleBFM {ok['scalebfm']} (both {ok['both']}, only SONIC "
            f"{ok['only_sonic']}, only ScaleBFM {ok['only_scalebfm']}, exact McNemar p {ok['p']}); point estimate >= SONIC: "
            f"{'yes' if ok['scalebfm'] >= ok['sonic'] else 'no'}; falls {g['fell']['scalebfm']} vs {g['fell']['sonic']} "
            f"({'pass' if g['fell']['scalebfm'] <= g['fell']['sonic'] + 1 else 'FAIL'}); drops {g['dropped']['scalebfm']} vs "
            f"{g['dropped']['sonic']} ({'pass' if g['dropped']['scalebfm'] <= g['dropped']['sonic'] + 1 else 'FAIL'}); ego p95 "
            f"{g['ego_p95']['scalebfm']} vs {g['ego_p95']['sonic']} "
            f"({'pass' if g['ego_p95']['scalebfm'] <= 1.2 * g['ego_p95']['sonic'] else 'FAIL'}); measured end state "
            f"{r['measured']['scalebfm']} vs {r['measured']['sonic']}; RTF {r['rtf']}; void {r['void']}"
        )
    a, b = rep["final"]["rounds"], rep["start"]["rounds"]
    keys = sorted(set(a) & set(b))
    fixed = sum(a[k]["scalebfm"]["ok"] and not b[k]["scalebfm"]["ok"] for k in keys)
    broken = sum(b[k]["scalebfm"]["ok"] and not a[k]["scalebfm"]["ok"] for k in keys)
    from l2_measure import mcnemar_exact

    lines.append(
        f"final vs start (ScaleBFM): {sum(a[k]['scalebfm']['ok'] for k in keys)} vs {sum(b[k]['scalebfm']['ok'] for k in keys)} "
        f"of {len(keys)}; fixed {fixed}, broken {broken}, exact McNemar p {round(mcnemar_exact(fixed, broken), 6)}"
    )
    for name, r in rep.items():
        lines.append(f"{name} failures: ScaleBFM {r['why_counts']['scalebfm']}; SONIC {r['why_counts']['sonic']}")
    out = dict(report={k: {x: y for x, y in v.items() if x != "rounds"} for k, v in rep.items()}, lines=lines)
    (OUT / "sealed.json").write_text(json.dumps(out, indent=1, default=str))
    (OUT / "sealed.md").write_text("\n".join(f"- {x}" for x in lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "judge"):
        raise SystemExit("usage: sealed.py run|judge")
    run() if sys.argv[1] == "run" else judge()
