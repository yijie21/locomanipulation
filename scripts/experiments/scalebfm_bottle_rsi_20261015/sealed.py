"""TODO 26.3.4 (docs/handoff/scalebfm-bottle-rsi-20261015.md §4, §11): the bottle sealed set, untouched by the loop. `run`
plays a configuration of the loop's sandbox on the 80 sealed starts with the sandbox's own code: `start` (the L2 port the
loop began from) and `final` (its last incumbent). `judge` scores both against SONIC by controller-compare §2's bottle
rules (point estimate >= SONIC, exact McNemar; falls and drops <= SONIC + 1; ego p95 <= SONIC x 1.2) and pairs final with
start.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/scalebfm_bottle_rsi_20261015/sealed.py run /workspace/code/rsi-phase4/bottle-1 start final
    $PY scripts/experiments/scalebfm_bottle_rsi_20261015/sealed.py judge
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "scripts" / "experiments" / "controller_compare_20261013"), str(REPO / "src")]

from l2_measure import Torso, compare, mcnemar_exact  # noqa: E402

POOL = REPO / "outputs" / "scalebfm_bottle_rsi_20261015_hidden" / "pool_sealed" / "states"
OUT = REPO / "outputs" / "scalebfm_bottle_rsi_20261015" / "sealed"
REL = Path("scripts/experiments/scalebfm_bottle_rsi_20261015")
SCENE = "sim_lowcom"


def spec_of(sandbox: Path, name: str) -> Path:
    if name == "start":
        return sandbox / REL / "incumbent.yaml"
    inc = sandbox / "outputs" / "scalebfm_bottle_rsi_20261015_hidden" / "gate" / "incumbent.json"
    return Path(json.loads(inc.read_text())["spec"]) if inc.exists() else sandbox / REL / "incumbent.yaml"


def run(sandbox: Path, names: list[str], procs: int = 12) -> None:
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", TMPDIR="/workspace/tmp", NUMPY_MADVISE_HUGEPAGE="0")
    py = sandbox / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
    for name in names:
        spec = spec_of(sandbox, name)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}.yaml").write_text(spec.read_text())
        with (OUT / f"{name}.log").open("a") as fh:
            ps = [subprocess.Popen([str(py), str(sandbox / REL / "bottle_round.py"), str(POOL), "--spec", str(spec),
                                    "--out", str(OUT / name), "--shard", f"{i}/{procs}"], cwd=sandbox,
                                   env={**env, "CUDA_VISIBLE_DEVICES": str(i % 2)}, stdout=fh, stderr=subprocess.STDOUT)
                  for i in range(procs)]  # fmt: skip
            for p in ps:
                p.wait()
        print(name, len(list((OUT / name).glob("*.json"))), "rounds", flush=True)


def judge() -> dict:
    torso = Torso(SCENE)
    rep = {n: compare(POOL, OUT / n, "bottle", torso) for n in ("start", "final") if (OUT / n).is_dir()}
    lines = []
    for name, r in rep.items():
        ok, g = r["ok"], r
        lines.append(
            f"{name}: n {r['n']} SONIC {ok['sonic']} ScaleBFM {ok['scalebfm']} (only SONIC {ok['only_sonic']}, only "
            f"ScaleBFM {ok['only_scalebfm']}, exact McNemar p {ok['p']}); falls {g['fell']['scalebfm']} vs "
            f"{g['fell']['sonic']}; drops {g['dropped']['scalebfm']} vs {g['dropped']['sonic']} "
            f"({'pass' if g['dropped']['scalebfm'] <= g['dropped']['sonic'] + 1 else 'FAIL'}); ego p95 "
            f"{g['ego_p95']['scalebfm']} vs {g['ego_p95']['sonic']}; failures {r['why_counts']['scalebfm']}; SONIC "
            f"{r['why_counts']['sonic']}; void {r['void']}"
        )
    if {"start", "final"} <= set(rep):
        a, b = rep["final"]["rounds"], rep["start"]["rounds"]
        keys = sorted(set(a) & set(b))
        fixed = sum(a[k]["scalebfm"]["ok"] and not b[k]["scalebfm"]["ok"] for k in keys)
        broken = sum(b[k]["scalebfm"]["ok"] and not a[k]["scalebfm"]["ok"] for k in keys)
        lines.append(
            f"final vs start: only final {fixed}, only start {broken}, exact McNemar p {round(mcnemar_exact(fixed, broken), 6)}"
        )
    out = dict(report={k: {x: y for x, y in v.items() if x != "rounds"} for k, v in rep.items()}, lines=lines)
    (OUT / "sealed.json").write_text(json.dumps(out, indent=1, default=str))
    (OUT / "sealed.md").write_text("\n".join(f"- {x}" for x in lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "judge":
        judge()
    elif len(sys.argv) >= 4 and sys.argv[1] == "run":
        run(Path(sys.argv[2]).resolve(), sys.argv[3:])
    else:
        sys.exit(__doc__)
