"""TODO 32.4.4: why r1_slide_then_grasp lost 38 -> 31 / 80 when walks went from 0.25 to 0.5 m/s (all nine flips the
book under the shelf, `pick_thick_novel`: entry timeouts, the book knocked). Hypothesis: after a 0.5 m/s walk the body
still sways when the reach starts. Diagnosis only -- no skill is changed: the same catalog runs under variants that
patch the motor layer at run time, on every start of the family's three pools (book and wallet, 95 each):

  v05_s0     0.5 m/s, no wait (today's setting; its rounds should equal the registry's sleep_20261021 records)
  v05_s05 / v05_s10 / v05_s15
             0.5 m/s, then a hold of 0.5 / 1.0 / 1.5 s after a walk's last leg (the leg that ends at a heading)
  v035_s0 / v025_s0
             0.35 / 0.25 m/s (MotorConfig.walk_speed), no wait

    python3 scripts/experiments/slide_speed_20261021/probe.py pools        # local: copy the starts
    PYTHONPATH=src $PY scripts/experiments/slide_speed_20261021/probe.py launch [--procs 48] [--gpus 0,1,2,3]
    python3 scripts/experiments/slide_speed_20261021/probe.py report
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "scripts/rsi"), str(REPO / "src")]

FAMILY = "r1_slide_then_grasp"
OUT = REPO / "outputs/slide_speed_20261021"
POOL = OUT / "pool"
SB = Path("/workspace/code/rsi-scene/r1-slide-a/outputs")
SOURCES = [SB / "scene_rsi_20261016_hidden" / FAMILY / "pool_sealed", SB / "scene_rsi_20261016" / FAMILY / "pool_diag",
           SB / "scene_rsi_20261016_hidden" / FAMILY / "pool_accept"]  # fmt: skip
VARIANTS = {"v05_s0": (0.5, 0.0), "v05_s05": (0.5, 0.5), "v05_s10": (0.5, 1.0), "v05_s15": (0.5, 1.5),
            "v035_s0": (0.35, 0.0), "v025_s0": (0.25, 0.0)}  # fmt: skip


def pools() -> None:
    POOL.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in SOURCES:
        for p in src.glob("*.json"):
            if not p.name.startswith("_"):
                shutil.copy2(p, POOL / p.name)
                n += 1
    print(n, "starts ->", POOL)


def work(variant: str, labels: list[str]) -> None:
    """One process: the catalog under `variant`'s motor patch, on `labels`."""
    import scene_round as SR

    from locomani.backends.scalebfm import motor as MM

    speed, settle = VARIANTS[variant]
    walk = MM.Motor.walk_to

    def walk_to(self, xy, yaw=None, *, speed=None, face="travel"):
        out = walk(self, xy, yaw, speed=speed, face=face)
        if settle > 0 and yaw is not None and out.ok:
            self.hold(settle)
            out.info["settle_s"] = settle
        return out

    MM.Motor.walk_to = walk_to
    fam = SR.F.load(FAMILY)
    catalog = REPO / "src/locomani/tasks/scene/skills"
    skills = SR.load_skills(catalog)
    h = SR.skills_hash(catalog)
    world = SR.make_world(fam, "scalebfm")
    world.motor.cfg = replace(world.motor.cfg, walk_speed=speed)
    out = OUT / "rounds" / variant
    out.mkdir(parents=True, exist_ok=True)
    for x in labels:
        SR.run_round(fam, world, skills, h, POOL, out, x, {}, False, "scalebfm")


def launch(procs: int, gpus: str) -> None:
    labels = sorted(p.stem for p in POOL.glob("*.json") if not p.name.startswith("_"))
    jobs = []
    per = max(1, procs // len(VARIANTS))
    for v in VARIANTS:
        for k in range(per):
            jobs.append((v, labels[k::per]))
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", NUMPY_MADVISE_HUGEPAGE="0")
    devs = gpus.split(",")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    ps = []
    for i, (v, ls) in enumerate(jobs):
        cmd = [sys.executable, __file__, "work", v, ",".join(ls)]
        fh = open(OUT / "logs" / f"{v}_{i}.log", "w")
        ps.append(subprocess.Popen(cmd, cwd=REPO, env={**env, "CUDA_VISIBLE_DEVICES": devs[i % len(devs)]}, stdout=fh,
                                   stderr=subprocess.STDOUT))  # fmt: skip
    t0 = time.time()
    for p in ps:
        p.wait()
    print(f"{len(ps)} workers, {round(time.time() - t0)} s")


def report() -> dict:
    from gate import mcnemar_one_sided

    rd = lambda v: {p.stem: json.loads(p.read_text()) for p in (OUT / "rounds" / v).glob("*.json")}  # noqa: E731
    base = rd("v05_s0")
    out = {}
    for v in VARIANTS:
        r = rd(v)
        row = {}
        for task in ("pick_thick_novel", "pick_wallet"):
            xs = sorted(x for x in r if x.startswith(task) and x in base)
            b = sum(r[x]["ok"] and not base[x]["ok"] for x in xs)
            c = sum(base[x]["ok"] and not r[x]["ok"] for x in xs)
            fail = collections.Counter(next((s["step"] for s in r[x].get("steps") or [] if not s.get("ok", True)), "-")
                                       for x in xs if not r[x]["ok"])  # fmt: skip
            row[task] = dict(ok=sum(bool(r[x]["ok"]) for x in xs), n=len(xs), b=b, c=c, p_better=mcnemar_one_sided(b, c),
                             fell=sum(bool(r[x].get("fell")) for x in xs), failed_steps=dict(fail.most_common(4)))  # fmt: skip
        out[v] = row
        print(v, json.dumps(row))
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pools", "launch", "work", "report"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--gpus", default="0,1,2,3")
    a = ap.parse_args()
    if a.cmd == "pools":
        pools()
    elif a.cmd == "launch":
        launch(a.procs, a.gpus)
    elif a.cmd == "work":
        work(a.args[0], [x for x in a.args[1].split(",") if x])
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
