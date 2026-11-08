"""The hug-lift height probe (2026-11-07, after gso_pack_all_s1's sealed pool: 0 of 20 two-hand pack rounds put the
object in, most ending in put_raise "kept in both hands"): is the two-hand raise limited by the load's mass or by
the hands' height? Each box of `--tasks` (gso_bimanual_s<n>'s lift tasks) is hugged at each mass of `--masses` by the
main library's `bimanual.pick`, then both hands climb the rungs of probe_hug_skills.RUNGS_Z with the put-in's own raise
move (probe_hug_skills/__init__.py). Every (mass, start) is one round through scene_round.py, so the records carry the
usual steps and the 50 Hz traces; the same starts are used at every mass.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src $PY scripts/rsi/probe_hug_lift.py \\
        [--family gso_bimanual_s1] [--tasks lift_nor_cal_box,lift_gso_0356] [--n 3] [--masses 0.3,0.8,1.5] \\
        [--seed 880000] [--out <dir>] [--library <skills dir>] [--pool <another run's pool>] [--mass <one mass>]

Writes <out>/pool/ (the starts), <out>/m<mass>/ (records and traces per mass) and <out>/summary.json: per task and
mass, per rung, the rounds that still held the box with both hands and the median / max of the jaw height error, the
reach's attitude and IK error, the box's slip and tilt; and each round's highest rung held. capability_write.py reads
the summary into reach_bimanual.hug_lift.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import family as F  # noqa: E402

SKILLS = HERE / "probe_hug_skills"
OUT = REPO / "outputs" / "scene_rsi_20261016" / "capability" / "hug_lift"
FIELDS = ("jaw_dz_cm", "left_jaw_err_cm", "right_jaw_err_cm", "left_dz_cm", "right_dz_cm", "left_dxy_cm", "right_dxy_cm", "attitude_err_deg", "ik_err_cm", "slip_cm", "slip_dz_cm",
          "tilt_deg", "box_rise_cm", "hand_gap_cm", "pelvis_z")  # fmt: skip


def mass_dir(out: Path, mass: float) -> Path:
    return out / f"m{mass:.1f}"


def run_mass(fam_name: str, out: Path, labels: list[str], mass: float, library: str) -> None:
    import scene_round as SR

    sets = {"MASS_KG": mass, **({"LIBRARY": library} if library else {})}
    SR.run(fam_name, out / "pool", SKILLS, mass_dir(out, mass), labels, sets, True)


def _stat(vals: list) -> dict | None:
    v = [float(x) for x in vals if x is not None]
    return None if not v else dict(median=round(float(np.median(v)), 2), max=round(float(np.max(v)), 2))


def summarize(out: Path, masses: list[float], labels: list[str]) -> dict:
    rows: dict = {}
    for mass in masses:
        for label in labels:
            f = mass_dir(out, mass) / f"{label}.json"
            if not f.exists():
                continue
            rec = json.loads(f.read_text())
            steps = rec.get("steps") or []
            hug = next((s for s in steps if s.get("step") == "hugged"), {})
            failed = next((s for s in steps if s.get("step", "").startswith("hug_") and not s.get("ok")), None)
            rungs = [s for s in steps if s.get("step") == "rung"]
            top = next((s for s in steps if s.get("step") == "ladder_top"), {})
            rows.setdefault(rec["task"], {}).setdefault(f"{mass:.1f}", []).append(dict(
                label=label, hug_ok=hug.get("pick_ok"), hug_touching=hug.get("touching"),
                hug_failed_at=None if failed is None else f"{failed['step']}: {failed.get('why')}",
                fell=rec.get("fell"), top_rung=top.get("rung"), top_jaw_z=top.get("jaw_z"),
                top_under_z=top.get("under_z"),
                rungs=[{k: s.get(k) for k in ("target_z", "jaw_z", "under_z", "touching", "fell", *FIELDS)}
                       for s in rungs]))  # fmt: skip
    table: dict = {}
    for task, by_mass in sorted(rows.items()):
        table[task] = {}
        for mass, rs in sorted(by_mass.items()):
            hugged = [r for r in rs if r["hug_touching"] == ["left", "right"]]
            ladder: dict = {}
            for r in hugged:
                for s in r["rungs"]:
                    ladder.setdefault(f"{s['target_z']:.2f}", []).append(s)
            per_rung = {}
            for z, ss in sorted(ladder.items()):
                held = [s for s in ss if s["touching"] == ["left", "right"] and not s["fell"]]
                per_rung[z] = dict(
                    reached=len(ss), held=len(held),
                    jaw_err_cm=_stat([max(s["left_jaw_err_cm"], s["right_jaw_err_cm"]) for s in ss]),
                    jaw_dz_cm=_stat([s["jaw_dz_cm"] for s in ss]),
                    hand_dz_cm=_stat([min(s["left_dz_cm"], s["right_dz_cm"]) for s in ss]),  # the lower hand's miss
                    hand_dxy_cm=_stat([max(s["left_dxy_cm"], s["right_dxy_cm"]) for s in ss]),
                    box_rise_cm=_stat([s["box_rise_cm"] for s in ss]),
                    attitude_err_deg=_stat([s["attitude_err_deg"] for s in ss]),
                    ik_err_cm=_stat([s["ik_err_cm"] for s in ss]),
                    slip_cm=_stat([s["slip_cm"] for s in ss]),
                    slip_dz_cm=_stat([s["slip_dz_cm"] for s in ss]),
                    tilt_deg=_stat([s["tilt_deg"] for s in ss]),
                    under_z=_stat([s["under_z"] for s in ss]),
                    pelvis_z=_stat([s["pelvis_z"] for s in ss]),
                )  # fmt: skip
            table[task][mass] = dict(
                rounds=len(rs), hugged=len(hugged),
                hug_failures=[r["hug_failed_at"] for r in rs if r["hug_touching"] != ["left", "right"]],
                top_rung=[r["top_rung"] for r in hugged], top_jaw_z=[r["top_jaw_z"] for r in hugged],
                top_under_z=[r["top_under_z"] for r in hugged], per_rung=per_rung)  # fmt: skip
    return dict(masses=masses, labels=labels, table=table, rounds=rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="gso_bimanual_s1")
    ap.add_argument("--tasks", default="lift_nor_cal_box,lift_gso_0356")
    ap.add_argument("--n", type=int, default=3, help="starts per task")
    ap.add_argument("--masses", default="0.3,0.8,1.5")
    ap.add_argument("--seed", type=int, default=880000, help="first start seed (clear of every pool's range)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--library", default="", help="a skills package directory to hug with (an RSI incumbent's skills/)")
    ap.add_argument("--pool", type=Path, default=None, help="reuse this pool's starts (another run's <out>/pool)")
    ap.add_argument("--mass", type=float, default=None, help="worker: run this mass only")
    a = ap.parse_args(argv)
    fam = F.load(a.family)
    tasks = [t for t in a.tasks.split(",") if t]
    masses = [float(x) for x in a.masses.split(",") if x]
    pool = a.out / "pool"
    if a.mass is not None:
        run_mass(a.family, a.out, F.labels(pool), a.mass, a.library)
        return 0
    if a.pool is not None and not pool.exists():
        import shutil

        shutil.copytree(a.pool, pool)
    if not (pool / "_pool.json").exists():
        F.draw(fam, "diag", pool, seed=a.seed, n=a.n * len(tasks), tasks=tasks)
    labels = F.labels(pool)
    procs = []
    for mass in masses:  # one process per mass, in parallel: each loads the scene and the controller once
        log = (a.out / f"m{mass:.1f}.log").open("w")
        cmd = [
            sys.executable,
            __file__,
            "--family",
            a.family,
            "--out",
            str(a.out),
            "--mass",
            str(mass),
            "--library",
            a.library,
        ]
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()))
    codes = [p.wait() for p in procs]
    summary = dict(library=a.library or "main", **summarize(a.out, masses, labels))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["table"], indent=1))
    return max(codes)


if __name__ == "__main__":
    raise SystemExit(main())
