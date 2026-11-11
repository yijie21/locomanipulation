"""Stance after a walk (2026-10-31, canonical starts): gen_reach.py's home step cannot square the feet -- mode 4 pins the
ankles where the gait stopped. Does marching in place (mode 2, the pelvis reference held still) bring the legs back to
one stance? Same random walks as gen_reach.py, then one of
    none    stand 0.3 s
    march   mode 2 at the current pelvis xy and yaw for MARCH_S, then stand 0.3 s
    march2  the same for 2 * MARCH_S
and record the leg joints, the feet in the pelvis heading frame and the arm joints. Writes
outputs/reachwm_20261030/stance/<variant>.npz.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/stance_probe.py march --resets 60
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
import env as E  # noqa: E402

OUT = E.REPO / "outputs" / "reachwm_20261030" / "stance"
MARCH_S = 2.0


def march(m, seconds: float) -> None:
    from locomani.backends.scalebfm.motor import MODE_WALK, Outcome, Pose, yaw_quat

    p, _ = m.pelvis()
    pose = Pose(p.copy(), yaw_quat(m.yaw()), m.ref(m.t).joints.copy(), np.zeros(29))
    m.pinned = None
    m._set_mode(MODE_WALK)
    m.ref = lambda _t, P=pose: P
    t0 = m.t
    m._run("march", lambda t: Outcome(True) if t - t0 >= seconds else None, seconds + 1.0)
    if m.fell_at is None:
        m._stand()


def feet(m) -> np.ndarray:
    p, _ = m.pelvis()
    yaw = m.yaw()
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for side in ("left", "right"):
        a = m.sim.body(f"{side}_ankle_roll_link")[0] - p
        out.append([c * a[0] + s * a[1], -s * a[0] + c * a[1], a[2]])
    return np.array(out).ravel()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["reset", "none", "march", "march2"])
    ap.add_argument("--resets", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7000000)
    a = ap.parse_args(argv)
    env = E.Env("open")
    m = env.m
    rows = dict(q=[], feet=[], fell=[], walked_m=[])
    for i in range(a.resets):
        rng = np.random.default_rng(a.seed + i)
        env.reset(None)
        ang, d = rng.uniform(-math.pi, math.pi), rng.uniform(0.3, 1.2)
        if a.variant != "reset":
            m.walk_to(np.array([d * math.cos(ang), d * math.sin(ang)]), yaw=float(rng.uniform(-math.pi, math.pi)))
        if a.variant.startswith("march") and m.fell_at is None:
            march(m, MARCH_S * (2 if a.variant == "march2" else 1))
        if m.fell_at is None:
            m.hold(0.3)
        rows["q"].append(m.joints().copy())
        rows["feet"].append(feet(m))
        rows["fell"].append(m.fell_at is not None)
        rows["walked_m"].append(d if a.variant != "reset" else 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{a.variant}.npz", **{k: np.array(v) for k, v in rows.items()})
    print(a.variant, "fell", int(np.sum(rows["fell"])), "of", a.resets, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
