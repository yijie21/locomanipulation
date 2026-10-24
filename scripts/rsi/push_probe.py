"""TODO 33.24.2 (user 2026-10-24: push chairs and stools aside before a grasp; measure whether ScaleBFM stays up first):
can the robot move a piece of floor furniture in the movable scene variant (scripts/rsi/movable_scene.py), and how?

For each piece, method and trial: the robot is reset facing the piece from the open side with the most clearance on
the navigation map (STAND_M from it, the trial's lateral offset), settles, and
  hands  -- puts both open hands on the piece's near face at HAND_Z (or the piece's top, lower), level, SHOULDER_HALF
            apart, then walks PUSH_M ahead keeping its facing;
  drag   -- reaches the right hand over the near top edge, grips, and steps PUSH_M back;
  body   -- walks PUSH_M + its depth ahead into it, hands where they rest.
Recorded per round: fell (and when), how far the piece moved and turned, whether it tipped (tilt), where the pelvis
ended against where it was sent, the simulated seconds. The verdict per method is the plan's: stable (no fall) and
moved (>= MOVED_M) in most trials. Nothing here is a skill or a task; it says what a preparatory action could rely on.

    PYTHONPATH=src $PY scripts/rsi/push_probe.py --out <dir> [--keys K ...] [--methods hands drag body] [--trials 4]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

SCENE = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars_movable"
KEYS = ("BlackKitchenChair", "BlackBarStool_B", "StepStool")
METHODS = ("hands", "drag", "body")
STAND_M = 0.30  # the pelvis this far from the piece's near face
HAND_Z = 0.75
SHOULDER_HALF = 0.15
PUSH_M = 0.30
MOVED_M = 0.15
OFFSETS = (0.0, 0.06, -0.06, 0.12)  # the trials' lateral offsets of the stance, metres
SETTLE_S = 1.0


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def approach(world, key: str):
    """(unit vector robot -> piece, the piece's half depth along it, stance xy) for the clearest of 8 sides."""
    from locomani.tasks.scene import choose as C
    from locomani.tasks.scene.graspability import _quat_mat

    o = world.meta["objects"][key]  # the scene file's pose: every trial puts the pieces back there (object_pose
    p, q = np.asarray(o["init_pos"], float), np.asarray(o["init_quat"], float)  # reads zeros before a forward pass)
    half = np.asarray(o["box"]["half_extent"], float)
    Rb = _quat_mat(q)
    best = None
    for t in np.radians(np.arange(0.0, 360.0, 45.0)):
        u = np.array([math.cos(t), math.sin(t)])
        depth = float(np.abs(Rb[:2, :2].T @ u) @ half[:2])
        xy = np.asarray(p[:2], float) - (depth + STAND_M) * u
        cl = C.clearance(world, xy)
        if best is None or cl > best[0]:
            best = (cl, u, depth, xy)
    return best[1], best[2], best[3], best[0]


def trial(world, key: str, method: str, offset: float, all_keys) -> dict:
    m = world.motor
    home = {k: (np.asarray(world.meta["objects"][k]["init_pos"], float), np.asarray(world.meta["objects"][k]["init_quat"], float))
            for k in all_keys}  # fmt: skip
    u, depth, xy, clear = approach(world, key)
    lat = np.array([-u[1], u[0]])
    xy = xy + offset * lat
    yaw = math.atan2(u[1], u[0])
    m.reset(xy, yaw)
    for k, (pos, quat) in home.items():
        m.place(world.meta["objects"][k]["body"], pos, quat)
    m.hold(SETTLE_S)
    p0, q0 = world.object_pose(key)
    p0, q0 = np.asarray(p0, float).copy(), np.asarray(q0, float).copy()
    t0 = m.t
    top = float(p0[2] + world.meta["objects"][key]["box"]["half_extent"][2])
    face = np.asarray(p0[:2], float) - depth * u
    steps = []
    if method == "hands":
        z = min(HAND_Z, top - 0.03)
        pts = [np.r_[face + SHOULDER_HALF * lat - 0.02 * u, z], np.r_[face - SHOULDER_HALF * lat - 0.02 * u, z]]
        rot = _rz(yaw)
        steps.append(("reach", m.reach("both", pts, [rot, rot])))
        if m.fell_at is None:
            steps.append(("walk", m.walk_to(xy + PUSH_M * u, yaw, face="fixed")))
    elif method == "drag":
        g = np.r_[face + 0.02 * u, top - 0.02]
        down, across = np.array([0.0, 0.0, -1.0]), np.r_[u, 0.0]  # wrist down, closing across the top rail's thickness
        rot = np.stack([down, across, np.cross(down, across)], axis=1)
        steps.append(("reach", m.reach("right", g, rot, via=[g + np.array([0.0, 0.0, 0.12])])))
        if m.fell_at is None:
            steps.append(("grip", m.grip("right")))
        if m.fell_at is None:
            steps.append(("step_back", m.sidestep(forward=-PUSH_M)))
    else:
        steps.append(("walk", m.walk_to(xy + (PUSH_M + STAND_M) * u, yaw, face="fixed")))
    if m.fell_at is None:
        steps.append(("hold", m.hold(1.0)))
    p1, q1 = world.object_pose(key)
    p1, q1 = np.asarray(p1, float), np.asarray(q1, float)
    yaw_of = lambda q: math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))  # noqa: E731
    from locomani.tasks.scene.graspability import _quat_mat

    up = _quat_mat(q1)[:, 2]
    pel, _ = m.pelvis()
    moved = float(np.linalg.norm(p1[:2] - p0[:2]))
    return dict(key=key, method=method, offset=offset, clear=round(clear, 3), fell=m.fell_at is not None,
                fell_at=None if m.fell_at is None else round(m.fell_at - t0, 2), moved_m=round(moved, 3),
                along_m=round(float((p1[:2] - p0[:2]) @ u), 3),
                turned_deg=round(math.degrees((yaw_of(q1) - yaw_of(q0) + math.pi) % (2 * math.pi) - math.pi), 1),
                tilt_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, float(up[2]))))), 1),
                pelvis_err_m=round(float(np.linalg.norm(np.asarray(pel[:2]) - (xy + PUSH_M * u))), 3),
                seconds=round(m.t - t0, 1), steps=[(n, bool(o.ok), o.why[:60]) for n, o in steps])  # fmt: skip


def summary(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        s = out.setdefault(f"{r['key']} / {r['method']}", dict(n=0, fell=0, moved=0, moved_m=[]))
        s["n"] += 1
        s["fell"] += r["fell"]
        s["moved"] += (not r["fell"]) and r["moved_m"] >= MOVED_M
        s["moved_m"].append(r["moved_m"])
    for s in out.values():
        s["moved_m"] = round(float(np.median(s["moved_m"])), 3)
        s["works"] = s["moved"] * 2 > s["n"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--keys", nargs="+", default=list(KEYS))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--trials", type=int, default=len(OFFSETS))
    a = ap.parse_args(argv)
    from scene_round import make_world

    world = make_world({"scene_dir": a.scene.resolve()}, "scalebfm")
    movable = [
        k for k, o in world.meta["objects"].items() if (o.get("semantic") or {}).get("movable") and o.get("free")
    ]
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (a.out / "rounds.jsonl").open("a") as fh:
        for key in a.keys:
            for method in a.methods:
                for off in OFFSETS[: a.trials]:
                    wall = time.time()
                    r = trial(world, key, method, off, movable)
                    r["wall_s"] = round(time.time() - wall, 1)
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    print(json.dumps({k: r[k] for k in ("key", "method", "offset", "fell", "moved_m", "turned_deg", "tilt_deg")}),
                          flush=True)  # fmt: skip
    (a.out / "summary.json").write_text(json.dumps(summary(rows), indent=1))
    print(json.dumps(summary(rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
