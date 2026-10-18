#!/usr/bin/env python3
"""Settle a built ADT scene's free objects into a rest the simulator keeps, and write those poses back (2026-10-18,
plan docs/handoff/scene-rsi-plan-20261016.md §10.2). Runs in the sim venv.

    $VENV_SIM_PY scripts/scene_settle.py --scene outputs/scene_discovery/adt_apartment_mars [--write]

The ADT poses are the recorded ones, and the colliders are convex hulls a little larger than the objects: books stand
up to 1.3 cm into their neighbours in a row, objects up to 8 mm into their surface slab, the chopping board hangs
clear of the island. Left alone, the solver pushes them apart in the first steps: rows of books fell over end to
end and the carrot rolled off the tilted board onto the floor. Here every free object's six degrees of freedom get
heavy damping on their translation (DAMPING) and light damping on their rotation (DAMPING_ROT) for SETTLE_S, so
penetrations come apart and objects come to rest without gaining speed, while a round object can still roll onto
the face it rests on (with heavy rotational damping too, the carrot stayed where it was put and rolled off as soon
as the damping went); then the damping goes and the scene runs CHECK_S as the runner has it, which must leave every
object where it settled. Side panels (scene_add_panels.py) go in first: a row of books rests against them.
Loads `<scene>/env_only.xml` (no robot) at the runner's timestep. Writes `<scene>/settle.json` (per object: how far
it moved and turned while settling, its drift in the check, whether it still rests on its surface) and, with
`--write`, the settled poses into `scene_meta.json` (`init_pos`, `init_quat`), `scene_locomani.xml` and
`env_only.xml` (the free bodies' `pos` / `quat`).
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import mujoco
import numpy as np

DT = 0.005  # the ScaleBFM runner's physics step (backends/scalebfm/runtime.py)
DAMPING = 5.0  # N s/m on every free object's translation while settling
DAMPING_ROT = 0.001  # N m s/rad on its rotation
SETTLE_S = 4.0
CHECK_S = 20.0
DRIFT_M, DRIFT_DEG = 0.005, 3.0  # the check's limits: an object that moves more in CHECK_S does not rest


def load(scene: Path):
    # env_only.xml names the free objects' textured visuals' materials without defining them; visuals do not collide
    xml = re.sub(r' material="[^"]*"', "", (scene / "env_only.xml").read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=scene) as fh:  # its mesh paths are absolute anyway
        fh.write(xml)
        fh.flush()
        m = mujoco.MjModel.from_xml_path(fh.name)
    m.opt.timestep = DT
    return m


def angle_deg(q0, q1) -> float:
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q0, q1)))))))


def settle(m) -> tuple[dict, dict]:
    """(settled {key: (pos, quat)}, per-object report)."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    joints = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    keys = {j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])).removesuffix("_object") for j in joints}  # fmt: skip
    pose = lambda j: (d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + 3].copy(), d.qpos[m.jnt_qposadr[j] + 3 : m.jnt_qposadr[j] + 7].copy())  # noqa: E731  # fmt: skip
    start = {j: pose(j) for j in joints}
    dofs = np.concatenate([np.arange(m.jnt_dofadr[j], m.jnt_dofadr[j] + 6) for j in joints])
    damping = m.dof_damping.copy()
    m.dof_damping[dofs] = np.tile([DAMPING] * 3 + [DAMPING_ROT] * 3, len(joints))
    for _ in range(int(SETTLE_S / DT)):
        mujoco.mj_step(m, d)
    settled = {j: pose(j) for j in joints}
    m.dof_damping[:] = damping
    d.qvel[:] = 0.0
    for _ in range(int(CHECK_S / DT)):
        mujoco.mj_step(m, d)
    report = {}
    for j in joints:
        (p0, q0), (p1, q1), (p2, q2) = start[j], settled[j], pose(j)
        drift_m, drift_deg = float(np.linalg.norm(p2 - p1)), angle_deg(q1, q2)
        report[keys[j]] = dict(
            moved_mm=round(1000 * float(np.linalg.norm(p1 - p0)), 1), turned_deg=round(angle_deg(q0, q1), 1),
            dz_mm=round(1000 * float(p1[2] - p0[2]), 1), drift_mm=round(1000 * drift_m, 1),
            drift_deg=round(drift_deg, 2), rests=bool(drift_m <= DRIFT_M and drift_deg <= DRIFT_DEG),
        )  # fmt: skip
    return {keys[j]: settled[j] for j in joints}, report


def on_surface(meta: dict, key: str, pos) -> bool:
    """X's settled centre still over its surface's slab (within its half extents) and not below its top."""
    o = meta["objects"][key]
    s = meta["surfaces"].get(o.get("surface") or "")
    if s is None:
        return True  # a floor object or one held by a container: no slab to stay on
    c, sn = np.cos(-float(s.get("yaw", 0.0))), np.sin(-float(s.get("yaw", 0.0)))
    dx, dy = float(pos[0]) - s["xy"][0], float(pos[1]) - s["xy"][1]
    local = np.array([c * dx - sn * dy, sn * dx + c * dy])
    return bool(np.all(np.abs(local) <= np.asarray(s["half_extent"][:2], float) + 0.02) and pos[2] >= s["top"] - 0.01)


def write(scene: Path, meta: dict, settled: dict) -> None:
    fmt = lambda v: " ".join(f"{float(x):.6g}" for x in v)  # noqa: E731
    for key, (p, q) in settled.items():
        meta["objects"][key]["init_pos"] = [round(float(x), 6) for x in p]
        meta["objects"][key]["init_quat"] = [round(float(x), 6) for x in q]
    (scene / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    for name in ("scene_locomani.xml", "env_only.xml"):
        path = scene / name
        s = path.read_text()
        for key, (p, q) in settled.items():
            pat = rf'(<body name="{re.escape(key)}_object") pos="[^"]*" quat="[^"]*"'
            s, n = re.subn(pat, rf'\1 pos="{fmt(p)}" quat="{fmt(q)}"', s)
            assert n == 1, (name, key, n)
        path.write_text(s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--write", action="store_true", help="write the settled poses into the scene's three files")
    a = ap.parse_args(argv)
    meta = json.loads((a.scene / "scene_meta.json").read_text())
    settled, report = settle(load(a.scene))
    for key, (p, _) in settled.items():
        report[key]["on_surface"] = on_surface(meta, key, p)
    bad = sorted(k for k, r in report.items() if not (r["rests"] and r["on_surface"]))
    moved = sorted(report.items(), key=lambda kv: -kv[1]["moved_mm"])
    out = dict(scene=str(a.scene), dt=DT, damping=DAMPING, damping_rot=DAMPING_ROT, settle_s=SETTLE_S, check_s=CHECK_S, written=a.write,
               not_resting=bad, objects=report)  # fmt: skip
    (a.scene / "settle.json").write_text(json.dumps(out, indent=1))
    print(
        f"[settle] {len(report)} free objects; moved most: {[(k, r['moved_mm'], r['turned_deg']) for k, r in moved[:8]]}"
    )
    print(f"[settle] not resting or off their surface after the check: {[(k, report[k]) for k in bad]}")
    if a.write:
        write(a.scene, meta, settled)
        print(f"[settle] wrote the settled poses into {a.scene}/scene_meta.json, scene_locomani.xml, env_only.xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
