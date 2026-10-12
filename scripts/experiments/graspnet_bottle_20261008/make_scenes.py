"""TODO 21.3.4: one bottle scene per GraspNet object, the object standing where the water bottle stood.

For every object of SIMPLE's GraspNet set (third_party/SIMPLE/data/assets/graspnet, 75 objects, licence
non-commercial research) that fits the open Dex1 jaw (9.5 cm) across its narrower horizontal side in one of its stable
poses, the bottle scene (configs/scenes.yaml `bottle`, outputs/scene/sim_lowcom) is copied with the bottle's body
re-dressed: the object's own mesh as the visual (manifold.obj, true size), its CoACD pieces (coacd_0.05) as collision,
in the tallest such stable pose, its bounding-box centre at the body origin, resting on the 0.62 m table top. SIMPLE's
stable poses do not all rest in this simulator (14 of 65 tipped 12-134 deg in 2 s, 2026-10-08), so each scene is
dropped for 2 s and rebuilt in the pose it came to rest in, until it stays within 1 cm and 3 deg; a pose that then no
longer fits the jaw falls back to the next stable pose. The body
keeps the bottle's name, place and contact parameters, so the stack, the spawn and the held criterion read it as the
target unchanged. Mass from the pieces' volume at 400 kg/m^3, clipped to 0.03-0.40 kg (the bottle is 0.35).

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/make_scenes.py outputs/graspnet_bottle_20261008
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "outputs" / "scene" / "sim_lowcom"
GN = REPO / "third_party" / "SIMPLE" / "data" / "assets" / "graspnet"
BODY, KEY = "water_bottle_object", "water_bottle"
JAW = 0.095  # DEX1_OPENING_M
TABLE_TOP = 0.62
DENSITY, MASS_RANGE = 400.0, (0.03, 0.40)
PALETTE = [
    (0.85, 0.33, 0.10), (0.00, 0.45, 0.74), (0.93, 0.69, 0.13), (0.49, 0.18, 0.56), (0.47, 0.67, 0.19),
    (0.30, 0.75, 0.93), (0.64, 0.08, 0.18), (0.20, 0.20, 0.20), (0.90, 0.50, 0.70), (0.55, 0.40, 0.25),
]  # fmt: skip


def obj_vertices(path: Path) -> np.ndarray:
    return np.array([list(map(float, ln.split()[1:4])) for ln in open(path) if ln.startswith("v ")])


def narrow_width(v: np.ndarray) -> float:
    a = np.linspace(0, np.pi, 72, endpoint=False)
    proj = v[:, :2] @ np.c_[np.cos(a), np.sin(a)].T
    return float((proj.max(0) - proj.min(0)).min())


def candidate_poses(oid: int, stable: dict, verts: np.ndarray) -> list:
    """Stable poses whose narrower horizontal side fits the jaw, tallest first."""
    out = []
    for k, row in enumerate(np.asarray(stable[oid])):
        R = Rotation.from_quat([row[4], row[5], row[6], row[3]]).as_matrix()  # rows are xyz + wxyz
        v = verts @ R.T
        w, h = narrow_width(v), float(np.ptp(v[:, 2]))
        if w <= JAW:
            out.append((k, R, h, w))
    return sorted(out, key=lambda c: -c[2])


def settle(xml: Path, seconds: float = 2.0):
    """(moved m, rotated deg, world rotation of the body at rest) after `seconds` of free physics."""
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    adr = m.jnt_qposadr[m.body_jntadr[m.body(BODY).id]]
    mujoco.mj_forward(m, d)
    p0 = d.qpos[adr : adr + 3].copy()
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    q = d.qpos[adr + 3 : adr + 7]
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    return float(np.linalg.norm(d.qpos[adr : adr + 3] - p0)), float(np.degrees(R.magnitude())), R.as_matrix()


def mass_properties(pieces: list[Path], R: np.ndarray, offset: np.ndarray):
    """Volume, centre of mass and inertia (unit density) of the rotated, shifted pieces, compiled by MuJoCo."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="o")
    body.add_freejoint()
    q = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
    for i, p in enumerate(pieces):
        spec.add_mesh(name=f"m{i}", file=str(p))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{i}", density=1.0, quat=q, pos=R @ offset)
    m = spec.compile()
    return float(m.body_mass[1]), m.body_ipos[1].copy(), m.body_iquat[1].copy(), m.body_inertia[1].copy()


def build(oid: int, choice, out: Path, base_meta: dict) -> dict:
    k, R, h, w = choice
    d = GN / "models" / f"{oid:03d}" / "coacd_0.05"
    pieces = sorted(d.glob("convex_piece_*.obj"), key=lambda p: int(p.stem.split("_")[-1]))
    visual = d / "manifold.obj"
    v = obj_vertices(visual) @ R.T
    centre = (v.max(0) + v.min(0)) / 2
    offset = -R.T @ centre  # mesh-frame shift that puts the rotated box centre at the body origin
    half = (v.max(0) - v.min(0)) / 2
    vol, ipos, iquat, inertia = mass_properties(pieces, R, offset)
    mass = float(np.clip(vol * DENSITY, *MASS_RANGE))

    spec = mujoco.MjSpec.from_file(str(BASE / "scene_locomani.xml"))
    body = spec.body(BODY)
    template = body.geoms[2]  # a collision geom of the bottle: its contact parameters carry over
    contact = dict(
        friction=template.friction.copy(),
        solref=template.solref.copy(),
        solimp=template.solimp.copy(),
        condim=template.condim,
    )
    for g in list(body.geoms):
        spec.delete(g)
    q = Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
    pos = R @ offset
    rgba = [*PALETTE[oid % len(PALETTE)], 1.0]
    spec.add_mesh(name=f"gn{oid:03d}_vis", file=str(visual))
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"gn{oid:03d}_vis", contype=0, conaffinity=0, group=1,
        rgba=rgba, quat=q, pos=pos, density=0.0,
    )  # fmt: skip
    for i, p in enumerate(pieces):
        spec.add_mesh(name=f"gn{oid:03d}_col{i}", file=str(p))
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"gn{oid:03d}_col{i}", rgba=[0.8, 0.8, 0.8, 0.0], quat=q, pos=pos,
            **contact,
        )  # fmt: skip
    body.explicitinertial = True
    body.mass = mass
    body.ipos = ipos
    body.iquat = iquat
    body.inertia = inertia * (mass / vol)
    x, y = base_meta["objects"][KEY]["init_pos"][:2]
    z = TABLE_TOP + half[2] + 0.002
    body.pos = [x, y, z]
    body.quat = [1.0, 0.0, 0.0, 0.0]
    model = spec.compile()

    out.mkdir(parents=True, exist_ok=True)
    xml = out / "scene_locomani.xml"
    xml.write_text(spec.to_xml())
    meta = json.loads(json.dumps(base_meta))
    o = meta["objects"][KEY]
    o.update(
        init_pos=[x, y, z],
        init_quat=[1.0, 0.0, 0.0, 0.0],
        height=float(2 * half[2]),
        scale=1.0,
        mesh=str(visual),
        box=dict(half_extent=half.tolist(), centre=[0.0, 0.0, 0.0]),
        inertial=dict(body=BODY, mass=mass, com_before=ipos.tolist(), com_after=ipos.tolist(), com_drop_m=0.0),
        graspnet=dict(id=oid, stable_index=int(k), width=w, pieces=len(pieces)),
    )
    meta["object_phrase"][KEY] = f"graspnet object {oid:03d}"
    meta.update(scene_xml=str(xml), nbody=int(model.nbody), ngeom=int(model.ngeom), nq=int(model.nq))
    (out / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    return dict(id=oid, stable_index=int(k), height=float(2 * half[2]), width=w, mass=mass, pieces=len(pieces),
                scene=str(out))  # fmt: skip


def main(out_root: Path) -> int:
    base_meta = json.loads((BASE / "scene_meta.json").read_text())
    stable = np.load(GN / "new_stable_object_poses.npy", allow_pickle=True).item()
    rows, skipped = [], []
    for oid in sorted(int(p.name) for p in (GN / "models").iterdir()):
        verts = obj_vertices(GN / "models" / f"{oid:03d}" / "coacd_0.05" / "manifold.obj")
        dest = out_root / "scenes" / f"obj_{oid:03d}"
        row = None
        for k, R, _, _ in candidate_poses(oid, stable, verts):
            for attempt in range(4):
                v = verts @ R.T
                w = narrow_width(v)
                if w > JAW:
                    break
                if dest.exists():
                    shutil.rmtree(dest)
                row = build(oid, (k, R, float(np.ptp(v[:, 2])), w), dest, base_meta)
                moved, turned, R_rest = settle(dest / "scene_locomani.xml")
                row.update(settle_moved_m=moved, settle_deg=turned, settle_rebuilds=attempt)
                if moved <= 0.01 and turned <= 3.0:
                    break
                yaw = np.arctan2(R_rest[1, 0], R_rest[0, 0])  # keep the tilt it came to rest in, not its spin
                R = Rotation.from_euler("z", -yaw).as_matrix() @ R_rest @ R
                row = None
            if row is not None:
                break
        if row is None:
            skipped.append(oid)
            if dest.exists():
                shutil.rmtree(dest)
            continue
        rows.append(row)
        r = rows[-1]
        print(
            f"obj {oid:03d}: pose {r['stable_index']} h {r['height']:.3f} w {r['width']:.3f} m {r['mass']:.3f} kg"
            f" rebuilt {r['settle_rebuilds']}x, rest {100 * r['settle_moved_m']:.1f} cm {r['settle_deg']:.1f} deg",
            flush=True,
        )
    (out_root / "objects.json").write_text(json.dumps(dict(objects=rows, too_wide=skipped), indent=1))
    print(f"{len(rows)} scenes; {len(skipped)} too wide for the jaw or never at rest: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
