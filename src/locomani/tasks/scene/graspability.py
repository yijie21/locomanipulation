"""TODO 33.18 (docs/handoff/scene-rsi-strategy-plan-20261021.md §1.1, D13): can the Dex1 jaw take this object, and from
which approach -- a check on the object's collision hulls, no simulation (DexAgent's thin_grasp_diagnosis in our terms).
Infrastructure, written by hand: skills and roles call it, the loop does not change it.

The jaw closes along the wrist's y axis (rules.md 12). A side grasp's wrist x is the approach (horizontal, shoulder to
object), so its closing axis is the approach's horizontal normal; the jaw must straddle the object's section at the grasp
height, measured across that axis. The section is cut from the collision geoms of the object's body (mesh hulls, boxes,
cylinders, spheres), composed with the object's pose now, inside a band of +-SECTION_BAND_M around the height.

    fit(world, key, approach_yaw)      -> the section's width across the jaw, the margin a side, a verdict
    section_widths(world, key)         -> {approach yaw (deg): width (m)}
    assess(world, key, approach_yaw)   -> numeric verdicts for every grasp kind and an ordered `recommend` list

`world` is a SceneWorld (sim.m, meta, object_pose) or a StaticScene (below: the physics model and a start's placements,
for roles' and reports' use without a simulation). Yaws are radians in, degrees out; widths and margins metres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

JAW_OPEN_M = 0.095  # Motor.jaw_opening at runtime.GRIP_OPEN: 2 * (GRIP_OPEN - GRIP_CLOSED) (test_rsi_graspability)
INSERTION_ERROR_M = 0.010  # where the reach servo lands the jaw centre: about 1 cm off (rules.md 14)
FIT_MARGIN_MIN_M = 0.5 * INSERTION_ERROR_M  # a margin a side below this goes in only if the reach lands well: "tight"
JAW_ABOVE_TOP_M = 0.039  # a level jaw's lowest centre above the support top (housing 3.1-3.4 cm below the jaw axis)
SECTION_BAND_M = 0.010  # the section is the union of cuts at the height and +- this
ROTATE_MAX_DEG = 35.0  # `assess`: an approach rotated by up to this about the object counts as the same stance
FLAT_HALF_M = 0.02  # resting half-height below this: a side jaw passes above it, a top pinch takes it
RIM_WALL_MAX_M = 0.02  # a container wall at most this thick can be pinched across
STEP_DEG = 5.0
_CIRCLE = 24  # points per circle for cylinders, capsules, spheres, ellipsoids


@dataclass
class _Part:
    verts: np.ndarray  # (n, 3) in the body frame
    edges: np.ndarray  # (k, 2) index pairs: the convex hull's edges (all pairs for tiny parts)


def _quat_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def _round_points(kind: int, size) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, _CIRCLE, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = size[0]
        rings = [(r * np.cos(p), r * np.sin(p)) for p in np.linspace(-np.pi / 2, np.pi / 2, 9)]
        return np.vstack([np.stack([rr * c, rr * s, np.full_like(c, zz)], 1) for zz, rr in rings])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        pts = _round_points(mujoco.mjtGeom.mjGEOM_SPHERE, [1.0])
        return pts * np.asarray(size[:3])
    r, h = size[0], size[1]
    z = [-h, h] if kind == mujoco.mjtGeom.mjGEOM_CYLINDER else [-h - r, -h, h, h + r]
    out = []
    for zz in z:
        rr = r if abs(zz) <= h + 1e-12 else 0.35 * r  # a capsule's caps, coarsely
        out.append(np.stack([rr * c, rr * s, np.full_like(c, zz)], 1))
    return np.vstack(out)


def _hull_edges(v: np.ndarray) -> np.ndarray:
    n = len(v)
    if n <= 12:
        return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], int).reshape(-1, 2)
    try:
        from scipy.spatial import ConvexHull

        simp = ConvexHull(v).simplices
    except Exception:  # degenerate (flat) point sets: every pair
        return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], int).reshape(-1, 2)
    e = np.vstack([simp[:, [0, 1]], simp[:, [1, 2]], simp[:, [0, 2]]])
    return np.unique(np.sort(e, 1), axis=0)


_CACHE: dict = {}


def _model(world):
    return world.sim.m if hasattr(world, "sim") else world.m


def body_parts(world, key: str) -> list[_Part]:
    """The object's collision geoms as convex point sets in its body frame (cached per model and body)."""
    m = _model(world)
    bid = m.body(world.meta["objects"][key]["body"]).id
    ck = (id(m), bid)
    if ck in _CACHE:
        return _CACHE[ck]
    parts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        kind = int(m.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
            local = np.asarray(m.mesh_vert[a : a + n], float)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = m.geom_size[g]
            local = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], float)
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                      mujoco.mjtGeom.mjGEOM_ELLIPSOID):  # fmt: skip
            local = _round_points(kind, m.geom_size[g])
        else:
            continue
        v = np.asarray(m.geom_pos[g], float) + local @ _quat_mat(m.geom_quat[g]).T
        parts.append(_Part(v, _hull_edges(v)))
    _CACHE[ck] = parts
    return parts


def world_parts(world, key: str) -> list[np.ndarray]:
    """Each part's vertices in the world, at the object's pose now."""
    p, q = world.object_pose(key)
    Rb = _quat_mat(q)
    return [np.asarray(p, float) + part.verts @ Rb.T for part in body_parts(world, key)]


def resting_half_height(world, key: str) -> float:
    z = np.concatenate([v[:, 2] for v in world_parts(world, key)])
    return float(0.5 * (z.max() - z.min())) if len(z) else 0.0


def grasp_height(world, key: str) -> float:
    """A side grasp's jaw height: the object's centre, or JAW_ABOVE_TOP_M over its support's top if that is higher."""
    p, _ = world.object_pose(key)
    sup = world.meta["objects"][key].get("surface")
    top = float(world.meta["surfaces"][sup]["top"]) if sup in world.meta.get("surfaces", {}) else -np.inf
    return max(float(p[2]), top + JAW_ABOVE_TOP_M)


def section(world, key: str, z: float | None = None) -> np.ndarray:
    """(k, 2) world xy points of the object's section in the band z +- SECTION_BAND_M (empty: nothing at that height)."""
    z = grasp_height(world, key) if z is None else float(z)
    out = []
    p, q = world.object_pose(key)
    Rb = _quat_mat(q)
    for part in body_parts(world, key):
        v = np.asarray(p, float) + part.verts @ Rb.T
        a, b = v[part.edges[:, 0]], v[part.edges[:, 1]]
        for lev in (z - SECTION_BAND_M, z, z + SECTION_BAND_M):
            on = np.abs(v[:, 2] - lev) < 1e-9
            out.append(v[on, :2])
            lo, hi = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
            cut = (lo < lev) & (hi > lev)
            if cut.any():
                t = (lev - a[cut, 2]) / (b[cut, 2] - a[cut, 2])
                out.append((a[cut] + t[:, None] * (b[cut] - a[cut]))[:, :2])
    pts = np.vstack(out) if out else np.zeros((0, 2))
    return pts


def width_across(points: np.ndarray, closing_xy) -> float:
    """The section's extent along a horizontal closing axis (0 for an empty section)."""
    if len(points) == 0:
        return 0.0
    c = np.asarray(closing_xy, float)[:2]
    c = c / np.linalg.norm(c)
    s = points @ c
    return float(s.max() - s.min())


def _closing(approach_yaw: float) -> np.ndarray:
    return np.array([-np.sin(approach_yaw), np.cos(approach_yaw)])


def verdict(width: float) -> tuple[str, float]:
    """("fits" | "tight" | "too_wide", margin a side) of a width against the open jaw."""
    margin = 0.5 * (JAW_OPEN_M - width)
    return ("fits" if margin >= FIT_MARGIN_MIN_M else "tight" if margin >= 0 else "too_wide"), margin


def fit_axis(world, key: str, closing_xy, z: float | None = None, points: np.ndarray | None = None) -> dict:
    """The fit of the jaw closing along `closing_xy` (any horizontal axis; a wrist's y axis projected)."""
    z = grasp_height(world, key) if z is None else float(z)
    pts = section(world, key, z) if points is None else points
    if len(pts) == 0:
        return dict(z=round(z, 4), width_m=0.0, margin_m=None, verdict="below_jaw", fits=False,
                    reason=f"nothing of the object at the jaw height {z:.3f} m: a level jaw passes above it")  # fmt: skip
    w = width_across(pts, closing_xy)
    v, mg = verdict(w)
    why = {
        "fits": "fits",
        "tight": "goes in only if the reach lands within the margin",
        "too_wide": "wider than the jaw",
    }[v]
    return dict(z=round(z, 4), width_m=round(w, 4), margin_m=round(mg, 4), verdict=v, fits=v == "fits",
                reason=f"{w * 100:.1f} cm across the jaw against {JAW_OPEN_M * 100:.1f} cm open: "
                       f"{mg * 100:+.2f} cm a side, {why}")  # fmt: skip


def fit(world, key: str, approach_yaw: float, z: float | None = None) -> dict:
    """The side grasp's fit at a horizontal approach (radians: the direction from the shoulder to the object)."""
    out = fit_axis(world, key, _closing(approach_yaw), z)
    out["approach_deg"] = round(float(np.degrees(approach_yaw)), 1)
    return out


def section_widths(world, key: str, z: float | None = None, step_deg: float = STEP_DEG) -> dict[float, float]:
    """{approach yaw in degrees, -180 to 180: the section's width across the jaw} (the jaw is symmetric, so a yaw and
    its opposite agree)."""
    pts = section(world, key, z)
    return {float(d): width_across(pts, _closing(np.radians(d))) for d in np.arange(-180.0, 180.0, step_deg)}


def _wrap_deg(d: float) -> float:
    return float((d + 180.0) % 360.0 - 180.0)


def rim(world, key: str) -> dict | None:
    """A container's thin walls: from thin upright box geoms, else from the scene file's box and opening."""
    o = world.meta["objects"][key]
    sem = o.get("semantic") or {}
    container = bool(o.get("container") or (sem.get("container") or {}).get("is"))
    if not container:
        return None
    walls = []
    for part in body_parts(world, key):
        ext = part.verts.max(0) - part.verts.min(0)
        if len(part.verts) == 8 and min(ext[0], ext[1]) <= RIM_WALL_MAX_M and ext[2] >= 0.03:
            walls.append(float(min(ext[0], ext[1])))
    if not walls:
        op = (sem.get("container") or {}).get("opening_half_extent")
        box = (o.get("box") or {}).get("half_extent")
        if op and box:
            walls = [float(min(np.asarray(box[:2]) - np.asarray(op[:2])))]
    if not walls or min(walls) <= 0:
        return dict(available=False, reason="a container, but no wall thin enough to pinch across")
    w = min(walls)
    v, mg = verdict(w)
    return dict(available=w <= RIM_WALL_MAX_M, wall_m=round(w, 4), margin_m=round(mg, 4), fits=v == "fits",
                reason=f"rim pinch across a {w * 100:.1f} cm wall ({mg * 100:.2f} cm a side)")  # fmt: skip


def top_pinch(world, key: str) -> dict | None:
    """A flat object's top pinch, across its narrowest horizontal axis (the whole hull, projected)."""
    if resting_half_height(world, key) >= FLAT_HALF_M:
        return None
    pts = np.vstack([v[:, :2] for v in world_parts(world, key)])
    widths = {float(d): width_across(pts, _closing(np.radians(d))) for d in np.arange(-90.0, 90.0, STEP_DEG)}
    d, w = min(widths.items(), key=lambda kv: kv[1])
    v, mg = verdict(w)
    return dict(width_m=round(w, 4), margin_m=round(mg, 4), fits=v == "fits", closing_deg=round(_wrap_deg(d + 90.0), 1),
                reason=f"top pinch across the {w * 100:.1f} cm axis ({mg * 100:+.2f} cm a side)")  # fmt: skip


def assess(world, key: str, approach_yaw: float | None = None, z: float | None = None) -> dict:
    """Every grasp kind's numeric verdict and an ordered `recommend` list (plain strings with numbers)."""
    z = grasp_height(world, key) if z is None else float(z)
    pts = section(world, key, z)
    widths = {float(d): width_across(pts, _closing(np.radians(d))) for d in np.arange(-180.0, 180.0, 1.0)}
    hh = resting_half_height(world, key)
    out = dict(object=key, z=round(z, 4), jaw_open_m=JAW_OPEN_M, resting_half_height_m=round(hh, 4),
               flat=hh < FLAT_HALF_M, section_points=int(len(pts)))  # fmt: skip
    rec: list[str] = []
    if len(pts):
        best_d, best_w = min(widths.items(), key=lambda kv: kv[1])
        out["narrowest"] = dict(approach_deg=best_d, width_m=round(best_w, 4), margin_m=round(verdict(best_w)[1], 4))
        out["widest_m"] = round(max(widths.values()), 4)
        fit_yaws = [d for d, w in widths.items() if verdict(w)[0] == "fits"]
        out["fits_from_deg"] = fit_yaws[:: max(1, len(fit_yaws) // 24)] if fit_yaws else []
    if approach_yaw is not None and len(pts):
        a0 = _wrap_deg(float(np.degrees(approach_yaw)))
        side = fit_axis(world, key, _closing(approach_yaw), z, pts)
        side["approach_deg"] = round(a0, 1)
        out["side"] = side
        window = sorted(((abs(r), r, widths[_wrap_deg(round(a0 + r))]) for r in np.arange(-ROTATE_MAX_DEG, ROTATE_MAX_DEG + 1)),
                        key=lambda t: t[0])  # fmt: skip
        first = next(((r, w) for _, r, w in window if verdict(w)[0] == "fits"), None)
        wide = min(window, key=lambda t: t[2])
        out["rotate"] = dict(
            smallest_deg=None if first is None else round(float(first[0]), 1),
            smallest_margin_m=None if first is None else round(verdict(first[1])[1], 4),
            widest_margin_deg=round(float(wide[1]), 1), widest_margin_m=round(verdict(wide[2])[1], 4),
        )  # fmt: skip
        if side["fits"]:
            rec.append(f"side grasp at this approach fits ({side['margin_m'] * 100:.2f} cm a side)")
        elif first is not None:
            rec.append(f"rotate the approach {first[0]:+.0f}° ({verdict(first[1])[1] * 100:.2f} cm a side)")
        elif out.get("fits_from_deg"):
            d = min(out["fits_from_deg"], key=lambda x: abs(_wrap_deg(x - a0)))
            rec.append(f"fits only {_wrap_deg(d - a0):+.0f}° off this approach ({verdict(widths[d])[1] * 100:.2f} cm a "
                       f"side): another stance")  # fmt: skip
    elif len(pts) and out.get("fits_from_deg"):
        d = out["narrowest"]["approach_deg"]
        rec.append(f"side grasp fits from approach {d:+.0f}° ({out['narrowest']['margin_m'] * 100:.2f} cm a side)")
    rm = rim(world, key)
    if rm is not None:
        out["rim"] = rm
        if rm.get("available"):
            rec.append(f"rim pinch across the {rm['wall_m'] * 100:.1f} cm wall")
    tp = top_pinch(world, key)
    if tp is not None:
        out["top_pinch"] = tp
        if tp["fits"]:
            rec.append(f"top pinch across the {tp['width_m'] * 100:.1f} cm axis ({tp['margin_m'] * 100:.2f} cm a side)")
    if not len(pts) and tp is None:
        rec.append(f"nothing of the object at the jaw height {z:.3f} m: lower the jaw or pinch from above")
    if len(pts) and not out.get("fits_from_deg") and not (rm or {}).get("available") and not (tp or {}).get("fits"):
        n = out["narrowest"]["width_m"]
        rec.append(f"nothing fits the {JAW_OPEN_M * 100:.1f} cm jaw at any yaw (narrowest {n * 100:.1f} cm): needs a "
                   f"prep action (tip or reorient it) or another grasp")  # fmt: skip
    out["recommend"] = rec
    return out


class StaticScene:
    """The check without a simulation: the scene's physics model (loaded once, about 7 s and 1.5 GB for the ADT
    apartment), its scene file, and object poses from a start's placements (else the scene file's)."""

    def __init__(self, scene_dir, placements: dict | None = None, model: mujoco.MjModel | None = None):
        d = Path(scene_dir)
        phys = d.parent / f"{d.name}_physics" / "scene_locomani.xml"
        self.meta = json.loads((d / "scene_meta.json").read_text())
        self.m = (
            model
            if model is not None
            else mujoco.MjModel.from_xml_path(str(phys if phys.exists() else d / "scene_locomani.xml"))
        )
        self.placements = {}
        self.place(placements or {})

    def place(self, placements: dict) -> None:
        self.placements = {k: (np.asarray(p, float), np.asarray(q, float)) for k, (p, q) in placements.items()}

    def object_pose(self, key: str):
        if key in self.placements:
            return self.placements[key]
        o = self.meta["objects"][key]
        return np.asarray(o["init_pos"], float), np.asarray(o["init_quat"], float)
