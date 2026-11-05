"""Google Scanned Objects as free objects of a scene (the `logistics_room_1030` scenes, 2026-11-03).

`assets/mujoco_scanned_objects/models/<name>/` holds kevinzakka's obj2mjcf conversion of the 1030 scanned household
objects: `model.obj` + `texture.png` (the scan, visual only) and up to 32 V-HACD convex hulls `model_collision_*.obj`.
The MJCF they ship has no free joint, no mass and the same element names in every file, so a scene does not include
it; this module reads the meshes and writes the object into the scene's own MJCF with its own names.

Three things live here, each a plain function over plain data:
- the catalog (`build_catalog`): every object's size, hull volume, mass, rest test, dataset category, graspability and
  train / held-out split, from the meshes, the thresholds and `categories.json` (the dataset's own labels, fetched from
  Gazebo Fuel by `scripts/gso_categories.py`; an object whose category is not in `Thresholds.used_categories` is
  `none`: the user keeps shoes, toys and the like out of the logistics room, 2026-11-04 / 05); written once to `assets/mujoco_scanned_objects/catalog.json` by
  `scripts/gso_catalog.py` and read by the scene builder;
- the draw (`sample_keys`): which objects a scene instance gets, from a seed, a pool and the single / bimanual ratio;
- the placement (`place_objects`) and the MJCF / scene-file records (`add_to_scene`, `object_record`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from locomani.tasks.scene.start import (
    OBJECT_GAP_M,
    SURFACE_MARGIN_M,
    _rot2,
    edge_depths,
    footprint,
    inside_top,
    overlap,
)

CATALOG_VERSION = 2  # 2: `thresholds.used_categories` (the categories drawn) replaces `excluded_categories`
KEY_FORMAT = (
    "gso_{index:04d}"  # object keys: short, stable, not an English word (rules.md 30 voids skills naming objects)
)
POOLS = ("train", "eval_unseen", "all")
GRASPABLE = ("single", "bimanual", "none")
UNLABELLED = "(none)"  # how a model without a dataset label is named in a category list (categories.json `counts`)
# the dataset's labels (categories.json) kept out of the logistics room (user 2026-11-04: shoes; 2026-11-05: also toys,
# stuffed toys, mice, media cases, Legos, keyboards, the car seat, action figures and hats) ...
NOT_USED_CATEGORIES = (
    "Shoe", "Toys", "Stuffed Toys", "Mouse", "Media Cases", "Legos", "Keyboard", "Car Seat", "Action Figures", "Hat",
)  # fmt: skip
# ... and the categories drawn by default: every other label of categories.json and the unlabelled models (216, which
# no list names, so "the rest" keeps them; drop UNLABELLED from the list to leave them out too)
USED_CATEGORIES = (
    "Consumer Goods", "Bottles and Cans and Cups", "Bag", "Board Games", "Headphones", "Camera", UNLABELLED,
)  # fmt: skip
FRICTION = [0.8, 0.005, 0.0001]  # as the ADT apartment's free objects (scripts/adt_to_scene.py)
REST_DT = 0.005  # the ScaleBFM runner's physics step (scene_settle.DT)
REST_SETTLE_S, REST_CHECK_S = 1.0, 23.0  # dropped 1 mm onto a plane, then watched; scene_settle.py settles for 24 s
REST_TILT_DEG, REST_SLIDE_M = 3.0, 0.005  # scene_settle.DRIFT_DEG / DRIFT_M: more than this at rest and the scene lint
# calls the object drifting; an object that tips over on its own (the Jenga box standing on its 8 x 8 cm end) fails here
# and is `none`, instead of failing scene_lint.py's rest test in every instance that draws it
HASH_TOKEN = re.compile(r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])[A-Za-z0-9]{10,13}$")  # GSO's trailing ids: `NJnjGna4u1a`


@dataclass(frozen=True)
class Thresholds:
    """What makes an object single-hand, two-hand or neither, and how heavy it is. Recorded in the catalog.

    Single: the narrow side of its footprint fits the open Dex1 jaw (tasks/scene/graspability.JAW_OPEN_M) and its
    grasp point (half its height over the platform top) is inside the measured grip band's floor (capability table
    `grip_height_accepted` 0.25-1.25 m; the platform is 0.371 m, so 0.40 m keeps a flat object out, user Q10).
    Bimanual: wider than a jaw, its wide side between the two hands' plausible span and tall enough to press on
    (user Q17, 2026-11-03: 0.20-0.45 m, >= 0.10 m; the measured two-hand width is only the carton's 0.43-0.47 m,
    capability.yaml `box_grasp.width`, so this range is `untested`).
    Categories: an object whose dataset category (UNLABELLED when it has none) is not in `used_categories` is `none`
    whatever its shape (user 2026-11-04: no shoes; 2026-11-05: only USED_CATEGORIES, the labels left once
    NOT_USED_CATEGORIES are taken out). None uses every category and needs no labels."""

    jaw_m: float = 0.095
    min_height_m: float = 0.03
    surface_top_m: float = 0.371
    min_grasp_height_m: float = 0.40
    bimanual_width_m: tuple[float, float] = (0.20, 0.45)
    bimanual_min_height_m: float = 0.10
    density_kg_m3: float = 300.0  # a hull is hollow-ish scanned plastic, cardboard, cloth: well under water
    mass_range_kg: tuple[float, float] = (0.05, 1.5)  # the upper bound is the single arm's measured mass_lifted_max
    held_out_fraction: float = 0.10
    used_categories: tuple[str, ...] | None = USED_CATEGORIES


def phrase(name: str) -> str:
    """A readable phrase from a GSO directory name: `Lovable_Huggable_..._Teddy_Bear_Beige` -> `lovable huggable ...
    teddy bear beige`; a trailing scan id (`_NJnjGna4u1a`) goes."""
    tokens = name.split("_")
    if len(tokens) > 1 and HASH_TOKEN.match(tokens[-1]):
        tokens = tokens[:-1]
    return " ".join(t for t in tokens if t).lower()


def _read_obj(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    verts, faces = [], []
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith(b"f "):
                faces.append([int(t.split(b"/")[0]) - 1 for t in line.split()[1:]])
    return np.asarray(verts, float).reshape(-1, 3), faces


def hull_volume(path: Path) -> float:
    """Volume of a closed mesh (m^3) by the signed tetrahedra of its faces; a V-HACD hull is closed and convex."""
    v, faces = _read_obj(path)
    if not len(v) or not faces:
        return 0.0
    total = 0.0
    for f in faces:
        for i in range(1, len(f) - 1):  # fan: V-HACD writes triangles, an OBJ may carry quads
            a, b, c = v[f[0]], v[f[i]], v[f[i + 1]]
            total += float(np.dot(a, np.cross(b, c)))
    return abs(total) / 6.0


def split_of(name: str, held_out_fraction: float) -> str:
    """`train` or `held_out` by the name's hash: stable across catalog rebuilds and independent of the threshold
    changes above (user Q22: the evaluation may draw only objects no training instance ever held)."""
    h = int.from_bytes(hashlib.sha1(name.encode()).digest()[:4], "big")
    return "held_out" if h % 10_000 < int(round(held_out_fraction * 10_000)) else "train"


def category_used(category: str | None, thresholds: Thresholds) -> bool:
    """Whether an object of this dataset category (None: unlabelled) may be drawn under `thresholds.used_categories`."""
    used = thresholds.used_categories
    return used is None or (category or UNLABELLED) in used


def graspable(extent, thresholds: Thresholds, rest: dict | None = None, category: str | None = None) -> tuple[str, str]:
    """(`single` | `bimanual` | `none`, why) from the object's resting extents (x, y, z in metres), its rest test
    (`rest_test`; None skips that check) and its dataset category (None: unlabelled, the category UNLABELLED)."""
    narrow, wide = sorted(float(e) for e in extent[:2])
    height = float(extent[2])
    grasp_height = thresholds.surface_top_m + height / 2
    if not category_used(category, thresholds):
        return "none", f"category {category or UNLABELLED}: not among the used categories (user 2026-11-05)"
    if rest is not None and not rest["stands"]:
        return "none", f"does not stand at rest on a plane: tilted {rest['tilt_deg']} deg, slid {rest['slid_mm']} mm"
    if height < thresholds.min_height_m:
        return "none", f"{height * 100:.1f} cm tall: flatter than {thresholds.min_height_m * 100:.0f} cm"
    if narrow <= thresholds.jaw_m:
        if grasp_height < thresholds.min_grasp_height_m:
            return "none", (f"narrow side {narrow * 100:.1f} cm fits the jaw but the grasp point {grasp_height:.3f} m "
                            f"is under {thresholds.min_grasp_height_m} m")  # fmt: skip
        return "single", f"narrow side {narrow * 100:.1f} cm fits the {thresholds.jaw_m * 100:.1f} cm jaw"
    lo, hi = thresholds.bimanual_width_m
    if lo <= wide <= hi and height >= thresholds.bimanual_min_height_m:
        return "bimanual", f"wide side {wide * 100:.1f} cm between two hands ({lo}-{hi} m), {height * 100:.1f} cm tall"
    return "none", f"narrow side {narrow * 100:.1f} cm > jaw; wide side {wide * 100:.1f} cm, {height * 100:.1f} cm tall: outside the two-hand range"  # fmt: skip


def density_of(mass_kg: float, hull_volume_m3: float) -> float:
    """The hull density that gives the catalog mass: the hulls carry the mass, so MuJoCo's centre of mass and inertia
    follow the scanned shape (a solid box of the bounding box put the 12-inch action figure's mass too high and it
    tipped over; with its hulls' own distribution it stands, 2026-11-04)."""
    return float(mass_kg) / max(float(hull_volume_m3), 1e-9)


def object_spec(model_dir: Path, mass_kg: float, hull_volume_m3: float):
    """The object's own MJCF as an MjSpec with absolute asset paths, a free joint, the scene's friction and the
    catalog mass over its hulls (the scan geom massless)."""
    d = Path(model_dir).resolve()
    spec = mujoco.MjSpec.from_file(str(d / "model.xml"))
    for m in spec.meshes:
        if m.file and not os.path.isabs(m.file):
            m.file = str(d / m.file)
    for t in spec.textures:
        if t.file and not os.path.isabs(t.file):
            t.file = str(d / t.file)
    body = spec.worldbody.bodies[0]
    body.add_freejoint()
    density = density_of(mass_kg, hull_volume_m3)
    for g in body.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            g.density = 0.0
        else:
            g.density = density
            g.friction = FRICTION
    return spec, body


def rest_test(model_dir: Path, mass_kg: float, hull_volume_m3: float) -> dict:
    """Drop the object 1 mm onto a plane and watch it REST_SETTLE_S + REST_CHECK_S: {tilt_deg, slid_mm, dz_mm, stands},
    `stands` when it neither tilts past REST_TILT_DEG nor slides past REST_SLIDE_M in the check (the scene lint's rest
    test, scripts/scene_lint.py, applied to the object alone)."""
    spec, body = object_spec(model_dir, mass_kg, hull_volume_m3)
    body.pos = [0.0, 0.0, 0.001]
    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2.0, 2.0, 0.1]
    floor.friction = [1.0, 0.02, 0.001]  # the platform's (assets/scene/logistics_room_box_v1)
    spec.option.timestep = REST_DT
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q_start = d.qpos[:7].copy()
    for _ in range(int(REST_SETTLE_S / REST_DT)):
        mujoco.mj_step(m, d)
    q0 = d.qpos[:7].copy()
    for _ in range(int(REST_CHECK_S / REST_DT)):
        mujoco.mj_step(m, d)
    q1 = d.qpos[:7].copy()
    tilt = float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q_start[3:], q1[3:])))))))
    slid = float(np.linalg.norm(q1[:2] - q0[:2]))
    turned = float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q0[3:], q1[3:])))))))
    stands = tilt <= REST_TILT_DEG and slid <= REST_SLIDE_M and turned <= REST_TILT_DEG
    return {"tilt_deg": round(tilt, 2), "slid_mm": round(slid * 1000, 1), "dz_mm": round(float(q1[2] - q_start[2]) * 1000, 1),
            "stands": bool(stands)}  # fmt: skip


def load_categories(path: Path) -> dict[str, str | None]:
    """{model name: category or None} from `categories.json` (scripts/gso_categories.py)."""
    doc = json.loads(Path(path).read_text())
    return dict(doc["categories"])


def catalog_entry(
    index: int, model_dir: Path, thresholds: Thresholds, rest: bool = True, category: str | None = None
) -> dict:
    """One object's record from its directory (`model.obj`, the hulls); `index` gives its key. `rest` runs the rest
    test (the simulator; off for a unit test of the geometry alone); `category` is the dataset's label for it."""
    name = model_dir.name
    v, _ = _read_obj(model_dir / "model.obj")
    mn, mx = v.min(0), v.max(0)
    extent = mx - mn
    hulls = sorted(model_dir.glob("model_collision_*.obj"))
    volume = sum(hull_volume(h) for h in hulls)
    mass = float(np.clip(volume * thresholds.density_kg_m3, *thresholds.mass_range_kg))
    rested = rest_test(model_dir, mass, volume) if rest else None
    kind, why = graspable(extent, thresholds, rested, category)
    return {
        "key": KEY_FORMAT.format(index=index),
        "name": name,
        "phrase": phrase(name),
        "category": category,
        "extent_m": [round(float(e), 4) for e in extent],
        "bbox_min_m": [round(float(e), 4) for e in mn],
        "bbox_max_m": [round(float(e), 4) for e in mx],
        "hull_volume_m3": round(volume, 7),
        "mass_kg": round(mass, 4),
        "grasp_height_m": round(thresholds.surface_top_m + float(extent[2]) / 2, 4),
        "n_vertices": int(len(v)),
        "n_hulls": len(hulls),
        "rest": rested,
        "graspable": kind,
        "why": why,
        "split": split_of(name, thresholds.held_out_fraction),
    }


def _entry(args) -> dict:
    return catalog_entry(*args)


def build_catalog(
    models: Path,
    thresholds: Thresholds | None = None,
    source: dict | None = None,
    workers: int | None = None,
    categories: dict[str, str | None] | None = None,
) -> dict:
    """Every `models/<name>/model.obj` as a record, keyed `gso_NNNN` in sorted-name order (the key never changes while
    the model set does not); `workers` processes (default: the machine's cores, at most 16). `categories` (from
    `load_categories`) must name every model unless `thresholds.used_categories` is None: a model missing from the
    labels would silently count as unlabelled and be drawn, or dropped, by mistake. `counts.excluded` is the number of
    objects whose category is not used; `not_used_categories` lists those categories."""
    thresholds = thresholds or Thresholds()  # fmt: skip
    dirs = sorted(p for p in models.iterdir() if (p / "model.obj").is_file())
    if not dirs:
        raise FileNotFoundError(f"no models under {models}: run scripts/install_gso.sh")
    categories = categories or {}
    if thresholds.used_categories is not None:
        unknown = [d.name for d in dirs if d.name not in categories]
        if unknown:
            raise ValueError(f"{len(unknown)} models have no entry in categories.json (scripts/gso_categories.py), "
                             f"e.g. {unknown[:3]}: cannot select {list(thresholds.used_categories)}")  # fmt: skip
    jobs = [(i, d, thresholds, True, categories.get(d.name)) for i, d in enumerate(dirs)]
    workers = workers or min(16, os.cpu_count() or 1)
    with ProcessPoolExecutor(workers) as pool:
        records = list(pool.map(_entry, jobs, chunksize=8))
    objects = {rec["key"]: rec for rec in records}
    counts = {k: sum(1 for o in objects.values() if o["graspable"] == k) for k in GRASPABLE}
    counts["held_out"] = sum(1 for o in objects.values() if o["split"] == "held_out")
    out = [o["category"] or UNLABELLED for o in objects.values() if not category_used(o["category"], thresholds)]
    counts["excluded"] = len(out)
    return {
        "version": CATALOG_VERSION,
        "source": source or {},
        "thresholds": asdict(thresholds),
        "not_used_categories": sorted(set(out)),
        "counts": {"objects": len(objects), **counts},
        "objects": objects,
    }


def load_catalog(path: Path) -> dict:
    cat = json.loads(Path(path).read_text())
    if cat.get("version") != CATALOG_VERSION:
        raise ValueError(f"{path}: catalog version {cat.get('version')!r}, this code reads {CATALOG_VERSION}")
    return cat


def pool_keys(catalog: dict, kind: str, pool: str) -> list[str]:
    """The keys of one graspability kind in one pool, in catalog order."""
    if pool not in POOLS:
        raise ValueError(f"pool {pool!r}: one of {POOLS}")
    want = {"train": {"train"}, "eval_unseen": {"held_out"}, "all": {"train", "held_out"}}[pool]
    return [k for k, o in catalog["objects"].items() if o["graspable"] == kind and o["split"] in want]


def sample_keys(
    catalog: dict, seed: int, n: int = 10, n_bimanual: int = 4, pool: str = "train", attempt: int = 0
) -> list[str]:
    """`n` objects for one scene instance: `n_bimanual` two-hand ones and the rest single-hand, without replacement,
    from `pool` (user Q18: the ratio is a parameter, 6 / 4 by default; Q22: `train`, `eval_unseen` or `all`).
    `attempt` > 0 is the instance's next draw with the same seed, for when the first set does not fit on the
    platform (place_objects; seed 4's 0.44 x 0.24 m game box with the carton and three more wide boxes did not)."""
    if not 0 <= n_bimanual <= n:
        raise ValueError(f"--bimanual {n_bimanual} must be between 0 and --objects {n}")
    rng = np.random.default_rng(seed if attempt == 0 else [seed, attempt])
    out = []
    for kind, count in (("single", n - n_bimanual), ("bimanual", n_bimanual)):
        keys = pool_keys(catalog, kind, pool)
        if count > len(keys):
            raise ValueError(f"pool {pool!r} holds {len(keys)} {kind} objects, {count} asked")
        out += [keys[i] for i in sorted(rng.choice(len(keys), size=count, replace=False))]
    return out


def place_objects(
    boxes: dict[str, list[float]],
    surface: dict,
    seed: int,
    reach_depth_m: float,
    gap_m: float = 0.03,
    tries: int = 4000,
) -> dict[str, dict]:
    """A spot and yaw on `surface` for every object (`boxes`: key -> half extents x, y of its footprint), largest
    footprint first, by rejection sampling: inside the top by start.SURFACE_MARGIN_M, `gap_m` from every placed
    footprint (more than start.OBJECT_GAP_M, so a drawn start can still move one object among the others), and within
    `reach_depth_m` of an approach edge when such a spot exists (D13: the middle of the 1.2 x 0.9 m platform is out of
    reach from every edge; user Q9: a distractor may stand there, a target may not). Returns
    {key: {xy, yaw, in_reach_band}}; raises when an object finds no spot at all."""  # fmt: skip
    rng = np.random.default_rng(seed)
    flat = np.asarray(surface.get("flat_half_extent") or surface["half_extent"], float)[:2]
    yaw_s = float(surface.get("yaw", 0.0))
    top = (np.asarray(surface["xy"], float), flat - SURFACE_MARGIN_M, yaw_s)
    room = np.maximum(flat - SURFACE_MARGIN_M, 0.0)
    taken, out = [], {}
    for key in sorted(boxes, key=lambda k: -float(np.prod(boxes[k][:2]))):
        obj = {"box": {"half_extent": list(boxes[key][:2])}}
        found = None
        for in_band in (True, False):
            for _ in range(tries):
                xy = top[0] + _rot2(yaw_s) @ rng.uniform(-room, room)
                yaw = float(rng.uniform(-np.pi, np.pi))
                quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
                fp = footprint(obj, xy, quat)
                if not inside_top(fp, top) or any(overlap(fp, t, max(gap_m, OBJECT_GAP_M)) for t in taken):
                    continue
                if in_band and min(edge_depths(surface, xy).values()) > reach_depth_m:
                    continue
                found = (xy, yaw, in_band)
                break
            if found:
                break
        if found is None:
            raise RuntimeError(f"no spot on the surface for {key} after {2 * tries} tries ({len(taken)} placed)")
        taken.append(footprint(obj, found[0], np.array([np.cos(found[1] / 2), 0.0, 0.0, np.sin(found[1] / 2)])))
        out[key] = {"xy": [float(found[0][0]), float(found[0][1])], "yaw": found[1], "in_reach_band": found[2]}
    return out


def yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def add_to_scene(root: ET.Element, scene_dir: Path, models: Path, key: str, rec: dict, pos, yaw: float) -> str:
    """Copy the object's meshes into `<scene_dir>/gso/<key>/` and write its assets and free body into the MJCF tree
    (paths relative to the scene directory, as the pack's: a run copies the directory). The scan is one visual geom
    (no contact, no mass), the hulls collide (group 3, drawn by neither MuJoCo's cameras nor the Unreal export, which
    drops a body's colliders when it has visuals), and the hulls carry the catalog mass as their density (`density_of`:
    the centre of mass and inertia follow the scanned shape). Returns the body name `<key>_object` (scene_settle.py rewrites `pos` / `quat` by it)."""
    src = models / rec["name"]
    dst = scene_dir / "gso" / key
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.suffix in (".obj", ".png"):
            shutil.copy2(f, dst / f.name)
    rel = f"gso/{key}"
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name=f"{key}_tex", type="2d", file=f"{rel}/texture.png")
    ET.SubElement(asset, "material", name=f"{key}_mat", texture=f"{key}_tex", specular="0.5", shininess="0.5")
    ET.SubElement(asset, "mesh", name=f"{key}_vis", file=f"{rel}/model.obj")
    hulls = sorted(dst.glob("model_collision_*.obj"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    for i, h in enumerate(hulls):
        ET.SubElement(asset, "mesh", name=f"{key}_col_{i:02d}", file=f"{rel}/{h.name}")
    body = ET.SubElement(root.find("worldbody"), "body", name=f"{key}_object", pos=_fmt(pos), quat=_fmt(yaw_quat(yaw)))
    ET.SubElement(body, "freejoint", name=f"{key}_free")
    ET.SubElement(body, "geom", name=f"{key}_vis", type="mesh", mesh=f"{key}_vis", material=f"{key}_mat", contype="0",
                  conaffinity="0", group="1", density="0")  # fmt: skip
    density = f"{density_of(rec['mass_kg'], rec['hull_volume_m3']):.6g}"
    for i in range(len(hulls)):
        ET.SubElement(body, "geom", name=f"{key}_col_{i:02d}", type="mesh", mesh=f"{key}_col_{i:02d}", group="3",
                      friction=_fmt(FRICTION), density=density, rgba="0.8 0.8 0.8 0.3")  # fmt: skip
    return f"{key}_object"


def object_box(rec: dict) -> dict:
    """The scene file's `box` (half extents and centre in the body frame) from the catalog's bounding box."""
    mn, mx = np.asarray(rec["bbox_min_m"], float), np.asarray(rec["bbox_max_m"], float)
    return {
        "half_extent": [round(float(v), 4) for v in (mx - mn) / 2],
        "centre": [round(float(v), 4) for v in (mn + mx) / 2],
    }


def object_record(rec: dict, room: str, placed: dict, body_id: int, qpos_adr: int, init_pos, init_quat) -> dict:
    """The scene file's object entry: the legacy keys every reader uses plus the v2 `semantic` / `provenance` (every
    semantic field with a source) and a `gso` block naming the scan."""
    box = object_box(rec)
    kind = rec["graspable"]
    return {
        "body": f"{rec['key']}_object",
        "body_id": int(body_id),
        "qpos_adr": int(qpos_adr),
        "free": True,
        "surface": "A",
        "init_pos": [float(v) for v in init_pos],
        "init_quat": [float(v) for v in init_quat],
        "height": float(rec["extent_m"][2]),
        "scale": 1.0,
        "bimanual": kind == "bimanual",
        "container": False,
        "box": box,
        "semantic": {
            "phrase": rec["phrase"],
            "room": room,
            "movable": True,
            "graspable": kind,
            "mass_kg": float(rec["mass_kg"]),
            "upright_axis": [0.0, 0.0, 1.0],
        },
        "provenance": {k: "rule" for k in ("phrase", "room", "movable", "graspable", "mass_kg", "upright_axis")},
        "gso": {
            "name": rec["name"],
            "split": rec["split"],
            "capability": "untested" if kind == "bimanual" else "measured_envelope",
            "grasp_height_m": rec["grasp_height_m"],
            "in_reach_band": bool(placed["in_reach_band"]),
        },
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
