"""Aria Digital Twin sequence -> objects in the sim frame, with the rule-derived fields of the scene file.

ADT's world is y-up; the sim is z-up, so every pose goes through `ADT_TO_SIM` (a +90 deg turn about x, recorded
in the scene file's `source.transform`). Poses are each object's first row in `scene_objects.csv` (static rows
carry timestamp -1). Existing readers treat an object's box `half_extent[2]` as its height and `[0]`/`[1]` as
its footprint, so each object gets a body frame that is its ADT frame with the axes permuted: body z is the
local axis that points up at the initial pose. `upright_axis` is ADT's canonical up written in that body frame,
which is what the `upright` predicate tests. Numpy only; the meshes are read in `scripts/adt_to_scene.py`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ADT_TO_SIM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
SHELL = "ApartmentEnv"
# fixed in the sim: furniture, things on or in walls, cabinet parts and doors (no joints in round 1, decision 6)
FIXED_CATEGORIES = frozenset(
    {
        "shelter", "shelf", "wall artwork", "part of a cabinet/wardrobe", "door", "cabinets and shelves", "curtain",
        "thermostat", "hook", "light", "table", "dining table", "coffee table", "side table", "console table",
        "dressing table", "tv stand", "bookcase", "refrigerator", "bed frame", "mattress", "couch", "armchair",
        "chair", "dining chair", "bar stool", "shoe rack", "built in", "wall clock", "wall-mounted coat rack",
        "picture", "television", "safety sign", "surveillance camera", "cabinet", "router", "smart home video",
        "dehumidifier", "air purifier", "step ladder", "spice rack", "wine rack", "rack",
    }
)  # fmt: skip
FIXED_SIZE = 1.0  # m; anything longer than this is furniture whatever its category says


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m) -> list[float]:
    """Rotation matrix -> (w, x, y, z), w >= 0."""
    m = np.asarray(m, dtype=float)
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    q = np.array([w, x, y, z])
    return (q / np.linalg.norm(q)).round(6).tolist()


@dataclass(frozen=True)
class AdtObject:
    uid: str
    name: str
    prototype: str
    category: str
    motion: str
    rigidity: str
    rot: np.ndarray  # object -> sim world
    pos: np.ndarray  # sim world
    box_lo: np.ndarray  # object frame
    box_hi: np.ndarray
    up_local: np.ndarray  # ADT canonical up, object frame

    def corners(self) -> np.ndarray:
        """The 8 corners of the ADT box, object frame."""
        lo, hi = self.box_lo, self.box_hi
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    @property
    def body_axes(self) -> np.ndarray:
        """Rows = the body frame's x, y, z axes in the object frame (a signed permutation, det +1)."""
        world_up = self.rot.T @ np.array([0.0, 0.0, 1.0])
        k = int(np.argmax(np.abs(world_up)))
        z = np.zeros(3)
        z[k] = np.sign(world_up[k])
        x = np.zeros(3)
        x[(k + 1) % 3] = 1.0
        return np.array([x, np.cross(z, x), z])

    def body(self) -> dict:
        """The sim body: its origin at the box centre (what every hand-built scene has and the grasp code assumes:
        `obj_top` is init_pos z + half height, `box.centre` is read under another spelling and so is always 0).
        Returns init_pos (world), init_quat, half_extent, centre [0, 0, 0], upright_axis, and `mesh_offset`: what
        to add to a mesh's vertices once they are in the body axes, so the ADT origin (often the base) moves."""
        a = self.body_axes
        body = self.corners() @ a.T
        lo, hi = body.min(axis=0), body.max(axis=0)
        centre = (hi + lo) / 2  # in the body axes, relative to the ADT origin
        return {
            "init_pos": (self.pos + self.rot @ (a.T @ centre)).round(4).tolist(),
            "init_quat": matrix_to_quat(self.rot @ a.T),
            "half_extent": ((hi - lo) / 2).round(4).tolist(),
            "centre": [0.0, 0.0, 0.0],
            "upright_axis": (a @ self.up_local).round(4).tolist(),
            "mesh_offset": (-centre).round(5).tolist(),
        }

    def footprint(self) -> np.ndarray:
        """World xy of the 8 box corners (callers take a hull or a bounding box)."""
        return (self.corners() @ self.rot.T + self.pos)[:, :2]

    def world_z(self) -> tuple[float, float]:
        z = (self.corners() @ self.rot.T + self.pos)[:, 2]
        return float(z.min()), float(z.max())

    @property
    def size(self) -> float:
        return float(np.max(self.box_hi - self.box_lo))

    @property
    def movable(self) -> bool:
        return self.category not in FIXED_CATEGORIES and self.size <= FIXED_SIZE


UPRIGHT_OFF_DEG = 30.0  # an ADT upright axis this far off the object's resting up is replaced (resting_upright)


def resting_upright(objects: dict, max_deg: float = UPRIGHT_OFF_DEG) -> dict[str, tuple[list, list]]:
    """Free objects whose `upright_axis` (ADT's canonical up) is more than `max_deg` off the up they rest with at
    `init_quat` get, in place, the body axis (+-x, +-y, +-z) nearest to that resting up (TODO 19.15, user
    2026-10-18): ADT's canonical up had 28 of the 64 free objects lying down at rest -- every book standing in a
    shelf, the potted plants, fruit, keys, remotes, the wallet -- so `upright(X)` failed even where X never moved.
    Returns {key: (old axis, new axis)}; the old one stays as `semantic.upright_axis_adt`."""
    out = {}
    for key, o in sorted(objects.items()):
        sem = o.get("semantic") or {}
        if not o.get("free") or not sem:
            continue
        ax = np.asarray(sem.get("upright_axis") or (0.0, 0.0, 1.0), dtype=float)
        ax = ax / np.linalg.norm(ax)
        up = quat_to_matrix(o["init_quat"]).T @ np.array([0.0, 0.0, 1.0])  # world up in the body frame
        if np.degrees(np.arccos(np.clip(float(ax @ up), -1.0, 1.0))) <= max_deg:
            continue
        i = int(np.argmax(np.abs(up)))
        new = [0.0, 0.0, 0.0]
        new[i] = float(np.sign(up[i]))
        if np.allclose(new, ax):  # it rests tilted from every body axis (the supply bins, 36 deg): nothing nearer
            continue
        out[key] = (sem.get("upright_axis"), new)
        sem["upright_axis_adt"] = sem.get("upright_axis")
        sem["upright_axis"] = new
        (o.setdefault("provenance", {}))["upright_axis"] = "rule (resting pose)"
    return out


def read_sequence(seq_dir) -> list[AdtObject]:
    """Every object instance of a sequence at its first recorded pose, in the sim frame."""
    seq_dir = Path(seq_dir)
    inst = json.loads((seq_dir / "instances.json").read_text())
    first, boxes = {}, {}
    with open(seq_dir / "scene_objects.csv") as f:
        for r in csv.DictReader(f):
            first.setdefault(r["object_uid"], r)
    with open(seq_dir / "3d_bounding_box.csv") as f:
        for r in csv.DictReader(f):
            boxes.setdefault(r["object_uid"], r)
    out = []
    for uid, i in inst.items():
        if i.get("instance_type") != "object" or uid not in first or uid not in boxes:
            continue
        r, b = first[uid], boxes[uid]
        rot = quat_to_matrix([float(r[k]) for k in ("q_wo_w", "q_wo_x", "q_wo_y", "q_wo_z")])
        t = np.array([float(r[k]) for k in ("t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]")])
        out.append(
            AdtObject(
                uid=uid,
                name=i["instance_name"],
                prototype=i["prototype_name"],
                category=i["category"],
                motion=i["motion_type"],
                rigidity=i["rigidity"],
                rot=ADT_TO_SIM @ rot,
                pos=ADT_TO_SIM @ t,
                box_lo=np.array([float(b[f"p_local_obj_{a}min[m]"]) for a in "xyz"]),
                box_hi=np.array([float(b[f"p_local_obj_{a}max[m]"]) for a in "xyz"]),
                up_local=np.asarray(i["canonical_pose"]["up_vector"], dtype=float),
            )
        )
    names = [o.name for o in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate instance names: {sorted(dup)}")
    return sorted(out, key=lambda o: o.name)


def graspable(half_extent, movable: bool, capability: dict) -> str:
    """`single` / `bimanual` / `none` from the box and the measured envelope (configs/scene_discovery/capability.yaml)."""
    if not movable:
        return "none"
    narrow = 2 * min(half_extent[0], half_extent[1])
    if narrow <= capability["reach_single_arm"]["jaw_opening"]:
        return "single"
    lo, hi = capability["reach_bimanual"]["box_grasp"]["width"]
    wide = 2 * max(half_extent[0], half_extent[1])
    if lo <= wide <= hi or lo <= narrow <= hi:
        return "bimanual"
    return "none"
