"""Portable exact sampling inputs, tied to immutable scene metadata."""

import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = 2


def scene_hash(path):
    """Schema 1: the scene_meta.json bytes. They hold absolute paths (`scene_xml`, object meshes), so a bank
    written on one checkout never loads on another, however identical the scene. Kept to read old banks."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable_scene_hash(path, root=None):
    """Schema 2: the scene_meta.json content with every path under the repository made relative to it, in
    canonical JSON -- the same scene built in two checkouts hashes the same (checked 2026-09-25: a rebuild of
    `sim_lowcom` differs from the original only in its own directory)."""
    if root is None:
        from locomani import paths

        root = paths.REPO
    prefix = str(Path(root).resolve()) + "/"

    def rel(o):
        if isinstance(o, dict):
            return {k: rel(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rel(v) for v in o]
        if isinstance(o, str) and o.startswith(prefix):
            return o[len(prefix) :]
        return o

    canonical = json.dumps(rel(json.loads(Path(path).read_text())), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


HASHES = {1: scene_hash, 2: portable_scene_hash}


def validate(cases):
    ids = set()
    for c in cases:
        if c["case_id"] in ids:
            raise ValueError("duplicate case_id")
        ids.add(c["case_id"])
        for key in ("can_xy", "stand", "rel"):
            if len(c[key]) != 2 or not all(math.isfinite(x) for x in c[key]):
                raise ValueError(f"invalid {key}")
        for key in ("psi", "spawn_off", "spawn_lat", "dpsi"):
            if not math.isfinite(c[key]):
                raise ValueError(f"invalid {key}")
        if "object_yaw" in c and (type(c["object_yaw"]) not in (int, float) or not math.isfinite(c["object_yaw"])):
            raise ValueError("invalid object_yaw")
        if c["squat_hint"] is not None and not math.isfinite(c["squat_hint"]):
            raise ValueError("invalid squat_hint")
        if not isinstance(c["appearance"], int) or not isinstance(c["target"], str):
            raise ValueError("invalid appearance or target")
    return cases


def save_cases(path, cases, scene):
    data = dict(schema_version=SCHEMA_VERSION, scene_sha256=portable_scene_hash(scene), cases=validate(cases))
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def load_cases(path, scene):
    data = json.loads(Path(path).read_text())
    digest = HASHES.get(data["schema_version"])
    if digest is None or data["scene_sha256"] != digest(scene):
        raise ValueError("case bank schema or scene mismatch")
    return validate(data["cases"])
