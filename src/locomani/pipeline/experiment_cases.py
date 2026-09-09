"""Portable exact sampling inputs, tied to immutable scene metadata."""

import hashlib
import json
import math
from pathlib import Path


def scene_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    data = dict(schema_version=1, scene_sha256=scene_hash(scene), cases=validate(cases))
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def load_cases(path, scene):
    data = json.loads(Path(path).read_text())
    if data["schema_version"] != 1 or data["scene_sha256"] != scene_hash(scene):
        raise ValueError("case bank schema or scene mismatch")
    return validate(data["cases"])
