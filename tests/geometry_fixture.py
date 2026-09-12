"""Explicit portable geometry inputs for CPU control tests; no generated output paths."""

import json
from pathlib import Path

from locomani.primitives.arm_ik import ArmIK

DIRECTORY = Path(__file__).parent / "fixtures/geometry"


def scene_meta():
    return json.loads((DIRECTORY / "scene_meta.json").read_text())


def scene_ik():
    return ArmIK(str(DIRECTORY / "scene_locomani.xml"))
