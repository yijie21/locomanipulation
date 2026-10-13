"""B3 (TODO 25.2.2, plan §4.1): the bottle reach-envelope cells both probes walk -- the 24.2 positions (probe_cells.cells:
forward 0.30 / 0.38 / 0.46 x left -0.20 / -0.12 / -0.04 x z 0.67 / 0.74 / 0.81, in the stance's heading frame) at four
palm attitudes (probe_cells.palm_rotation_at: elevation below horizontal, yaw left of the heading, roll about the
approach). Deliverable = position error <= 5 cm and attitude error <= 15 deg (24.6's servo range).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "wbc_compare_20261012"))

import probe_cells as C  # noqa: E402

ATTITUDES = (
    dict(id=0, elev=30, yaw_dev=0, roll=0, name="down 30, jaw left (24.2)"),
    dict(id=1, elev=30, yaw_dev=0, roll=180, name="down 30, jaw right"),
    dict(id=2, elev=0, yaw_dev=0, roll=0, name="level, jaw left"),
    dict(id=3, elev=0, yaw_dev=-45, roll=0, name="level, turned 45 right (toward the bottle side)"),
)
POS_TOL_CM, ROT_TOL_DEG = 5.0, 15.0


def cells() -> list[tuple[dict, dict]]:
    """(position, attitude) pairs, attitude-major so a probe turns the palm only four times."""
    return [(pos, att) for att in ATTITUDES for pos in C.cells()]


def deliverable(row: dict) -> bool:
    return (
        row.get("fell_at_s") is None
        and row.get("err_norm_cm") is not None
        and row["err_norm_cm"] <= POS_TOL_CM
        and row.get("rot_err_deg") is not None
        and row["rot_err_deg"] <= ROT_TOL_DEG
    )
