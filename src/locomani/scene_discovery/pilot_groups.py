"""Option C of round 1: group bucket-2 tasks by what is missing, pick representatives to pilot with existing skills.

A gap is either an *envelope* gap (an existing skill run outside its measured setting: a taller table, a wider
object, a longer carry; a pilot measures it) or a *skill* gap (no existing skill does it: putting on a surface,
reorienting; it needs a new skill, plan decision 11). Only tasks whose gaps are all envelope gaps can be piloted
now. They are grouped by their set of gaps; within a group the representative is the task closest to the
measured envelope (the summed distance of its untested checks from their measured ranges, each over a scale),
and a second one is taken from the far end so a pilot sees the group's spread. A pilot's result is written back
to capability.yaml, which moves the whole group to bucket 1 or 3 on the next check.
"""

from __future__ import annotations

import numpy as np

# the skill refuses these before the robot moves (or there is no skill): a pilot cannot measure them
SKILL_GAPS = frozenset({"place_on_surface.single", "place_into.bimanual", "reorient", "grasp_single.grip_height"})
SCALE = {  # what counts as one unit of extrapolation per check (m, kg)
    "support_top": 0.10,
    "object_size": 0.03,
    "inboard": 0.05,
    "mass": 0.2,
    "route": 1.0,
    "carry_distance": 1.0,
    "rim_height": 0.10,
    "opening": 0.05,
}


def is_skill_gap(gap: str) -> bool:
    return gap in SKILL_GAPS or gap.startswith("goal.")


def _outside(measured, bound) -> float:
    """Distance of a measured value (or list of values) outside a bound ([lo, hi] range or a maximum)."""
    m = np.atleast_1d(np.asarray(measured, dtype=float))
    if isinstance(bound, (int, float)):
        return float(np.max(np.maximum(m - bound, 0.0)))
    b = np.asarray(bound, dtype=float)
    if b.ndim == 2:  # a range per value, e.g. object_size [[w_lo, w_hi], [h_lo, h_hi]]
        return float(np.sum(np.maximum(b[:, 0] - m, 0.0) + np.maximum(m - b[:, 1], 0.0)))
    if b.shape == (2,) and m.size == 1:
        return float(max(b[0] - m[0], m[0] - b[1], 0.0))
    return float(np.sum(np.maximum(b - m, 0.0)))  # opening: a size per axis that must reach the measured one


def extrapolation(task: dict) -> float:
    total = 0.0
    for c in task["checks"]:
        if c["outcome"] == "untested" and c["name"] in SCALE and "bound" in c:
            total += _outside(c["measured"], c["bound"]) / SCALE[c["name"]]
    return round(total, 3)


def groups(results: list[dict]) -> list[dict]:
    """[{gaps, n, representatives: [{id, goal, extrapolation, checks}], skill_gaps}] for bucket-2 tasks."""
    by: dict = {}
    for t in results:
        if t["bucket"] != 2:
            continue
        by.setdefault(tuple(t["gaps"]), []).append(t)
    out = []
    for gaps, tasks in by.items():
        skill = [g for g in gaps if is_skill_gap(g)]
        ranked = sorted(tasks, key=lambda t: (extrapolation(t), t["id"]))
        reps = [ranked[0]] + ([ranked[-1]] if len(ranked) > 1 else [])
        out.append(
            {
                "gaps": list(gaps),
                "skill_gaps": skill,
                "pilotable": not skill,
                "n": len(tasks),
                "representatives": [
                    {
                        "id": t["id"],
                        "goal": t["goal"],
                        "extrapolation": extrapolation(t),
                        "untested": {c["name"]: c.get("measured") for c in t["checks"] if c["outcome"] == "untested"},
                    }
                    for t in reps
                ],
            }
        )
    return sorted(out, key=lambda g: (not g["pilotable"], len(g["gaps"]), -g["n"]))
