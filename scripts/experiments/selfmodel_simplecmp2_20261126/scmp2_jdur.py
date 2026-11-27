"""selfmodel_simplecmp2_20261126 V1: simple_bench.py's --joint-dur <json>. Off by default; off, simple_bench never imports
this file and nothing of an episode changes.

The joint-path pregrasp (--avoid --joint, Motor.reach_joints) lasts max(reach_min_s, jaw arc length / speed). Here the
speed is set so that it lasts the duration the file gives for the episode ("<task>_L<level>_e<index>"); the path, its
start-gap fade and the minimum-jerk profile are unchanged. A value is either
  - a number: that many seconds, or
  - {"plan": <another plan2.py plan npz>, "speed": <its arm's speed or null>, "dur_s": <its recorded duration>}: the
    duration reach_joints gives that other path here, at the same command (the same reference, the same arc arithmetic
    as reach_joints: wrist + R.OFF over the path, max(reach_min_s, arc / (speed or reach_speed))), computed live and so
    not rounded; dur_s is only recorded beside it, for the check.
Used to give one arm's planned paths another arm's durations (J0's paths at J_D's and the other way round): the arms are
identical up to the pregrasp's command, so the other path's duration here is the one that arm had. The asked duration
comes from another arm's plan of the same episode: an intervention on the timing, not a decision of the robot's.
"""

from __future__ import annotations

import json
import sys

import numpy as np

_MAP: dict = {}


def durations(path: str) -> dict:
    if path not in _MAP:
        _MAP[path] = json.load(open(path))
    return _MAP[path]


def arc_length(m, side: str, path, joint_idx) -> float:
    """reach_joints' jaw-centre arc length of the path (m), from the reference at the command."""
    MO = sys.modules[type(m).__module__]
    base = m.ref(m.t)
    P, idx = np.asarray(path, float), list(joint_idx)

    def jaw_of(q):
        j = base.joints.copy()
        j[idx] = q
        p, quat = m.kp(base.pos, base.quat, j)[MO.WRIST[side]]
        return p + MO.R.mat(quat) @ MO.R.OFF

    jaw = np.array([jaw_of(q) for q in P])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(jaw, axis=0), axis=1))]
    return float(arc[-1])


def speed_for(m, side: str, path, joint_idx, names, path_json: str, name: str) -> tuple[float, dict]:
    """-> (reach_joints' speed for this path, the stage record's joint_dur_set)."""
    D = durations(path_json)
    assert name in D, f"--joint-dur: no duration for {name} in {path_json}"
    v = D[name]
    rec = {}
    if isinstance(v, dict):
        pl = np.load(v["plan"])
        assert [str(n) for n in pl["names"]] == [str(n) for n in names], (name, v["plan"])
        other = arc_length(m, side, pl["Q"], joint_idx)
        dur = max(m.cfg.reach_min_s, other / (v.get("speed") or m.cfg.reach_speed))
        rec = dict(plan=str(v["plan"]), other_arc_cm=round(100 * other, 3), other_dur_rec_s=v.get("dur_s"))
    else:
        dur = float(v)
    assert dur >= m.cfg.reach_min_s, f"--joint-dur: {name} asks {dur} s, under reach_min_s {m.cfg.reach_min_s}"
    length = arc_length(m, side, path, joint_idx)
    speed = length / dur
    return speed, dict(asked_s=round(dur, 6), arc_cm=round(100 * length, 3), speed=round(speed, 8), **rec)
