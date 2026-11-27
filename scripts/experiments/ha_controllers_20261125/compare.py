"""Side-by-side table of the controller smoke: HumanoidArena (probe_ctrl.py) vs LocoManiLab (lml_ref.py), 2026-11-25.

    # notebook: skip ha controller integration smoke
    python compare.py --ha $R/probe/ctrl_{c} --lml $R/probe/lml_ref/{c}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def r(x, n=1):
    if x is None:
        return "-"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(r(v, n) for v in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k[0]}:{r(v, n)}" for k, v in x.items()) + "}"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{n}f}"
    return str(x)


ROWS = [
    ("stand", "fell_at_s", "stand: fell at (s)", 1),
    ("stand", "root_z_min", "stand: pelvis z min (m)", 3),
    ("stand", "tilt_max_deg", "stand: tilt max (deg)", 1),
    ("stand", "xy_drift_m", "stand: xy drift 30 s (m)", 3),
    ("stand", "joint_err_vs_ref_end_mean_rad", "stand: |q - ref| mean at end (rad)", 3),
    ("stand", "joint_err_vs_ref_end_max_rad", "stand: |q - ref| max at end (rad)", 3),
    ("lag", "median_lag50_ms", "arm ramps: lag at 50 % (ms)", 0),
    ("lag", "median_lag90_ms", "arm ramps: lag at 90 % (ms)", 0),
    ("lag", "median_reached", "arm ramps: reached share", 2),
    ("lag", "ref_delay_s", "reference buffer delay (s)", 2),
    ("reach", "fell_at_s", "reach: fell at (s)", 1),
    ("reach", "settled", "reach: settled within 3 s", 0),
    ("reach", "label_after_ref_end_s", "reach: label after ref end (s)", 2),
    ("reach", "err_cm", "reach: settled hand-point error (cm)", 1),
    ("reach", "err_heading_cm", "reach: error fwd/left/up (cm)", 1),
    ("reach", "hand_moved_heading_cm", "reach: hand moved fwd/left/up (cm)", 1),
    ("reach", "hand_lag50_ms", "reach: hand-point lag 50 % (ms)", 0),
    ("reach", "hand_lag90_ms", "reach: hand-point lag 90 % (ms)", 0),
    ("reach", "arm_median_lag50_ms", "reach: arm joints lag 50 % (ms)", 0),
    ("reach", "median_lag50_ms", "reach: all moved joints lag 50 % (ms)", 0),
    ("reach", "pelvis_moved_cm", "reach: pelvis moved (cm)", 1),
    ("reach", "feet_moved_cm", "reach: feet moved (cm)", 1),
    ("reach", "max_tilt_deg", "reach: tilt max (deg)", 1),
    ("walk", "fell_at_s", "walk: fell at (s)", 1),
    ("walk", "ref_moved_heading_cm", "walk: reference moved fwd/left (cm)", 1),
    ("walk", "pelvis_moved_heading_cm", "walk: pelvis moved fwd/left (cm)", 1),
    ("walk", "goal_err_cm", "walk: final error to goal (cm)", 1),
    ("walk", "max_dev_from_ref_cm", "walk: max deviation from ref path (cm)", 1),
    ("walk", "yaw_change_deg", "walk: heading change (deg)", 1),
    ("walk", "feet_moved_cm", "walk: feet moved (cm)", 1),
    ("walk", "max_tilt_deg", "walk: tilt max (deg)", 1),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ha", type=Path, nargs="+", required=True)
    ap.add_argument("--lml", type=Path, nargs="+", required=True)
    a = ap.parse_args()
    cols = []
    for h, m in zip(a.ha, a.lml):
        H = json.load(open(h / "report.json"))
        M = json.load(open(m / "report.json"))
        cols += [(f"HA {H['controller']}", H), (f"LML {M['backend']}", M)]
    print("| quantity | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for ph, key, label, n in ROWS:
        print(f"| {label} | " + " | ".join(r((d.get(ph) or {}).get(key), n) for _, d in cols) + " |")
    print(f"| reach/walk: PD torque / limit max | " + " | ".join(
        r(((d.get("reach") or {}).get("host") or {}).get("pd_over_limit_max"), 2) + " / "
        + r(((d.get("walk") or {}).get("host") or {}).get("pd_over_limit_max"), 2) for _, d in cols) + " |")  # fmt: skip
    print(f"| ms per control step (stand) | " + " | ".join(
        r((d.get("stand") or {}).get("ms_per_control_step") or 1000 * d["stand"]["wall_s"] / max(d["stand"]["control_steps"], 1), 0)
        for _, d in cols) + " |")  # fmt: skip
    for _, d in cols:
        lg = d.get("lag") or {}
        if "moves" in lg:
            print(d.get("controller") or d.get("backend"), "HA" if "controller" in d else "LML", "lag moves (joint, lag50, lag90, reached, end err):",
                  [(m["joint"].replace("_joint", ""), m["lag50_ms"], m["lag90_ms"], round(m["reached"], 2), round(m["end_err_rad"], 3)) for m in lg["moves"]])  # fmt: skip


if __name__ == "__main__":
    main()
