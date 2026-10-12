"""TODO 21.3.4 probe: which grasp orientations can our IK reach on the bottle table from a standing stance.

A grasp is a tool-point position plus a jaw attitude: approach axis (palm x, the fingers' direction) at elevation
e below horizontal and azimuth az in the pelvis frame (0 = straight ahead, + = toward the robot's left), rolled by r
about it. For each, ArmIK (right arm + waist, tool point = ready_grasp_policy.OFFSET) is solved from the ready pose
and a near-ready pose; reachable = position error <= 1 cm and attitude error <= 10 deg. Pelvis frame, standing
pelvis 0.79 m, table top 0.62 m. Offline, no stack. Two stricter readings are reported beside it: the waist held to
the model's own joint limits (ArmIK widens waist pitch/roll 1.6x on purpose, as a lean the policy under-delivers),
and the Dex1 hand (outputs/scene/hand_points_dex1.npy, grasp-centre frame) at least 5 mm above the table top.

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/orientation_reach.py outputs/graspnet_bottle_20261008
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, ready_q

PELVIS_Z, TABLE = 0.79, 0.62
POINTS = {  # tool point (pelvis-frame x ahead, y left) and height above the table top
    "near_right_low": (0.38, -0.15, 0.03),
    "near_right_mid": (0.38, -0.15, 0.07),
    "far_right_mid": (0.48, -0.15, 0.07),
    "centre_mid": (0.40, 0.0, 0.07),
}
ELEVATIONS = (0, 30, 60, 90)
AZIMUTHS = range(-180, 180, 30)
ROLLS = (0, 45, 90, 135)


def attitude(elev, az, roll):
    """Palm frame: x along the approach (down by elev, turned by az), rolled about x."""
    return (
        Rotation.from_euler("ZY", [az, elev], degrees=True) * Rotation.from_euler("x", roll, degrees=True)
    ).as_matrix()


HAND = np.load(Path(__file__).resolve().parents[3] / "outputs/scene/hand_points_dex1.npy")


def solve_all(ik, target, R, seed):
    return ik.solve("right", target, R, seed, iters=300, rot_weight=2.0, use_waist=True, point_offset=OFFSET)


def main(out: Path) -> int:
    ik, strict = ArmIK(), ArmIK()
    for j in (13, 14):
        strict.lo[j] /= 1.6
        strict.hi[j] /= 1.6
    seeds = [ready_q()]
    rows = []
    for name, (x, y, h) in POINTS.items():
        target = np.array([x, y, TABLE + h - PELVIS_Z])
        for elev in ELEVATIONS:
            for az in AZIMUTHS if elev < 90 else (0,):
                ok = {"ik": False, "strict": False, "strict_clear": False}
                for roll in ROLLS:
                    R = attitude(elev, az, roll)
                    clear = float((target + HAND @ R.T)[:, 2].min() - (TABLE - PELVIS_Z)) >= 0.005
                    for key, solver in (("ik", ik), ("strict", strict)):
                        _, pos, rot = solve_all(solver, target, R, seeds[0])
                        hit = pos <= 0.01 and np.degrees(rot) <= 10
                        ok[key] |= bool(hit)
                        if key == "strict":
                            ok["strict_clear"] |= bool(hit and clear)
                rows.append({"point": name, "elev": elev, "az": az, **ok})
        sub = [r for r in rows if r["point"] == name]
        counts = {k: sum(r[k] for r in sub) for k in ("ik", "strict", "strict_clear")}
        print(name, f"of {len(sub)} approach directions:", counts, flush=True)
        for elev in ELEVATIONS:
            ok = [r["az"] for r in sub if r["elev"] == elev and r["strict_clear"]]
            print(f"  elev {elev:2d}: strict + table-clear azimuths {ok}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "orientation_reach.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
