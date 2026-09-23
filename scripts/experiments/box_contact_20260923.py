"""Would the brake-then-reach trials have touched the warehouse box or pallet? Offline, from the recorded states.

The walk scene has no box, so the reach runs through empty air. This rebuilds, per trial, the walk scene plus
static copies of the warehouse box's five 1 cm collision walls (nor_cal_box: 0.45 x 0.32 x 0.33, open top)
and the pallet (1.2 x 0.9 x 0.371), placed with the box's front face on the trial's red line and centred on
the settled pelvis, then replays every recorded state through MuJoCo's collision detection. Contacts are
split by the reach controller's phase at that frame (joined on the privileged `seq`).

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/box_contact_20260923.py <run_dir>
Writes <run_dir>/box_contacts.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "outputs" / "scene" / "walk" / "scene_locomani.xml"
PALLET_TOP = 0.371
# Box-local (x across the approach, y along it, z up) half sizes and centres, from the warehouse scene's geoms.
WALLS = {
    "box_floor": ((0.225, 0.16, 0.005), (0.0, 0.0, 0.005)),
    "box_side_a": ((0.005, 0.16, 0.165), (0.22, 0.0, 0.165)),
    "box_side_b": ((0.005, 0.16, 0.165), (-0.22, 0.0, 0.165)),
    "box_back": ((0.225, 0.005, 0.165), (0.0, 0.155, 0.165)),
    "box_front": ((0.225, 0.005, 0.165), (0.0, -0.155, 0.165)),
}


def placed(line_xy, heading, lateral_xy):
    """World (name, size, pos, yaw) for the walls and the pallet: box front face on the line, centred on lateral_xy."""
    f = np.array([np.cos(heading), np.sin(heading)])
    left = np.array([-f[1], f[0]])
    across = float((np.asarray(lateral_xy) - line_xy) @ left)
    centre = np.asarray(line_xy) + 0.16 * f + across * left  # box centre on the floor of the box
    out = []
    for name, (size, (bx, by, bz)) in WALLS.items():
        # box x runs across the approach (the robot's left is +x here, as the pair order does not matter)
        xy = centre + bx * left + by * f
        out.append((name, size, (*xy, PALLET_TOP + bz), heading - np.pi / 2))
    pallet_xy = np.asarray(line_xy) + (0.45 - 0.005) * f + across * left
    out.append(("pallet", (0.6, 0.45, PALLET_TOP / 2), (*pallet_xy, PALLET_TOP / 2), heading - np.pi / 2))
    return out


def model_with(geoms):
    xml = SCENE.read_text()
    extra = "".join(
        f'<geom name="{n}" type="box" size="{s[0]} {s[1]} {s[2]}" pos="{p[0]} {p[1]} {p[2]}" '
        f'euler="0 0 {yaw}" contype="1" conaffinity="1" group="0" rgba="0.8 0.6 0.35 0.6"/>'
        for n, s, p, yaw in geoms
    )
    xml = xml.replace("<worldbody>", "<worldbody>" + extra, 1)
    # the XML's meshdir/assets are relative to the scene directory
    spec = mujoco.MjSpec.from_string(xml)
    spec.meshdir = str(SCENE.parent / (spec.meshdir or ""))
    spec.texturedir = str(SCENE.parent / (spec.texturedir or ""))
    return spec.compile()


def main() -> int:
    run = Path(sys.argv[1])
    rows = [r for r in json.loads((run / "results.json").read_text())["rows"] if "status" in r]
    out = {}
    for r in rows:
        trace = json.loads((run / "traces" / f"{r['label']}.json").read_text())["traces"]
        phase_of = {t["seq"]: t["controller_phase"] for t in trace}
        first_seq = min(phase_of)
        pelvis0 = np.asarray(trace[0]["pelvis_position_m"][:2])
        model = model_with(placed(np.asarray(r["line_xy"]), r["heading"], pelvis0))
        data = mujoco.MjData(model)
        obstacle = {model.geom(n).id: n for n in [*WALLS, "pallet"]}
        robot = {g for g in range(model.ngeom) if model.geom_bodyid[g] != 0}
        hits, deepest = {}, {}
        for part in ("prep", "reach"):
            path = run / "trace" / f"{r['label']}_{part}_state_trace.npz"
            if not path.exists():
                continue
            z = np.load(path)
            for seq, q in zip(z["seq"], z["qpos"]):
                if seq < first_seq:
                    continue  # before the reach controller: arm preparation, hands near the body
                phase = phase_of.get(int(seq)) or phase_of[max(s for s in phase_of if s <= seq)]
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                for c in data.contact[: data.ncon]:
                    pair = {c.geom1, c.geom2}
                    wall = pair & obstacle.keys()
                    body = pair & robot
                    if not wall or not body:
                        continue
                    key = f"{phase}:{obstacle[wall.pop()]}:{model.body(model.geom_bodyid[body.pop()]).name}"
                    hits[key] = hits.get(key, 0) + 1
                    deepest[key] = min(deepest.get(key, 0.0), float(c.dist))
        out[r["label"]] = dict(
            reach_path=r.get("reach_path"),
            status=r["status"],
            best_position_error_cm=r["best_position_error_cm"],
            contact_frames=hits,
            deepest_mm={k: round(1000 * v, 1) for k, v in deepest.items()},
        )
        print(r["label"], r.get("reach_path"), f"err {r['best_position_error_cm']:.1f} cm", hits or "no contact")
    (run / "box_contacts.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
