"""Closest approach of each hand's fingertips to its grasp point, per round of a warehouse box run.

Replays <run>/trace/*_state_trace.npz through the run's own scene copy (every 5th row) and prints, per round and
hand, the smallest distance between the midpoint of the two Dex1 finger links and the grasp point on the box's
end wall (8 mm inside the wall, 1 cm under the rim -- the warehouse config's box_grasp_below_rim), with its world
components. The warehouse box stands with its walk-in axis on world +y, so dy < 0 is "short of the box", dz > 0
"above the grasp point". Written for the 2026-09-23 offline pilots (docs/handoff/warehouse-render-cost-20260923.md).

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/reach_gap_20260923.py outputs/runs/<run>
"""

import glob
import sys

import mujoco
import numpy as np

run = sys.argv[1]
m = mujoco.MjModel.from_xml_path(run + "/scene_reverse_camera/scene_locomani.xml")
d = mujoco.MjData(m)
fingers = {s: [m.body(f"{s}_dex1_finger_{i}_link").id for i in (1, 2)] for s in ("left", "right")}
box = m.body("nor_cal_box_object").id
hx, hz = 0.225, 0.165  # nor_cal_box half extents (0.45 x 0.32 x 0.33 m), body origin on its floor

for trace in sorted(glob.glob(run + "/trace/*_state_trace.npz")):
    qpos = np.load(trace)["qpos"]
    best = {}
    for k in range(0, len(qpos), 5):
        d.qpos[:] = qpos[k]
        mujoco.mj_kinematics(m, d)
        rot, origin = d.xmat[box].reshape(3, 3), d.xpos[box]
        for side, sx in (("left", -1), ("right", 1)):
            target = origin + rot @ np.array([sx * (hx - 0.008), 0.0, 2 * hz - 0.01])
            tip = np.mean([d.xpos[b] for b in fingers[side]], axis=0)
            err = tip - target
            if np.linalg.norm(err) < best.get(side, (9.0,))[0]:
                best[side] = (np.linalg.norm(err), err)
    label = trace.split("/")[-1].replace("_state_trace.npz", "")
    print(
        label,
        " ".join(
            f"{s}: {100 * n:.1f}cm (dx {100 * e[0]:+.0f} dy {100 * e[1]:+.0f} dz {100 * e[2]:+.0f})"
            for s, (n, e) in best.items()
        ),
    )
