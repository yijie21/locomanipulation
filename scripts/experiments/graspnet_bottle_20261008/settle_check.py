"""TODO 21.3.4: does each generated object rest where it was put? 2 s of free physics per scene (robot ignored).

PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/settle_check.py outputs/graspnet_bottle_20261008
"""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

root = Path(sys.argv[1])
rows = json.loads((root / "objects.json").read_text())["objects"]
out = []
for r in rows:
    m = mujoco.MjModel.from_xml_path(str(Path(r["scene"]) / "scene_locomani.xml"))
    d = mujoco.MjData(m)
    b = m.body("water_bottle_object").id
    adr = m.jnt_qposadr[m.body_jntadr[b]]
    mujoco.mj_forward(m, d)
    p0, q0 = d.qpos[adr : adr + 3].copy(), d.qpos[adr + 3 : adr + 7].copy()
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    p1, q1 = d.qpos[adr : adr + 3], d.qpos[adr + 3 : adr + 7]
    tilt = float(np.degrees(2 * np.arccos(min(1.0, abs(float(q0 @ q1))))))
    out.append(dict(id=r["id"], moved_m=float(np.linalg.norm(p1 - p0)), dz_m=float(p1[2] - p0[2]), rot_deg=tilt))
    print(
        f"obj {r['id']:03d}: moved {out[-1]['moved_m'] * 100:5.1f} cm dz {out[-1]['dz_m'] * 100:5.1f} cm rot {tilt:5.1f} deg",
        flush=True,
    )
(root / "settle_check.json").write_text(json.dumps(out, indent=1))
bad = [o["id"] for o in out if o["moved_m"] > 0.02 or o["rot_deg"] > 10]
print(f"{len(out) - len(bad)}/{len(out)} rest within 2 cm and 10 deg; unsettled: {bad}")
