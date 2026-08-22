"""Do the objects sit still when nothing touches them?

The grasp diagnostics kept reporting the target moving 5 cm with no robot link
in contact — during the alignment phase, which barely moves the hand at all.
Once the contact report was widened past the hand, the only thing touching the
can turned out to be the table. So the question is not how the hand approaches;
it is whether the scene's own contact model holds a cylinder still.

Pure MuJoCo: load the scene, step it, watch. No controller, no ZMQ, no deploy —
so nothing here can be blamed on control, and it runs in seconds.

    python src/teleop_free/diag_settle.py [--seconds 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np
import paths

SCENE_META = (f"{paths.REPO}/outputs/scene/sim/"
              "scene_meta.json")
TOL = 0.01          # a millimetre or two of settling is fine; a centimetre is not


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    meta = json.load(open(SCENE_META))
    model = mujoco.MjModel.from_xml_path(meta["scene_xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tracked = {k: v for k, v in meta["objects"].items() if v.get("qpos_adr", -1) >= 0}
    start = {k: data.qpos[v["qpos_adr"]: v["qpos_adr"] + 3].copy()
             for k, v in tracked.items()}
    print(f"timestep {model.opt.timestep}, {len(tracked)} free objects")

    n = int(args.seconds / model.opt.timestep)
    every = max(1, int(2.0 / model.opt.timestep))
    for step in range(n + 1):
        if step % every == 0:
            row = []
            for k, v in tracked.items():
                a = v["qpos_adr"]
                moved = float(np.linalg.norm(data.qpos[a:a + 3] - start[k]))
                _, x, y, _ = data.qpos[a + 3: a + 7]
                up = 1 - 2 * (x * x + y * y)
                row.append(f"{k} {moved * 100:5.2f}cm up{up:+.2f}")
            print(f"t={step * model.opt.timestep:5.1f}  " + "  ".join(row))
        mujoco.mj_step(model, data)

    print()
    bad = False
    for k, v in tracked.items():
        a = v["qpos_adr"]
        moved = float(np.linalg.norm(data.qpos[a:a + 3] - start[k]))
        _, x, y, _ = data.qpos[a + 3: a + 7]
        up = 1 - 2 * (x * x + y * y)
        verdict = "ok" if moved < TOL and up > 0.9 else "UNSTABLE"
        if verdict != "ok":
            bad = True
        print(f"  {k:8s} drifted {moved * 100:5.2f} cm, up {up:+.3f}  {verdict}")
    print("\n" + ("FAIL: something moves on its own — fix the scene, not the grasp"
                  if bad else "PASS: every object holds still"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
