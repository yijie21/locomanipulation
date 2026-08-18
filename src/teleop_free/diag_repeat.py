"""Is "command -> where the palm ends up" a function at all?

Everything downstream of this question is different depending on the answer.

The reach has always carried a hand-fitted inverse of that mapping — the
REACH_X_GAIN / REACH_Z_GAIN constants, which say the palm lands at roughly
0.55x - 0.02 of what was asked for. Replacing those with a model fitted from
data is only worth doing if the mapping is a MAPPING: if the same command,
issued from different arm configurations and pelvis heights, puts the palm in
the same place.

There is already a reason to doubt it. Sweeping the wrist orientation on SONIC
v1.1 returned 66 deg and 41 deg for two identical zero commands, which is not
noise on a function, it is history dependence. If the position mapping is the
same, no static inverse can help and the effort belongs in the closed loop
instead — lower gain, shorter dead time — rather than in a better feedforward.

So: pick a few target commands, approach each from several different starting
poses, and measure the spread of where the palm actually settles.

    python src/teleop_free/diag_repeat.py [--side right] [--settle 4]

Reading the result:
    spread < 1 cm    a static inverse model is worth fitting
    spread > 3 cm    it is not; fix the loop, not the seed
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_server import LocomotionMode  # noqa: E402
from primitives import DT, Primitives  # noqa: E402

# Commands to probe, in the pelvis frame. Spread across the region the grasp
# actually uses rather than the whole box.
TARGETS = [
    np.array([0.55, -0.20, 0.00]),
    np.array([0.75, -0.35, -0.10]),
    np.array([0.40, -0.10, 0.10]),
]
# Where the arm is sent BEFORE each probe. Same command, different history —
# that difference is the whole experiment.
STARTS = [
    ("tucked", np.array([0.16, -0.16, -0.02])),
    ("high", np.array([0.30, -0.20, 0.35])),
    ("low", np.array([0.30, -0.20, -0.35])),
    ("across", np.array([0.35, 0.20, 0.00])),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--settle", type=float, default=4.0)
    args = ap.parse_args()

    p = Primitives()
    p.cmd.start_control()
    p.cmd.reset_pose()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)

    def hold(pos, seconds):
        p.cmd.set_wrists(**{f"{args.side}_pos": np.asarray(pos, np.float32)})
        end = time.time() + seconds
        while time.time() < end:
            s = p.obs()
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(s))
            time.sleep(DT)

        return p.obs()

    print(f"\nsettling {args.settle:.1f} s per probe, {len(STARTS)} starts per "
          f"target\n")
    all_spread = []
    for target in TARGETS:
        print(f"command {np.round(target, 2)}")
        landed = []
        for label, start in STARTS:
            hold(start, args.settle)                     # establish the history
            s = hold(target, args.settle)                # then the probe
            palm = p.world_to_root_full(s["palms"][args.side]["pos"], s)
            landed.append(palm)
            print(f"   from {label:8s} -> palm {np.round(palm, 3)}  "
                  f"pelvis_z {s['pelvis']['pos'][2]:.3f}")
        landed = np.array(landed)
        spread = float(np.linalg.norm(landed - landed.mean(0), axis=1).max())
        rng = landed.max(0) - landed.min(0)
        all_spread.append(spread)
        print(f"   spread {spread * 100:.1f} cm   per-axis range "
              f"{np.round(rng * 100, 1)} cm")
        print(f"   mean offset from the command "
              f"{np.round((landed.mean(0) - target) * 100, 1)} cm\n")

    worst = max(all_spread)
    print(f"worst spread across all targets: {worst * 100:.1f} cm")
    if worst < 0.01:
        print("  -> the mapping is a function; fit a static inverse model")
    elif worst < 0.03:
        print("  -> borderline; a model would help but will not close the gap "
              "alone")
    else:
        print("  -> NOT a function of the command alone. A static inverse "
              "cannot fix this; the effort belongs in the closed loop "
              "(lower gain, shorter dead time) rather than a better seed.")

    p.cmd.pause_control()
    p.close()


if __name__ == "__main__":
    main()
