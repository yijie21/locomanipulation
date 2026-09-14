"""Build the crate scene for the bimanual carry, with the finger pads hardened.

    python src/locomani/tasks/crate/bimanual_scene.py <out_dir> [--seed N]

Two things on top of build_sim_scene.py, neither of which belongs in that
shared file yet:

* SCENE_PRESET is forced to crate_bimanual, so the output is always the crate
  alone on the table with its near wall at the table edge (scene_lib).
* The four dex1 pad geoms get the CRATE contact calibration
  (`build_sim_scene.harden_finger_pads(spec, "crate")`: priority 1, condim 4,
  solref 0.002, solimp 0.995/0.9999). With MuJoCo's default soft contacts a
  held 0.30 kg bottle creeps out of the closed jaw at ~0.5 cm/s; the 1.44 kg
  crate hangs on two pads by friction and would not survive a lift at all.
  Applied to the WRITTEN xml and verified on the compiled model, because a
  pattern that misses is silent. The hardening itself is no longer this file's
  own code -- it was a second implementation with its own constants until
  2026-09-11; the gripper work below (coupling, finger range, actuator gear)
  still is, and is why this pass re-opens the emitted XML at all.

Point the sim at the result with LOCOMANI_SIM_DIR=<out_dir> (run_sim_locomani
reads scene_meta.json / scene_locomani.xml from there), so the shared
outputs/scene/sim that the single-hand collection uses is never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import mujoco

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)  # reach src/ for `python src/locomani/tasks/crate/bimanual_scene.py`
from locomani import paths  # noqa: E402
from locomani.sim.build_sim_scene import PAD_CONDIM, PAD_PRIORITY, harden_finger_pads  # noqa: E402
from locomani.tasks.crate.config import DEFAULT_CRATE_TASK  # noqa: E402

# The deploy drives the dex1 fingers with kp 1.5 / kd 0.1 at 50 Hz; the sim's
# actuator gear turns that into 375 N/m at gear 250. Gear 1000 (pilots
# 24-28) made the finger PD unstable under the 20 ms control hold (1500 N/m
# on an 87 g finger: 48 ms natural period) and the fingers chattered between
# their limits with the command sitting at "open" the whole time -- the
# "erratic open/close" in those videos. 250 it stays. Grip force comes from
# FINGER_RANGE below instead: with the pad faces able to meet, the closed
# target sits 8 mm past a 16 mm bar, 3 N per finger, 6 N per coupled jaw,
# 28 N of friction for two hands against the 0.64 kg crate's 6.3 N.
# Module-level default, not an env read (decision 8): CrateTaskConfig.gripper_gear
# is never actually overridden in any script today, so this constant IS that default.
GRIPPER_GEAR = DEFAULT_CRATE_TASK.gripper_gear
# The two dex1 fingers are INDEPENDENT slide joints in the MJCF; the real
# Dex1 is a single-motor parallel jaw. Independent fingers cannot clamp a bar
# that is off the jaw centre: the near finger stops on the bar, the far one
# runs to its limit 2.9 mm from centre and never touches it (offline pinch
# test from the pilot-22 dump: one pad per hand, pinch force 0, that is what
# every "closed with contact, lift 0 cm" round was). A joint equality couples
# them, so the jaw self-centres on the bar as a parallel gripper does.
# CrateTaskConfig.couple_fingers's default (decision 8); never overridden in
# any script today, so this constant IS that default.
COUPLE_FINGERS = DEFAULT_CRATE_TASK.couple_fingers
# Finger travel: the MJCF's closed limit (-0.020) leaves the pad faces 5.8 mm
# apart, so a 16 mm bar more than ~3 mm off the jaw centre is met by one pad
# only. -0.023 lets the faces meet (0.1 mm), and the open limit is widened so
# the standard open target (0.0245) is not sitting ON the limit.
FINGER_RANGE = (-0.023, 0.030)


def harden_pads(xml_path: str) -> int:
    spec = mujoco.MjSpec.from_file(xml_path)
    # One implementation, in build_sim_scene.py, with the crate's calibration
    # (2026-09-11): this file used to carry a second copy with its own constants,
    # applied on top of the shared one. It also matched `dex1_col*` by prefix,
    # which would have caught a ghost hand's collision copies too; the shared
    # one requires contype/conaffinity, so it cannot.
    n = harden_finger_pads(spec, "crate")
    if COUPLE_FINGERS:
        for side in ("left", "right"):
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.name = f"{side}_gripper_couple"
            eq.name1 = f"{side}_hand_gripper_1_joint"
            eq.name2 = f"{side}_hand_gripper_2_joint"
            eq.data[:5] = [0.0, 1.0, 0.0, 0.0, 0.0]  # q1 = q2 (axes are mirrored, so this is symmetric closing)
    nj = 0
    for j in spec.joints:
        if "hand_gripper" in j.name:
            j.range = list(FINGER_RANGE)
            nj += 1
    if nj != 4:
        raise SystemExit(f"finger range: {nj} gripper joints found, want 4")
    ng = 0
    for act in spec.actuators:
        if "hand_gripper" in act.name:
            act.gear = [GRIPPER_GEAR, 0, 0, 0, 0, 0]
            ng += 1
    model = spec.compile()
    gears = [
        float(model.actuator_gear[a][0])
        for a in range(model.nu)
        if "hand_gripper" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    ]
    if ng != 4 or any(abs(g - GRIPPER_GEAR) > 1e-6 for g in gears):
        raise SystemExit(f"gripper gear did not take: {ng} actuators patched, gears {gears}")
    if (
        COUPLE_FINGERS
        and sum(
            1 for e in range(model.neq) if "gripper_couple" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e)
        )
        != 2
    ):
        raise SystemExit("finger coupling equality did not take")
    ids = [g for g in range(model.ngeom) if model.geom_priority[g] == PAD_PRIORITY]
    if len(ids) < 4 or any(model.geom_condim[g] != PAD_CONDIM for g in ids):
        raise SystemExit(
            f"pad hardening did not take: priority-{PAD_PRIORITY} geoms {ids} (want >= 4 with condim {PAD_CONDIM})"
        )
    with open(xml_path, "w") as f:
        f.write(spec.to_xml())
    return n


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--crate-setback",
        type=float,
        default=DEFAULT_CRATE_TASK.crate_setback,
        help="basket inset from the near table edge, in metres; the continuous 6/10 reference uses 0.04",
    )
    args = parser.parse_args()
    if not 0 <= args.crate_setback <= 0.20:
        parser.error("--crate-setback must be within 0..0.20 m")
    out = os.path.abspath(args.out_dir)
    # SCENE_TABLE_TOP / CRATE_SETBACK (scene_lib.CRATE_SETBACK) are knobs for
    # the walk-in geometry. A 0.45 m table with the crate 10 cm in from the
    # edge was tried on 2026-09-07 so the policy's default walking pose
    # (finger tips 0.56 m high, 0.19 m ahead) would clear the edge: the wall
    # grasp is then out of standing reach (IK 5-15 cm short at every stand;
    # needs a 17 cm squat, whose knees meet the set-back table). Shipped
    # values stay 0.62 / 0 -- crate_setback's own default (scene_lib.py), so
    # only --table-top needs passing here (decision 8: build_sim_scene.py's
    # CLI, not env).
    build_argv = [
        sys.executable,
        os.path.join(paths.SRC, "sim", "build_sim_scene.py"),
        os.path.join(paths.OUTPUTS, "scene"),
        out,
        "--preset",
        "crate_bimanual",
        "--table-top",
        "0.62",
    ]
    if args.seed is not None:
        build_argv += ["--seed", str(args.seed)]
    subprocess.run(build_argv, check=True)
    xml = os.path.join(out, "scene_locomani.xml")
    print(
        f"[bimanual_scene] finger pads hardened: {harden_pads(xml)} geoms, gripper gear {GRIPPER_GEAR:.0f}, fingers coupled {COUPLE_FINGERS}, range {FINGER_RANGE} -> {xml}"
    )
    set_crate_setback(out, args.crate_setback)


def set_crate_setback(directory, setback):
    """Set the initial edge inset in both physics and reset metadata, once."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import numpy as np

    directory = Path(directory)
    if not np.isfinite(setback) or not 0 <= setback <= 0.20:
        raise ValueError("crate setback must be within 0..0.20 m")
    meta_path = directory / "scene_meta.json"
    meta = json.loads(meta_path.read_text())
    crate = meta["objects"]["blue_crate"]
    # Builder emits the default inset. Reapplying this operation is idempotent.
    old = float(meta.get("crate_setback", DEFAULT_CRATE_TASK.crate_setback))
    crate["init_pos"][0] += setback - old
    tree = ET.parse(directory / "scene_locomani.xml")
    body = tree.find(f"./worldbody/body[@name='{crate['body']}']")
    if body is None:
        raise ValueError("crate body missing from scene")
    pos = np.fromstring(body.get("pos"), sep=" ")
    pos[0] += setback - old
    body.set("pos", " ".join(f"{value:.15g}" for value in pos))
    tree.write(directory / "scene_locomani.xml", encoding="unicode")
    meta["crate_setback"] = setback
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
