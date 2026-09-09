"""Build the crate scene for the bimanual carry, with the finger pads hardened.

    python src/teleop_free/bimanual_scene.py <out_dir> [--seed N]

Two things on top of build_sim_scene.py, neither of which belongs in that
shared file yet:

* SCENE_PRESET is forced to crate_bimanual, so the output is always the crate
  alone on the table with its near wall at the table edge (scene_lib).
* The four dex1 pad geoms get the agentic branch's contact hardening
  (priority 1, condim 4, solref 0.002, solimp 0.995/0.9999). With MuJoCo's
  default soft contacts a held 0.30 kg bottle creeps out of the closed jaw at
  ~0.5 cm/s; the 1.44 kg crate hangs on two pads by friction and would not
  survive a lift at all. Applied to the WRITTEN xml and verified on the
  compiled model, because a pattern that misses is silent.

Point the sim at the result with LOCOMANI_SIM_DIR=<out_dir> (run_sim_locomani
reads scene_meta.json / scene_locomani.xml from there), so the shared
outputs/scene/sim that the single-hand collection uses is never touched.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

PAD_PRIORITY = 1
PAD_CONDIM = 4
PAD_SOLREF = (0.002, 1.0)
PAD_SOLIMP = (0.995, 0.9999, 0.001, 0.5, 2.0)
# The deploy drives the dex1 fingers with kp 1.5 / kd 0.1 at 50 Hz; the sim's
# actuator gear turns that into 375 N/m at gear 250. Gear 1000 (pilots
# 24-28) made the finger PD unstable under the 20 ms control hold (1500 N/m
# on an 87 g finger: 48 ms natural period) and the fingers chattered between
# their limits with the command sitting at "open" the whole time -- the
# "erratic open/close" in those videos. 250 it stays. Grip force comes from
# FINGER_RANGE below instead: with the pad faces able to meet, the closed
# target sits 8 mm past a 16 mm bar, 3 N per finger, 6 N per coupled jaw,
# 28 N of friction for two hands against the 0.64 kg crate's 6.3 N.
GRIPPER_GEAR = float(os.environ.get("GRIPPER_GEAR", "250"))  # env override for A/B runs
# The two dex1 fingers are INDEPENDENT slide joints in the MJCF; the real
# Dex1 is a single-motor parallel jaw. Independent fingers cannot clamp a bar
# that is off the jaw centre: the near finger stops on the bar, the far one
# runs to its limit 2.9 mm from centre and never touches it (offline pinch
# test from the pilot-22 dump: one pad per hand, pinch force 0, that is what
# every "closed with contact, lift 0 cm" round was). A joint equality couples
# them, so the jaw self-centres on the bar as a parallel gripper does.
COUPLE_FINGERS = os.environ.get("COUPLE_FINGERS", "1") == "1"
# Finger travel: the MJCF's closed limit (-0.020) leaves the pad faces 5.8 mm
# apart, so a 16 mm bar more than ~3 mm off the jaw centre is met by one pad
# only. -0.023 lets the faces meet (0.1 mm), and the open limit is widened so
# the standard open target (0.0245) is not sitting ON the limit.
FINGER_RANGE = (-0.023, 0.030)


def harden_pads(xml_path: str) -> int:
    spec = mujoco.MjSpec.from_file(xml_path)
    n = 0
    for g in spec.geoms:
        if str(getattr(g, "meshname", "")).startswith("dex1_col"):
            g.priority = PAD_PRIORITY
            g.condim = PAD_CONDIM
            g.solref = PAD_SOLREF
            g.solimp = PAD_SOLIMP
            n += 1
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
    args = [a for a in sys.argv[1:]]
    if not args or args[0].startswith("-"):
        raise SystemExit(__doc__)
    out = os.path.abspath(args[0])
    env = dict(os.environ, SCENE_PRESET="crate_bimanual")
    # SCENE_TABLE_TOP / CRATE_SETBACK are knobs for the walk-in geometry. A
    # 0.45 m table with the crate 10 cm in from the edge was tried on
    # 2026-09-07 so the policy's default walking pose (finger tips 0.56 m
    # high, 0.19 m ahead) would clear the edge: the wall grasp is then out of
    # standing reach (IK 5-15 cm short at every stand; needs a 17 cm squat,
    # whose knees meet the set-back table). Shipped values stay 0.62 / 0.
    env.setdefault("SCENE_TABLE_TOP", "0.62")
    env.setdefault("CRATE_SETBACK", "0")
    if "--seed" in args:
        env["SCENE_SEED"] = args[args.index("--seed") + 1]
    subprocess.run(
        [sys.executable, os.path.join(paths.SRC, "build_sim_scene.py"), os.path.join(paths.OUTPUTS, "scene"), out],
        check=True,
        env=env,
    )
    xml = os.path.join(out, "scene_locomani.xml")
    print(
        f"[bimanual_scene] finger pads hardened: {harden_pads(xml)} geoms, gripper gear {GRIPPER_GEAR:.0f}, fingers coupled {COUPLE_FINGERS}, range {FINGER_RANGE} -> {xml}"
    )


if __name__ == "__main__":
    main()
