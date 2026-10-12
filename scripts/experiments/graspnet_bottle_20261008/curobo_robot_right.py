"""TODO 21.3.4: the bottle robot (stock G1 + Dex1) for CuRobo, right arm + waist, pelvis fixed.

The warehouse study's robot builder (scripts/experiments/curobo_warehouse_20261008/curobo_robot.py) for the single-arm
line: the same inscribed-sphere fit and FK check against the MuJoCo samples, but only the torso and the right arm
collide and move; the left arm hangs where the deploy holds it and is left out of the chain. Runs in SIMPLE's venv
(CuRobo 0.7.4).

    PYTHONPATH=scripts/experiments/curobo_warehouse_20261008 third_party/SIMPLE/.venv/bin/python \
        scripts/experiments/graspnet_bottle_20261008/curobo_robot_right.py \
        outputs/graspnet_bottle_20261008/curobo/g1_geometry.json outputs/graspnet_bottle_20261008/curobo/g1_right_curobo.json
"""

from __future__ import annotations

import argparse
import json
import os

import curobo_robot
import numpy as np
from curobo_robot import ARM, MAX_SPHERES, check_fk, coverage, fit_link, tolerance

# The stock MJCF puts the torso frame 1 cm higher than the URDF the warehouse study used (waist_roll_link at z 0.035
# and torso_link 0.019 above it, against 0.044 and 0; shoulders 0.23778 above the torso, against 0.24778) and the
# wrist yaw joint 0.046 along the pitch link, against 0.051: FK off 10-15 mm (2026-10-08). The copy that carries the
# MJCF offsets lives in the outputs (writes it if missing); its meshes resolve against the original's directory.
ORIGINAL = curobo_robot.URDF
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
URDF = os.path.join(REPO, "outputs", "graspnet_bottle_20261008", "curobo", "g1_stock.urdf")
curobo_robot.URDF = URDF


def write_stock_urdf():
    import re

    s = open(ORIGINAL).read()

    def origin(text, joint, old, new):
        i = text.index(f'<joint name="{joint}"')
        j = text.index("<origin", i)
        k = text.index("/>", j)
        return text[:j] + re.sub(old, new, text[j:k]) + text[k:]

    s = origin(s, "waist_roll_joint", r'xyz="-0\.0039635 0 0\.044"', 'xyz="-0.0039635 0 0.035"')
    s = origin(s, "waist_pitch_joint", r'xyz="0 0 0"', 'xyz="0 0 0.019"')
    for side in ("left", "right"):
        s = origin(s, f"{side}_shoulder_pitch_joint", r"0\.24778", "0.23778")
        s = origin(s, f"{side}_wrist_yaw_joint", r'xyz="0\.051 ', 'xyz="0.046 ')
    os.makedirs(os.path.dirname(URDF), exist_ok=True)
    open(URDF, "w").write(s)


JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [f"right_{j}_joint" for j in ARM]
LINKS = ["torso_link"] + [f"right_{j}_link" for j in ARM]


def robot_config(spheres, buffer):
    links = [k for k in LINKS if k in spheres]
    ignore = {link: [o for o in LINKS[i + 1 : i + 3] if o in links] for i, link in enumerate(LINKS) if link in links}
    return dict(
        kinematics=dict(
            urdf_path=URDF,
            asset_root_path=os.path.dirname(ORIGINAL),
            base_link="pelvis",
            ee_link="right_wrist_yaw_link",
            collision_link_names=links,
            collision_spheres={k: spheres[k] for k in links},
            collision_sphere_buffer=buffer,
            self_collision_ignore={k: v for k, v in ignore.items() if v},
            self_collision_buffer={k: 0.0 for k in links},
            use_global_cumul=True,
            mesh_link_names=None,
            lock_joints=None,
            extra_links=None,
            cspace=dict(
                joint_names=JOINTS,
                retract_config=[0.0] * len(JOINTS),
                null_space_weight=[1.0] * len(JOINTS),
                cspace_distance_weight=[1.0] * len(JOINTS),
                max_jerk=500.0,
                max_acceleration=15.0,
            ),
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("geometry")
    ap.add_argument("out")
    ap.add_argument("--buffer", type=float, default=0.003)
    args = ap.parse_args()
    if not os.path.exists(URDF):
        write_stock_urdf()
    geo = json.load(open(args.geometry))
    rng = np.random.default_rng(0)
    spheres, report = {}, {}
    for link in LINKS:
        spheres[link] = fit_link(geo["links"][link], tolerance(link), rng)
        gap_max, gap_p95 = coverage(geo["links"][link], spheres[link], rng)
        report[link] = dict(spheres=len(spheres[link]), uncovered_max=gap_max, uncovered_p95=gap_p95)
        print(
            f"{link:28s} {len(spheres[link]):4d} spheres  uncovered max {1000 * gap_max:5.1f} mm  p95 {1000 * gap_p95:4.1f} mm"
        )
    total = sum(len(v) for v in spheres.values())
    print("spheres in total", total)
    if total > MAX_SPHERES:
        raise SystemExit(f"{total} spheres; CuRobo takes at most {MAX_SPHERES}")
    fk = check_fk(geo["fk_samples"], list(geo["joints"]))
    for link, e in fk.items():
        print(f"FK {link:22s} pos max {1000 * e['pos_max']:.2f} mm  rot max {e['rot_max_deg']:.3f} deg")
    lower = {n: geo["joints"][n][0] for n in JOINTS}
    upper = {n: geo["joints"][n][1] for n in JOINTS}
    with open(args.out, "w") as f:
        json.dump(
            dict(robot_cfg=robot_config(spheres, args.buffer), fit=report, fk_check=fk, lower=lower, upper=upper), f
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
