"""Kinematics check, comparison side (SIMPLE's venv, cuRobo): cuRobo's forward kinematics of the G1 with SIMPLE's Dex3
config (data/robots/g1/curobo/g1_29dof_with_dex3.yml, as plan2.py builds it) and our numpy chain (g1arm.py on a URDF),
both against Isaac Lab's of HumanoidArena's G1 (kin_isaac.py), for the right wrist yaw link and the right palm link in
the pelvis frame, in mm and deg.

cuRobo is built as curobo_avoid_20261120/plan2.py builds it (SIMPLE's patched cuRobo: the collision links in link_names,
the locked joints' links added first), except that only the right hand's joints are locked (at 0, the open hand): the
waist's 3 and the right arm's 7 are free, as kin_isaac.py samples them.

    cd $SIMPLE && CUDA_VISIBLE_DEVICES=3 .venv/bin/python -I <this file> <kin_isaac.npz> [--yml Y --urdf U] [--out J]
-> one JSON line (and --out): per link and model, mean / p95 / max of the position (mm) and rotation (deg) errors
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1arm  # noqa: E402

S = Path("/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE")


def mat_of(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def err(pa, Ra, pb, Rb):
    mm = 1000 * np.linalg.norm(pa - pb, axis=1)
    deg = np.degrees([np.linalg.norm(g1arm.rotvec(A.T @ B)) for A, B in zip(Ra, Rb)])
    return dict(mean_mm=round(float(mm.mean()), 3), p95_mm=round(float(np.percentile(mm, 95)), 3),
                max_mm=round(float(mm.max()), 3), mean_deg=round(float(np.mean(deg)), 3),
                p95_deg=round(float(np.percentile(deg, 95)), 3), max_deg=round(float(np.max(deg)), 3))  # fmt: skip


def curobo_fk(yml: Path, urdf_dir: Path, names, Q, links):
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import load_yaml

    td = TensorDeviceType()
    cfg = load_yaml(str(yml))["robot_cfg"]
    kin = cfg["kinematics"]
    kin["external_asset_path"] = str(urdf_dir)  # SIMPLE's (an absolute urdf_path, as the adjusted yml has, wins)
    kin["external_robot_configs_path"] = str(S / "data/robots/g1/curobo")  # SIMPLE's collision spheres
    kin["ee_link"] = "right_wrist_yaw_link"
    kin["link_names"] = list(dict.fromkeys(["right_wrist_yaw_link", "right_hand_palm_link"]
                                           + [n for n in kin["collision_link_names"] if n.startswith("right_")]))  # fmt: skip
    cs = kin["cspace"]
    cspace = list(cs["joint_names"])
    free = g1arm.ARM["right"] + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    lock = [n for n in cspace if n not in free and n.startswith("right_hand_")]
    keep = [i for i, n in enumerate(cspace) if n in free or n in lock]
    for k_, v_ in list(cs.items()):
        if isinstance(v_, list) and len(v_) == len(cspace):
            cs[k_] = [v_[i] for i in keep]
    kin["lock_joints"] = {n: 0.0 for n in lock}
    kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n.startswith("right_")]
    urdf = ET.parse(urdf_dir / kin["urdf_path"]).getroot()
    lk = [
        x.get("link") for j in urdf.iter("joint") if j.get("name") in lock for x in (j.find("parent"), j.find("child"))
    ]
    kin["link_names"] = list(dict.fromkeys(kin["link_names"] + lk))
    model = CudaRobotModel(RobotConfig.from_dict(cfg, td).kinematics)
    jn = list(model.joint_names)
    qt = torch.tensor([[q[names.index(n)] for n in jn] for q in Q], **td.as_torch_dict())
    st = model.get_state(qt)
    out = {}
    for ln in links:
        i = list(st.link_names).index(ln)
        p = st.links_position[:, i].cpu().numpy().astype(np.float64)
        R = np.array([mat_of(q) for q in st.links_quaternion[:, i].cpu().numpy().astype(np.float64)])
        out[ln] = (p, R)
    return out, jn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kin")
    ap.add_argument("--yml", default=str(S / "data/robots/g1/curobo/g1_29dof_with_dex3.yml"))
    ap.add_argument("--urdf", default=str(g1arm.SIMPLE_URDF))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    z = np.load(a.kin)
    names = [str(n) for n in z["names"]]
    Q = z["q"]
    links = ["right_wrist_yaw_link", "right_hand_palm_link"]
    iso = {ln: (z[f"pose_{ln}"][:, :3], np.array([mat_of(q) for q in z[f"pose_{ln}"][:, 3:]])) for ln in links}
    rep = dict(n=len(Q), yml=a.yml, urdf=a.urdf)
    cu, jn = curobo_fk(Path(a.yml), S / "data/robots/g1", names, Q, links)
    rep["curobo_joints"] = jn
    for ln in links:
        rep[f"curobo_{ln}"] = err(*cu[ln], *iso[ln])
    for ln in links:
        ch = g1arm.Chain(a.urdf, tip=ln)
        P, Rr = [], []
        for q in Q:
            T = ch.fk(dict(zip(names, q)))
            P.append(T[:3, 3])
            Rr.append(T[:3, :3])
        rep[f"numpy_{ln}"] = err(np.array(P), np.array(Rr), *iso[ln])
        # the constant part, if any: the mean offset in the link's own frame (Isaac's), and the mean rotation offset
        d = np.array([Ri.T @ (pc - pi) for pc, pi, Ri in zip(cu[ln][0], iso[ln][0], iso[ln][1])])
        rep[f"curobo_{ln}_offset_in_link_mm"] = (1000 * d.mean(0)).round(3).tolist()
    print(json.dumps(rep), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
