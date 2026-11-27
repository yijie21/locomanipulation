"""simple_relook_20261126 development: replay relook.py's registration on the looks simple_bench.py --relook saved
(RELOOK_DUMP), with other settings, against the simulator's pelvis (record only).

    cd <repo> && SIMPLE_EVAL_OUT=<tree> SIMPLE_LEVELS=... PYTHONPATH=src $PY -I replay.py <dump dir> [variant ...]
variants: default, nocoarse, prior10 (PRIOR_SIG xy 10 cm), coarse36 (search +-36 cm); each also with "+zfeet" (the
believed pelvis height from the feet, the dump's h; dumps without h: the simulator's, a diagnosis only) or "+zodo" (the
odometry's z, relook.py before FEET_Z)
-> one line a look and variant: the prior's and the estimate's error (cm, deg), inliers, rms
"""

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parents[1] / "simple_eval_20261112"
sys.path.insert(0, str(HERE))
import episodes as EP  # noqa: E402
import relook as RL  # noqa: E402

D = Path(sys.argv[1])
VARS = sys.argv[2:] or ["default"]
_m = {}


def model(name: str):
    if name not in _m:
        t, rest = name.split("_L")
        lv, ix = rest.split("_e")
        ep = next(e for e in EP.episodes(t) if str(e["level"]) == lv and int(e["index"]) == int(ix))
        info = EP.build(ep)
        m = mujoco.MjModel.from_xml_path(info["dir"] + "/scene_locomani.xml")
        _m[name] = m
    return _m[name]


for f in sorted(D.glob("*.npz")):
    if f.stem.endswith("_step"):  # a step's ticks, not a look
        continue
    z = np.load(f)
    name = str(z["name"])
    m = model(name)
    if "qadr" in z.files:
        qadr = z["qadr"]
    else:  # the first dumps: the 29 joints in MuJoCo order (runtime.Sim's qadr)
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
    body = RL.Body(m, qadr)
    cl = np.load(EP.OUT / "cloud_blind" / f"{name}.npz")
    ref = RL.Reference(dict(np.load(EP.OUT / "look_dump_blind" / f"{name}.npz")), cl["mask"], np.asarray(cl["pts"], float),
                       body, z["joints0"], z["L0"])  # fmt: skip
    pb0, yb, pt, yt = z["pb"], float(z["yb"]), z["pt"], float(z["yt"])
    for vz in VARS:
        v, _, zm = vz.partition("+")
        pb = pb0.copy()
        if zm == "zfeet":
            pb[2] = float(z["h"]) if "h" in z.files else float(pt[2])
        saved = (RL.PRIOR_SIG, dict(RL.COARSE))
        if v == "prior10":
            RL.PRIOR_SIG = (0.10, 0.10, 0.05, math.radians(5.0))
        if v == "coarse36":
            RL.COARSE.update(xy=0.36)
        M = RL.place(z["P_lev"], pb, yb)
        M = M[M[:, 2] > RL.FLOOR_Z]
        est = dict(pb=pb, yb=yb)
        for _ in range(2):
            Mw = RL.apply(np.r_[est["pb"] - pb, est["yb"] - yb], M, pb)
            d, _ = ref.target_tree.query(Mw, distance_upper_bound=RL.TGT_PAD)
            rec = RL.register(ref, RL.voxel(M[~np.isfinite(d)], RL.VOX_MOV), pb, v != "nocoarse")
            x = np.asarray(rec["x"])
            est = dict(pb=pb + x[:3], yb=yb + float(x[3]))
        RL.PRIOR_SIG, RL.COARSE = saved[0], saved[1]
        e = 100 * np.linalg.norm(est["pb"][:2] - pt[:2])
        print(json.dumps(dict(f=f.stem, v=vz, dz_cm=round(100 * float(pb[2] - pt[2]), 2), h_dump="h" in z.files, prior_cm=round(100 * float(np.linalg.norm(pb[:2] - pt[:2])), 2),
                              prior_deg=round(math.degrees(yb - yt), 3), est_cm=round(float(e), 2),
                              est_deg=round(math.degrees(est["yb"] - yt), 3), inlier=rec["inlier"], rms_mm=rec["rms_mm"],
                              sd_cm=rec["sd_cm"], coarse=rec.get("coarse"))), flush=True)  # fmt: skip
