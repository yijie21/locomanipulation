"""The kit's examples from the benchmark's Isaac looks (LOOK_DUMP files, blind.py look) and their grasps.

    python make_examples.py <outputs/simple_eval_20261112> <kit examples dir>      (any Python with numpy)

Not shipped (dev/ stays out of the package). Packing, from the repository's root, with the examples in
outputs/grasp_kit_build/examples:
    tar -czf outputs/g1_grasp_kit_<date>.tar.gz --exclude __pycache__ --exclude dev --exclude .gitignore \
        -s ',^grasp_kit,g1_grasp_kit,' -s ',^outputs/grasp_kit_build/examples,g1_grasp_kit/examples,' \
        grasp_kit outputs/grasp_kit_build/examples        (BSD tar; GNU tar: --transform)

Per example: looks/<name>.npz (g1grasp/look.py: rgb = the left image, right, baseline, depth = LingBot's (the depth
the benchmark's points came from), K, T_base_cam, instruction) and expected/<name>.{npz,json}: the benchmark's mask,
points and table height in the kit's base frame, and the grasp pick.graspgen_grasp chose.

T_base_cam is fitted (least squares, rigid) to the points the benchmark itself worked out from that depth, so the kit
must reproduce them; the robot's frame there is the simulator's, the base frame the pelvis's turned by its heading.
"""

import json
import sys
from pathlib import Path

import numpy as np

O, EX = Path(sys.argv[1]), Path(sys.argv[2])
NAMES = {  # name: what to expect
    "XMovePickTeleop_L0_e3": "",
    "BendPickMP_L1_e3": "",
    "XMoveBendPickTeleop_L1_e0": "",
    "TabletopGraspMP_L0_e0": "",
    "TabletopGraspMP_L1_e3": "",
    "TabletopGraspMP_L1_e7": "wrong target",
}
NOTE = {"TabletopGraspMP_L1_e7": "SAM 3's largest 'object' is not the one on the table (IoU with the truth 0.32)"}
W, H = 640, 360
f = (W / 2) / np.tan(np.radians(110) / 2)
K = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]])
AHEAD, ROLL, CLEAR, PRE = 0.7, (-30.0, 110.0), 0.06, 0.08  # pick.py


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def backproject(depth):
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=2)


def kabsch(A, B):  # R, t with B = A R^T + t
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def graspgen_grasp(z, cloud_pts, T_wo, yaw):  # pick.graspgen_grasp
    support = np.percentile(cloud_pts[:, 2], 2)
    ahead, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    n_ok, first = 0, None
    for i, (T_og, score) in enumerate(zip(z["T_op"], z["score"])):
        Rw = (T_wo @ T_og)[:3, :3]
        roll = np.degrees(np.arctan2(-Rw[2, 1], Rw[:, 1] @ left))
        if Rw[:, 0] @ ahead >= AHEAD and ROLL[0] <= roll <= ROLL[1] and (T_wo @ T_og)[2, 3] >= support + CLEAR:
            n_ok += 1
            if first is None:
                first = (i, T_og, float(score))
    return first, n_ok


(EX / "looks").mkdir(parents=True, exist_ok=True)
(EX / "expected").mkdir(parents=True, exist_ok=True)
for n, expect in NAMES.items():
    d = np.load(O / ("look_dump_eyes_sam" if n.startswith("Tabletop") else "look_dump_eyes") / f"{n}.npz")
    c = np.load(O / "cloud_eyes_final" / f"{n}.npz")
    g = np.load(O / "cloud_eyes_final" / f"grasp_{n}.npz")
    lb, ok, pts = d["lb"], d["ok"], d["pts"]
    assert np.array_equal(ok, (lb > 0.2) & (lb < 5)), n
    assert len(c["pts"]) == len(d["cloud"]) and np.allclose(c["pts"], d["cloud"]), n
    cam = backproject(lb)
    R, t = kabsch(cam[ok], pts[ok])
    res = np.linalg.norm(cam[ok] @ R.T + t - pts[ok], axis=1)
    yaw, pb = float(d["yaw"]), d["pb"]
    T_bw = np.eye(4)
    T_bw[:3, :3], T_bw[:3, 3] = rz(yaw).T, -rz(yaw).T @ pb
    T_wc = np.eye(4)
    T_wc[:3, :3], T_wc[:3, 3] = R, t
    T_bc = T_bw @ T_wc
    # against the simulator's camera (MuJoCo's camera frame: x right, y up, looking along -z)
    Rmj = d["cam_R"] @ np.diag([1, -1, -1])
    cam_cm = 100 * np.linalg.norm(t - d["cam_p"])
    cam_deg = np.degrees(np.arccos(np.clip((np.trace(R.T @ Rmj) - 1) / 2, -1, 1)))
    np.savez_compressed(EX / "looks" / f"{n}.npz", rgb=d["left"], right=d["right"], baseline=0.05, depth=lb.astype(np.float32),
                        K=K, T_base_cam=T_bc, instruction=str(d["instruction"]))  # fmt: skip
    to_b = lambda P: P @ T_bw[:3, :3].T + T_bw[:3, 3]  # noqa: E731
    cloud_b = to_b(c["pts"])
    T_wo = np.eye(4)
    T_wo[:3, 3] = c["p_o"]
    first, n_ok = graspgen_grasp(g, c["pts"], T_wo, yaw)
    out = dict(name=n, instruction=str(d["instruction"]), phrase=str(d["named"]), mask_score=round(float(c["score"]), 3),
               iou_with_truth=round(float(c["iou"]), 3), points=len(cloud_b), table_top=float(d["top"] - pb[2]),
               centre=np.median(cloud_b, axis=0).round(4).tolist(), grasps=len(g["score"]), takeable=n_ok,
               fit_residual_mm=round(1000 * float(res.max()), 4), camera_vs_simulator=dict(cm=round(cam_cm, 2), deg=round(cam_deg, 2)))  # fmt: skip
    if expect:
        out.update(expect=expect, note=NOTE[n])
    if first is not None:
        i, T_og, s = first
        P = T_bw @ T_wo @ T_og
        out.update(grasp_rank=i, grasp_score=round(s, 4), grasp_palm=np.round(P, 6).tolist())
    np.savez_compressed(EX / "expected" / f"{n}.npz", mask=c["mask"], cloud=cloud_b, table_top=out["table_top"])
    json.dump(out, open(EX / "expected" / f"{n}.json", "w"), indent=1)
    print(n, "fit max mm", out["fit_residual_mm"], "camera vs simulator", out["camera_vs_simulator"], "takeable", n_ok,
          "grasp", out.get("grasp_rank"), out.get("grasp_score"), "table", round(out["table_top"], 3))  # fmt: skip
