"""bigym_20261124: check the generated cuRobo robot (make_config.py) against BiGym's own MuJoCo model.

    cd <loco checkout on the 4090>
    $BIGYM_PY -I scripts/experiments/bigym_20261124/curobo/fk_check.py all [outputs/bigym_20261124/curobo]  # any python with numpy
      = BIGYM_EGL_GPU=0 (BiGym's venv)  fk_check.py mujoco <dir>   poses, BiGym's FK, the sphere coverage
        CUDA_VISIBLE_DEVICES=0 (SIMPLE's venv, cuRobo) fk_check.py curobo <dir>   cuRobo's FK of the same poses, compare

The MuJoCo side is the model bigym.loco.make("reach_target_multi_modal") builds (the scene, the robot attached under its
namespace, BiGym's right_end_effector site added at load), FK by mj_kinematics on a scratch MjData from qpos0 with the
pose's joints set; every pose in the pelvis frame. Poses: 200 with the waist at 0, then 200 with the waist uniform in
+-0.3 rad (each of yaw, roll, pitch); the 7 right-arm joints uniform within their MJCF limits; the right fingers open
(0.0245); seed 0.

Sphere coverage, per right-side collision link, in the link frame, against the collision geoms as MuJoCo collides them
(make_config.Piece: mesh hulls, boxes, cylinders): 20000 fresh surface points (seed 7) -> the share inside the union of
the link's spheres, the largest distance of a point outside it (mm); 400 points on each sphere's surface that are on the
union's boundary (inside no other sphere of the link) -> their distance to the geoms (0 inside a piece, else to the
nearest of 60000 dense surface points; resolution ~1 mm) as the bloat: max, 95th percentile, mean (mm). cuRobo adds
collision_sphere_buffer (5 mm) to every radius on top.

cuRobo side: the yml with the right arm, the waist and the right fingers free, nothing locked; ee right_end_effector and
link right_wrist_yaw_link; also every right-side sphere's centre by cuRobo against MuJoCo's link pose applied to the yml's
centre (the sphere file is read the way it is meant).
-> <dir>/fk_check.json and a table on stdout
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BIGYM_PY = os.environ.get("BIGYM_PY", "/wuji-vefps-D/wuji-il/yijie/tools/bigym2/BiGym2/.venv/bin/python")
CUROBO_PY = os.environ.get("CUROBO_PY", "/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE/.venv/bin/python")
BIGYM_ROOT = "/wuji-vefps-D/wuji-il/yijie/tools/bigym2/BiGym2"
ENV_SH = "/wuji-vefps-D/wuji-il/yijie/tools/bigym2/env.sh"  # BiGym's caches and headless EGL
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
FINGERS = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]
OPEN = 0.0245
N = 200


def right_links() -> list[str]:
    sys.path.insert(0, str(HERE))
    import make_config as M

    return [n for n in M.collision_links() if n.startswith("right_")]


def rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    c = (np.einsum("nij,nij->n", Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ----------------------------------------------------------------------------- MuJoCo (BiGym's venv)
def stage_mujoco(out: Path) -> None:
    import mujoco
    from scipy.spatial import cKDTree

    sys.path.insert(0, str(HERE))
    import make_config as M
    import yaml
    from bigym.loco import make

    env = make("reach_target_multi_modal", camera_keys=())
    m = env.inner_env.model
    ns = env.inner_env.robot.namespace or ""
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ns + n)  # noqa: E731
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ns + n)  # noqa: E731
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, ns + "right_end_effector")
    assert site >= 0 and all(jid(n) >= 0 for n in ARM + WAIST + FINGERS)
    rng = np.random.default_rng(0)
    lo = np.array([m.jnt_range[jid(n)][0] for n in ARM])
    hi = np.array([m.jnt_range[jid(n)][1] for n in ARM])
    Q = rng.uniform(lo, hi, size=(2 * N, 7))
    W = np.zeros((2 * N, 3))
    W[N:] = rng.uniform(-0.3, 0.3, size=(N, 3))
    links = right_links()
    d = mujoco.MjData(m)
    ee_p, ee_R, wy_p, wy_R = [], [], [], []
    link_p, link_R = [], []
    for k in range(2 * N):
        d.qpos[:] = m.qpos0
        for n, v in zip(ARM, Q[k]):
            d.qpos[m.jnt_qposadr[jid(n)]] = v
        for n, v in zip(WAIST, W[k]):
            d.qpos[m.jnt_qposadr[jid(n)]] = v
        for n in FINGERS:
            d.qpos[m.jnt_qposadr[jid(n)]] = OPEN
        mujoco.mj_kinematics(m, d)
        pb = bid("pelvis")
        Rp, pp = d.xmat[pb].reshape(3, 3), d.xpos[pb]
        ee_p.append(Rp.T @ (d.site_xpos[site] - pp))
        ee_R.append(Rp.T @ d.site_xmat[site].reshape(3, 3))
        b = bid("right_wrist_yaw_link")
        wy_p.append(Rp.T @ (d.xpos[b] - pp))
        wy_R.append(Rp.T @ d.xmat[b].reshape(3, 3))
        link_p.append([Rp.T @ (d.xpos[bid(n)] - pp) for n in links])
        link_R.append([Rp.T @ d.xmat[bid(n)].reshape(3, 3) for n in links])
    np.savez(out / "fk_check_mujoco.npz", Q=Q, W=W, ee_p=ee_p, ee_R=ee_R, wy_p=wy_p, wy_R=wy_R, link_p=link_p,
             link_R=link_R, links=np.array(links))  # fmt: skip
    env.close()

    # sphere coverage on the raw MJCF's collision geoms (the same bodies; make() only prefixes the names)
    raw = mujoco.MjModel.from_xml_path(str(Path(BIGYM_ROOT) / M.MJCF))
    sp = yaml.safe_load((out / "spheres/g1_29dof_dex1.yml").read_text())["collision_spheres"]
    cov = {}
    for n in links:
        pieces = M.link_pieces(raw, raw.body(n).id)
        C = np.array([s["center"] for s in sp[n]])
        r = np.array([s["radius"] for s in sp[n]])
        S = M.surface_samples(pieces, 20000, seed=7)
        out_d = (np.linalg.norm(S[:, None] - C[None], axis=-1) - r[None]).min(1)
        dense = M.surface_samples(pieces, 60000, seed=11)
        tree = cKDTree(dense)
        fib = fibonacci(400)
        B = np.concatenate([c + rr * fib for c, rr in zip(C, r)])
        own = np.repeat(np.arange(len(C)), len(fib))
        dd = np.linalg.norm(B[:, None] - C[None], axis=-1) - r[None]
        dd[np.arange(len(B)), own] = np.inf
        bnd = B[dd.min(1) > 0]  # on the union's boundary
        inside = M.depth_union(pieces, bnd) > 0
        bl = np.where(inside, 0.0, tree.query(bnd)[0])
        cov[n] = dict(geoms=[p.label for p in pieces], spheres=len(C), covered_pct=round(100 * float((out_d <= 0).mean()), 2),
                      uncovered_max_mm=round(1000 * float(max(0.0, out_d.max())), 2),
                      bloat_max_mm=round(1000 * float(bl.max()), 1), bloat_p95_mm=round(1000 * float(np.percentile(bl, 95)), 1),
                      bloat_mean_mm=round(1000 * float(bl.mean()), 1), r_min_mm=round(1000 * float(r.min()), 1),
                      r_max_mm=round(1000 * float(r.max()), 1))  # fmt: skip
        print(n, cov[n], flush=True)
    (out / "fk_check_coverage.json").write_text(json.dumps(cov, indent=1))


def fibonacci(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


# ----------------------------------------------------------------------------- cuRobo (SIMPLE's venv)
def stage_curobo(out: Path) -> None:
    import torch
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import load_yaml

    z = np.load(out / "fk_check_mujoco.npz")
    links = [str(x) for x in z["links"]]
    td = TensorDeviceType()
    cfg = load_yaml(str(out / "g1_29dof_with_dex1_1.yml"))["robot_cfg"]
    kin = cfg["kinematics"]
    kin["external_asset_path"] = str(out)
    kin["external_robot_configs_path"] = str(out)
    kin["ee_link"] = "right_end_effector"
    kin["link_names"] = ["right_end_effector", "right_wrist_yaw_link"] + links
    kin["collision_link_names"] = links
    cs = kin["cspace"]
    free = WAIST + ARM + FINGERS
    keep = [i for i, n in enumerate(cs["joint_names"]) if n in free]
    for k_, v_ in list(cs.items()):
        if isinstance(v_, list) and len(v_) == len(cs["joint_names"]):
            cs[k_] = [v_[i] for i in keep]
    robot = RobotConfig.from_dict(cfg, td)
    model = CudaRobotModel(robot.kinematics)
    names = list(model.joint_names)
    assert sorted(names) == sorted(free), names
    full = {}
    q = np.zeros((2 * N, len(names)))
    for k in range(2 * N):
        full = dict(zip(ARM, z["Q"][k])) | dict(zip(WAIST, z["W"][k])) | {n: OPEN for n in FINGERS}
        q[k] = [full[n] for n in names]
    st = model.get_state(torch.tensor(q, **td.as_torch_dict()))

    def mats(qq: np.ndarray) -> np.ndarray:
        w, x, y, zz = qq.T
        return np.stack([np.stack([1 - 2 * (y * y + zz * zz), 2 * (x * y - zz * w), 2 * (x * zz + y * w)], -1),
                         np.stack([2 * (x * y + zz * w), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - x * w)], -1),
                         np.stack([2 * (x * zz - y * w), 2 * (y * zz + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)  # fmt: skip

    ee_p = st.ee_position.cpu().numpy()
    ee_R = mats(st.ee_quaternion.cpu().numpy())
    lp = st.link_poses["right_wrist_yaw_link"]
    wy_p, wy_R = lp.position.cpu().numpy(), mats(lp.quaternion.cpu().numpy())
    res = {}
    for tag, sl in (("waist_0", slice(0, N)), ("waist_rand_0.3", slice(N, 2 * N))):
        e_mm = 1000 * np.linalg.norm(ee_p[sl] - z["ee_p"][sl], axis=1)
        e_deg = rot_deg(ee_R[sl], z["ee_R"][sl])
        w_mm = 1000 * np.linalg.norm(wy_p[sl] - z["wy_p"][sl], axis=1)
        w_deg = rot_deg(wy_R[sl], z["wy_R"][sl])
        res[tag] = dict(right_end_effector=dict(max_mm=float(e_mm.max()), median_mm=float(np.median(e_mm)),
                                                max_deg=float(e_deg.max()), median_deg=float(np.median(e_deg))),
                        right_wrist_yaw_link=dict(max_mm=float(w_mm.max()), median_mm=float(np.median(w_mm)),
                                                  max_deg=float(w_deg.max()), median_deg=float(np.median(w_deg))))  # fmt: skip
    # the spheres where cuRobo puts them vs MuJoCo's link pose applied to the yml's centres
    import yaml

    sp = yaml.safe_load((out / "spheres/g1_29dof_dex1.yml").read_text())["collision_spheres"]
    S = st.link_spheres_tensor.cpu().numpy()  # (n, s, 4) in the pelvis frame
    buf = float(kin["collision_sphere_buffer"])
    worst = 0.0
    for k in range(0, 2 * N, 20):
        exp = np.concatenate([np.array([s["center"] for s in sp[n]]) @ z["link_R"][k][i].T + z["link_p"][k][i]
                              for i, n in enumerate(links)])  # fmt: skip
        got = S[k][S[k][:, 3] > 0, :3]
        dmin = np.linalg.norm(exp[:, None] - got[None], axis=-1).min(1)
        worst = max(worst, float(dmin.max()))
    n_exp = sum(len(sp[n]) for n in links)
    res["spheres"] = dict(n_expected=n_exp, n_curobo=int((S[0][:, 3] > 0).sum()), centre_max_mm=round(1000 * worst, 3),
                          radius_buffer_m=buf)  # fmt: skip
    cov = json.loads((out / "fk_check_coverage.json").read_text())
    report = dict(mujoco_model='bigym.loco.make("reach_target_multi_modal") (namespaced robot, BiGym\'s right_end_effector site)',
                  poses=dict(n_per_set=N, seed=0, arm="uniform within the MJCF limits", fingers=OPEN), fk=res, coverage=cov)  # fmt: skip
    (out / "fk_check.json").write_text(json.dumps(report, indent=1))
    print("cuRobo FK vs BiGym MuJoCo, pelvis frame (200 poses each)")
    for tag in ("waist_0", "waist_rand_0.3"):
        for ln in ("right_end_effector", "right_wrist_yaw_link"):
            r = res[tag][ln]
            print(f"  {tag:15s} {ln:22s} max {r['max_mm']:.4f} mm / {r['max_deg']:.4f} deg   median {r['median_mm']:.4f} mm / {r['median_deg']:.4f} deg")  # fmt: skip
    print("  spheres:", res["spheres"])
    print(f"{'link':28s} {'n':>3s} {'cov%':>7s} {'unc_mm':>7s} {'bloat_max':>9s} {'p95':>6s} {'mean':>6s}")
    for n, c in cov.items():
        print(f"{n:28s} {c['spheres']:3d} {c['covered_pct']:7.2f} {c['uncovered_max_mm']:7.2f} {c['bloat_max_mm']:9.1f} {c['bloat_p95_mm']:6.1f} {c['bloat_mean_mm']:6.1f}")  # fmt: skip


def main() -> None:
    which = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/bigym_20261124/curobo").resolve()
    if which == "mujoco":
        stage_mujoco(out)
    elif which == "curobo":
        stage_curobo(out)
    elif which == "all":
        me = str(Path(__file__).resolve())
        sh = f"export BIGYM_EGL_GPU=0; source {ENV_SH}; {BIGYM_PY} -I {me} mujoco {out}"
        subprocess.run(["bash", "-c", sh], check=True)
        subprocess.run(
            [CUROBO_PY, "-I", me, "curobo", str(out)], check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        )
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
