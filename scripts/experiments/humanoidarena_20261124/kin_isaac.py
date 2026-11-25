"""Kinematics check, Isaac side: Isaac Lab's forward kinematics of HumanoidArena's G1 (g1_29dof_with_dex3_rev_1_0_m2.usd,
ROBOT_USD_OVERRIDE) for random right-arm poses, in the pelvis frame; kin_compare.py puts cuRobo's (SIMPLE's Dex3 config)
and our numpy chain (g1arm.py) against it.

Per sample: the 7 right-arm joints uniform inside Isaac Lab's joint limits (5 % in from each end), the 3 waist joints
uniform in +-0.3 rad (inside their limits), every other joint at SONIC's standing default, the hands open. The joints are
written to the simulation and only the kinematics are updated (sim.forward(): PhysX's update_articulations_kinematic, no
physics step); the link frames are read from the PhysX view (link frames, not centres of mass). Also the Dex3 finger
links in the palm frame with the hand open and closed (the provider's close pose): where the grasp centre lies.

    GPU=3 bash run_ha.sh kin_isaac.py --ha_task ppbox --out <dir> [--n 200]
-> <dir>/kin_isaac.npz
"""

import argparse
import json
from pathlib import Path

import ha_sim as H
import numpy as np


def main() -> int:
    p = H.build_parser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    H.finish_args(a, a.ha_task)
    H.check_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = H.Sim(a)
    torch = sim.torch
    rob = sim.robot
    jn = sim.joint_names
    bn = sim.body_names
    lim = rob.data.joint_pos_limits[0].cpu().numpy().astype(np.float64)  # (J, 2), robot joint order
    q_def = rob.data.default_joint_pos[0].cpu().numpy().astype(np.float64).copy()
    for i, n in enumerate(H.ORDER):
        q_def[jn.index(n)] = sim.default[i]
    rng = np.random.default_rng(20261124)
    links = ["pelvis", "torso_link", "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link",
             "right_hand_palm_link", "right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]  # fmt: skip
    li = [bn.index(n) for n in links]

    def fk(qfull):
        rob.write_joint_state_to_sim(
            torch.tensor(qfull[None], dtype=torch.float32, device=rob.device),
            torch.zeros((1, len(jn)), dtype=torch.float32, device=rob.device),
        )
        sim.env.sim.forward()
        T = rob.root_physx_view.get_link_transforms().cpu().numpy()[0].astype(np.float64)  # (L, 7): p, quat xyzw
        out = {}
        for n, i in zip(links, li):
            x, y, z, w = T[i, 3:]
            out[n] = (T[i, :3].copy(), H.mat_of([w, x, y, z]))
        pp, pR = out["pelvis"]
        return {n: (pR.T @ (p_ - pp), pR.T @ R_) for n, (p_, R_) in out.items()}

    Q, POSE = [], {n: [] for n in links}
    for _ in range(a.n):
        qf = q_def.copy()
        for n in H.ARM["right"]:
            j = jn.index(n)
            lo, hi = lim[j]
            m = 0.05 * (hi - lo)
            qf[j] = rng.uniform(lo + m, hi - m)
        for n in H.WAIST:
            j = jn.index(n)
            lo, hi = lim[j]
            qf[j] = rng.uniform(max(lo, -0.3), min(hi, 0.3))
        f = fk(qf)
        Q.append([qf[jn.index(n)] for n in H.ORDER])
        for n in links:
            p_, R_ = f[n]
            POSE[n].append(np.concatenate([p_, H.quat_of(R_)]))
    # the Dex3 fingers in the palm frame, hand open and closed (the provider's binary poses), arm at the default
    hand = {}
    ridx = sim.provider._right_hand_idx.cpu().numpy().tolist()
    for closed in (False, True):
        qf = q_def.copy()
        qf[ridx] = sim.provider._get_hand_pose_from_binary("right", closed)
        f = fk(qf)
        pp, pR = f["right_hand_palm_link"]
        hand["closed" if closed else "open"] = {
            n: (pR.T @ (f[n][0] - pp)).tolist()
            for n in ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link")
        }
    lim_order = np.array([lim[jn.index(n)] for n in H.ORDER])
    np.savez_compressed(out / "kin_isaac.npz", names=np.array(H.ORDER), q=np.array(Q), lim=lim_order,
                        right_hand_joints=np.array([jn[i] for i in ridx]),
                        **{f"pose_{n}": np.array(v) for n, v in POSE.items()}, hand=json.dumps(hand))  # fmt: skip
    print(json.dumps(dict(n=a.n, hand=hand, startup_s=round(sim.startup_s, 1), joints=jn, bodies=bn)), flush=True)
    print("[kin_isaac] DONE", flush=True)
    sim.close()
    return 0


if __name__ == "__main__":
    H.main_guard(main)
