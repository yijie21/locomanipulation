"""selfmodel_faildiag_20261126 revision 6: the new tuck pose (T1b / T2b) by the registered rule, on LocoManiLab's own G1
model (g1_unified/robot.xml: its collision geoms; the hand point = wrist_yaw_link + 13 cm along its x, HAND_OFF).

Rule (fixed before any result with the pose):
  * both hand points not ahead of the pelvis's front edge (the largest forward extent of the pelvis body's collision
    geoms, pelvis frame), at LML's default stance (DEFAULT_Q legs and waist; pelvis at the origin);
  * no self-collision of either arm or hand with the torso, the pelvis, the legs or the other arm, by MuJoCo's own
    collision detection on the model's geoms (parent-child pairs filtered as the model does) -- at the default stance
    and at every 5th frame of the legs / waist of a real walk reference (lmlcmp3's SONIC walks of watering can ep 46);
  * every arm joint inside its range with 10 % of the range kept free at each end;
  * both hand points no higher than the pelvis (z <= 0 in the pelvis frame; `--zmax-cm 0`);
  * among those, the least horizontal distance of the hand points from the pelvis's vertical axis (the larger
    sqrt(x^2 + y^2) of the two, pelvis frame; `--objective radial`); candidates within TIE_CM of the least count as a
    tie, broken by the least elbow bend (the angle between the upper arm, shoulder roll link -> elbow link, and the
    forearm, elbow link -> wrist_yaw_link), then by grid order.
  This is the coordinator's second rule (11-26). The first rule (the least outward distance, the larger |y|, no height
  limit; `--objective out`, no --zmax-cm) picked both arms raised straight back, which SONIC did not follow in the
  3-episode smoke (revision 6 records its numbers); the change: the first rule had no height limit.
  The arms are mirror images (right = left with roll, yaw, wrist roll, wrist yaw negated: simple_bench.TUCK's
  convention). Search: a coarse grid over shoulder pitch / roll / yaw and elbow (wrists 0), then a finer grid over all
  seven joints from the best 12 coarse candidates (each step kept only if feasible at the stance and the walk poses).

    python fd_tuckpose.py --out <json>      (the chosen rule: --objective radial --zmax-cm 0, the defaults)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

XML = "/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend/assets/g1_unified/robot.xml"
WALK = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126/q1hit/rerun/sonic/steplog/IThorLivingRoomApproachTableAndLiftWateringCan/alone/ep+0046.npz"
MJ_NAMES = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
DEFAULT_Q = np.array([-0.2, 0, 0, 0.4, -0.2, 0, -0.2, 0, 0, 0.4, -0.2, 0, 0, 0, 0,
                      0, 0.4, 0, 1.2, 0, 0, 0, 0, -0.4, 0, 1.2, 0, 0, 0])  # lml_selfmodel.DEFAULT_Q  # fmt: skip
HAND_OFF = np.array([0.13, 0.0, 0.0])
MARGIN = 0.10
TIE_CM = 0.5
START_N = 12
OBJECTIVE = "radial"  # the chosen rule (11-26); "out" is the first rule
ZMAX = None
MIRROR = np.array([1, -1, -1, 1, -1, 1, -1], float)  # left -> right arm
SIMPLE_TUCK = np.array([0.30, 0.30, 0.0, 0.60, 0.0, 0.0, 0.0])


class Body:
    def __init__(self):
        m = self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(m)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.qadr = np.array([m.jnt_qposadr[jid(n)] for n in MJ_NAMES])
        self.lim = np.array([m.jnt_range[jid(n)] for n in MJ_NAMES])
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.pel, self.torso = bid("pelvis"), bid("torso_link")
        self.wrist = {s: bid(f"{s}_wrist_yaw_link") for s in ("left", "right")}
        self.sh = {s: bid(f"{s}_shoulder_roll_link") for s in ("left", "right")}
        self.el = {s: bid(f"{s}_elbow_link") for s in ("left", "right")}
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        arm_parts = ("shoulder", "elbow", "wrist", "hand")
        self.side = np.array([("left" if n.startswith("left") else "right" if n.startswith("right") else "") if any(p in n for p in arm_parts) else "" for n in names])
        self.free = int(m.jnt_qposadr[0]) if m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else None
        self.names = names
        self.front = self._pelvis_front()

    def set(self, q29: np.ndarray) -> None:
        d = self.d
        d.qpos[:] = 0
        if self.free is not None:
            d.qpos[self.free + 3] = 1.0
        d.qpos[self.qadr] = q29
        mujoco.mj_kinematics(self.m, d)

    def _pelvis_front(self) -> float:
        """The pelvis body's collision geoms' largest forward (pelvis x) extent, at the default stance."""
        m, d = self.m, self.d
        self.set(DEFAULT_Q)
        Rp, pp = d.xmat[self.pel].reshape(3, 3), d.xpos[self.pel]
        xs = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != self.pel or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                w = pg + v @ Rg.T
            else:
                c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                w = np.array([pg + Rg @ (c + h * np.array(s)) for s in itertools.product((-1, 1), repeat=3)])
            xs.append(((w - pp) @ Rp)[:, 0].max())
        return float(max(xs))

    def hands(self) -> dict:
        d = self.d
        Rp, pp = d.xmat[self.pel].reshape(3, 3), d.xpos[self.pel]
        out = {}
        for s in ("left", "right"):
            R = d.xmat[self.wrist[s]].reshape(3, 3)
            hp = d.xpos[self.wrist[s]] + R @ HAND_OFF
            ua, fa = d.xpos[self.el[s]] - d.xpos[self.sh[s]], d.xpos[self.wrist[s]] - d.xpos[self.el[s]]
            bend = np.degrees(np.arccos(np.clip(ua @ fa / (np.linalg.norm(ua) * np.linalg.norm(fa)), -1, 1)))
            out[s] = dict(hp=(hp - pp) @ Rp, bend=float(bend))
        return out

    def collides(self) -> list:
        """Contacts (penetrating) between an arm / hand body and any other robot body not of the same arm."""
        m, d = self.m, self.d
        mujoco.mj_collision(m, d)
        bad = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            s1, s2 = self.side[b1], self.side[b2]
            if (s1 or s2) and s1 != s2:
                bad.append((self.names[b1], self.names[b2], float(c.dist)))
        return bad


def arms(q29: np.ndarray, left7: np.ndarray) -> np.ndarray:
    q = q29.copy()
    q[15:22] = left7
    q[22:29] = left7 * MIRROR
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--coarse", type=float, default=0.12)
    ap.add_argument("--objective", choices=["out", "radial"], default="radial")
    ap.add_argument("--zmax-cm", type=float, default=0.0, help="hand points no higher than this above the pelvis; < -900: none")
    a = ap.parse_args()
    global OBJECTIVE, ZMAX
    OBJECTIVE = a.objective
    ZMAX = None if a.zmax_cm < -900 else a.zmax_cm / 100
    B = Body()
    lo, hi = B.lim[15:22, 0], B.lim[15:22, 1]
    lo_m, hi_m = lo + MARGIN * (hi - lo), hi - MARGIN * (hi - lo)
    # the right arm (left value x MIRROR) must sit inside its own margins too: its range in left terms, intersected
    rl, rh = B.lim[22:29, 0], B.lim[22:29, 1]
    rl_m, rh_m = rl + MARGIN * (rh - rl), rh - MARGIN * (rh - rl)
    lo_m = np.maximum(lo_m, np.where(MIRROR > 0, rl_m, -rh_m))
    hi_m = np.minimum(hi_m, np.where(MIRROR > 0, rh_m, -rl_m))
    z = np.load(WALK)
    C = {str(c): i for i, c in enumerate(z["cols"])}
    push = z["rows"][:, [C[f"push{7 + j}"] for j in range(29)]].astype(float)
    walk_poses = push[np.isfinite(push).all(1)][::5]
    walk_poses = walk_poses[np.abs(walk_poses[:, :15] - DEFAULT_Q[:15]).max(1) > 1e-6]  # the frames off the default stance

    def evaluate(l7: np.ndarray, walk: bool) -> dict | None:
        B.set(arms(DEFAULT_Q, l7))
        h = B.hands()
        fwd = max(h["left"]["hp"][0], h["right"]["hp"][0])
        if fwd > B.front:
            return None
        if ZMAX is not None and max(h["left"]["hp"][2], h["right"]["hp"][2]) > ZMAX:  # the chosen rule: z <= 0
            return None
        if B.collides():
            return None
        if walk:
            for wq in walk_poses:
                B.set(arms(wq, l7))
                if B.collides():
                    return None
            B.set(arms(DEFAULT_Q, l7))
        if OBJECTIVE == "radial":  # the chosen rule: horizontal distance from the pelvis's vertical axis
            out = max(float(np.linalg.norm(h["left"]["hp"][:2])), float(np.linalg.norm(h["right"]["hp"][:2])))
        else:
            out = max(abs(h["left"]["hp"][1]), abs(h["right"]["hp"][1]))
        return dict(q=l7.tolist(), out_cm=100 * out, fwd_cm=100 * fwd, bend=max(h["left"]["bend"], h["right"]["bend"]),
                    hp_left=h["left"]["hp"].tolist(), hp_right=h["right"]["hp"].tolist())  # fmt: skip

    def grid(lo_, hi_, step):
        return [np.arange(lo_[i], hi_[i] + 1e-9, step) if hi_[i] > lo_[i] else np.array([lo_[i]]) for i in range(len(lo_))]

    # coarse: shoulder pitch / roll / yaw, elbow; wrists 0 (inside the margins)
    g = grid(lo_m[:4], hi_m[:4], a.coarse)
    cands = []
    for sp, sr, sy, el in itertools.product(*g):
        r = evaluate(np.array([sp, sr, sy, el, 0.0, 0.0, 0.0]), walk=False)
        if r is not None:
            cands.append(r)
    cands.sort(key=lambda r: r["out_cm"])
    print(f"pelvis front edge {100 * B.front:.1f} cm; coarse grid {np.prod([len(v) for v in g])}, feasible at the stance {len(cands)}")
    better = lambda r, b: r["out_cm"] < b["out_cm"] - 1e-9 or (abs(r["out_cm"] - b["out_cm"]) <= 1e-9 and r["bend"] < b["bend"])  # noqa: E731
    starts, fine = [], []
    for r in cands:  # the best START_N coarse candidates that also pass the walk poses
        if evaluate(np.array(r["q"]), walk=True) is not None:
            starts.append(r)
        if len(starts) >= START_N:
            break
    # fine: from each, a pattern search over all seven joints (steps coarse/2 .. coarse/16), every try inside the margins
    # and feasible at the stance and the walk poses; a move is kept when it lowers out_cm (equal: lowers the bend)
    for r in starts:
        b = r
        for step in (a.coarse / 2, a.coarse / 4, a.coarse / 8, a.coarse / 16):
            moved = True
            while moved:
                moved = False
                for i in range(7):
                    for sg in (-1.0, 1.0):
                        q = np.array(b["q"])
                        q[i] = float(np.clip(q[i] + sg * step, lo_m[i], hi_m[i]))
                        e = evaluate(q, walk=True)
                        if e is not None and better(e, b):
                            b, moved = e, True
        fine.append(b)
    fine.sort(key=lambda r: r["out_cm"])
    tied = [r for r in fine if r["out_cm"] <= fine[0]["out_cm"] + TIE_CM] if fine else []
    pick = min(tied, key=lambda r: r["bend"]) if tied else None
    B.set(DEFAULT_Q)
    base_coll = B.collides()
    B.set(arms(DEFAULT_Q, SIMPLE_TUCK))
    hs = B.hands()
    B.set(DEFAULT_Q)
    hd = B.hands()
    res = dict(rule=__doc__, pelvis_front_cm=100 * B.front, margins=[lo_m.tolist(), hi_m.tolist()], n_coarse=len(cands), n_fine=len(fine), n_tied=len(tied),
               pick=pick, mirror=MIRROR.tolist(), right_q=(np.array(pick["q"]) * MIRROR).tolist() if pick else None,
               lml_default=dict(hp_left=hd["left"]["hp"].tolist(), hp_right=hd["right"]["hp"].tolist(), bend=hd["left"]["bend"]),
               simple_tuck=dict(hp_left=hs["left"]["hp"].tolist(), hp_right=hs["right"]["hp"].tolist(), bend=hs["left"]["bend"]),
               n_walk_poses=len(walk_poses), default_collisions=base_coll, starts=[r["q"] for r in starts], fine=fine)  # fmt: skip
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rule"}, indent=1))


if __name__ == "__main__":
    main()
