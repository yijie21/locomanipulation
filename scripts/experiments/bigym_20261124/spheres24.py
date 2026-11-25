"""bigym_selfmodel_20261124: the 24 spheres of the right arm and gripper the self model describes, picked from the cuRobo
spheres fitted to BiGym's own collision geoms (curobo/make_config.py: 79 on the right side), to match SIMPLE's 24
(SIMPLE_ROOT/data/robots/g1/curobo/spheres/g1_29dof.yml, selfmodel_via_20261120/arm_fk.py) one for one.

The pick (deterministic, from the two sphere files only):
  arm, 9    SIMPLE's per-link counts (shoulder pitch 1, roll 3, yaw 2, elbow 1, wrist pitch 2; none on the wrist roll and
            yaw links, as in SIMPLE). Both robots are the G1 29-dof with the same link frames (BiGym's arm links match
            SIMPLE's within 3.9 mm), so for each SIMPLE sphere the BiGym sphere of the same link whose centre is nearest
            in the link's frame, each used once (greedy in SIMPLE's order).
  palm, 3   Dex1's base link: its largest sphere (the core, at 4 cm) and the two front spheres (x >= 0.05, where the
            fingers start) at the two ends of its z axis: a line across the palm, as SIMPLE's three palm spheres (along
            the Dex3 palm's z).
  fingers, 12  6 a finger (Dex1 has two): farthest-point sampling over the finger's spheres from its tip (largest x),
            then ordered base to tip.
Labels: via_point.py / via_model.py read the arm by SIMPLE's link names (KEY_LINKS, "hand_palm_link",
"hand_index_1_link"), so each picked sphere carries the label of the SIMPLE sphere it stands for: the arm's by link, the
palm's "hand_palm_link", the fingers' SIMPLE's 12 finger labels in SIMPLE's order (thumb_0 .. index_1): finger 1 takes
the first six, finger 2 the last six, base to tip, so "hand_index_1_link" (the fingertip via_point uses for the hand's
attitude) is finger 2's two distal spheres. The BiGym link of each sphere is kept beside its label.

    S = Spheres24(<bigym spheres yml>, <simple spheres yml>)
    S.links (labels), S.bodies (BiGym links), S.local (24, 3), S.radius (24,)
    S.world(model, data, ns)   # (24, 3) centres in the world for MuJoCo's body poses
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

ARM = ("shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link", "elbow_link", "wrist_pitch_link")
SIMPLE_HAND = ("hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link", "hand_middle_0_link", "hand_middle_1_link",
               "hand_index_0_link", "hand_index_1_link")  # fmt: skip
PALM_FRONT_X = 0.05
RIGHT = tuple(f"right_{n}" for n in ("shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link", "elbow_link",
              "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link", "dex1_base_link", "dex1_finger_link_1",
              "dex1_finger_link_2"))  # fmt: skip
REPO = Path(__file__).resolve().parents[3]
BIGYM_YML = REPO / "outputs/bigym_20261124/curobo/spheres/g1_29dof_dex1.yml"
SIMPLE_YML = Path("/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE/data/robots/g1/curobo/spheres/g1_29dof.yml")


def _load(path) -> dict:
    s = yaml.safe_load(open(path))
    s = s.get("collision_spheres", s)
    return {k: [(np.asarray(x["center"], float), float(x["radius"])) for x in v] for k, v in s.items()}


def _fps(C: np.ndarray, first: int, n: int) -> list[int]:
    pick = [first]
    while len(pick) < min(n, len(C)):
        d = np.min(np.linalg.norm(C[:, None] - C[pick][None], axis=2), axis=1)
        d[pick] = -1.0
        pick.append(int(np.argmax(d)))
    return pick


class Spheres24:
    def __init__(self, bigym_yml=BIGYM_YML, simple_yml=SIMPLE_YML):
        B, S = _load(bigym_yml), _load(simple_yml)
        links, bodies, local, radius = [], [], [], []

        def add(label, body, c, r):
            links.append(label)
            bodies.append(body)
            local.append(c)
            radius.append(r)

        for n in ARM:  # nearest in the link's frame, each BiGym sphere once
            cand, used = B[f"right_{n}"], set()
            for c_s, _ in S[f"right_{n}"]:
                d = [np.linalg.norm(c - c_s) if i not in used else np.inf for i, (c, _) in enumerate(cand)]
                i = int(np.argmin(d))
                used.add(i)
                add(n, f"right_{n}", *cand[i])
        base = B["right_dex1_base_link"]
        C, R = np.array([c for c, _ in base]), np.array([r for _, r in base])
        front = np.where(C[:, 0] >= PALM_FRONT_X)[0]
        for i in (int(np.argmax(R)), int(front[np.argmax(C[front, 2])]), int(front[np.argmin(C[front, 2])])):
            add("hand_palm_link", "right_dex1_base_link", C[i], R[i])
        labels = [n for n in SIMPLE_HAND for _ in S[f"right_{n}"]]
        assert len(labels) == 12, labels
        for f, lab in (("right_dex1_finger_link_1", labels[:6]), ("right_dex1_finger_link_2", labels[6:])):
            C = np.array([c for c, _ in B[f]])
            R = np.array([r for _, r in B[f]])
            pick = sorted(_fps(C, int(np.argmax(C[:, 0])), 6), key=lambda i: C[i, 0])  # base to tip
            for lb, i in zip(lab, pick):
                add(lb, f, C[i], R[i])
        assert len(links) == 24
        self.links, self.bodies = links, bodies
        self.local, self.radius = np.array(local), np.array(radius, np.float32)
        # every fitted sphere of the right arm and gripper (what the planner checks): the collection's box clearance
        full = [(b, c, r) for b, v in B.items() if b.startswith("right_") and b in RIGHT for c, r in v]
        self.full_bodies = [b for b, _, _ in full]
        self.full_local, self.full_radius = np.array([c for _, c, _ in full]), np.array([r for _, _, r in full])

    def world(self, m, d, ns: str = "") -> np.ndarray:
        """The 24 centres (world) at the data's body poses (mj_kinematics done)."""
        ids = self._ids(m, ns)
        Rm = d.xmat[ids].reshape(-1, 3, 3)
        return d.xpos[ids] + np.einsum("kij,kj->ki", Rm, self.local)

    def world_full(self, m, d, ns: str = "") -> np.ndarray:
        """Every right-arm and gripper sphere's centre (world), radii in full_radius."""
        if getattr(self, "_fkey", None) != (id(m), ns):
            self._fkey, self._fid = (id(m), ns), np.array([m.body(ns + b).id for b in self.full_bodies])
        Rm = d.xmat[self._fid].reshape(-1, 3, 3)
        return d.xpos[self._fid] + np.einsum("kij,kj->ki", Rm, self.full_local)

    def _ids(self, m, ns: str) -> np.ndarray:
        key = (id(m), ns)
        if getattr(self, "_key", None) != key:
            self._key, self._idv = key, np.array([m.body(ns + b).id for b in self.bodies])
        return self._idv

    def table(self) -> str:
        return "\n".join(f"{i:2d} {lb:20s} {b:26s} {np.round(c, 4).tolist()} r {r:.3f}"
                         for i, (lb, b, c, r) in enumerate(zip(self.links, self.bodies, self.local, self.radius)))  # fmt: skip


if __name__ == "__main__":
    S = Spheres24()
    print(S.table())
    print(f"all right-arm spheres (the box clearance): {len(S.full_bodies)}")
