"""The scene as the robot's own senses give it, for track "official" (escalation plan, stage A-S, 2026-11-17).

simple_eval_20261112/blind.py's perception behind skill_simple's view interface: the head camera's colour and depth
images, SAM 3 on the colour image for the target the instruction names and for the basket, GraspGen-X's Dex3 grasps on
the target's seen points (worked out between the look pass and the run: bed_simple.py look, then grasps_graspgen.py),
the table's level and edge from the points around the target, leg odometry and joint readings for pelvis and palm
(place.Belief with odo). Nothing here reads the simulator's state of the scene.

Needs the SAM 3 service on port 18832 (scripts/rsi/sam3_server.py).
"""

from pathlib import Path

import numpy as np

from locomani.perception import camera_tools as B
from locomani.perception import head as SEE

from . import skill as S


def cloud_name(episode_id: str) -> str:
    task, kind, n = episode_id.split("/")
    return f"{task}_L{kind.lstrip('L')}_e{n.lstrip('e')}"


def instances(sam3, rgb, phrases, floor: float = 0.3):
    """Every instance SAM 3 gives for the phrases, best score first: [(score, mask, phrase)]."""
    found = SEE.sam_results(sam3, rgb, phrases)
    out = []
    for phrase in phrases:
        for score, packed in found.get(phrase, []):
            if score > floor:
                bits = np.unpackbits(packed["bits"], count=int(np.prod(packed["shape"])))
                out.append((float(score), bits.astype(bool).reshape(packed["shape"]), phrase))
    return sorted(out, key=lambda r: -r[0])


class CameraView:
    search = True  # a basket on another table is looked for by turning

    def __init__(self, env, episode_id: str, instruction: str):
        self.env, self.name, self.instruction, self.s, self.far = env, cloud_name(episode_id), instruction, None, None
        self.cloud = Path(env.cloud_dir)  # the look cache folder of this package and these parameters

    def heading(self) -> float:
        return float(self.env.bel.pelvis()[1])

    def look(self) -> dict:
        """The start's look (blind.first_look): the target's points and the table under it, in the robot's frame."""
        if self.s is None:
            bel = self.env.bel
            rgb, pts, ok = bel.scene()
            pb, yaw = bel.pelvis()
            ahead = np.array([np.cos(yaw), np.sin(yaw)])
            mask, score = B.target_mask(
                self.env.sam3, self.instruction, rgb, pts, ok, ahead, pb, post=self.env.mask_post
            )
            s = dict(rgb=rgb, pts=pts, ok=ok, pb=pb, yaw=yaw, mask=mask, score=score)
            if mask is None or (mask & ok).sum() < 100:
                s["mask"] = None
            else:
                cloud = pts[mask & ok]
                s["cloud"] = cloud[np.linalg.norm(cloud - np.median(cloud, axis=0), axis=1) < 0.3]
                s["top"] = SEE.support_of(pts, ok, mask)
            self.s = s
        return self.s

    def target(self):
        s = self.look()
        if s["mask"] is None or not (self.cloud / f"grasp_{self.name}.npz").exists():
            return None
        T_wo = np.eye(4)
        T_wo[:3, 3] = np.load(self.cloud / f"{self.name}.npz")["p_o"]  # the frame the grasps were worked out in
        lo, hi = np.percentile(s["cloud"], [1, 99], axis=0)  # the head sees the front and the top: their span
        return dict(T_wo=T_wo, lo=lo, hi=hi, top=s["top"])

    def grasp(self, tgt, yaw: float):
        pts = np.load(self.cloud / f"{self.name}.npz")["pts"]
        support = float(np.percentile(pts[:, 2], 2))  # the cloud's foot
        high = float(np.percentile(pts[:, 2], 99))  # the cloud's top
        return S.choose_grasp(
            np.load(self.cloud / f"grasp_{self.name}.npz"), tgt["T_wo"], yaw, support, self.env.K, high=high
        )

    def edge(self, ahead, lateral: float, table: str = "table"):
        if table == "table":
            s = self.s
            return SEE.near_edge(s["pts"], s["ok"], s["top"], ahead, lateral, s["pb"][:2] @ ahead + B.OWN)
        f, pb = self.far, self.env.bel.pelvis()[0]
        return SEE.near_edge(f["pts"], f["ok"], float(np.percentile(f["seen"][:, 2], 2)), ahead, lateral, pb[:2] @ ahead + B.HELD)  # fmt: skip

    def _basket(self, rgb, pts, ok, yaw2: float):
        m, score, _ = SEE.find(self.env.sam3, rgb, B.BOWL)
        self.env.row["basket_score"] = score
        if m is None or (m & ok).sum() < 200:
            return None
        self.far = dict(pts=pts, ok=ok, seen=pts[m & ok])
        return B.basket_from(self.far["seen"], np.array([np.cos(yaw2), np.sin(yaw2)]))

    def basket(self, yaw2: float):
        """A basket on the same table: it was in the start's look."""
        s = self.s
        return self._basket(s["rgb"], s["pts"], s["ok"], yaw2)

    def find_basket(self, yaw: float):
        """The other table is out of view: turn (behind first, then to either side) and look. -> (heading, basket)."""
        bel = self.env.bel
        for k, turn in enumerate(B.SEARCH):
            yaw2 = yaw + turn
            bel.walk(bel.pelvis()[0][:2], yaw2, face="fixed")
            found = self._basket(*bel.scene(), yaw2)
            if found is not None:
                self.env.row["turns"] = k + 1
                return yaw2, found
        return None

    def basket_near(self, yaw2: float):
        """A second look from near, the object held aside: (centre, rim) or None."""
        rgb, pts, ok = self.env.bel.scene()
        m, _, _ = SEE.find(self.env.sam3, rgb, B.BOWL)
        self.env.row["basket_px"] = int((m & ok).sum()) if m is not None else 0
        if self.env.row["basket_px"] < 200:
            return None
        c, rim, _ = B.basket_from(pts[m & ok], np.array([np.cos(yaw2), np.sin(yaw2)]))
        return c, rim

    # ---- the articulated tasks (simple_art): an object the instruction names, a part of it to take hold of ----

    def art(self, after: float = 0.2) -> dict:
        """One look at the object and its parts (head camera, SAM 3, depth; the robot's frame). `after`: points nearer than
        this along the heading are the robot's own body. -> dict(obj=cloud or None, parts=[dict(p, lo, hi, n, score)], ...);
        the parts best first by K["art_pick"]."""
        K, bel = self.env.K, self.env.bel
        rgb, pts, ok = bel.scene()
        pb, yaw = bel.pelvis()
        ahead = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-ahead[1], ahead[0]])
        near = pts[..., :2] @ ahead > pb[:2] @ ahead + after
        out = dict(rgb=rgb, pts=pts, ok=ok, pb=pb, yaw=yaw, obj=None, parts=[])
        objs = [
            r
            for r in instances(self.env.sam3, rgb, K["art_obj"].split("|"), K["art_floor"])
            if (r[1] & ok & near).sum() >= 100
        ]
        if not objs:
            return out
        m_obj = objs[0][1]
        out.update(obj=pts[m_obj & ok & near], obj_score=objs[0][0], obj_mask=m_obj)
        ys, xs = np.nonzero(m_obj)
        box = (xs.min(), xs.max(), ys.min(), ys.max())
        pad = 0.1 * max(box[1] - box[0], box[3] - box[2])
        for score, m, phrase in instances(self.env.sam3, rgb, K["art_part"].split("|"), K["art_floor"]):
            v = m & ok & near
            yy, xx = np.nonzero(m)
            inside = (xx >= box[0] - pad) & (xx <= box[1] + pad) & (yy >= box[2] - pad) & (yy <= box[3] + pad)
            if v.sum() < K["art_min_px"] or inside.mean() < 0.8:
                continue
            c = pts[v]
            c = c[np.linalg.norm(c - np.median(c, axis=0), axis=1) < 0.4]
            lo, hi = np.percentile(c, [5, 95], axis=0)
            out["parts"].append(
                dict(c=c, p=np.median(c, axis=0), lo=lo, hi=hi, n=int(v.sum()), score=score, phrase=phrase)
            )
        pick = {"score": lambda r: -r["score"], "high": lambda r: -r["p"][2], "low": lambda r: r["p"][2],
                "left": lambda r: -(r["p"][:2] @ left), "right": lambda r: r["p"][:2] @ left,
                "near": lambda r: (r["p"][:2] - pb[:2]) @ ahead, "big": lambda r: -r["n"]}[K["art_pick"]]  # fmt: skip
        out["parts"].sort(key=pick)
        return out
