"""The scene as the robot's own senses give it on LocoManiLab (camera track).

What the robot has: the instruction, the head camera's 320x240 colour and depth image, its joint readings, the IMU's
tilt, leg odometry and its own model. From these:

  the pose of the camera   the robot's own model at the joint readings, on the pelvis of leg odometry (`LmlBelief.scene`)
  what the instruction     `parse`: navigation ("walk to the bed"), pick or lift ("pick up the soap bottle") or place
  names                    ("put the apple into the box"): the object, the receptacle or fixture, the support
  the target               SAM 3 on the colour image with the object's name; turned to look for it when it is not in
                           view (`LmlCameraView.look`); its points from the depth image
  its grasps               GraspGen-X (Dex3) candidates on those points, read from the grasp files
  the edge                 the nearest thing ahead between knee and chest height, from the depth points, not the target

Needs the SAM 3 service (the address comes with the episode: `env.sam3`). The robot's own camera geometry and its belief of where it
is (LmlHeadKin, LmlBelief, cam_points of the old file) are fixed modules of locomani now: perception/head.py and bench/locomanilab/body.py.
"""

import re

import numpy as np

from locomani.perception import head as SEE

from . import skill as S

OWN = 0.2  # m ahead of the pelvis: nearer points are the robot's own body
HELD = 0.6  # m ahead of the pelvis: with an object carried in front, nearer points are the hand and the object
PELVIS_UP = 0.793  # m, the pelvis above the floor in the standing pose (the robot's own model)
BAND = (0.15, 1.10)  # m above the floor: what the legs, the pelvis and the hanging hands can run into
SEARCH = (0.9, -0.9, 1.8, -1.8, np.pi)  # rad from the heading at the start: where the robot turns to look for a thing
MIN_PX = 60
FLOOR = 0.25  # SAM 3's least score for an instance


# ---- the instruction


def parse(prompt: str) -> dict:
    """kind (navigation | pick | place | articulate), the object's name, the receptacle's or fixture's name, the surface it is on.
    articulate (a door or a drawer): action (open | close | pnp), target, handle, dest, approach (walk to dest first), and for
    the drawer tasks `obj` (the apple of the pnp, else None) and `steps` (the instruction's order of sub-goals)."""
    s = re.sub(r"^bend down and ", "", prompt.lower().strip().rstrip("."))
    door = re.match(r"(walk to the fridge from far away and )?(open|close) the fridge door(?: in front of you)?$", s)
    if door:  # the three fridge-door tasks
        return dict(kind="articulate", action=door.group(2), target="fridge door", handle="fridge door handle",
                    dest="fridge", approach=bool(door.group(1)), obj=None)  # fmt: skip
    low = re.match(
        r"walk to the (lower-(?:left|right) drawer) of the ([a-z ]+), then crouch down and (open|close) it$", s
    )
    if low:  # the two walk-and-crouch drawer tasks: walk to the dresser first, then work the drawer
        return dict(kind="articulate", action=low.group(3), target=low.group(1), handle="drawer handle", dest=low.group(2),
                    approach=True, obj=None, steps=("approach", low.group(3)))  # fmt: skip
    pnp = re.match(r"pick up the ([a-z ]+) from the top of the ([a-z ]+), open its (lower-(?:left|right) drawer), put the \1 inside, and close the drawer$", s)  # fmt: skip
    if pnp:  # the drawer pick-and-place: the instruction's own order
        return dict(kind="articulate", action="pnp", target=pnp.group(3), handle="drawer handle", dest=pnp.group(2),
                    approach=False, obj=pnp.group(1), steps=("pick", "open", "place", "close"))  # fmt: skip
    m = re.match(r"(?:walk|go)(?: over)? to the ([a-z ]+?)(?: and .*)?$", s)
    if m:
        return dict(kind="navigation", obj=None, dest=m.group(1), on=None)
    obj = re.search(r"(?:pick up|put|place|move|lift) the ([a-z ]+?)(?= from| beside| and| in| into| on| to|$)", s)
    dest = re.search(r"(?:put|place|move)(?: it| the [a-z ]+?)? (?:in|into|on|onto|to) (?:either )?(?:the )?([a-z -]+?)(?= on the| in the|$)", s)  # fmt: skip
    on = re.search(r"(?:on|onto) the [a-z ]+? on the ([a-z ]+)$", s)
    return dict(kind="place" if dest else "pick", obj=obj.group(1) if obj else None, dest=dest.group(1) if dest else None,
                on=on.group(1) if on else None)  # fmt: skip


def phrases(name: str) -> list[str]:
    """What SAM 3 is asked for, most specific first: the name, then its last word ("scrub brush" -> "brush")."""
    out = [name]
    if " " in name:
        out.append(name.split()[-1])
    return out


# ---- the robot's own senses (the head camera's geometry and the Belief are fixed modules; what is read from a picture is here)


def floor_of(pts, ok, pb) -> float:
    """The floor's height: the robot's own model puts the standing pelvis 0.793 m above it; the depth points' lowest
    level (a 1 cm histogram's mode among points below the knee) is the check, returned when it is near."""
    z = pts[ok & (pts[..., 2] < pb[2] - 0.55)][:, 2]
    if len(z) < 200:
        return float(pb[2] - PELVIS_UP)
    h, e = np.histogram(z, bins=np.arange(z.min(), z.min() + 0.3, 0.01))
    seen = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    return float(seen) if abs(seen - (pb[2] - PELVIS_UP)) < 0.08 else float(pb[2] - PELVIS_UP)


def obstacle_edge(pts, ok, floor: float, ahead, lateral: float, after: float, skip=None, half: float = 0.30):
    """The coordinate along `ahead` of the nearest thing between knee and chest height in a band of `half` to either side
    of `lateral` beyond `after`, over the depth points (not the pixels `skip`); None when nothing is seen."""
    u, w = pts[..., :2] @ ahead[:2], pts[..., :2] @ np.array([-ahead[1], ahead[0]])
    sel = (
        ok
        & (pts[..., 2] > floor + BAND[0])
        & (pts[..., 2] < floor + BAND[1])
        & (np.abs(w - lateral) < half)
        & (u > after)
    )
    if skip is not None:
        sel &= ~skip
    return float(np.percentile(u[sel], 0.5)) if sel.sum() >= 30 else None


def complete(cloud, cam, how: str = "none"):
    """The seen points as they are (a depth image sees the near surface only; nothing is added)."""
    return cloud


def cloud_name(episode_id: str) -> str:
    """The name of an episode's point and grasp files; a name already in that form is returned as it is."""
    if "/" not in episode_id:
        return episode_id
    task, kind, n = episode_id.split("/")
    return f"{task}_L{kind}_e{n}"


def have_grasps(folder, name: str) -> bool:
    """Whether the grasp candidates for the named points have been worked out (in the look cache `folder`)."""
    return (folder / f"grasp_{name}.npz").exists()


def load_frame(folder, name: str) -> np.ndarray:
    """The place the named points' frame is anchored at: the one the grasps were worked out in."""
    return np.load(folder / f"{name}.npz")["p_o"]


def load_grasps(folder, name: str) -> dict:
    """The grasp candidates of the named points: palm poses, scores and the finger closings."""
    z = np.load(folder / f"grasp_{name}.npz")
    return {k: z[k] for k in z.files}


class Sam:
    """SAM 3's best instance over a list of phrases (`sam3`: the service's (host, port), from the episode)."""

    @staticmethod
    def best(sam3, rgb, names, floor: float = FLOOR):
        """The best-scored instance over all the phrases at once: (mask, score, phrase)."""
        m, score, p = SEE.find(sam3, rgb, list(names), floor=floor)
        return m, score, p

    @staticmethod
    def find(sam3, rgb, names):
        for p in names:
            m, score, _ = SEE.find(sam3, rgb, [p], floor=FLOOR)
            if m is not None:
                return m, score, p
        return None, 0.0, None


# ---- the pick's view


class LmlCameraView:
    """The target, its grasp and the edge ahead, from the head camera."""

    search = True

    def __init__(self, env, source: str, prompt: str):
        self.env, self.name, self.prompt, self.s = env, cloud_name(source), prompt, None
        self.cloud = env.cloud_dir  # the look cache folder of this package and these parameters
        self.want = parse(prompt)
        self.yaw0, self.dest, self.chosen = None, None, None

    def heading(self) -> float:
        return self.yaw0 if self.yaw0 is not None else float(self.env.bel.pelvis()[1])

    def _see(self, names, after: float):
        bel = self.env.bel
        rgb, pts, ok = bel.scene()
        pb, yaw = bel.pelvis()
        mask, score, phrase = Sam.find(self.env.sam3, rgb, names)
        found = mask is not None and (mask & ok).sum() >= MIN_PX
        return dict(rgb=rgb, pts=pts, ok=ok, pb=pb, yaw=yaw, mask=mask if found else None, score=score, phrase=phrase,
                    after=after, floor=floor_of(pts, ok, pb), cam_b=bel.cam_b)  # fmt: skip

    def look(self) -> dict:
        """The start's look, then a turn to either side and behind until the target is seen: the target's points, the
        level it stands on and the floor, in the robot's frame."""
        if self.s is not None:
            return self.s
        bel, row = self.env.bel, self.env.row
        names = phrases(self.want["obj"]) if self.want["obj"] else []
        pb, yaw0 = bel.pelvis()
        row["prompt_parse"] = self.want
        for k, turn in enumerate((0.0,) + SEARCH):
            if k:
                bel.walk(bel.pelvis()[0][:2], yaw0 + turn, face="fixed")
            s = self._see(names, bel.pelvis()[0][:2] @ np.array([np.cos(yaw0 + turn), np.sin(yaw0 + turn)]) + OWN)
            if s["mask"] is not None:
                row["turns"], self.yaw0 = k, float(yaw0 + turn)
                keep = s["mask"] & s["ok"]
                cloud = s["pts"][keep]
                s["cloud"] = cloud[np.linalg.norm(cloud - np.median(cloud, axis=0), axis=1) < 0.3]
                s["top"] = SEE.support_of(s["pts"], s["ok"], s["mask"])
                row["target_px"], row["target_score"], row["target_phrase"] = int(keep.sum()), round(s["score"], 2), s["phrase"]  # fmt: skip
                if self.want["kind"] == "place":  # the receptacle, if it is in the image the target was seen in
                    from . import flows as SC

                    views = [(s["rgb"], s["pts"], s["ok"], s["pb"], s["yaw"])]
                    self.dest = SC.dest_of(self.want, views, row, self.env.sam3)
                    row["dest_in_view"] = self.dest is not None
                self.s = s
                return s
        self.s = dict(mask=None)
        return self.s

    def target(self):
        s = self.look()
        if s["mask"] is None or not have_grasps(self.cloud, self.name):
            return None
        T_wo = np.eye(4)
        T_wo[:3, 3] = load_frame(self.cloud, self.name)  # the frame the grasps were worked out in
        lo, hi = np.percentile(s["cloud"], [1, 99], axis=0)  # the head sees the front and the top: their span
        return dict(T_wo=T_wo, lo=lo, hi=hi, top=float(np.percentile(s["cloud"][:, 2], 2)))

    def grasp(self, tgt, yaw: float):
        g = S.choose_grasp(load_grasps(self.cloud, self.name), tgt["T_wo"], yaw, tgt["top"], self.env.K)
        self.chosen = (
            None if g is None else dict(T_wo=tgt["T_wo"], T_og=g[0], f_close=g[2], yaw=yaw)
        )  # the place needs it
        return g

    def edge(self, ahead, lateral: float, table: str = "table", half: float = 0.30):
        s = self.s
        return obstacle_edge(s["pts"], s["ok"], s["floor"], ahead, lateral, s["pb"][:2] @ ahead + OWN, skip=s["mask"], half=half)  # fmt: skip
