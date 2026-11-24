"""What a camera view needs to read a picture of SIMPLE's scenes (blind.py's pieces and the bed's table_objects, TODO 49.2).

The instruction's object name, the phrases for the basket, where the robot turns to look for it, the object's mask from
SAM 3 (or, for an instruction that names no object, from the depth points alone) and the basket's centre and rim from its
seen points. The old bed replaced `blind.target_mask` by its own version at run time; here that version is the function,
and what the run adds (the robot's own pixels taken out of the mask: `post`) is an argument.
"""

import re

import numpy as np

from locomani.perception import head

BOWL = ("gray bowl", "basket", "bowl", "container")
SEARCH = (np.pi, 0.5 * np.pi, -0.5 * np.pi)  # rad from the start heading: where the robot turns to look for the basket
OWN = 0.2  # m ahead of the pelvis: nearer points are the robot's own body
HELD = 0.6  # m ahead of the pelvis: with an object carried in front, nearer points are the hand and the object


def named(instruction: str):
    """The object an instruction names, or None ("the object")."""
    s = instruction.lower().strip()
    for pat in (r"pick up the ([a-z ]+?)(?: from|,|\.|$)", r"hand over (?:the )?([a-z ]+?) from"):
        m = re.search(pat, s)
        if m:
            return m.group(1).strip()
    return None


def basket_from(seen: np.ndarray, ahead):
    """The basket's centre (xy), rim height and near side (along `ahead`) from its seen points."""
    lo, hi = np.percentile(seen, [1, 99], axis=0)
    return 0.5 * (lo + hi)[:2], float(hi[2]), float(np.percentile(seen[:, :2] @ ahead, 1))


def table_objects(pts, ok, ahead, pb, cap: float = 0.12, min_px: int = 500, half: float = 0.12):
    """For an instruction that names no object: the thing standing on the table nearest the robot, from the depth points alone.
    The table's level is the mode of the heights ahead (5 mm bins); what stands 1.5-12 cm above it (the robot's own arms rise
    higher, that is the cap) is cut into image components; of those with at least `min_px` pixels and within `half` m of the robot's
    midline, the nearest is the object ("the object on the tabletop": the practice episodes' target is nearest the robot
    and in front of it, 9 of 10 episodes). -> mask or None."""
    from scipy import ndimage

    left = np.array([-ahead[1], ahead[0]])
    a, lat = (pts[..., :2] - pb[:2]) @ ahead, (pts[..., :2] - pb[:2]) @ left
    reg = ok & (a > 0.2) & (a < 1.6) & (abs(lat) < 0.9)
    z = pts[reg][:, 2]
    if len(z) < 500:
        return None
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.005, 0.005))
    zt = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    zt = float(np.median(z[abs(z - zt) < 0.01]))
    above = ndimage.binary_opening(reg & (pts[..., 2] > zt + 0.015) & (pts[..., 2] < zt + cap), iterations=2)
    lab, n = ndimage.label(above)
    best = None
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() >= min_px and abs(np.median(lat[m])) < half and (best is None or np.median(a[m]) < best[0]):
            best = (float(np.median(a[m])), m)
    return None if best is None else best[1]


def target_mask(sam3, instruction: str, rgb, pts, ok, ahead, pb, post=None):
    """The target's mask and its score. An instruction that names an object: SAM 3's best instance of that name. One that
    names none ("grasp the object on the tabletop"): `table_objects` (the largest thing on the surface of the old blind.py is
    the table itself or the robot's arms on the glossy table's stereo depth). `post(mask) -> mask`, when given, is applied to
    the result (see_self: the robot's own pixels out)."""
    if named(instruction) is None:
        m, score = table_objects(pts, ok, ahead, pb), 1.0
    else:
        m, score, _ = head.find(sam3, rgb, [named(instruction)])
    if post is not None and m is not None:
        m = post(m)
    return m, score
