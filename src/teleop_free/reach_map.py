"""The graspable-region map: where can the hands GRASP, in the pelvis frame.

Navigation needs a stand-position criterion, and the scattered scalar
thresholds it uses today (GRIP_X band, REACH_XY_TOL, aim()'s shoulder
distances) are hand-tuned shadows of one object: the set R of pelvis-frame
points where some arm can put the palm WITH a grasp-quality orientation.
This module computes R once, offline, and exposes a lookup.

Method: forward Monte-Carlo. Sample arm joint configurations uniformly,
FK them on the scene model (ArmIK.fk — the same model the sim runs), keep
samples whose palm orientation could actually take an upright object:

    - approach (the palm-frame grasp-offset direction, world) dives at most
      MAX_DESCEND below horizontal — steeper is a top-down strike;
    - the jaw axis (palm +y) leans at most JAW_TILT_MAX out of the
      horizontal plane — past that it closes along the can, not across it.

Torso: the pelvis frame makes legs irrelevant here — squatting moves R
DOWN IN THE WORLD, not in the pelvis frame — so world-frame coverage is the
pelvis-frame map extruded over the delivered pelvis-height range
[PELVIS_MIN, PELVIS_MAX]. The waist joints DO change R in the pelvis frame,
so waist pitch is enumerated over the range SONIC actually delivers.

Everything here is SONIC-legal by construction: the arms are exact (the
sim-side pin runs them in position mode), and the torso parameters are the
measured delivered ranges, not commanded ones.
"""

from __future__ import annotations

import os

import numpy as np

from arm_ik import ArmIK
from g1_joints import ARM_MJ
from primitives_const import GRASP_CENTRE_OFFSET_CONST

from primitives_const import HAND as _HAND
MAP_PATH = ("/workspace/code/locomanipulation/outputs/scene/"
            + ("reach_map_dex1.npz" if _HAND == "dex1" else "reach_map.npz"))

OFFSET_DIR = (np.asarray(GRASP_CENTRE_OFFSET_CONST)
              / np.linalg.norm(GRASP_CENTRE_OFFSET_CONST))

MAX_DESCEND = np.deg2rad(25.0)
JAW_TILT_MAX = np.deg2rad(25.0)

# Delivered torso capability (measured on this stack, not commanded):
PELVIS_MAX = 0.793            # standing pelvis height
PELVIS_MIN = 0.598            # deepest DELIVERED squat: stepped commands to
                              # 13 cm measured pelvis drops of 19.5-19.9 cm
                              # (diag_reach_volume / diag_reach_sweep) — the
                              # earlier 0.660 assumed the single-shot 13 cm
WAIST_PITCH = (0.0, 0.05)     # DELIVERED lean while standing: commanding
                              # 0.25 measures 0.03-0.06, and the policy also
                              # wanders waist yaw ~0.2 rad in response to
                              # unusual arm poses — the map keeps the waist
                              # nearly neutral and the navigator erodes the
                              # lookup by 5 cm instead of trusting lean

# Voxel grid, pelvis frame.
VOX = 0.025
X0, X1 = 0.12, 0.75
Y0, Y1 = -0.75, 0.75
Z0, Z1 = -0.55, 0.45

N_SAMPLES = 600_000           # per arm per waist config: at 120k each voxel
                              # averaged 1.6 hits, so the erosion and the
                              # azimuth-coverage statistics were pure noise

# Sweet-zone ("core") criteria: the envelope is where a grasp is POSSIBLE;
# the core is where it is COMFORTABLE — room left for servo corrections.
AZ_BINS = 9                   # approach-azimuth bins over the front sector
AZ_SECTOR = np.deg2rad(60.0)  # matches the front-sector sample filter
CORE_JMARGIN = np.deg2rad(15.0)   # min distance of any arm joint to a limit
CORE_AZBINS = 3               # distinct approach azimuths seen at the voxel
CORE_X_MIN = 0.16             # inboard cut: closer voxels put the pregrasp
                              # against the torso (self-collision entries)


def _erode6(o: np.ndarray) -> np.ndarray:
    """Keep a voxel only if all 6 face-neighbours are also occupied."""
    p = np.pad(o, 1)
    return (o & p[2:, 1:-1, 1:-1] & p[:-2, 1:-1, 1:-1]
            & p[1:-1, 2:, 1:-1] & p[1:-1, :-2, 1:-1]
            & p[1:-1, 1:-1, 2:] & p[1:-1, 1:-1, :-2])


def _dilate6(o: np.ndarray) -> np.ndarray:
    p = np.pad(o, 1)
    return (o | p[2:, 1:-1, 1:-1] | p[:-2, 1:-1, 1:-1]
            | p[1:-1, 2:, 1:-1] | p[1:-1, :-2, 1:-1]
            | p[1:-1, 1:-1, 2:] | p[1:-1, 1:-1, :-2])


def _close6(o: np.ndarray) -> np.ndarray:
    """Morphological closing: fills single-voxel Monte-Carlo sampling holes
    without extending the true boundary — the occupancy is a ~5-hit/voxel
    MC estimate whose interior holes are noise, not unreachability."""
    return _erode6(_dilate6(o))


def _grid_shape():
    nx = int(round((X1 - X0) / VOX))
    ny = int(round((Y1 - Y0) / VOX))
    nz = int(round((Z1 - Z0) / VOX))
    return nx, ny, nz


def build(seed: int = 0) -> dict:
    import mujoco
    ik = ArmIK()
    rng = np.random.default_rng(seed)
    # Self-collision filter. mj_kinematics alone accepted configurations
    # with the forearm through the chest — the probe measured 30+ cm of
    # "delivery error" there because the sim, unlike bare FK, has a torso.
    # Robot-robot contact pairs (non-adjacent bodies) reject a sample.
    m = ik.model
    robot_geoms = set()
    for g in range(m.ngeom):
        nm = m.body(m.geom_bodyid[g]).name
        if nm.startswith(("left_", "right_", "pelvis", "torso", "waist",
                          "logo", "head", "imu")):
            robot_geoms.add(g)

    def self_collides() -> bool:
        mujoco.mj_collision(m, ik.data)
        d = ik.data
        for ci in range(d.ncon):
            g1, g2 = d.contact.geom1[ci], d.contact.geom2[ci]
            if g1 not in robot_geoms or g2 not in robot_geoms:
                continue
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
                continue
            return True
        return False
    nx, ny, nz = _grid_shape()
    occ = {"left": np.zeros((nx, ny, nz), dtype=bool),
           "right": np.zeros((nx, ny, nz), dtype=bool)}
    # Exemplar evidence per voxel: the joint solution that made it feasible.
    # The verification probe replays exactly this — the promise, not a
    # re-derivation of it.
    ex_q = {s2: np.zeros((nx, ny, nz, 7), dtype=np.float32)
            for s2 in ("left", "right")}
    ex_wp = {s2: np.full((nx, ny, nz), np.nan, dtype=np.float32)
             for s2 in ("left", "right")}
    ex_pos = {s2: np.zeros((nx, ny, nz, 3), dtype=np.float32)
              for s2 in ("left", "right")}
    ex_appr = {s2: np.zeros((nx, ny, nz, 3), dtype=np.float32)
               for s2 in ("left", "right")}
    # Comfort evidence per voxel: the BEST joint-limit margin any sample
    # achieved there, and a bitmask of which approach-azimuth bins produced
    # a hit — forward-MC's version of "how many directions work here".
    jmargin = {s2: np.zeros((nx, ny, nz), dtype=np.float32)
               for s2 in ("left", "right")}
    azmask = {s2: np.zeros((nx, ny, nz), dtype=np.uint16)
              for s2 in ("left", "right")}

    for side in ("left", "right"):
        arm = ARM_MJ[side]
        lo, hi = ik.lo[arm].copy(), ik.hi[arm].copy()
        for wp in WAIST_PITCH:
            qs = rng.uniform(lo, hi, size=(N_SAMPLES, 7))
            q29 = np.zeros(29)
            q29[14] = wp                       # waist_pitch
            kept = 0
            for row in qs:
                q29[arm] = row
                pos, rot = ik.fk(q29, side)
                appr = rot @ OFFSET_DIR
                dh = float(np.hypot(appr[0], appr[1]))
                if -appr[2] > np.tan(MAX_DESCEND) * max(dh, 1e-9):
                    continue                   # diving too steeply
                if appr[0] < dh * 0.5:
                    # Approach must come from the FRONT sector (within ~60
                    # deg of pelvis +x): the executed pipeline plans its
                    # travel from a virtual start on the robot->object
                    # bearing, so a voxel only reachable by a sideways or
                    # backwards approach is a promise the servo cannot keep
                    # — the map accepted one and the planner then refused
                    # "a direction this object cannot be gripped from".
                    continue
                if appr[2] > 0.5:
                    continue                   # grasping upward: not a table
                jaw_tilt = abs(float(np.arcsin(np.clip(rot[2, 1], -1, 1))))
                if jaw_tilt > JAW_TILT_MAX:
                    continue
                if self_collides():
                    continue
                i = int((pos[0] - X0) / VOX)
                j = int((pos[1] - Y0) / VOX)
                k = int((pos[2] - Z0) / VOX)
                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                    marg = float(np.minimum(row - lo, hi - row).min())
                    if marg > jmargin[side][i, j, k] or not occ[side][i, j, k]:
                        # Exemplar = the most comfortable solution seen, not
                        # the first: replays and demos then avoid limit-pinned
                        # postures.
                        jmargin[side][i, j, k] = marg
                        ex_q[side][i, j, k] = row
                        ex_wp[side][i, j, k] = wp
                        ex_pos[side][i, j, k] = pos
                        ex_appr[side][i, j, k] = appr
                    az = float(np.arctan2(appr[1], appr[0]))
                    b = int((az + AZ_SECTOR) / (2 * AZ_SECTOR) * AZ_BINS)
                    if 0 <= b < AZ_BINS:
                        azmask[side][i, j, k] |= np.uint16(1 << b)
                    occ[side][i, j, k] = True
                    kept += 1
            print(f"  {side} waist {wp:.2f}: kept {kept}/{N_SAMPLES}")

    both = occ["left"] | occ["right"]
    # Sweet zone: eroded envelope, comfortable joints, several approach
    # azimuths, and clear of the torso.
    xc = X0 + (np.arange(nx) + 0.5) * VOX
    x_ok = (xc >= CORE_X_MIN)[:, None, None]
    sweet = {}
    for s2 in ("left", "right"):
        bits = azmask[s2]
        nb = np.zeros(bits.shape, dtype=np.uint8)
        for b in range(AZ_BINS):
            nb += ((bits >> b) & 1).astype(np.uint8)
        # Erode the CLOSED occupancy (sampling holes are noise), but keep
        # only genuinely sampled voxels — the comfort criteria are only
        # defined where a sample landed.
        sweet[s2] = (_erode6(_close6(occ[s2])) & occ[s2]
                     & (jmargin[s2] >= CORE_JMARGIN)
                     & (nb >= CORE_AZBINS) & x_ok)
        print(f"  {s2}: envelope {occ[s2].sum()} -> core {sweet[s2].sum()}")
    np.savez_compressed(
        MAP_PATH, left=occ["left"], right=occ["right"], both=both,
        sweet_left=sweet["left"], sweet_right=sweet["right"],
        jmargin_left=jmargin["left"], jmargin_right=jmargin["right"],
        azmask_left=azmask["left"], azmask_right=azmask["right"],
        vox=VOX, origin=np.array([X0, Y0, Z0]),
        pelvis_range=np.array([PELVIS_MIN, PELVIS_MAX]),
        ex_q_left=ex_q["left"], ex_q_right=ex_q["right"],
        ex_wp_left=ex_wp["left"], ex_wp_right=ex_wp["right"],
        ex_pos_left=ex_pos["left"], ex_pos_right=ex_pos["right"],
        ex_appr_left=ex_appr["left"], ex_appr_right=ex_appr["right"])
    print(f"saved {MAP_PATH}: both {both.sum()} voxels "
          f"({both.sum() * VOX**3 * 1e3:.1f} litres)")
    return {"left": occ["left"], "right": occ["right"], "both": both}


class ReachMap:
    """Runtime lookup: is a world point graspable from a candidate stand?"""

    def __init__(self, path: str = MAP_PATH):
        z = np.load(path)
        self.left, self.right, self.both = z["left"], z["right"], z["both"]
        # Sweet zone (comfortable core) — present on maps built after the
        # comfort-metrics rebuild; fall back to the envelope on old files.
        self.sweet_left = z["sweet_left"] if "sweet_left" in z else self.left
        self.sweet_right = (z["sweet_right"] if "sweet_right" in z
                            else self.right)
        self.vox = float(z["vox"])
        self.origin = z["origin"]
        self.pelvis_min, self.pelvis_max = z["pelvis_range"]

    def _grid(self, arm: str, core: bool):
        if core:
            return {"left": self.sweet_left, "right": self.sweet_right,
                    "both": self.sweet_left | self.sweet_right}[arm]
        return {"left": self.left, "right": self.right, "both": self.both}[arm]

    def _at(self, grid, p) -> bool:
        idx = ((np.asarray(p) - self.origin) / self.vox).astype(int)
        if ((idx < 0).any() or (idx >= np.array(grid.shape)).any()):
            return False
        return bool(grid[tuple(idx)])

    def graspable(self, p_pelvis, arm: str = "both",
                  core: bool = False) -> bool:
        """p in the pelvis frame at STANDING height. The squat extends reach
        DOWNWARD: a point below the map is re-tested as if the pelvis had
        come down by up to (PELVIS_MAX - PELVIS_MIN). core=True tests the
        sweet zone instead of the full envelope."""
        grid = self._grid(arm, core)
        p = np.asarray(p_pelvis, dtype=float)
        drop = self.pelvis_max - self.pelvis_min
        for dz in np.arange(0.0, drop + 1e-9, self.vox):
            if self._at(grid, p + [0.0, 0.0, dz]):
                return True
        return False

    def plan_core_stand(self, obj_z_pf, arm: str = "right",
                        aim_ang_max: float = np.deg2rad(33.0),
                        min_margin: float = 0.05):
        """Joint (aim, squat) stand plan against the sweet core.

        The core narrows below the elbow band, so forcing the object into
        the STANDING core drags the stand toward the table. Instead: pick
        the smallest squat drop that brings a THICK core slice down onto
        the object, and aim THAT slice at it. obj_z_pf is the object's z
        in the pelvis frame at STANDING height.

        Returns (aim_cells_xy, drop, margin) or None:
        aim_cells_xy — pelvis-frame (x, y) core cells of the chosen slice,
        deep inside it (in-slice chebyshev clearance >= margin) and within
        aim_ang_max of straight ahead so the ego camera keeps the object;
        drop — the pelvis drop to command before grasping.
        """
        grid = self._grid(arm, core=True)
        grid_env = _close6(self._grid(arm, core=False))
        nz = grid.shape[2]
        drop_max = float(self.pelvis_max - self.pelvis_min)
        layers = []
        for dz in np.arange(0.0, drop_max + 1e-9, self.vox):
            k = int((float(obj_z_pf) + dz - self.origin[2]) / self.vox)
            if not (0 <= k < nz):
                continue
            sl = grid[:, :, k]
            occ = np.argwhere(sl)
            if len(occ) == 0:
                continue
            # Aim cells come from the CORE (comfort); the scatter margin is
            # measured against the ENVELOPE boundary (feasibility): a stand
            # that lands 1-2 cells off a core aim still only needs the grasp
            # to be POSSIBLE there — the core slice itself is too ragged
            # (~30 cells/layer) to have an interior.
            occ_set = {(int(i), int(j))
                       for i, j in np.argwhere(grid_env[:, :, k])}

            def cheb_margin(i, j):
                # Chebyshev distance to the nearest non-core cell (off-grid
                # counts as non-core).
                for r in range(1, 7):
                    for di in range(-r, r + 1):
                        for dj in range(-r, r + 1):
                            if (max(abs(di), abs(dj)) == r
                                    and (i + di, j + dj) not in occ_set):
                                return (r - 1) * self.vox
                return 6 * self.vox

            pts, margs = [], []
            for i, j in occ:
                x = self.origin[0] + (i + 0.5) * self.vox
                y = self.origin[1] + (j + 0.5) * self.vox
                if abs(np.arctan2(y, x)) > aim_ang_max:
                    continue
                pts.append((x, y))
                margs.append(cheb_margin(int(i), int(j)))
            if pts:
                layers.append((float(dz), np.asarray(pts),
                               np.asarray(margs)))
        if not layers:
            return None
        # Selection: the FARTHEST overlap, not the least squat. Squatting is
        # free; standing close to the table is what collides. Across all
        # squat depths pick the layer whose usable cells reach the largest
        # horizontal distance, and aim at that layer's far band. R_CAP keeps
        # the aim under the servo's walk-in depth band (GRIP_X_MAX 0.30) and
        # the pregrasp staging limit (~0.32).
        R_CAP = 0.34            # execution-side staging limit, deep-squat
                                # era: the servo's depth band now follows
                                # the plan (plan_r), so only the hard
                                # pregrasp-staging wall (~0.36) remains
        # Hand-side bias: a can on the torso centreline forces the arm to
        # reach ACROSS the chest (wrist yaw at its limit — the "twisted"
        # pregrasp), while a can in front of the shoulder is a short
        # straight entry. Prefer aim cells >= 6 cm on the grasping arm's
        # side; the ego cone (33 deg < the ~36.7 deg view gate) already
        # bounds how far out they can sit. Fall back to the full band only
        # when no layer has side cells.
        def _pick(side_only):
            best = None
            for dz, pts, margs in layers:
                r = np.hypot(pts[:, 0], pts[:, 1])
                use = (margs >= min_margin) & (r <= R_CAP)
                if side_only:
                    use &= (pts[:, 1] <= -0.06 if arm == "right"
                            else pts[:, 1] >= 0.06)
                if not use.any():
                    continue
                r_max = float(r[use].max())
                if best is None or r_max > best[0]:
                    far = use & (r >= r_max - 0.03)
                    best = (r_max, pts[far], dz, float(margs[far].max()))
            return best

        best = _pick(True) or _pick(False)
        if best is None:
            dz, pts, margs = max(layers, key=lambda l: float(l[2].max()))
            return pts[margs >= margs.max()], dz, float(margs.max())
        return best[1], best[2], best[3]

    def margin(self, p_pelvis, arm: str = "both") -> float:
        """Distance (m) to the nearest non-graspable voxel, xy-plane, at the
        best squat level — the stand-quality score the navigator maximises."""
        if not self.graspable(p_pelvis, arm):
            return 0.0
        m = 0.0
        while m < 0.15:
            m += self.vox
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                q = np.asarray(p_pelvis) + [m * np.cos(a), m * np.sin(a), 0]
                if not self.graspable(q, arm):
                    return m - self.vox
        return m


if __name__ == "__main__":
    build()


def best_stand(rm: "ReachMap", obj_w, obj_z_pf: float, pelvis_xy, yaw_now: float,
               edge_dist_fn, arm: str = "both",
               margin_floor: float = 0.05, edge_floor: float = 0.10):
    """Pick (stand_xy, yaw) near the current stand that puts the object deep
    inside the reach map.

    `edge_dist_fn(xy) -> m` gives the distance from a candidate stand to the
    nearest table edge. The floor is 0.10 — the squat-era knee clearance of
    0.26 CONFLICTS with the reach core (the can sits only 0.12 m inboard, so
    a core stand necessarily hugs the edge), and the no-squat flow's
    successful episodes all stood at 0.10-0.20. The facing offset
    from the object's bearing is capped at 40 deg — the body faces the table
    edge, the hand crosses it from above, and the object stays in the ego
    camera. Returns (stand_xy, yaw, margin) or None if nothing qualifies.
    """
    obj_w = np.asarray(obj_w, dtype=float)
    best = None
    for dx in np.arange(-0.45, 0.46, 0.05):
        for dy in np.arange(-0.45, 0.46, 0.05):
            s = np.asarray(pelvis_xy, dtype=float) + [dx, dy]
            if edge_dist_fn(s) < edge_floor:
                continue
            d = obj_w[:2] - s
            if np.linalg.norm(d) < 0.05:
                continue
            bearing = float(np.arctan2(d[1], d[0]))
            for delta in np.arange(-0.7, 0.71, 0.1):
                psi = bearing + delta
                c, sn = np.cos(psi), np.sin(psi)
                rel = np.array([c * d[0] + sn * d[1],
                                -sn * d[0] + c * d[1], obj_z_pf])
                m = rm.margin(rel, arm)
                if m < margin_floor:
                    continue
                J = m - 0.3 * abs(delta) - 0.2 * float(np.hypot(dx, dy))
                if best is None or J > best[0]:
                    best = (J, s, psi, m)
    if best is None:
        return None
    return best[1], best[2], best[3]
