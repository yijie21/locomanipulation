"""The navigation and place flows on the head camera.

  approach  the fixture the instruction names (SAM 3), the free space from the depth points seen so far (an occupancy
            map in the robot's frame, grown at every look; what has not been seen is taken as free and replanned when it
            turns out not to be), a plan to the nearest free place beside the seen fixture, a walk of at most 2.5 m, a
            new look. A fixture that is not in view is looked for by turning, then by walking toward open floor.
  place     the receptacle the instruction names, from the start's image or, when it was not in view, from a look
            after turning with the object carried; the object's foot in the palm from the seen points; the edge to
            stay behind from the depth points.

The flows read the instruction, the head camera's image, joint readings and leg odometry through `env.bel`, and hold
no reading of the scene's state. The numbers they leave in `env.row` are their own estimates. (Moved from skill_lml_cam.py,
TODO 49.4: the fingers go through `env.hand`, SAM 3's address is `env.sam3`, and the ends of `approach` and `pick_only` tell the
bed with `env.flow_done` so that its recorder can read its numbers; the old bed wrapped these two functions for that.)
"""

import numpy as np

from . import camera_view as V
from . import skill as S

P = S.P
RES = 0.05
REACH = 6.0
SEG = 2.5  # m, the longest walk before a new look
WEAK = 0.12  # SAM 3's least score for a fixture (the pick uses 0.25)
MIN_PX_NAV = 25  # a fixture far off is a few dozen pixels


# ---- the map


class Map:
    """What the robot cannot walk through and where the named fixture is, in the robot's frame, from every look so far."""

    def __init__(self, centre_xy):
        self.n = int(2 * REACH / RES)
        self.origin = np.asarray(centre_xy, float) - REACH
        self.occ = np.zeros((self.n, self.n), np.int16)  # depth points between knee and head height, per cell
        self.fix = np.zeros((self.n, self.n), np.int16)  # of them the fixture's
        self.floor = np.zeros((self.n, self.n), np.int16)  # floor points seen
        self.fix_pts = []

    def cell(self, xy):
        c = np.floor((np.asarray(xy, float) - self.origin) / RES).astype(int)
        return c[..., 0], c[..., 1]

    def _count(self, arr, xy):
        i, j = self.cell(xy)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        np.add.at(arr, (i[ok], j[ok]), 1)

    def add(self, pts, ok, floor: float, pb, yaw: float, mask=None, far: float = 4.0) -> None:
        ahead = np.array([np.cos(yaw), np.sin(yaw)])
        d = np.linalg.norm(pts[..., :2] - pb[:2], axis=-1)
        use = ok & (pts[..., :2] @ ahead > pb[:2] @ ahead + V.OWN) & (d < far)
        h = pts[..., 2] - floor
        band = use & (h > V.BAND[0]) & (h < 1.35)
        self._count(self.occ, pts[band][:, :2])
        self._count(self.floor, pts[use & (h < 0.06)][:, :2])
        if mask is not None:
            fx = band & mask
            self._count(self.fix, pts[fx][:, :2])
            self.fix_pts.append(pts[fx])

    def taken(self, min_pts: int = 3):
        return self.occ >= min_pts

    def fixture(self, min_pts: int = 1):
        """The cells the fixture's masks gave; when they fall in separate pieces (a mask that also caught another piece of
        furniture) the biggest piece, the one with the most cells."""
        from scipy.ndimage import label

        f = self.fix >= min_pts
        if not f.any():
            return f
        lab, n = label(dilate(f, 0.25))
        if n > 1:
            f = f & (lab == 1 + int(np.argmax(np.bincount(lab[f], minlength=n + 1)[1:])))
        return f

    def centre_of(self):
        """The seen fixture's centre (xy): the middle of the span of the cells that are its."""
        c = np.argwhere(self.fixture())
        if len(c) == 0:
            return None
        lo, hi = np.percentile(c, [2, 98], axis=0)
        return self.origin + (0.5 * (lo + hi) + 0.5) * RES


class PlanGrid:
    """A grid of cells that are free or taken, and the shortest way over the free ones."""

    def xy(self, c) -> np.ndarray:
        return self.origin + (np.asarray(c, float) + 0.5) * self.res

    def cell(self, xy) -> tuple:
        return tuple(np.clip(((np.asarray(xy) - self.origin) / self.res).astype(int), 0, self.n - 1))

    def clear(self, a, b) -> bool:
        """Whether the straight line between two cells stays in free cells."""
        k = int(max(abs(a[0] - b[0]), abs(a[1] - b[1]))) + 1
        return all(
            self.free[int(round(a[0] + (b[0] - a[0]) * t / k)), int(round(a[1] + (b[1] - a[1]) * t / k))]
            for t in range(k + 1)
        )

    def path(self, start_xy, goals: np.ndarray):
        """The shortest way over free cells from `start_xy` to any cell of the mask `goals`, as corner points (the
        start left out), or None."""
        import heapq

        s = self.cell(start_xy)
        free = self.free.copy()
        k = self.disk.shape[0] // 2
        free[max(0, s[0] - k) : s[0] + k + 1, max(0, s[1] - k) : s[1] + k + 1] |= ~self.taken[
            max(0, s[0] - k) : s[0] + k + 1, max(0, s[1] - k) : s[1] + k + 1
        ]  # the robot stands where it stands
        dist, prev, heap, end = {s: 0.0}, {}, [(0.0, s)], None
        while heap:
            du, u = heapq.heappop(heap)
            if du > dist.get(u, np.inf):
                continue
            if goals[u]:
                end = u
                break
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    v = (u[0] + di, u[1] + dj)
                    if (di or dj) and 0 <= v[0] < self.n and 0 <= v[1] < self.n and free[v]:
                        dv = du + self.res * (1.4142 if di and dj else 1.0)
                        if dv < dist.get(v, np.inf):
                            dist[v], prev[v] = dv, u
                            heapq.heappush(heap, (dv, v))
        if end is None:
            return None
        cells = [end]
        while cells[-1] != s:
            cells.append(prev[cells[-1]])
        cells = cells[::-1]
        keep, i = [], 0
        while i < len(cells) - 1:  # the farthest cell seen in a straight free line from each corner
            j = len(cells) - 1
            while j > i + 1 and not self.clear(cells[i], cells[j]):
                j -= 1
            keep.append(cells[j])
            i = j
        return [self.xy(c) for c in keep], float(dist[end])


class MapGrid(PlanGrid):
    """The planner's grid over a map built from the camera."""

    def __init__(self, mp: Map, radius: float):
        from scipy.ndimage import binary_dilation

        self.res, self.origin, self.n = RES, mp.origin, mp.n
        k = int(np.ceil(radius / RES))
        yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
        self.disk = xx**2 + yy**2 <= k**2
        self.taken = mp.taken() | mp.fixture()
        self.free = ~binary_dilation(self.taken, structure=self.disk)


def dilate(mask, radius: float):
    from scipy.ndimage import binary_dilation

    k = int(np.ceil(radius / RES))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    return binary_dilation(mask, structure=xx**2 + yy**2 <= k**2)


def near_fixture(mp: Map, grid, fixture, gap: float, here=None):
    """The way to the free places beside the seen fixture (just outside the robot's clearance round it): to those within
    0.85 m of its seen centre when that is not much longer than to the nearest place beside it. A rule that asks for the
    robot near a fixture's centre ("walk to the desk": within 1 m of it) and one that asks for it beside the fixture
    (iTHOR: within 0.35 m of its surface) are both met there. -> (corners, length) or None."""
    near = fixture | (dilate(fixture, 0.30) & grid.taken)  # the fixture's cells and what touches them (mask edges)
    ring = dilate(near, gap + 2 * RES) & grid.free
    known = ring & dilate(mp.floor > 0, 0.10)  # places on floor that has been seen, not behind the fixture's face
    ring = known if known.any() else ring
    ii, jj = np.mgrid[0 : mp.n, 0 : mp.n]
    centres = mp.origin + (np.stack([ii, jj], -1) + 0.5) * RES
    central = ring & (np.linalg.norm(centres - mp.centre_of(), axis=-1) <= 0.85)
    f_ring = grid.path(here, ring) if ring.any() else None
    f_mid = grid.path(here, central) if central.any() else None
    return f_mid if f_mid is not None and (f_ring is None or f_mid[1] <= f_ring[1] + 0.6) else f_ring


def polyline_cut(here, corners, length: float):
    """The point `length` along the path from `here` through `corners` (the last corner if it is shorter), and whether
    that is the path's end."""
    p, left = np.asarray(here, float), length
    for q, c in enumerate(corners):
        seg = float(np.linalg.norm(c - p))
        if seg >= left:
            return p + (c - p) * (left / max(seg, 1e-9)), False
        p, left = c, left - seg
    return p, True


def clear_heading(mp: Map, here, n: int = 16, reach: float = 3.0) -> float:
    """The heading with the farthest run of cells that are not taken (open floor to walk toward when nothing is found)."""
    taken, best = mp.taken(), (-1.0, 0.0)
    for a in np.linspace(-np.pi, np.pi, n, endpoint=False):
        d = 0.0
        while d < reach:
            i, j = mp.cell(here + (d + RES) * np.array([np.cos(a), np.sin(a)]))
            if not (0 <= i < mp.n and 0 <= j < mp.n) or taken[i, j]:
                break
            d += RES
        if d > best[0]:
            best = (d, float(a))
    return best[1]


def approach(env) -> None:
    """A navigation task from the image: find the named fixture, plan over what has been seen, walk, look again."""
    try:
        _approach(env)
    finally:
        env.flow_done("approach")  # the bed's recorder reads the end place (record only), also after an exception


def _approach(env) -> None:
    K, row, bel, phase = env.K, env.row, env.bel, env.clock.phase
    want = env.want
    names = V.phrases(want["dest"])
    pb0, yaw0 = bel.pelvis()
    mp = Map(pb0[:2])
    nav = row["nav"] = dict(looks=0, seen=None, turns=0, explored=0, segments=[], names=names)
    turn_k, base = 0, yaw0
    with phase("walk"):
        for it in range(18):
            rgb, pts, ok = bel.scene()
            pb, yaw = bel.pelvis()
            floor = V.floor_of(pts, ok, pb)
            # a mask counts when it stands up from the floor
            mask, score, ph = V.Sam.best(env.sam3, rgb, names, floor=WEAK)
            if mask is not None:
                got = mask & ok
                if got.sum() < MIN_PX_NAV or np.median(pts[got][:, 2] - floor) < 0.25:
                    mask = None
            mp.add(pts, ok, floor, pb, yaw, mask)
            nav["looks"] += 1
            c_ = mp.centre_of()
            nav.setdefault("log", []).append((None if mask is None else round(float(score), 2), None if c_ is None else c_.round(2).tolist(), pb[:2].round(2).tolist()))  # fmt: skip
            if mask is not None and nav["seen"] is None:
                nav["seen"] = it
                nav["score"], nav["phrase"] = round(score, 2), ph
            here = pb[:2].copy()
            if not mp.fixture().any():  # not seen yet: turn on the spot, then walk toward open floor and turn again
                if turn_k < len(V.SEARCH):
                    bel.walk(here, base + V.SEARCH[turn_k], face="fixed")
                    turn_k += 1
                    nav["turns"] += 1
                    continue
                base = clear_heading(mp, here)
                nav["explored"] += 1
                turn_k = 0
                bel.walk(
                    here + SEG * np.array([np.cos(base), np.sin(base)]), base, face="travel", speed=K["long_walk_speed"]
                )
                continue
            grid = MapGrid(mp, K["nav_gap"])
            fixture = mp.fixture()
            found = near_fixture(mp, grid, fixture, K["nav_gap"], here)
            if found is None and K["nav_gap"] > 0.22:  # no way at this clearance: one at a tighter one
                grid = MapGrid(mp, 0.22)
                found = near_fixture(mp, grid, fixture, 0.22, here)
            if found is None:
                nav["segments"].append(None)
                turn_k = 0
                bel.walk(here, yaw + 0.9, face="fixed")  # nothing reachable from here: turn and look again
                continue
            corners, length = found
            centre = mp.centre_of()
            nav["segments"].append(round(length, 2))
            if length < 0.15:  # at the place
                break
            stop, last = polyline_cut(here, corners, SEG)
            v = centre - stop if last else stop - here
            bel.walk(
                stop, float(np.arctan2(v[1], v[0])), face="travel" if not last else "fixed", speed=K["long_walk_speed"]
            )
            if last:  # the end of the plan: one more look from there decides whether it is the place
                continue
    centre = mp.centre_of()
    nav["centre"] = None if centre is None else centre.round(3).tolist()  # the fixture as the robot last took it


# ---- place


def dest_of(want: dict, views: list, row: dict, sam3):
    """(centre xy, the level the object is let go over, near side along the first view's heading, seen xy points) of the
    receptacle the instruction names, from the views (rgb, points, ok, pelvis, yaw): the middle of what is seen of it in
    the first view that shows it, and its highest point. None when no view shows it."""
    names = V.phrases(want["dest"])
    yaw = views[0][4]
    ahead = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-ahead[1], ahead[0]])
    for rgb, pts, ok, pb, _ in views:
        mask, score, phrase = V.Sam.find(sam3, rgb, names)
        row.setdefault("dest_score", round(score, 2))
        row.setdefault("dest_phrase", phrase)
        if mask is None or (mask & ok).sum() < 150:
            continue
        seen = pts[mask & ok]
        seen = seen[np.linalg.norm(seen - np.median(seen, axis=0), axis=1) < 1.2]
        hi = np.percentile(seen, 99, axis=0)
        a_lo, a_hi = np.percentile(seen[:, :2] @ ahead, [1, 99])
        w_lo, w_hi = np.percentile(seen[:, :2] @ left, [1, 99])
        return 0.5 * (a_lo + a_hi) * ahead + 0.5 * (w_lo + w_hi) * left, float(hi[2]), float(a_lo), seen[:, :2]
    return None


def place(env) -> None:
    """Carry, walk to the receptacle, the object over it, its foot to the rim, let go, the hand away. The receptacle
    comes from the image, the object's foot from its seen points."""
    hand, mo, bel, K, row, view, phase = env.hand, env.mo, env.bel, env.K, env.row, env.view, env.clock.phase
    side = S.SIDE
    ch = view.chosen
    if ch is None:
        row["why"] = "no grasp to carry from"
        return
    st = env.stance or {}
    pb, yb = bel.pelvis()
    palm, Rm = bel.palm(side)
    Tg = ch["T_wo"] @ ch["T_og"]  # the palm at the grasp, in the robot's frame
    cloud = view.s["cloud"]
    lo, hi = np.percentile(cloud, [1, 99], axis=0)
    foot = Tg[:3, :3].T @ (np.r_[0.5 * (lo + hi)[:2], np.percentile(cloud[:, 2], 2)] - Tg[:3, 3])  # in the palm's frame
    row["foot"] = foot.round(4).tolist()  # the object's foot in the palm's frame, from the seen cloud
    env.mark("before_carry")
    dest = view.dest
    if dest is None:  # not in the start's image: carry the object and turn to look for it
        with phase("carry"):
            carry = pb[:2] + S.PL.rot_z(yb)[:2, :2] @ [K["carry_x"], K["carry_y"]]
            bel.reach(side, np.r_[carry, palm[2]], Rm, speed=K["carry_speed"])
        with phase("search"):
            for k, turn in enumerate(V.SEARCH):
                bel.walk(bel.pelvis()[0][:2], view.yaw0 + turn, face="fixed")
                rgb, pts, ok = bel.scene()
                pbk, yk = bel.pelvis()
                dest = dest_of(env.want, [(rgb, pts, ok, pbk, yk)], row, env.sam3)
                if dest is not None:
                    row["dest_turns"] = k + 1
                    break
        pb, yb = bel.pelvis()
        palm, Rm = bel.palm(side)
        if dest is None:
            row["why"] = "no receptacle seen"
            return
    centre, rim, near, seen_xy = dest
    row["dest"] = dict(centre=np.round(centre, 3).tolist(), rim=round(rim, 3))
    bearing = float(np.arctan2(*(centre - pb[:2])[::-1]))
    stay = bool(st.get("stay"))
    turned = abs(P.R.wrap(bearing - yb)) > np.radians(60) and not stay
    yaw2 = bearing if turned else yb
    ahead2 = np.array([np.cos(yaw2), np.sin(yaw2)])
    left2 = np.array([-ahead2[1], ahead2[0]])
    Rr = S.PL.rot_z(yaw2 - yb) @ Rm
    over = np.r_[centre, rim + K["over_rim"] + K["over_extra"]] - Rr @ foot
    release = np.r_[centre, rim - K["seat"]] - Rr @ foot
    stance2 = (
        release[:2]
        - S.PL.rot_z(yaw2)[:2, :2] @ [K["stand"] + st.get("d2", 0.0), K["place_side"]]
        + st.get("t2", 0.0) * left2
    )
    if turned:  # a fresh look from the turn for what stands between the robot and the receptacle
        with phase("turn"):
            bel.walk(pb[:2], yaw2, face="fixed")
        rgb, pts, ok = bel.scene()
        pbt, _ = bel.pelvis()
        limit = V.obstacle_edge(
            pts, ok, V.floor_of(pts, ok, pbt), ahead2, float(stance2 @ left2), pbt[:2] @ ahead2 + V.HELD
        )
    else:
        limit = view.edge(ahead2, float(stance2 @ left2))
    band = np.abs(seen_xy @ left2 - stance2 @ left2) < 0.3
    near = float((seen_xy[band] @ ahead2).min()) if band.any() else None
    limit = near if limit is None else limit if near is None else min(limit, near)
    if limit is not None:
        stance2 = stance2 - max(0.0, stance2 @ ahead2 - (limit - K["table_gap"])) * ahead2
    if stay:
        stance2 = pb[:2].copy()
    row["place"] = dict(centre=centre.round(3).tolist(), rim=round(rim, 3), turned=bool(turned), stance2=stance2.round(3).tolist(),
                        move=round(float(np.linalg.norm(stance2 - pb[:2])), 2), limit=limit,
                        over=over.round(4).tolist(), release=release.round(4).tolist())  # fmt: skip
    walks = turned or np.linalg.norm(stance2 - pb[:2]) > 0.05
    with phase("carry"):
        carry = pb[:2] + S.PL.rot_z(yb)[:2, :2] @ [K["carry_x"], K["carry_y"]]
        bel.reach(side, np.r_[carry, max(palm[2], over[2])], Rm, speed=K["carry_speed"])
    if walks:
        with phase("walk2"):
            bel.walk(
                stance2, yaw2, face="travel" if turned else "fixed", speed=K["long_walk_speed"] if turned else None
            )
    with phase("over"):
        bel.reach(side, over, Rr, speed=K["over_speed"])
    if np.linalg.norm(release - over) > 1e-3:  # as skill_lml.place: let go from the place over the rim
        with phase("release"):
            bel.reach(side, release, Rr, speed=K["release_speed"])
    row["over_vec_cm"] = (100 * (release - over)).round(1).tolist()  # the plan's own step from over to release
    env.mark("before_open")
    with phase("open"):
        start = np.array(hand.cmd.get(side, hand.grip_target(side)), float)
        for u in np.linspace(0, 1, max(1, int(K["open_s"] / 0.02))):
            hand.cmd[side] = start + u * (S.PL.WIDE - start)
            S.one_step(mo, K)
        mo.hold(K["open_hold"])
    with phase("away"):
        p, Rh = bel.palm(side)
        if K["back_off"] > 0:
            bel.reach(side, p - K["back_off"] * Rh[:, 1], Rh, speed=K["back_speed"])
        pb, yb = bel.pelvis()
        home = pb[:2] + S.PL.rot_z(yb)[:2, :2] @ [K["carry_x"], K["carry_y"]]
        bel.reach(side, np.r_[home, bel.palm(side)[0][2] + K["away_up"]], Rh, speed=K["away_speed"])


HELD_DEFL = 1.5  # rad: the fingers' joints stopped this far (summed) short of where the closing commanded them


def finger_defl(env) -> float | None:
    """How far the right hand's finger joints are from where the closing took them: an empty hand closes all the way, a
    hand round an object is stopped by it. The joint readings are the robot's own."""
    ch = env.view.chosen
    if ch is None:
        return None
    return float(np.abs(env.hand.q(S.SIDE) - np.asarray(ch["f_close"], float)).sum())


def pick_only(env) -> None:
    """The pick of skill_simple.episode on the camera view; records the finger joints' deflection."""
    S.episode(env)
    env.row["finger_defl"] = finger_defl(env)
    env.flow_done("pick")


def pick_place(env) -> None:
    """The pick, then `place` if the finger joints say the hand holds something."""
    pick_only(env)
    row = env.row
    if "why" in row or (row["finger_defl"] or 0.0) < HELD_DEFL:
        return
    place(env)


# ---- a door or a drawer


def articulate(env) -> None:
    """A door or drawer task (open or close the fridge door, open or close the low drawer: the walk to the fixture first
    when the instruction starts with it; the drawer pick-and-place has no walk, its `want["steps"]` are the instruction's
    order: pick the apple, open, place it inside, close).

    The seed does not open or close anything; this is where the loop writes that flow. It walks to the fixture when asked
    (`approach`), takes one look for the handle and leaves in `env.row["art_look"]` whether it was found; no arm motion,
    no manipulation."""
    want, row, bel, phase = env.want, env.row, env.bel, env.clock.phase
    if want["approach"]:
        approach(env)
    with phase("look"):
        rgb, pts, ok = bel.scene()
        mask, score, phrase = V.Sam.find(env.sam3, rgb, V.phrases(want["handle"]))
        px = 0 if mask is None else int((mask & ok).sum())
        row["art_look"] = dict(
            found=bool(mask is not None and px >= V.MIN_PX), score=round(float(score), 2), phrase=phrase, px=px
        )
