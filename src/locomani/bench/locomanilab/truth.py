"""The scene as the simulator's own state gives it on LocoManiLab, and the truth track's flows (TODO 49.4; moved from skill_lml.py).

What the bed (never a skill package) needs of the mirrored model's geometry: the record's comparisons with the truth (the
recorder, record.py), the truth track's view (`LmlTruthView`) and its flows (`approach`, `place`, `pick_place`: the old bed_lml.py
`run` track), the object's point cloud for GraspGen-X (`object_cloud`, prep). Fixed infrastructure: a skill package cannot import
this module (guard rule R8). The motions are the seed skill's (walk to a stance, reach, close three fingers, lift); a
LocoManiLab room is not a table square to the robot, so the scene is read from the mirrored model's geometry, not a layout file:

  target   the body the task names; its extent from its collision meshes
  grasp    GraspGen-X's Dex3 palm poses on points sampled from the object's surface, in the object's frame
           (object_cloud -> perception/grasps.py, once per object), the best-scored the right hand can take
  edge     where the furniture begins ahead of the robot: the nearest hit of rays cast forward between knee and chest
           height in a band around the line to the stance, on anything solid that is not the robot or the target

The flows take the skill module as an argument (the package's, or the seed's): its parameters, `choose_grasp` and `one_step`.
"""

import inspect
from pathlib import Path

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.bench.simple import body as PL

BAND = (0.15, 1.10)  # m above the floor: what the legs, the pelvis and the hanging hands can run into


def T(p, q) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3], M[:3, :3] = p, R.mat(np.asarray(q, float))
    return M


def subtree(m, root: int) -> set:
    """Body ids of `root` and everything below it."""
    out = {root}
    for b in range(m.nbody):
        k = b
        while k not in out and k != 0:
            k = int(m.body_parentid[k])
        if k in out:
            out.add(b)
    return out


def geom_points(m, d, g: int) -> np.ndarray:
    """World points on a collision geom: a mesh's vertices, a primitive's bounding corners."""
    Rg, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    k = m.geom_dataid[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and k >= 0:
        v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
    else:
        s = m.geom_size[g]
        t = m.geom_type[g]
        h = s if t == mujoco.mjtGeom.mjGEOM_BOX else np.array([s[0], s[0], s[1] + s[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else s[1]]) if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE) else np.full(3, s[0])  # fmt: skip
        v = np.array([[sx, sy, sz] for sx in (-1, 0, 1) for sy in (-1, 0, 1) for sz in (-1, 0, 1)]) * h
    return p + v @ Rg.T


def solid(m, g: int) -> bool:
    return bool(m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE


def body_points(sim, bodies: set) -> np.ndarray:
    m, d = sim.m, sim.d
    return np.vstack([geom_points(m, d, g) for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and solid(m, g)])


def local_tris(m, g: int) -> np.ndarray:
    """A solid geom's surface as triangles in the geom's frame: a mesh's own, a box's twelve, a 16-sided prism for a
    cylinder or capsule, a box round anything else."""
    t, k, sz = m.geom_type[g], m.geom_dataid[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH and k >= 0:
        v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        return v[m.mesh_face[m.mesh_faceadr[k] : m.mesh_faceadr[k] + m.mesh_facenum[k]]]
    if t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        a = np.linspace(0, 2 * np.pi, 17)
        h = sz[1] + (sz[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)
        ring = np.stack([sz[0] * np.cos(a), sz[0] * np.sin(a)], 1)
        lo, hi, out = np.c_[ring, np.full(17, -h)], np.c_[ring, np.full(17, h)], []
        for i in range(16):
            out += [
                [lo[i], lo[i + 1], hi[i]],
                [hi[i], lo[i + 1], hi[i + 1]],
                [[0, 0, -h], lo[i + 1], lo[i]],
                [[0, 0, h], hi[i], hi[i + 1]],
            ]
        return np.array(out, float)
    h = sz if t == mujoco.mjtGeom.mjGEOM_BOX else np.full(3, sz[0])
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    return np.array([[c[q[0]], c[q[1]], c[q[2]]] for q in quads] + [[c[q[0]], c[q[2]], c[q[3]]] for q in quads])


def object_cloud(sim, body: int, n: int = 3000, seed: int = 0) -> np.ndarray:
    """`n` points on the surface of the solid geoms of the body and the bodies below it (LocoManiLab hangs an
    object's geoms on children of the body that carries its free joint), in the body's frame."""
    m, d = sim.m, sim.d
    rng, tris, below = np.random.default_rng(seed), [], subtree(m, body)
    for g in (g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in below and solid(m, g)):
        tris.append(d.geom_xpos[g] + local_tris(m, g) @ d.geom_xmat[g].reshape(3, 3).T)
    T = np.vstack(tris)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    i = rng.choice(len(T), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = T[i, 0] + u[:, None] * (T[i, 1] - T[i, 0]) + v[:, None] * (T[i, 2] - T[i, 0])
    return (pts - d.xpos[body]) @ d.xmat[body].reshape(3, 3)


class LmlTruthView:
    """The scene read from the mirrored model (which carries the environment's whole state)."""

    search = False

    def __init__(self, env, grasps: Path, skill):
        self.env, self.grasps, self.S = env, grasps, skill
        m = env.sim.m
        self.target_id = m.body(env.info["target"]).id
        self.own = subtree(m, m.body("pelvis").id) | subtree(m, self.target_id)

    def heading(self) -> float:
        return float(R.yaw_of(self.env.sim.d.qpos[3:7]))

    def target(self):
        sim, b = self.env.sim, self.target_id
        V = body_points(sim, subtree(sim.m, b))
        return dict(T_wo=T(sim.d.xpos[b], sim.d.xquat[b]), lo=V.min(0), hi=V.max(0), top=float(V[:, 2].min()))

    def grasp(self, tgt, yaw: float):
        if not self.grasps.exists():
            return None
        z, back = np.load(self.grasps), self.env.K.get("normal_back", -1.0)
        if back > -1.0:  # not the poses whose palm faces back at the robot: the hand would reach past the object and
            n = (tgt["T_wo"] @ z["T_op"])[:, :3, 1] @ [np.cos(yaw), np.sin(yaw), 0.0]  # come back over it (it swept a
            keep = n >= back  # cup 30-40 cm away before the fingers closed, 11-17)
            z = dict(T_op=z["T_op"][keep], score=z["score"][keep], f_close=z["f_close"], f_open=z["f_open"])
        S = self.S
        S = self.S
        kw = (
            dict(centre=0.5 * (tgt["lo"] + tgt["hi"]))
            if "centre" in inspect.signature(S.choose_grasp).parameters
            else {}
        )
        return S.choose_grasp(z, tgt["T_wo"], yaw, tgt["top"], self.env.K, **kw)  # (a skill package's has no `centre`)

    def edge(self, ahead, lateral: float, table: str = "table", half: float = 0.30):
        """The coordinate along `ahead` of the nearest solid thing ahead of the robot that is neither the robot nor the
        target, over rays between knee and chest height in a band `half` to either side of `lateral`; None when the
        way is clear for 3 m."""
        sim = self.env.sim
        m, d = sim.m, sim.d
        left, here = np.array([-ahead[1], ahead[0]]), float(d.qpos[:2] @ ahead)
        vec, gid, best = np.r_[ahead, 0.0], np.zeros(1, np.int32), None
        for z in np.linspace(*BAND, 9):
            for w in np.linspace(lateral - half, lateral + half, 9):
                o, gone = np.r_[here * ahead + w * left, z], 0.0
                for _ in range(20):  # past the robot's own geoms, the target's and the ones nothing collides with
                    dist = mujoco.mj_ray(m, d, o, vec, None, 1, -1, gid)
                    if dist < 0 or gone + dist > 3.0:
                        break
                    g = int(gid[0])
                    if solid(m, g) and int(m.geom_bodyid[g]) not in self.own:
                        best = gone + dist if best is None else min(best, gone + dist)
                        break
                    o, gone = o + (dist + 1e-3) * vec, gone + dist + 1e-3
        return None if best is None else here + best


class Grid:
    """What the robot cannot walk through, from above: a cell is taken when a ray dropped from head height meets
    something solid (not the robot) above knee height. `free` is that map grown by the robot's radius."""

    def __init__(self, sim, own: set, res: float = 0.05, reach: float = 5.0, radius: float = 0.30):
        from scipy.ndimage import binary_dilation

        m, d = sim.m, sim.d
        n = int(2 * reach / res)
        self.res, self.origin, self.n = res, sim.d.qpos[:2] - reach, n
        self.body = np.full((n, n), -1)
        vec, gid = np.array([0.0, 0.0, -1.0]), np.zeros(1, np.int32)
        for i in range(n):
            for j in range(n):
                o = np.r_[self.xy((i, j)), 1.35]
                for _ in range(8):
                    dist = mujoco.mj_ray(m, d, o, vec, None, 1, -1, gid)
                    if dist < 0 or o[2] - dist < BAND[0]:
                        break
                    g = int(gid[0])
                    if solid(m, g) and int(m.geom_bodyid[g]) not in own:
                        self.body[i, j] = int(m.geom_bodyid[g])
                        break
                    o[2] -= dist + 1e-3
        k = int(np.ceil(radius / res))
        yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
        self.disk = xx**2 + yy**2 <= k**2
        self.taken = self.body >= 0
        self.free = ~binary_dilation(self.taken, structure=self.disk)

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


def approach(env) -> None:
    """A navigation task: walk round what stands in the way to the nearest free place that meets the task's rule and
    face the fixture. The iTHOR tasks ask for the robot within 0.35 m of the fixture for a second: a place beside it.
    The Holodeck ones for the base within a radius of a point: a place inside that circle, 0.2 m in from its rim."""
    from scipy.ndimage import binary_dilation

    sim, K, row, view, bel, phase = env.sim, env.K, env.row, env.view, env.bel, env.clock.phase
    m = sim.m
    grid = Grid(sim, subtree(m, m.body("pelvis").id), radius=K["nav_gap"])
    fixture = np.isin(grid.body, list(subtree(m, view.target_id)))
    k = int(np.ceil(K["nav_gap"] / grid.res)) + 2
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    goals = binary_dilation(fixture, structure=xx**2 + yy**2 <= k**2) & grid.free
    tgt = (env.spec or {}).get("target") or {}
    if tgt.get("radius") and tgt.get("xy"):
        ii, jj = np.mgrid[0 : grid.n, 0 : grid.n]
        centres = grid.origin + (np.stack([ii, jj], -1) + 0.5) * grid.res
        goals = grid.free & (np.linalg.norm(centres - np.asarray(tgt["xy"]), axis=-1) <= tgt["radius"] - 0.2)
    here = sim.d.qpos[:2].copy()
    found = grid.path(here, goals) if fixture.any() and goals.any() else None
    row["nav"] = dict(fixture_cells=int(fixture.sum()), goal_cells=int(goals.sum()), corners=None if found is None else len(found[0]),
                      length=None if found is None else round(found[1], 2))  # fmt: skip
    if found is None:
        row["why"] = "no way to the fixture"
        return
    V = body_points(sim, subtree(m, view.target_id))
    centre = 0.5 * (V.min(0) + V.max(0))[:2]
    with phase("walk"):
        prev = here
        for q, p in enumerate(found[0]):  # each corner on the heading it is reached on, the last one facing the fixture
            v = centre - p if q == len(found[0]) - 1 else p - prev
            bel.walk(p, float(np.arctan2(v[1], v[0])), face="travel", speed=K["long_walk_speed"])
            prev = p


def surface_below(sim, xy, own: set, top: float = 1.6) -> float | None:
    """The height of the first solid thing under the point `xy`, seen from `top`, that is not one of `own`."""
    m, d = sim.m, sim.d
    o, vec, gid = np.r_[xy, top], np.array([0.0, 0.0, -1.0]), np.zeros(1, np.int32)
    for _ in range(12):
        dist = mujoco.mj_ray(m, d, o, vec, None, 1, -1, gid)
        if dist < 0:
            return None
        g = int(gid[0])
        if solid(m, g) and int(m.geom_bodyid[g]) not in own:
            return float(o[2] - dist)
        o[2] -= dist + 1e-3
    return None


def own_bodies(env) -> set:
    """The robot's and the target's bodies (the receptacle search leaves them out)."""
    m = env.sim.m
    return subtree(m, m.body("pelvis").id) | subtree(m, m.body(env.info["target"]).id)


def receptacle(env):
    """Where the task wants the object, from its spec: (centre xy, the height the object is let go over, the points of
    the receptacle's body or None). A basin or a basket is entered from its rim; a region on a surface from just above it."""
    sim = env.sim
    m, t = sim.m, env.spec["target"]
    reg = t.get("region") or {}
    V = body_points(sim, subtree(m, m.body(t["body"]).id)) if t.get("body") else None
    if reg.get("corners"):  # several regions count (the desk's two robot-side corners): the one nearest the object
        C = np.array(reg["corners"], float)[:, :2]
        centre = C[np.argmin(np.linalg.norm(C - sim.d.xpos[m.body(env.info["target"]).id][:2], axis=1))]
    elif reg.get("centre") is not None:
        centre = np.array(reg["centre"][:2], float)
    elif V is not None:
        centre = 0.5 * (V.min(0) + V.max(0))[:2]
    else:
        return None
    if reg.get("z"):
        rim = float(reg["z"][1])
    elif V is not None and reg.get("centre") is None:
        rim = float(V[:, 2].max())
    else:
        rim = surface_below(sim, centre, own_bodies(env))
    return (centre, rim, V) if rim is not None else None


def place(env, S) -> None:
    """After a pick: carry the object to the task's receptacle, lower its foot to the rim, let go, take the hand away.
    The motions and parameters are skill_simple.place's; what differs is where things are read from: the object's foot
    in the hand from the simulator after the lift (not from the look before it), the receptacle from the task's spec,
    and the robot turns to face the receptacle when it is more than 60 deg off its heading. With the stance override
    `stay` there is no second walk: the object is let go from the pick's stance (the pick's own d and t then decide
    whether both the object and the receptacle are within reach)."""
    sim, mo, bel, K, row, view, phase = env.sim, env.mo, env.bel, env.K, env.row, env.view, env.clock.phase
    m, side = sim.m, PL.SIDE
    rec = receptacle(env)
    if rec is None:
        row["why"] = "no receptacle in the spec"
        return
    centre, rim, V = rec
    pb, yb = bel.pelvis()
    palm, Rm = bel.palm(side)
    O = body_points(sim, subtree(m, view.target_id))
    foot = Rm.T @ (
        np.r_[0.5 * (O.min(0) + O.max(0))[:2], O[:, 2].min()] - palm
    )  # the object's foot in the palm's frame
    bearing = float(np.arctan2(*(centre - pb[:2])[::-1]))
    st = env.stance or {}
    stay = bool(st.get("stay"))  # a stance choice: let go from where the robot stood for the pick, no second walk
    turned = abs(R.wrap(bearing - yb)) > np.radians(60) and not stay
    yaw2 = bearing if turned else yb
    ahead2 = np.array([np.cos(yaw2), np.sin(yaw2)])
    left2 = np.array([-ahead2[1], ahead2[0]])
    Rr = PL.rot_z(yaw2 - yb) @ Rm  # the same hand attitude seen from the robot
    over = np.r_[centre, rim + K["over_rim"] + K["over_extra"]] - Rr @ foot
    release = np.r_[centre, rim - K["seat"]] - Rr @ foot
    stance2 = (
        release[:2]
        - PL.rot_z(yaw2)[:2, :2] @ [K["stand"] + st.get("d2", 0.0), K["place_side"]]
        + st.get("t2", 0.0) * left2
    )
    limit = view.edge(ahead2, float(stance2 @ left2))
    if V is not None:
        band = np.abs(V[:, :2] @ left2 - stance2 @ left2) < 0.3
        near = float((V[band][:, :2] @ ahead2).min()) if band.any() else None
        limit = near if limit is None else limit if near is None else min(limit, near)
    if limit is not None:
        stance2 = stance2 - max(0.0, stance2 @ ahead2 - (limit - K["table_gap"])) * ahead2
    if stay:
        stance2 = pb[:2].copy()
    row["place"] = dict(centre=centre.round(3).tolist(), rim=round(rim, 3), turned=bool(turned), stance2=stance2.round(3).tolist(),
                        move=round(float(np.linalg.norm(stance2 - pb[:2])), 2))  # fmt: skip
    env.mark("before_carry")
    walks = turned or np.linalg.norm(stance2 - pb[:2]) > 0.05
    if not K["carry_skip"] or (
        walks and K["carry_skip"] != "always"
    ):  # "always": walk with the hand where the lift left it
        with phase("carry"):
            carry = pb[:2] + PL.rot_z(yb)[:2, :2] @ [K["carry_x"], K["carry_y"]]
            bel.reach(side, np.r_[carry, max(palm[2], over[2])], Rm, speed=K["carry_speed"])
    if walks:
        with phase("walk2"):
            bel.walk(
                stance2, yaw2, face="travel" if turned else "fixed", speed=K["long_walk_speed"] if turned else None
            )
    with phase("over"):
        bel.reach(side, over, Rr, speed=K["over_speed"])
    if np.linalg.norm(release - over) > 1e-3:  # over_extra = 0, seat = -over_rim: let go from the place over the rim
        with phase("release"):
            bel.reach(side, release, Rr, speed=K["release_speed"])
    row["release_cm"] = float(100 * np.linalg.norm(mo.jaw_centre(side)[0] - bel.to_sim(release)))  # record
    env.mark("before_open")
    with phase("open"):
        start = np.array(sim.fingers_cmd.get(side, sim.grip_target[7 * ("left", "right").index(side) :][:7]), float)
        for u in np.linspace(0, 1, max(1, int(K["open_s"] / 0.02))):
            sim.fingers_cmd[side] = start + u * (PL.WIDE - start)
            S.one_step(mo, K)
        mo.hold(K["open_hold"])
    with phase("away"):  # off the object against the palm's normal, then back to the carrying place: the tasks that
        p, Rh = bel.palm(side)  # test the release want the hand 0.25 m from the object
        if K["back_off"] > 0:
            bel.reach(side, p - K["back_off"] * Rh[:, 1], Rh, speed=K["back_speed"])
        pb, yb = bel.pelvis()
        home = pb[:2] + PL.rot_z(yb)[:2, :2] @ [K["carry_x"], K["carry_y"]]
        bel.reach(side, np.r_[home, bel.palm(side)[0][2] + K["away_up"]], Rh, speed=K["away_speed"])


def pick_place(env, S, facade) -> None:
    """A pick_place, receptacle or composite task: the pick of the skill's `episode` (it is given `facade`: what a skill
    package gets of the episode), then `place` if the object is in the hand."""
    S.episode(facade)
    snap = env.snapshot()
    if "why" in env.row or not snap["held"]:
        return
    place(env, S)
