"""Where the robot passes through scenery it does not collide with (user 2026-10-19: the videos show a hand through
a wall and a head inside a bookcase). The walls (`adt_shell_*`) collide with nothing, and furniture collides with the
robot only below its lowest surface (adt_to_scene.furniture_solid), so the simulation lets an arm, a hand or the
head into them; the videos draw the visual meshes, where it shows.

Per sampled tick of a round's 50 Hz trace (scene_round --npz), rays against the scenery's visual meshes (static
group-1 geoms: walls and furniture), the robot's and free objects' geoms moved out of the ray's groups:
* from each robot body's origin to sampled vertices of its visual meshes: a hit short of the vertex means that part
  of the robot's surface is past a scenery surface, by the rest of the ray;
* from each robot body's parent's origin to the body's origin: a link wholly through a thin panel.
A hit within 3 cm of the floor is the soles on the floor and is skipped. Reports per round the deepest event per
robot part (head and torso, arms, hands and fingers, legs) and the scenery body it went into, and how many sampled
ticks had a part at least --depth deep.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/scene_rsi_20261016/clip_audit.py <round base without suffix> ... [--every 5] [--depth 0.02]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import report_media as RM  # noqa: E402
import visualize as V  # noqa: E402

PER_GEOM = 48  # vertices sampled per robot visual mesh, plus the 6 extremes along its axes
FLOOR_Z = 0.03
PARTS = (("head/torso", ("torso", "waist", "pelvis")), ("hand", ("wrist", "dex1")), ("arm", ("shoulder", "elbow")),
         ("leg", ("hip", "knee", "ankle")))  # fmt: skip


def part(body: str) -> str:
    return next((p for p, keys in PARTS if any(k in body for k in keys)), "other")


class Auditor:
    def __init__(self, sc: V.Scene, seed: int = 0):
        self.sc, m = sc, sc.m
        self.robot = sorted(sc.robot_bodies)
        free = {b for b in range(m.nbody) if m.body_jntnum[m.body_rootid[b]] > 0}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in free or m.geom_bodyid[g] in sc.robot_bodies:
                m.geom_group[g] = 5  # out of the rays' groups
        self.mask = np.array([0, 1, 0, 0, 0, 0], np.uint8)  # the scenery's visual meshes
        rng = np.random.default_rng(seed)
        self.samples: dict[int, list[tuple[int, np.ndarray]]] = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b not in sc.robot_bodies or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] != 0:
                continue
            mid = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pick = rng.choice(len(v), size=min(PER_GEOM, len(v)), replace=False)
            ext = [int(np.argmin(v[:, k])) for k in range(3)] + [int(np.argmax(v[:, k])) for k in range(3)]
            self.samples.setdefault(b, []).append((g, v[np.unique(np.r_[pick, ext])].astype(np.float64)))
        self.geomid = np.zeros(1, np.int32)

    def ray(self, a: np.ndarray, b: np.ndarray) -> tuple[float, int] | None:
        """(how far past the first scenery hit `b` lies, hit geom) on the segment a -> b, or None."""
        v = b - a
        n = float(np.linalg.norm(v))
        if n < 1e-4:
            return None
        d = mujoco.mj_ray(self.sc.m, self.sc.d, a, v / n, self.mask, 1, -1, self.geomid)
        if d < 0 or d >= n:
            return None
        hit = a + v / n * d
        if hit[2] < FLOOR_Z:
            return None
        return n - d, int(self.geomid[0])

    def tick(self, r, i: int) -> list[tuple[float, str, int]]:
        """(depth, robot body, scenery geom) of every ray past scenery at tick i."""
        self.sc.set(r, i)
        m, d = self.sc.m, self.sc.d
        out = []
        for b in self.robot:
            o = d.xpos[b].copy()
            p = int(m.body_parentid[b])
            if p in self.sc.robot_bodies:
                hit = self.ray(d.xpos[p].copy(), o)
                if hit:
                    out.append((hit[0], m.body(b).name, hit[1]))
            for g, verts in self.samples.get(b, ()):
                world = d.geom_xpos[g] + verts @ d.geom_xmat[g].reshape(3, 3).T
                for w in world:
                    hit = self.ray(o, w)
                    if hit:
                        out.append((hit[0], m.body(b).name, hit[1]))
        return out

    def scenery(self, g: int) -> str:
        m = self.sc.m
        return m.geom(g).name or m.body(int(m.geom_bodyid[g])).name

    def round(self, r, every: int, depth: float) -> dict:
        worst: dict[str, dict] = {}
        ticks_over = 0
        for i in range(0, r.n, every):
            ev = self.tick(r, i)
            if any(e[0] >= depth for e in ev):
                ticks_over += 1
            for dep, body, g in ev:
                k = part(body)
                if dep > worst.get(k, {}).get("depth_m", 0.0):
                    worst[k] = dict(
                        depth_m=round(dep, 3), body=body, into=self.scenery(g), t=round(float(r.z["t"][i]), 2)
                    )
        n = len(range(0, r.n, every))
        return dict(label=r.label, ok=bool(r.rec["ok"]), ticks=n, ticks_over=ticks_over, worst=worst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rounds", nargs="+", type=Path)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--depth", type=float, default=0.02)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    aud = Auditor(V.Scene())
    rows = []
    for base in a.rounds:
        row = aud.round(RM.Rd(base), a.every, a.depth)
        row["path"] = str(base)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
