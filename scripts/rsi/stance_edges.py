#!/usr/bin/env python3
"""TODO 27.3.1: which edges of each surface the robot can stand at, by the scene runner's own stance search.

The scene file's `approach_edges` (scripts/adt_to_scene.py, 2026-10-03) come from the SONIC-era rule: 3 of 5 points
along an edge need 0.26 m of clearance at the stand and 0.6 m of run-up behind it. The ADT coffee table has none, yet
the pick_single loop picked the TV remote off it 10/10 (plan §9.1), and that rule's "no edge of its surface has room
to stand" is the feasibility checker's largest bucket-3 class (610 of 2830 template tasks). This script keeps the same
five points per edge and the same 3-of-5 vote but asks of each point what the loop's final pick_single skill
(incumbent 0109bde905a8, `skills/pick.py` `plan_approach`) asks of a stance:

- an object at the point, DEPTHS_M in from the edge; the stance STAND_AHEAD_M behind it along the edge's inward
  normal, STAND_SIDE_M (else each of STAND_SIDE_RETRY_M) toward either hand;
- the stance at least STANCE_CLEAR_M from the static floor objects the scene map leaves out (`floor_boxes`) and
  OTHER_EDGE_CLEAR_M from every occupied cell but the surface's own furniture's tops (`other_clearance`);
- a route on the skill's map (body radius 0.35 m, plus those floor objects) from the surface's room's spawn region to
  the free cell nearest the stance (`navigate.path` walks there, then one straight leg in), and that last leg itself
  OTHER_EDGE_CLEAR_M clear of all but the own furniture (the runner does not check the leg; with furniture collision
  bodies, D3, a leg across another piece of furniture is a collision).

A point passes when any depth, lateral and hand passes. The result is an overlay (the scene file stays as it is: the
inner loop's sandboxes read it and D6 freezes the scene), read by `scripts/scene_feasibility.py --approach-edges` and
`scripts/scene_discover_tasks.py --approach-edges`. `--loop-root` also checks the overlay against every walk the
pick_single loop made: each edge the robot reached a stance at must be listed. `--walk K/N` (after the overlay is
written; shard K of N) walks the robot in the sim, on the skill's map, from a spawn 1-3 m away in the surface's room
to the middle passing stance of every edge the overlay adds, and holds there HOLD_S: one JSON per edge in
WALK_DIR. ADT-derived output stays under outputs/ (Demo 1 decision 17). Sim venv; no model calls.

    $PY scripts/rsi/stance_edges.py \\
        [--scene outputs/scene_discovery/adt_apartment_mars/scene_meta.json] \\
        [--out outputs/scene_rsi_20261016/capability/stance/adt_apartment_mars.json] \\
        [--loop-root /workspace/code/rsi-scene/pick-a]
    CUDA_VISIBLE_DEVICES=0 $PY scripts/rsi/stance_edges.py --walk 0/2   # and 1/2 on GPU 1
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.primitives.navigator import Navigator  # noqa: E402
from locomani.primitives.navigator import build as build_navigation  # noqa: E402
from locomani.tasks.scene.start import _inside, spawn_polygons  # noqa: E402
from locomani.tasks.scene.world import BODY_RADIUS_M  # noqa: E402

# the incumbent skill's stance constants (pick_single loop, skills/pick.py at 0109bde905a8)
STAND_AHEAD_M = 0.40
STAND_SIDE_M = 0.12
STAND_SIDE_RETRY_M = (0.22, 0.30)
STANCE_CLEAR_M = 0.12
OTHER_EDGE_CLEAR_M = 0.25
FLOOR_BOTTOM_M = 0.30
FLOOR_MAPPED_MAX = 0.5
NORMALS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}
# this script's own choices
DEPTHS_M = (0.05, 0.15)  # an object this far in from the edge (the skill reaches up to 0.30)
POINTS = np.linspace(-0.8, 0.8, 5)  # along the edge, x half its length: the 10-03 rule's points ...
VOTE = 3  # ... and its vote
SPAWN_STEP_M = 0.25  # spawn-region sampling grid
DEFAULT_SCENE = "outputs/scene_discovery/adt_apartment_mars/scene_meta.json"
DEFAULT_OUT = "outputs/scene_rsi_20261016/capability/stance/adt_apartment_mars.json"
WALK_DIR = "outputs/scene_rsi_20261016/capability/stance/walk"
SPAWN_DISTANCE_M = (1.0, 3.0)  # tasks/scene/start.py
HOLD_S = 3.0


class StanceMap:
    """The skill's map (scene map + floor objects), its navigator, per-room route fields, per-furniture clearance."""

    def __init__(self, meta: dict):
        self.meta = meta
        occ, _ = build_navigation(meta, body_radius=BODY_RADIUS_M)
        self.boxes = floor_boxes(meta, occ)
        for c, he, yaw in self.boxes:
            occ.add_box(c, he, yaw)
        occ.recompute()
        self.occ, self.nav = occ, Navigator(occ)
        self._fields: dict = {}
        self._clear: dict = {}

    def field(self, room: str | None) -> np.ndarray | None:
        """Geodesic distance to the room's spawn region on the free cells (inf = unreachable). Routes are symmetric
        on a static map, so one march per room answers 'can a spawn in this room walk to that cell'."""
        if room not in self._fields:
            occ, mask = self.occ, np.zeros_like(self.occ.free)
            for poly in spawn_polygons(self.meta, room):
                p = np.asarray(poly, float)
                lo, hi = p.min(axis=0), p.max(axis=0)
                for x in np.arange(lo[0], hi[0] + 1e-9, SPAWN_STEP_M):
                    for y in np.arange(lo[1], hi[1] + 1e-9, SPAWN_STEP_M):
                        if _inside(poly, (x, y)) and occ.is_free((x, y)):
                            c = occ.cell_of((x, y))
                            mask[c[0], c[1]] = True
            ok = self.nav.set_goal_mask(mask)
            self._fields[room] = np.where(self.nav.dist < 1e5, self.nav.dist, np.inf) if ok else None
        return self._fields[room]

    def other_clearance(self, furniture: str) -> np.ndarray:
        """Distance to the nearest occupied cell that is not a top of `furniture` (skills/pick.py other_clearance)."""
        if furniture not in self._clear:
            own = copy.copy(self.occ)
            own.occupied = np.zeros_like(self.occ.occupied)
            for k, s in self.meta["surfaces"].items():
                if k.split("__")[0] == furniture and s.get("half_extent") is not None:
                    own.add_box(s["xy"], s["half_extent"], s.get("yaw", 0.0))
            self._clear[furniture] = distance_transform_edt(~(self.occ.occupied & ~own.occupied), sampling=self.occ.res)
        return self._clear[furniture]

    def at(self, grid: np.ndarray, xy) -> float:
        c = self.occ.clip_cell(self.occ.cell_of(xy))
        return float(grid[c[0], c[1]])

    def stance_ok(self, key: str, stance) -> str:
        """'' when the skill would take this stance from a spawn in the surface's room, else why not."""
        surf = self.meta["surfaces"][key]
        if floor_clearance(self.boxes, stance) < STANCE_CLEAR_M:
            return "floor object"
        other = self.other_clearance(key.split("__")[0])
        if self.at(other, stance) < OTHER_EDGE_CLEAR_M:
            return "other furniture"
        field = self.field(surf.get("room"))
        if field is None:
            return "no spawn region"
        occ = self.occ
        entry = occ.xy_of(occ.nearest_free(stance)) if not occ.is_free(stance) else np.asarray(stance, float)
        if not np.isfinite(self.at(field, entry)):
            return "no route"
        n = max(2, int(np.ceil(np.linalg.norm(entry - stance) / (occ.res / 2))) + 1)
        if any(
            self.at(other, entry + u * (np.asarray(stance) - entry)) < OTHER_EDGE_CLEAR_M for u in np.linspace(0, 1, n)
        ):
            return "last leg"
        return ""


def floor_boxes(meta: dict, occ) -> list[tuple]:
    """skills/pick.py floor_boxes: static floor objects (bottom below FLOOR_BOTTOM_M) the scene map mostly misses."""
    out = []
    for k, o in meta["objects"].items():
        if o.get("free") or o.get("surface") is not None or k.startswith("Apartment"):
            continue
        p, he = np.asarray(o["init_pos"], float), np.asarray(o["box"]["half_extent"], float)
        if p[2] - he[2] > FLOOR_BOTTOM_M:
            continue
        yaw = R.yaw_of(np.asarray(o["init_quat"], float))
        Ro = R.rot_z(yaw)[:2, :2]
        g = np.linspace(-1.0, 1.0, 9)
        cells = [occ.cell_of(p[:2] + Ro @ (he[:2] * [u, v])) for u in g for v in g]
        if np.mean([not occ.in_bounds(c) or occ.occupied[c[0], c[1]] for c in cells]) < FLOOR_MAPPED_MAX:
            out.append((p[:2], he[:2], yaw))
    return out


def floor_clearance(boxes, xy) -> float:
    d = np.inf
    for c, he, yaw in boxes:
        q = np.maximum(np.abs(R.rot_z(yaw)[:2, :2].T @ (np.asarray(xy, float)[:2] - c)) - he, 0.0)
        d = min(d, float(np.linalg.norm(q)))
    return d


def edge_points(surf: dict, edge: str, depth: float) -> list[tuple[np.ndarray, float]]:
    """(object xy, robot yaw into the surface) at POINTS along `edge`, `depth` in from it."""
    Rs = R.rot_z(float(surf.get("yaw", 0.0)))[:2, :2]
    half = np.asarray(surf["half_extent"], float)
    n = np.asarray(NORMALS[edge])
    along = np.array([-n[1], n[0]])
    ax = 0 if "x" in edge else 1
    d = min(depth, 2 * half[ax] - 0.02)
    w = Rs @ n
    face = float(np.arctan2(-w[1], -w[0]))
    out = []
    for t in POINTS * half[1 - ax]:
        local = n * (half[ax] - d) + along * t
        out.append((np.asarray(surf["xy"], float) + Rs @ local, face))
    return out


def surface_edges(sm: StanceMap, key: str) -> dict:
    """Per edge: the points that pass, why the others do not (the refusal of their last candidate), and the passing
    stance [x, y, yaw] of the point nearest the edge's middle."""
    surf = sm.meta["surfaces"][key]
    out = {}
    for edge in NORMALS:
        passed, why, stances = 0, collections.Counter(), {}
        for i in range(len(POINTS)):
            last, hit = "no candidate", None
            for depth in DEPTHS_M:
                obj, face = edge_points(surf, edge, depth)[i]
                for lateral in (STAND_SIDE_M, *STAND_SIDE_RETRY_M):
                    for sign in (-1.0, 1.0):  # right hand, left hand
                        stance = obj - (R.rot_z(face)[:2, :2] @ np.array([STAND_AHEAD_M, sign * lateral]))
                        last = sm.stance_ok(key, stance)
                        if not last:
                            hit = [round(float(stance[0]), 3), round(float(stance[1]), 3), round(face, 4)]
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                passed += 1
                stances[i] = hit
            else:
                why[last] += 1
        mid = min(stances, key=lambda i: abs(i - len(POINTS) // 2)) if stances else None
        out[edge] = {"points": passed, "why": dict(why), "stance": stances.get(mid)}
    return out


def loop_walks(loop_root: Path, meta: dict) -> dict:
    """(surface, edge) -> [walks that reached a stance without a fall, walks that did not] over every round of the
    pick_single loop (diagnosis, gate, sealed)."""
    from locomani.tasks.scene.world import Task

    fam = json.loads((loop_root / "scripts/rsi/families/pick_single.json").read_text())
    rows = [json.loads(line) for line in (REPO / fam["tasks_file"]).read_text().splitlines() if line.strip()]
    obj = {str(r["id"]): Task.from_row(r, meta).objects[0] for r in rows if str(r["id"]) in fam["tasks"]}
    tally: dict = collections.defaultdict(lambda: [0, 0])
    for f in (loop_root / "outputs").rglob("*_s*.json"):
        try:
            r = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(r, dict) or r.get("task") not in obj:
            continue
        walk = next(
            (s for s in r.get("steps") or () if s.get("step") == "walk" and s.get("face_deg") is not None), None
        )
        if walk is None:
            continue
        s_key = meta["objects"][obj[r["task"]]]["surface"]
        tally[(s_key, edge_of(meta["surfaces"][s_key], np.radians(walk["face_deg"])))][0 if walk["ok"] else 1] += 1
    return dict(tally)


def edge_of(surf: dict, face: float) -> str:
    """The edge whose inward direction is nearest the robot's yaw `face`."""
    Rs = R.rot_z(float(surf.get("yaw", 0.0)))[:2, :2]
    best = max(NORMALS, key=lambda e: float(-(Rs @ np.asarray(NORMALS[e])) @ np.array([np.cos(face), np.sin(face)])))
    return best


def walk_cases(report: dict) -> list[dict]:
    """One case per edge the overlay lists and the scene file does not."""
    return [
        dict(surface=k, edge=e, room=v["room"], stance=v["edges"][e]["stance"])
        for k, v in sorted(report["surfaces"].items())
        for e in v["approach_edges"]
        if e not in v["old"]
    ]


def walk(scene: Path, report: dict, shard: str) -> None:
    """Walk to each case's stance from a spawn in its room (a fresh reset each), hold, record (`--walk`)."""
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
    from locomani.tasks.scene.navigate import path as nav_path
    from locomani.tasks.scene.world import SceneWorld

    k, n = (int(v) for v in shard.split("/"))
    cases = walk_cases(report)[k::n]
    out_dir = REPO / WALK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    world = SceneWorld(scene.parent, R.Policy("cuda", ScaleBfmBackendConfig()))
    sm, m = StanceMap(world.meta), world.motor
    for c in cases:
        f = out_dir / f"{c['surface']}_{c['edge']}.json"
        if f.exists():
            continue
        t0, rng = time.time(), np.random.default_rng(zlib.crc32(f.name.encode()))
        goal, face = np.asarray(c["stance"][:2], float), float(c["stance"][2])
        polys = spawn_polygons(world.meta, c["room"])
        pts = np.concatenate([np.asarray(p, float) for p in polys])
        start = corners = None
        for _ in range(2000):
            xy = rng.uniform(pts.min(axis=0), pts.max(axis=0))
            d = float(np.linalg.norm(xy - goal))
            if (
                not any(_inside(p, xy) for p in polys)
                or not sm.occ.is_free(xy)
                or not SPAWN_DISTANCE_M[0] <= d <= SPAWN_DISTANCE_M[1]
            ):
                continue
            corners = nav_path(sm.occ, sm.nav, xy, goal)
            if corners is not None:
                start = xy
                break
        rec = dict(c, start=None if start is None else np.round(start, 3).tolist())
        if start is None:
            rec.update(ok=False, why="no spawn 1-3 m away with a route", fell=False)
        else:
            m.reset(start, float(rng.uniform(-np.pi, np.pi)))
            m.hold(1.0)
            out = None
            for i, corner in enumerate(corners[1:], start=1):
                out = m.walk_to(corner, face if i == len(corners) - 1 else None)
                if not out.ok:
                    break
            p, q = m.pelvis()
            stop = float(np.linalg.norm(np.asarray(p[:2]) - goal))
            yaw_err = float(np.degrees(R.wrap(R.yaw_of(q) - face)))
            fell_walk = m.fell_at is not None
            if not fell_walk:
                m.hold(HOLD_S)
            rec.update(ok=bool(out.ok and m.fell_at is None), why=out.why, fell=m.fell_at is not None,
                       fell_walking=fell_walk, legs=len(corners) - 1, stop_err_cm=round(100 * stop, 1),
                       yaw_err_deg=round(yaw_err, 1), sim_s=round(m.t, 1))  # fmt: skip
        rec["wall_s"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(rec, indent=1, default=float))
        print(f"[walk] {f.stem}: ok {rec['ok']} {rec.get('why', '')} stop {rec.get('stop_err_cm')} cm "
              f"fell {rec['fell']} ({rec['wall_s']} s)", flush=True)  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--loop-root", help="a pick_single loop sandbox: check the overlay against its walks")
    ap.add_argument("--walk", metavar="K/N", help="walk-check shard K of N of the overlay at --out (sim, GPU)")
    args = ap.parse_args(argv)
    t0 = time.time()
    scene = REPO / args.scene if not Path(args.scene).is_absolute() else Path(args.scene)
    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    if args.walk:
        walk(scene, json.loads(out.read_text()), args.walk)
        return 0
    meta = json.loads(scene.read_text())
    sm = StanceMap(meta)
    surfaces = {}
    for key in sorted(meta["surfaces"]):
        s = meta["surfaces"][key]
        if s.get("half_extent") is None:
            continue
        per = surface_edges(sm, key)
        surfaces[key] = {
            "approach_edges": [e for e in NORMALS if per[e]["points"] >= VOTE],
            "old": list(s.get("approach_edges") or []),
            "room": s.get("room"),
            "top": s.get("top"),
            "edges": per,
        }
    gained = sorted(k for k, v in surfaces.items() if v["approach_edges"] and not v["old"])
    lost = sorted(k for k, v in surfaces.items() if v["old"] and not v["approach_edges"])
    changed = sorted(k for k, v in surfaces.items() if set(v["approach_edges"]) != set(v["old"]))
    report = {
        "scene": str(args.scene),
        "rule": (
            f"an edge counts when >= {VOTE} of {len(POINTS)} points along it pass skills/pick.py's stance search "
            f"(object {list(DEPTHS_M)} m in; stance {STAND_AHEAD_M} m behind it, {[STAND_SIDE_M, *STAND_SIDE_RETRY_M]} m "
            f"toward either hand; >= {STANCE_CLEAR_M} m from floor objects, >= {OTHER_EDGE_CLEAR_M} m from other "
            f"furniture along the last leg; route at body radius {BODY_RADIUS_M} m from the room's spawn region)"
        ),
        "body_radius": BODY_RADIUS_M,
        "floor_boxes": len(sm.boxes),
        "surfaces_total": len(surfaces),
        "approachable_old": sum(bool(v["old"]) for v in surfaces.values()),
        "approachable_new": sum(bool(v["approach_edges"]) for v in surfaces.values()),
        "gained": gained,
        "lost": lost,
        "changed": len(changed),
        "surfaces": surfaces,
    }
    if args.loop_root:
        walks = loop_walks(Path(args.loop_root), meta)
        check = []
        for (s_key, edge), (ok, bad) in sorted(walks.items()):
            check.append(dict(surface=s_key, edge=edge, walked=ok, failed=bad,
                              listed_old=edge in surfaces[s_key]["old"],
                              listed_new=edge in surfaces[s_key]["approach_edges"]))  # fmt: skip
        report["loop_check"] = check
        report["loop_missed"] = [c for c in check if c["walked"] and not c["listed_new"]]
    report["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[stance] {len(surfaces)} surfaces: approachable {report['approachable_old']} -> "
          f"{report['approachable_new']} (gained {len(gained)}, lost {len(lost)}, changed {len(changed)}) "
          f"in {report['seconds']} s -> {out}")  # fmt: skip
    if args.loop_root:
        for c in report["loop_check"]:
            print(f"   loop {c['surface']:42s} {c['edge']}  walked {c['walked']:3d} failed {c['failed']:3d}  "
                  f"listed old {c['listed_old']!s:5s} new {c['listed_new']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
