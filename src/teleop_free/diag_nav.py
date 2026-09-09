"""Offline check of the map and the planner, with no sim running.

Everything the navigator does is a pure function of scene_meta, so the whole
thing can be validated before it ever drives a robot — which is the point: the
failures it exists to fix (walking into a table, walking away from an
unreachable goal) are exactly the ones that are expensive to reproduce live.

    python src/teleop_free/diag_nav.py [--ascii]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import navigator as nav  # noqa: E402
import paths

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"
WANT = 0.45  # pelvis-to-object distance the grasp needs
BAND = 0.08


def rollout(planner: nav.Navigator, start, step=nav.STEP_SIZE, limit=80, stop=0.08):
    """Walk the field greedily, as the live follower will. Returns the path.

    The step shrinks as the goal nears, so the path actually arrives instead of
    stopping one full stride short — which is what the live follower does too,
    since a 0.45 m stride into a goal 0.2 m away overshoots it.
    """
    path = [np.asarray(start, dtype=float)[:2]]
    for _ in range(limit):
        d = planner.distance_at(path[-1])
        if d < stop:
            break
        wp = planner.waypoint(path[-1], min(step, max(d, stop)))
        if wp is None:
            break
        path.append(wp)

    return np.array(path)


def report(label, planner, occ, start, goal_xy) -> bool:
    ok = planner.reachable(start)
    print(f"\n--- {label} ---")
    print(f"  goal {np.round(goal_xy, 2)}  snap {planner.snap:.2f} m  reachable {ok}")
    if not ok:
        print("  FAIL: field is flat at the start cell — the follower would walk off in a straight line")
        return False

    path = rollout(planner, start)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    cells = [tuple(occ.clip_cell(occ.cell_of(p))) for p in path]
    clear = [occ.clearance[c] for c in cells]
    # path[0] is where the robot already is — if that is illegal it is a fact
    # about the reset, not about the plan.
    hits = sum(occ.occupied[c] for c in cells[1:])
    print(
        f"  geodesic {planner.distance_at(start):.2f} m, path {len(path)} "
        f"waypoints / {seg:.2f} m, min clearance {min(clear):.2f} m"
    )
    print(f"  end {np.round(path[-1], 2)}, {np.linalg.norm(path[-1] - goal_xy):.2f} m from the goal centre")
    # Clearance below the body radius is legal only while escaping: a robot
    # that starts at a table is inside the inflation by definition, so the
    # hard property to assert is that no waypoint is inside an actual
    # obstacle.
    if hits:
        print(f"  FAIL: {hits} waypoints land inside an obstacle")
        return False
    if min(clear[2:] or clear) < occ.body_radius - 1e-6:
        print("  FAIL: path re-enters the inflated region after escaping")
        return False

    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ascii", action="store_true")
    ap.add_argument("--radius", type=float, default=nav.BODY_RADIUS)
    args = ap.parse_args()

    meta = json.load(open(SCENE_META))
    occ, planner = nav.build(meta, body_radius=args.radius)
    print(f"map {occ.nx}x{occ.ny} cells @ {occ.res} m, body radius {occ.body_radius} m")
    print(
        f"  occupied {occ.occupied.sum()} cells, free after inflation {occ.free.sum()} ({100 * occ.free.mean():.0f}%)"
    )

    start = np.array(meta["robot_start"]["xy"], dtype=float)
    orange = np.array(meta["objects"][meta["target_object"]]["init_pos"][:2])
    plate = np.array(meta["objects"][meta["goal_object"]]["init_pos"][:2])

    ok = True
    # 1. the approach that has been failing 40% of the time
    planner.set_goal_ring(orange, WANT - BAND, WANT + BAND, here_xy=start)
    ok &= report("approach A (start -> beside the orange)", planner, occ, start, orange)
    stand_a = rollout(planner, start)[-1]

    # 2. the carry leg, starting from wherever leg 1 ended
    planner.set_goal_ring(plate, WANT - BAND, WANT + BAND, here_xy=stand_a)
    ok &= report("approach B (beside the orange -> beside the plate)", planner, occ, stand_a, plate)

    # 3. the case the orbit had no answer for: standing inside the inflation
    # Pressed against table A but not inside it — 0.09 m of clearance against
    # a 0.30 m inflation, i.e. exactly where every manipulation stage ends.
    inside = np.array([0.62, 0.62])
    planner.set_goal_ring(plate, WANT - BAND, WANT + BAND, here_xy=inside)
    ok &= report("recovery (pressed against table A -> beside the plate)", planner, occ, inside, plate)

    # 4. a goal that genuinely cannot be stood at, to prove it is detected
    planner.set_goal_ring(np.array([0.0, 0.0]), 0.05, 0.10, here_xy=start)
    print("\n--- unreachable goal (table centre) ---")
    print(
        f"  snap {planner.snap:.2f} m, reachable {planner.reachable(start)} "
        f"(the snap is the point: it widens until something is legal)"
    )

    if args.ascii:
        planner.set_goal_ring(orange, WANT - BAND, WANT + BAND, here_xy=start)
        path = rollout(planner, start)
        marks = {"R": start, "o": orange, "p": plate}
        marks.update({"+": p for p in path[1:-1]})
        marks["G"] = path[-1]
        print("\n" + nav.ascii_map(occ, marks))

    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
