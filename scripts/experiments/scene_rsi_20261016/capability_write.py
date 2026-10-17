"""TODO 27.3 (plan docs/handoff/scene-rsi-plan-20261016.md §9, D10): write the mars + ScaleBFM capability table,
configs/scene_discovery/capability_mars_scalebfm.yaml, from the phase-1 probes (capability_probe.py, cells under
outputs/scene_rsi_20261016/capability/) and the inner loops' sealed results (D10: every accepted skill writes its
sealed-pool numbers back). Same keys as the 10-03 SONIC table, so scene_discovery.feasibility reads either; the
numbers here are all measured on mars + ScaleBFM, or marked untested.

A reach cell counts when the robot stood, the jaw centre was within REACH_TOL_CM of the target after a 1.5 s hold and
the wrist within REACH_ATT_DEG of the asked attitude. A sweep value counts when >= 2 of its 3 jittered rounds did.
A pick_single task enters `grasp_settings` when its sealed success is >= BUCKET1_RATE (D8), else `grasp_failures`.
A put-down family's on(X, S) task (`--place-loop <sandbox>:<family>`, phase 2) enters
`place.on_surface_single.measured_tasks` the same way (else `failed_tasks`); the objects its rounds lifted and carried away in >= BUCKET1_RATE of them enter
`grasp_settings` where pick_single did not measure them, and its longest carry route among successes is
`carry.single_arm.route_measured_max`.

    $PY scripts/experiments/scene_rsi_20261016/capability_write.py --loop-root /workspace/code/rsi-scene/pick-a \
        [--place-loop /workspace/code/rsi-scene/r1-on-a:r1_on_single]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src")]

from capability_probe import EDGE_M, OUT, load  # noqa: E402

from locomani.scene_discovery import feasibility, scene_file  # noqa: E402
from locomani.scene_discovery.predicates import parse  # noqa: E402

DST = REPO / "configs" / "scene_discovery" / "capability_mars_scalebfm.yaml"
OLD = REPO / "configs" / "scene_discovery" / "capability.yaml"
ADT = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars"
REACH_TOL_CM = 2.0
REACH_ATT_DEG = 10.0
BUCKET1_RATE = 0.50
PROBE_SRC = "scripts/experiments/scene_rsi_20261016/capability_probe.py; outputs/scene_rsi_20261016/capability/"
PLAN_SRC = "docs/handoff/scene-rsi-plan-20261016.md §9"


def reach_ok(r: dict) -> bool:
    return bool(not r.get("fell") and not r.get("error") and r.get("jaw_err_after_hold_cm", 99) <= REACH_TOL_CM
                and r.get("attitude_after_hold_deg", 99) <= REACH_ATT_DEG)  # fmt: skip


def reach_table() -> dict:
    """Per side and wrist attitude: the heights some probed posture reached within the tolerance at a forward
    distance of 0.40-0.60 m, and per height the farthest forward reach (standing, level)."""
    rows = load("reach")
    out: dict = {}
    for side in ("right", "left"):
        for att in ("level", "down"):
            ok_z = collections.defaultdict(set)
            for r in rows.values():
                c = r["cell"]
                if c["side"] == side and c["att"] == att and reach_ok(r):
                    ok_z[c["z"]].add((c["pelvis"], c["x"]))
            heights = sorted(z for z, xs in ok_z.items() if any(0.40 <= x <= 0.60 for _, x in xs))
            band = _band_around(heights, 0.75 if att == "level" else 0.35)
            tested = sorted(
                {r["cell"]["z"] for r in rows.values() if r["cell"]["side"] == side and r["cell"]["att"] == att}
            )
            far = {}
            for r in rows.values():
                c = r["cell"]
                if c["side"] == side and c["att"] == att and c["pelvis"] is None and reach_ok(r):
                    far[c["z"]] = max(far.get(c["z"], 0.0), c["x"])
            out[f"{side}_{att}"] = dict(heights_reached=band, heights_tested=[tested[0], tested[-1]] if tested else None,
                                        forward_max_standing={f"{z:.2f}": far[z] for z in sorted(far)},
                                        cells=sum(1 for r in rows.values() if r["cell"]["side"] == side and r["cell"]["att"] == att),
                                        ok=sum(1 for r in rows.values() if r["cell"]["side"] == side and r["cell"]["att"] == att and reach_ok(r)))  # fmt: skip
    return out


def _band_around(heights: list[float], centre: float) -> list[float] | None:
    """The contiguous run of 0.1 m grid heights containing the one nearest `centre`."""
    if not heights:
        return None
    hs = sorted(heights)
    k = min(range(len(hs)), key=lambda i: abs(hs[i] - centre))
    lo = hi = k
    while lo > 0 and hs[lo] - hs[lo - 1] < 0.11:
        lo -= 1
    while hi < len(hs) - 1 and hs[hi + 1] - hs[hi] < 0.11:
        hi += 1
    return [hs[lo], hs[hi]]


def sweep(kind_rows: dict, key: str) -> dict:
    """{value: dict(n, lifted, carried, placed, fell)} for one grip sweep."""
    by: dict = {}
    for r in kind_rows.values():
        c = r["cell"]
        if c["sweep"] != key:
            continue
        v = c[key] if key != "top_down" else (c["top"], c["height"])
        row = by.setdefault(v, dict(n=0, lifted=0, carried=0, placed=0, fell=0, rise_cm=[], slip_cm=[]))
        row["n"] += 1
        row["lifted"] += bool(r.get("lifted"))
        row["carried"] += bool(r.get("carried"))
        row["placed"] += bool(r.get("placed"))
        row["fell"] += bool(r.get("fell"))
        row["rise_cm"].append(r.get("rise_cm"))
        row["slip_cm"].append(r.get("slip_hold_cm"))
    return dict(sorted(by.items()))


def passes(row: dict, what: str = "lifted") -> bool:
    return row[what] >= 2 and row["fell"] == 0


def frac(row: dict, what: str = "lifted") -> str:
    return f"{row[what]}/{row['n']}"


def sealed_rounds(sandbox: Path, family: str, rounds: Path | None = None) -> tuple[dict, Path, Path, str, float]:
    """(per task "ok/n", rounds dir, pool dir, version, rate): the family's final incumbent on its sealed pool, or
    `rounds` -- the same pool run again under a later skill catalog (a write-back's regression), which then decides:
    the table says what the catalog does now."""
    hidden = sandbox / "outputs" / "scene_rsi_20261016_hidden" / family
    if rounds is None:
        summ = json.loads((hidden / "sealed" / "summary.json").read_text())
        final = summ["final"]
        return (
            summ["versions"][final]["per_task"],
            hidden / "sealed" / final,
            hidden / "pool_sealed",
            final,
            summ["final_rate"],
        )
    per = collections.defaultdict(lambda: [0, 0])
    for f in sorted(rounds.glob("*_s*.json")):
        if f.name.startswith("_"):  # an older success rule's copy
            continue
        r = json.loads(f.read_text())
        per[r["task"]][0] += bool(r["ok"])
        per[r["task"]][1] += 1
    ok, n = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return (
        {t: f"{a}/{b}" for t, (a, b) in sorted(per.items())},
        rounds,
        hidden / "pool_sealed",
        f"catalog re-run ({rounds})",
        round(ok / n, 3) if n else 0.0,
    )


def held(sandbox: Path, family: str, checker, rounds: Path | None = None) -> tuple[list, list, dict]:
    """A pick family's sealed pool (`sealed_rounds`), per held(X) task: a measured grasp setting or a failure. A
    setting names its object and surface: the checker takes that measurement for that object where it rests, whatever
    its geometric proxies say (the coffee table has no approach edge in the scene file, yet the remote went 10/10). The
    inboard range and `stance_xy` come from the successful rounds."""
    per, rdir, pool, version, rate = sealed_rounds(sandbox, family, rounds)
    fam = json.loads((sandbox / "scripts/experiments/scene_rsi_20261016/families" / f"{family}.json").read_text())
    rows = {json.loads(line)["id"]: json.loads(line) for line in (REPO / fam["tasks_file"]).read_text().splitlines() if line.strip()}  # fmt: skip
    src = f"{PLAN_SRC} ({family}, sealed pool); <loop sandbox>/outputs/scene_rsi_20261016_hidden/{family}/sealed/summary.json"
    settings, failures, deepest, routes = [], [], 0.0, []
    for tid, frac_s in sorted(per.items()):
        x = next(p for p in map(parse, rows[tid]["goal"]) if p.name == "held").args[0]
        o = checker.objects[x]
        half, _ = checker.box(x)
        s_key = o["surface"]
        inboards, stances = [], []
        for f in sorted(rdir.glob(f"{tid}_s*.json")):
            r = json.loads(f.read_text())
            if not r.get("ok"):
                continue
            st = json.loads((pool / f.name).read_text())["start"]
            d = checker.edge_stand(s_key, np.asarray(st["placements"][x][0][:2], float))[1]
            if np.isfinite(d):
                inboards.append(float(d))
            walk = next((w for w in r.get("steps", []) if w.get("step") == "walk" and w.get("stance")), None)
            if walk is not None:
                stances.append([round(float(v), 3) for v in walk["stance"]])
                d = checker.route(np.asarray(st["robot_xy"], float), np.asarray(walk["stance"], float))
                if np.isfinite(d):
                    routes.append(float(d))
        deepest = max([deepest, *inboards])
        ok, n = map(int, frac_s.split("/"))
        entry = dict(name=f"{family}.{tid}", status="measured", objects=[x], surface=s_key,
                     setting=f"ADT apartment (mars): {x} on {s_key}, placement and yaw random on that surface, robot spawned 1-3 m away ({family} sealed pool); {version}",
                     success=f"{ok}/{n}", support_top=[round(float(checker.surfaces[s_key]["top"]), 3)] * 2,
                     object_width=[round(float(2 * min(half[:2])), 3)] * 2, object_height=[round(float(2 * half[2]), 3)] * 2,
                     mass_max=float((o.get("semantic") or {}).get("mass_kg", 0.0)),
                     inboard=[round(min(inboards), 3), round(max(inboards), 3)] if inboards else None,
                     stance_xy=stances[0] if stances else None, source=src)  # fmt: skip
        if n and ok / n >= BUCKET1_RATE:
            settings.append(entry)
        else:
            failures.append(dict(name=entry["name"], gap=f"grasp_single.{tid}", objects=[x], pilots=[tid],
                                 success=entry["success"], setting=entry["setting"], source=entry["source"]))  # fmt: skip
    summary = dict(final=version, rate=rate, per_task=per, deepest_success=deepest, route_max=max(routes, default=0.0))
    return settings, failures, summary


def put_on(
    loop_root: Path, family: str, checker, picked: set, rounds: Path | None = None
) -> tuple[list, list, list, float, dict]:
    """A put-down family's final incumbent on its sealed pool, per one-object on(X, S) task: (put-down settings,
    put-down failures, grasp settings for the objects it lifted and carried in >= BUCKET1_RATE of the rounds and
    `picked` lacks, the longest occupancy-map route from the pick stance to the put-down stance among successes,
    summary). A task with two objects gives no (object, surface) setting."""
    per, rdir, _pool, final, rate = sealed_rounds(loop_root, family, rounds)
    fam = json.loads((loop_root / "scripts/experiments/scene_rsi_20261016/families" / f"{family}.json").read_text())
    rows = {json.loads(line)["id"]: json.loads(line) for line in (REPO / fam["tasks_file"]).read_text().splitlines() if line.strip()}  # fmt: skip
    src = f"{PLAN_SRC.replace('§9', '§10')} ({family}, sealed pool); <loop sandbox>/outputs/scene_rsi_20261016_hidden/{family}/sealed/summary.json"
    settings, failures, grasps, longest = [], [], [], 0.0
    for tid, frac_s in sorted(per.items()):
        ons = [p for p in map(parse, rows[tid]["goal"]) if p.name == "on"]
        if len(ons) != 1:
            continue
        x, s_key = ons[0].args
        o = checker.objects[x]
        half, _ = checker.box(x)
        ok, n = map(int, frac_s.split("/"))
        lifted, routes = 0, []
        for f in sorted(rdir.glob(f"{tid}_s*.json")):
            r = json.loads(f.read_text())
            steps = r.get("steps") or []
            walk = next((w for w in steps if w.get("step") == "walk" and w.get("stance")), None)
            carry = next((w for w in steps if w.get("step") == "carry" and w.get("stance")), None)
            lifted += carry is not None  # the put-down carries only an object it holds
            if r.get("ok") and walk is not None and carry is not None:
                d = checker.route(np.asarray(walk["stance"], float), np.asarray(carry["stance"], float))
                if np.isfinite(d):
                    routes.append(float(d))
        longest = max([longest, *routes])
        setting = f"ADT apartment (mars): {x} from {o['surface']} onto {s_key}, placement and yaw random on its surface, robot spawned 1-3 m away ({family} sealed pool); {final}"
        entry = dict(name=f"{family}.{tid}", status="measured", objects=[x], surface=s_key, from_surface=o["surface"],
                     setting=setting, success=f"{ok}/{n}", support_top=[round(float(checker.surfaces[s_key]["top"]), 3)] * 2,
                     object_height=[round(float(2 * half[2]), 3)] * 2,
                     carry_route=[round(min(routes), 2), round(max(routes), 2)] if routes else None, source=src)  # fmt: skip
        if n and ok / n >= BUCKET1_RATE:
            settings.append(entry)
        else:
            failures.append(dict(name=entry["name"], gap=f"place_on_surface.{tid}", objects=[x], surface=s_key,
                                 pilots=[tid], success=entry["success"], setting=setting, source=src))  # fmt: skip
        if x not in picked and n and lifted / n >= BUCKET1_RATE:
            picked.add(x)
            grasps.append(dict(name=f"{family}.{tid}.pick", status="measured", objects=[x], surface=o["surface"],
                               setting=f"{setting}: lifted and carried away", success=f"{lifted}/{n}",
                               support_top=[round(float(checker.surfaces[o['surface']]["top"]), 3)] * 2,
                               object_width=[round(float(2 * min(half[:2])), 3)] * 2,
                               object_height=[round(float(2 * half[2]), 3)] * 2,
                               mass_max=float((o.get("semantic") or {}).get("mass_kg", 0.0)), inboard=None,
                               stance_xy=None, source=src))  # fmt: skip
    return settings, failures, grasps, longest, dict(final=final, rate=rate, per_task=per)


def into(
    loop_root: Path, family: str, checker, picked: set, rounds: Path | None = None
) -> tuple[list, list, list, float, dict]:
    """`put_on` for a put-in family's one-object in(X, C) tasks: (put-in settings, put-in failures, grasp settings for
    the objects lifted in >= BUCKET1_RATE of the rounds and `picked` lacks, the longest pick-to-drop route among
    successes, summary). A round lifted its object when it reached the carry or the move over the container (a drop
    next to the pick stance skips the carry)."""
    per, rdir, _pool, final, rate = sealed_rounds(loop_root, family, rounds)
    fam = json.loads((loop_root / "scripts/experiments/scene_rsi_20261016/families" / f"{family}.json").read_text())
    rows = {json.loads(line)["id"]: json.loads(line) for line in (REPO / fam["tasks_file"]).read_text().splitlines() if line.strip()}  # fmt: skip
    src = f"{PLAN_SRC.replace('§9', '§10')} ({family}, sealed pool); <loop sandbox>/outputs/scene_rsi_20261016_hidden/{family}/sealed/summary.json"
    settings, failures, grasps, longest = [], [], [], 0.0
    for tid, frac_s in sorted(per.items()):
        ins = [p for p in map(parse, rows[tid]["goal"]) if p.name == "in"]
        if len(ins) != 1:
            continue
        x, c = ins[0].args
        o = checker.objects[x]
        half, _ = checker.box(x)
        hc = checker.box(c)[0]
        wall = float(checker.objects[c]["box"].get("wall", 0.01))
        ok, n = map(int, frac_s.split("/"))
        lifted, routes = 0, []
        for f in sorted(rdir.glob(f"{tid}_s*.json")):
            r = json.loads(f.read_text())
            steps = r.get("steps") or []
            walk = next((w for w in steps if w.get("step") == "walk" and w.get("stance")), None)
            carry = next((w for w in steps if w.get("step") == "carry" and w.get("stance")), None)
            lifted += any(w.get("step") in ("carry", "over") for w in steps)
            if r.get("ok") and walk is not None and carry is not None:
                d = checker.route(np.asarray(walk["stance"], float), np.asarray(carry["stance"], float))
                if np.isfinite(d):
                    routes.append(float(d))
        longest = max([longest, *routes])
        setting = f"ADT apartment (mars): {x} from {o['surface']} into {c}, placement and yaw random on its surface, robot spawned 1-3 m away ({family} sealed pool); {final}"
        entry = dict(name=f"{family}.{tid}", status="measured", objects=[x], surface=c, from_surface=o["surface"],
                     setting=setting, success=f"{ok}/{n}", rim_height=[round(float(checker.top(c)), 3)] * 2,
                     opening=np.sort(2 * (hc[:2] - wall)).round(3).tolist(),
                     carry_route=[round(min(routes), 2), round(max(routes), 2)] if routes else None, source=src)  # fmt: skip
        if n and ok / n >= BUCKET1_RATE:
            settings.append(entry)
        else:
            failures.append(dict(name=entry["name"], gap=f"place_into.{tid}", objects=[x], surface=c, pilots=[tid],
                                 success=entry["success"], setting=setting, source=src))  # fmt: skip
        if x not in picked and n and lifted / n >= BUCKET1_RATE:
            picked.add(x)
            grasps.append(dict(name=f"{family}.{tid}.pick", status="measured", objects=[x], surface=o["surface"],
                               setting=f"{setting}: lifted and carried away", success=f"{lifted}/{n}",
                               support_top=[round(float(checker.surfaces[o['surface']]["top"]), 3)] * 2,
                               object_width=[round(float(2 * min(half[:2])), 3)] * 2,
                               object_height=[round(float(2 * half[2]), 3)] * 2,
                               mass_max=float((o.get("semantic") or {}).get("mass_kg", 0.0)), inboard=None,
                               stance_xy=None, source=src))  # fmt: skip
    return settings, failures, grasps, longest, dict(final=final, rate=rate, per_task=per)


def build(loop_root: Path, place_loops=(), held_loops=(), pick_rounds: Path | None = None, into_loops=()) -> dict:
    """`place_loops` / `held_loops` / `into_loops`: (sandbox, family, rounds dir or None) per finished family;
    `pick_rounds`: the pick_single sealed pool re-run under the current catalog (None: its final incumbent's rounds)."""
    old = yaml.safe_load(OLD.read_text())
    meta = scene_file.load(str(ADT / "scene_meta.json"))
    checker = feasibility.Checker(meta, dict(old, robot=dict(old["robot"], body_radius=0.35)))
    reach = reach_table()
    grip = load("grip")
    width, mass, top = sweep(grip, "width"), sweep(grip, "mass"), sweep(grip, "top")
    inboard, height, topdown = sweep(grip, "inboard"), sweep(grip, "height"), sweep(grip, "top_down")
    posture, loco = load("posture"), load("loco")
    settings, failures, ps = held(loop_root, "pick_single", checker, pick_rounds)
    loops = {}
    for root, family, rounds in held_loops:
        ok, bad, summ = held(root, family, checker, rounds)
        settings, failures = settings + ok, failures + bad
        ps["deepest_success"] = max(ps["deepest_success"], summ["deepest_success"])
        ps["route_max"] = max(ps["route_max"], summ["route_max"])
        loops[family] = dict(final_incumbent=summ["final"], sealed_rate=summ["rate"], per_task=summ["per_task"],
                             bucket1=summ["rate"] >= BUCKET1_RATE, source=f"{PLAN_SRC.replace('§9', '§10')}")  # fmt: skip
    picked = {x for e in settings for x in e["objects"]}
    placed, place_failed, carry_max = [], [], 0.0
    for root, family, rounds in place_loops:
        ok, bad, grasps, longest, summ = put_on(root, family, checker, picked, rounds)
        placed, place_failed, carry_max = placed + ok, place_failed + bad, max(carry_max, longest)
        settings = settings + grasps
        loops[family] = dict(final_incumbent=summ["final"], sealed_rate=summ["rate"], per_task=summ["per_task"],
                             bucket1=summ["rate"] >= BUCKET1_RATE, source=f"{PLAN_SRC.replace('§9', '§10')}")  # fmt: skip
    put_in, put_in_failed = [], []
    for root, family, rounds in into_loops:
        ok, bad, grasps, longest, summ = into(root, family, checker, picked, rounds)
        put_in, put_in_failed, carry_max = put_in + ok, put_in_failed + bad, max(carry_max, longest)
        settings = settings + grasps
        loops[family] = dict(final_incumbent=summ["final"], sealed_rate=summ["rate"], per_task=summ["per_task"],
                             bucket1=summ["rate"] >= BUCKET1_RATE, source=f"{PLAN_SRC.replace('§9', '§10')}")  # fmt: skip

    lvl = reach["right_level"]
    lows = [v["heights_reached"][0] for v in reach.values() if v["heights_reached"]]
    far_mid = [x for z, x in lvl["forward_max_standing"].items() if 0.65 <= float(z) <= 1.15]
    widths_ok = [w for w, r in width.items() if passes(r)]
    masses_ok = [m for m, r in mass.items() if passes(r)]
    masses_bad = [m for m, r in mass.items() if not passes(r) and m > max(masses_ok, default=0)]
    carried_ok = [m for m, r in mass.items() if passes(r, "carried")]
    tops_ok = [t for t, r in top.items() if passes(r)]
    placed_ok = [t for t, r in top.items() if passes(r, "placed")]
    heights_ok = [h for h, r in height.items() if passes(r)]
    inb_ok = [i for i, r in inboard.items() if passes(r)]
    squats = {v["cell"]["pelvis"]: v for v in posture.values() if "pelvis" in v["cell"]}
    lowest_pelvis = min((v["at"]["pelvis"]["z"] for v in squats.values() if not v["fell"]), default=None)

    def loco_row(prefix):
        rs = [v for k, v in loco.items() if k.startswith(prefix)]
        return dict(n=len(rs), ok=sum(bool(v["ok"]) for v in rs), fell=sum(bool(v["fell"]) for v in rs),
                    pos_err_cm=[min(v["pos_err_cm"] for v in rs), max(v["pos_err_cm"] for v in rs)] if rs else None,
                    yaw_err_deg=[min(v["yaw_err_deg"] for v in rs), max(v["yaw_err_deg"] for v in rs)] if rs else None)  # fmt: skip

    stand = loco.get("stand_30s", {})
    cap = {
        "robot": dict(
            old["robot"],
            body_radius=0.35,
            controller="ScaleBFM (mode 2 walking, mode 4 standing)",
            model="mars",
            body_radius_source="src/locomani/tasks/scene/world.py BODY_RADIUS_M (0.26 let a swinging arm catch the kitchen island)",
            single_arm_side="right; the left hand was probed too (reach_single_arm.hand_envelope)",
            body_radius_tight=0.26,
            body_radius_tight_note="a route that exists only at this radius (SONIC's navigator, arms tucked) is a gap, navigate.narrow_passage: never walked on mars + ScaleBFM, never shown impossible",
        ),  # fmt: skip
        "locomotion": {
            "walk": dict(
                status="measured",
                setting="walk_mars flat floor, walk_to from a standing reset, 0.5-4 m and one 1.8 m diagonal ending turned 90 deg",
                **loco_row("walk"),
                source=PROBE_SRC + "loco/",
            ),  # fmt: skip
            "step_back": dict(status="measured", **loco_row("back"), source=PROBE_SRC + "loco/"),
            "sidestep": dict(status="measured", **loco_row("side"), source=PROBE_SRC + "loco/"),
            "turn_in_place": dict(status="measured", **loco_row("turn"), source=PROBE_SRC + "loco/"),
            "stand_30s": dict(
                status="measured",
                ok=stand.get("ok"),
                drift_cm=stand.get("pos_err_cm"),
                feet_cm=stand.get("feet_cm"),
                note="mode 4 pins the ankles; mode 2 standing spreads the feet (plan 7.1)",
                source=PROBE_SRC + "loco/",
            ),  # fmt: skip
            "navigation": dict(
                status="measured_one",
                setting="ADT apartment, occupancy-map routes at body radius 0.35 m from a random spawn 1-3 m from the object to its stand (pick_single sealed pool, 80 rounds, 0 falls)",
                route_max_measured=round(float(ps["route_max"]), 2),
                source=f"{PLAN_SRC} (7.10)",
            ),  # fmt: skip
            "cross_room": dict(
                status="untested",
                note="the bedroom's spawn region had no reachable point at body radius 0.35 m (pick_single mouse_pickup unavailable)",
            ),
        },
        "reach_single_arm": {
            "jaw_opening": 0.095,
            "hand_envelope": dict(
                status="measured",
                setting=f"walk_mars, standing or after a squat to 0.62 / 0.52 m, jaw centre {0.15} m to the side, forward 0.30-0.90 m; counted within {REACH_TOL_CM} cm and {REACH_ATT_DEG} deg after a 1.5 s hold",
                **reach,
                source=PROBE_SRC + "reach/",
            ),  # fmt: skip
            "grip_height_accepted": lvl["heights_reached"],
            "grip_height_blocked_outside": True,
            "grip_height_source": "hand_envelope.right_level: the probe reached nothing level below or above this band (tested 0.05-1.55 m)",
            "mass_lifted_max": max(masses_ok) if masses_bad else None,
            "grasp_settings": settings,
            "grasp_failures": failures,
            "inboard_max": round(max(max(far_mid) - EDGE_M if far_mid else 0.0, ps["deepest_success"]), 2),
            "inboard_source": f"the larger of: hand_envelope.right_level.forward_max_standing at 0.65-1.15 m minus the {EDGE_M} m the grip probes stood from the edge ({round(max(far_mid) - EDGE_M, 2) if far_mid else None}); the deepest start a grasp setting won, as the checker measures inboard ({round(ps['deepest_success'], 3)}: the checker's nearest-approach-edge proxy can read deeper than the edge a skill used)",
            "motor_grasp": dict(
                status="measured",
                setting=f"probe table, object box placed {EDGE_M} m + inboard ahead and 0.15 m right of the pelvis, wrist level, pre-grasp 12 cm short, entry, grip, lift 10 cm, hold 3 s; three placement jitters per value; base: top 0.75 m, 6 cm wide, 15 cm tall, 0.3 kg, friction 0.6, inboard 0.10 m",
                width={f"{k:.3f}": frac(v) for k, v in width.items()},
                width_lifted=[min(widths_ok), max(widths_ok)] if widths_ok else None,
                mass={f"{k:.1f}": frac(v) for k, v in mass.items()},
                mass_lifted=[min(masses_ok), max(masses_ok)] if masses_ok else None,
                support_top={f"{k:.2f}": frac(v) for k, v in top.items()},
                support_top_lifted=[min(tops_ok), max(tops_ok)] if tops_ok else None,
                inboard={f"{k:.2f}": frac(v) for k, v in inboard.items()},
                inboard_lifted=[min(inb_ok), max(inb_ok)] if inb_ok else None,
                object_height={f"{k:.2f}": frac(v) for k, v in height.items()},
                object_height_lifted=[min(heights_ok), max(heights_ok)] if heights_ok else None,
                top_down={f"{t:.2f}/{h:.2f}": frac(v) for (t, h), v in topdown.items()},
                note="the motor layer under an ideal placement, no skill: the hard limits of bucket 3; what a skill reaches is in grasp_settings",
                source=PROBE_SRC + "grip/",
            ),  # fmt: skip
        },
        "reach_bimanual": {
            "box_grasp": dict(
                status="untested",
                note="no bimanual skill in the mars + ScaleBFM library (D4 starts single-hand); the warehouse box line (ScaleBFM, docs/handoff/scalebfm-warehouse-rsi-20261014.md) runs on its own runner. ADT holds one bimanual object.",
            ),
        },
        "carry": {
            "single_arm": dict(
                status="measured_one",
                setting="the motor grasp's mass sweep: after the lift, step back 0.5 m, turn 90 deg, walk 1 m with the jaw closed; flat floor",
                straight_or_backward=dict(
                    distance=[0.5, 1.0], success={f"{k:.1f} kg": frac(v, "carried") for k, v in mass.items()}
                ),
                carried_mass=[min(carried_ok), max(carried_ok)] if carried_ok else None,
                route_measured_max=round(carry_max, 2) if carry_max else None,
                route_measured_source="the longest occupancy-map route from the pick stance to the put-down stance among a put-down family's sealed successes (place.on_surface_single.measured_tasks)",
                source=PROBE_SRC + "grip/",
            ),  # fmt: skip
            "bimanual_box": dict(status="untested", usable_distance=0.0, note="no bimanual skill"),
        },
        "place": {
            "into_container_single_arm": dict(
                status="measured",
                setting="the put-in families' sealed pools: each setting is one object into one container (the checker takes it for that pair whatever the ranges say); the rim range spans the settings, the opening is the smallest one measured",
                rim_height_measured=[min(e["rim_height"][0] for e in put_in), max(e["rim_height"][0] for e in put_in)],
                opening_measured=[min(e["opening"][0] for e in put_in), min(e["opening"][1] for e in put_in)],
                measured_tasks=put_in,
                failed_tasks=put_in_failed,
                source=PLAN_SRC.replace("§9", "§10"),
            )
            if put_in
            else dict(
                status="untested",
                note="no place skill in the library yet (D4: the loop writes it)",
                failed_tasks=put_in_failed,
            ),
            "put_down_bimanual": dict(status="untested", note="no bimanual skill"),
            "on_surface_single": dict(
                status="measured",
                setting="the put-down families' sealed pools: each setting is one object onto one surface (the checker takes it for that pair whatever the ranges say); the ranges span the settings",
                support_top_measured=[
                    min(e["support_top"][0] for e in placed),
                    max(e["support_top"][0] for e in placed),
                ],
                object_height_measured=[
                    min(e["object_height"][0] for e in placed),
                    max(e["object_height"][0] for e in placed),
                ],
                measured_tasks=placed,
                failed_tasks=place_failed,
                source=PLAN_SRC.replace("§9", "§10"),
            )
            if placed
            else dict(
                status="untested",
                note="no place skill in the library yet; the motor layer put the probe box back down where it was lifted (motor_put_back)",
                failed_tasks=place_failed,
            ),
            "motor_put_back": dict(
                status="measured_one",
                setting="the motor grasp's support-top sweep: lower 9.5 cm back onto the same table, release, retract; counted when the box rests upright on the table and nothing touches it",
                support_top={f"{k:.2f}": frac(v, "placed") for k, v in top.items()},
                support_top_placed=[min(placed_ok), max(placed_ok)] if placed_ok else None,
                source=PROBE_SRC + "grip/",
            ),  # fmt: skip
        },
        "posture": {
            "squat": dict(
                status="measured",
                setting="walk_mars, posture(pelvis_z) by whole-body IK with the torso reference upright, mode 4; held 10 s, then stood up",
                per_pelvis_ref={
                    f"{p:.2f}": dict(
                        delivered=v["at"]["pelvis"]["z"], torso_pitch_deg=v["at"]["torso_pitch_deg"], fell=v["fell"]
                    )
                    for p, v in sorted(squats.items(), reverse=True)
                },
                lowest_pelvis_delivered=lowest_pelvis,
                hands_lowest=min(lows) if lows else None,
                hands_lowest_source="reach_single_arm.hand_envelope: the lowest jaw height any probed attitude and posture reached",
                source=PROBE_SRC + "posture/",
            ),  # fmt: skip
            "lean": dict(
                status="measured",
                per_ref_deg={
                    str(v["cell"]["lean_deg"]): dict(delivered_deg=v["at"]["torso_pitch_deg"], fell=v["fell"])
                    for v in sorted(
                        (v for v in posture.values() if "lean_deg" in v["cell"]), key=lambda v: v["cell"]["lean_deg"]
                    )
                },
                note="mode 4 tracks no torso keypoint: the controller sets the waist",
                source=PROBE_SRC + "posture/",
            ),  # fmt: skip
        },
        "overhead": old["overhead"],
        "inner_loops": dict(
            pick_single=dict(
                final_incumbent=ps["final"],
                sealed_rate=ps["rate"],
                per_task=ps["per_task"],
                bucket1=ps["rate"] >= BUCKET1_RATE,
                source=f"{PLAN_SRC} (7.10)",
            ),
            **loops,
        ),  # fmt: skip
    }
    return cap


HEADER = """# What the mars G1 + Dex1 + ScaleBFM stack can do, as measured (scene RSI plan D10, TODO 27.3; phase 1, {date}).
# Written by scripts/experiments/scene_rsi_20261016/capability_write.py from the phase-1 probes and the inner loops'
# sealed results -- edit the probes or the writer, not this file. The SONIC table of 2026-10-03 stays in
# capability.yaml; scripts/scene_feasibility.py --capability picks one.
#
# Same keys and status words as capability.yaml (measured / measured_one / untested). Differences in meaning:
#   reach_single_arm.grip_height_accepted is the hand's measured level-wrist envelope, not a skill's stance templates;
#     with grip_height_blocked_outside the checker blocks a grip outside it (the probe tried and missed there).
#   reach_single_arm.grasp_settings are the inner loop's tasks at >= 50 % sealed success (D8); the tasks below it are
#     grasp_failures naming their objects (a skill gap for that object as it rests).
#   reach_single_arm.mass_lifted_max (when set): the probe failed above it.
#   place.on_surface_single.measured_tasks are put-down families' tasks at >= 50 % sealed success, one object onto one
#     surface each (failed_tasks below it); grasp_settings also hold the objects those rounds lifted and carried away.
#   place.into_container_single_arm.measured_tasks are the same for put-in families (one object into one container).
#   Every skill the library does not have yet (bimanual) is untested: a gap, never bucket 1.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop-root", type=Path, required=True, help="the pick_single loop's sandbox checkout")
    ap.add_argument("--place-loop", action="append", default=[], metavar="SANDBOX:FAMILY[:ROUNDS]",
                    help="a finished put-down family's sandbox and name (repeatable); ROUNDS: its sealed pool re-run under the current catalog")  # fmt: skip
    ap.add_argument("--held-loop", action="append", default=[], metavar="SANDBOX:FAMILY[:ROUNDS]",
                    help="a finished pick family besides pick_single (repeatable)")  # fmt: skip
    ap.add_argument("--into-loop", action="append", default=[], metavar="SANDBOX:FAMILY[:ROUNDS]",
                    help="a finished put-in family (repeatable)")  # fmt: skip
    ap.add_argument("--pick-rounds", type=Path, help="pick_single's sealed pool re-run under the current catalog")
    ap.add_argument("--out", type=Path, default=DST)
    a = ap.parse_args(argv)
    import datetime

    def loop(v: str) -> tuple:
        parts = v.split(":")
        return Path(parts[0]), parts[1], Path(parts[2]) if len(parts) > 2 else None

    cap = build(a.loop_root, [loop(v) for v in a.place_loop], [loop(v) for v in a.held_loop], a.pick_rounds,
                [loop(v) for v in a.into_loop])  # fmt: skip

    class NoAlias(yaml.SafeDumper):  # the same list in two places stays two lists, not an anchor
        def ignore_aliases(self, data):
            return True

    text = (
        HEADER.format(date=datetime.date.today().isoformat())
        + "\n"
        + yaml.dump(cap, Dumper=NoAlias, sort_keys=False, width=120, allow_unicode=True)
    )
    a.out.write_text(text)
    print(f"wrote {a.out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
