"""TODO 27.3 (plan docs/handoff/scene-rsi-plan-20261016.md §9, D10): write the mars + ScaleBFM capability table,
configs/scene_discovery/capability_mars_scalebfm.yaml, from the phase-1 probes (capability_probe.py, cells under
outputs/scene_rsi_20261016/capability/) and the inner loops' sealed results (D10: every accepted skill writes its
sealed-pool numbers back). Same keys as the 10-03 SONIC table, so scene_discovery.feasibility reads either; the
numbers here are all measured on mars + ScaleBFM, or marked untested.

A reach cell counts when the robot stood, the jaw centre was within REACH_TOL_CM of the target after a 1.5 s hold and
the wrist within REACH_ATT_DEG of the asked attitude. A sweep value counts when >= 2 of its 3 jittered rounds did.
A pick_single task enters `grasp_settings` when its sealed success is >= BUCKET1_RATE (D8), else `grasp_failures`.

    $PY scripts/experiments/scene_rsi_20261016/capability_write.py --loop-root /workspace/code/rsi-scene/pick-a
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


def pick_single(loop_root: Path, checker) -> tuple[list, list, dict]:
    """The inner loop's final incumbent on its sealed pool, per task: a measured grasp setting or a failure. A setting
    names its object and surface: the checker takes that measurement for that object where it rests, whatever its
    geometric proxies say (the coffee table has no approach edge in the scene file, yet the remote went 10/10). The
    inboard range and `stance_xy` come from the successful rounds."""
    hidden = loop_root / "outputs" / "scene_rsi_20261016_hidden" / "pick_single"
    summ = json.loads((hidden / "sealed" / "summary.json").read_text())
    final = summ["final"]
    per = summ["versions"][final]["per_task"]
    fam = json.loads((HERE / "families" / "pick_single.json").read_text())
    rows = {json.loads(line)["id"]: json.loads(line) for line in (REPO / fam["tasks_file"]).read_text().splitlines() if line.strip()}  # fmt: skip
    settings, failures, deepest = [], [], 0.0
    for tid, frac_s in sorted(per.items()):
        x = rows[tid]["goal"][0].split("(", 1)[1].rstrip(")")
        o = checker.objects[x]
        half, _ = checker.box(x)
        s_key = o["surface"]
        inboards, stances = [], []
        for f in sorted((hidden / "sealed" / final).glob(f"{tid}_s*.json")):
            r = json.loads(f.read_text())
            if not r.get("ok"):
                continue
            st = json.loads((hidden / "pool_sealed" / f.name).read_text())["start"]
            d = checker.edge_stand(s_key, np.asarray(st["placements"][x][0][:2], float))[1]
            if np.isfinite(d):
                inboards.append(float(d))
            walk = next((w for w in r.get("steps", []) if w.get("step") == "walk" and w.get("stance")), None)
            if walk is not None:
                stances.append([round(float(v), 3) for v in walk["stance"]])
        deepest = max([deepest, *inboards])
        ok, n = map(int, frac_s.split("/"))
        entry = dict(name=f"pick_single.{tid}", status="measured", objects=[x], surface=s_key,
                     setting=f"ADT apartment (mars): {x} on {s_key}, placement and yaw random on that surface, robot spawned 1-3 m away (pick_single sealed pool); the loop's final incumbent {final}",
                     success=f"{ok}/{n}", support_top=[round(float(checker.surfaces[s_key]["top"]), 3)] * 2,
                     object_width=[round(float(2 * min(half[:2])), 3)] * 2, object_height=[round(float(2 * half[2]), 3)] * 2,
                     mass_max=float((o.get("semantic") or {}).get("mass_kg", 0.0)),
                     inboard=[round(min(inboards), 3), round(max(inboards), 3)] if inboards else None,
                     stance_xy=stances[0] if stances else None,
                     source=f"{PLAN_SRC} (7.10, sealed pool); <loop sandbox>/outputs/scene_rsi_20261016_hidden/pick_single/sealed/summary.json")  # fmt: skip
        if n and ok / n >= BUCKET1_RATE:
            settings.append(entry)
        else:
            failures.append(dict(name=entry["name"], gap=f"grasp_single.{tid}", objects=[x], pilots=[tid],
                                 success=entry["success"], setting=entry["setting"], source=entry["source"]))  # fmt: skip
    return settings, failures, dict(final=final, rate=summ["final_rate"], per_task=per, deepest_success=deepest)


def build(loop_root: Path) -> dict:
    old = yaml.safe_load(OLD.read_text())
    meta = scene_file.load(str(ADT / "scene_meta.json"))
    checker = feasibility.Checker(meta, dict(old, robot=dict(old["robot"], body_radius=0.35)))
    reach = reach_table()
    grip = load("grip")
    width, mass, top = sweep(grip, "width"), sweep(grip, "mass"), sweep(grip, "top")
    inboard, height, topdown = sweep(grip, "inboard"), sweep(grip, "height"), sweep(grip, "top_down")
    posture, loco = load("posture"), load("loco")
    settings, failures, ps = pick_single(loop_root, checker)

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
                route_max_measured=round(float(_route_max(loop_root, checker)), 2),
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
                source=PROBE_SRC + "grip/",
            ),  # fmt: skip
            "bimanual_box": dict(status="untested", usable_distance=0.0, note="no bimanual skill"),
        },
        "place": {
            "into_container_single_arm": dict(
                status="untested", note="no place skill in the library yet (D4: the loop writes it)"
            ),
            "put_down_bimanual": dict(status="untested", note="no bimanual skill"),
            "on_surface_single": dict(
                status="untested",
                note="no place skill in the library yet; the motor layer put the probe box back down where it was lifted (motor_put_back)",
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
            )
        ),  # fmt: skip
    }
    return cap


def _route_max(loop_root: Path, checker) -> float:
    """Longest occupancy-map route from a spawn to the stand among the final incumbent's sealed successes."""
    hidden = loop_root / "outputs" / "scene_rsi_20261016_hidden" / "pick_single"
    final = json.loads((hidden / "sealed" / "summary.json").read_text())["final"]
    best = 0.0
    for f in sorted((hidden / "sealed" / final).glob("*.json")):
        if f.name.startswith("_"):  # rounds of an older success rule
            continue
        r = json.loads(f.read_text())
        if not r.get("ok"):
            continue
        st = json.loads((hidden / "pool_sealed" / f.name).read_text())["start"]
        walk = next((s for s in r.get("steps", []) if s.get("step") == "walk" and s.get("stance")), None)
        if walk is None:
            continue
        d = checker.route(np.asarray(st["robot_xy"], float), np.asarray(walk["stance"], float))
        if np.isfinite(d):
            best = max(best, d)
    return best


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
#   Every skill the library does not have yet (place, place into, bimanual) is untested: a gap, never bucket 1.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop-root", type=Path, required=True, help="the pick_single loop's sandbox checkout")
    ap.add_argument("--out", type=Path, default=DST)
    a = ap.parse_args(argv)
    import datetime

    cap = build(a.loop_root)

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
