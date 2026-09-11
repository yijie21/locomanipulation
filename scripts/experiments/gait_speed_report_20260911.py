#!/usr/bin/env python3
"""Stage A report: the two checkpoints' speed maps, and what they imply.

Reads the per-leg JSON that diag_shuffle writes for each arm and answers the
three questions the re-calibration actually needs:

  1. what does each commanded speed DELIVER on this checkpoint (m/s, cm/step,
     post-IDLE carry), with the spread over repeats;
  2. which command value on v1.1 delivers the velocity release delivered for
     each of the three approach bands -- that is the re-pick, and it is an
     inverse lookup on the measured curve, not a guess;
  3. do the production constants still clear the approach loop's stall rule
     (round_phases_approach.py:155: less than 1.5 cm of depth progress in 5 s
     = "stalled", i.e. 0.003 m/s), and can _creep_to_band still close its
     error inside max_s.

    python scripts/experiments/gait_speed_report_20260911.py
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "outputs/diagnostics/gait_speed_v11_20260911"
# The knee (0.28-0.37) was measured in a second pass once the first showed the
# curve STEPS there; same script, same scene, same protocol, so the rows merge.
DIRS = (DIAG, ROOT / "outputs/diagnostics/gait_speed_v11_20260911_knee")
ARMS = ("sonic_v1_1", "release")

# round_phases_approach.py:155-158 -- the timer resets on 1.5 cm of progress and
# fires after 5 s without it.
STALL_M, STALL_S = 0.015, 5.0
STALL_SPEED = STALL_M / STALL_S
# The three bands and the velocity release delivered for each
# (primitives_tuning.py:306-317 + round_phases_approach.py:82-100).
BANDS = {"APPR_COARSE": (0.30, 0.247), "APPR_MID": (0.215, 0.040), "APPR_SHUFFLE": (0.20, 0.013)}
NAV = {"NAV_SPEED": 0.45, "NAV_SPEED_MIN": 0.35}
# The approach's own budget, all from round_phases_approach.py / the visible
# profile. NOT _creep_to_band: that function is DEAD in the production path
# (defined, exercised by tests, called by nothing in a round), so the "creep
# timeout" reading of ledger 17 was wrong.
APPROACH_BUDGET_S = 45.0  # do_approach(45.0, ramp=True)
SPAWN_DEPTH_M = 1.2  # edge_range 1.1-1.3 in configs/tasks/bottle_visible.json
PLAN_R_M = 0.32  # ledger 17's measured plan_r
MID_BAND_M = 0.30  # round_phases.py:85 mid_band default
BRAKE_DEPTH_M = 0.31  # reach_config.visible_brake_depth
DECEL_BAND_M = 0.12  # reach_config.visible_decel_band


def _policy_from_stack(d):
    """Which checkpoint was RUNNING -- the stack's own log, not the sweep
    process's locomani.paths lookup. On 2026-09-11 those disagreed for the
    release arm: the env var reached start_stack.sh but not the sweep."""
    log = d / "stack.log"
    if log.is_file():
        for line in log.read_text().splitlines():
            if line.strip().startswith("policy:"):
                return line.split("policy:")[1].strip()
    return None


def load(arm):
    out, seen = None, []
    for base in DIRS:
        p = base / arm / "legs.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        ran = _policy_from_stack(base / arm)
        seen.append((base.name, ran, d.get("policy")))
        if out is None:
            out = dict(d)
            out["rows"] = list(d["rows"])
            out["policy_from_stack"] = ran
            out["sources"] = []
        else:
            out["rows"] += d["rows"]
            # A merged pass that ran a DIFFERENT checkpoint would silently
            # average two curves -- refuse instead.
            if ran != out["policy_from_stack"]:
                raise SystemExit(f"{arm}: {base.name} ran {ran}, not {out['policy_from_stack']}")
        out["sources"].append(base.name)
    if out is not None:
        out["policy_labels_seen"] = seen
    return out


def curve(rows):
    """{speed: {metric: (median, min, max, n)}} over the walking legs."""
    by = {}
    for r in rows:
        if r.get("fell") or r["mode"] != "SLOW_WALK" or not r.get("move"):
            continue
        by.setdefault(r["speed"], []).append(r)
    out = {}
    for sp, rs in sorted(by.items()):
        out[sp] = {}
        for k in ("net", "per_step", "carry"):
            vals = [r[k] for r in rs if r.get(k) is not None and r[k] == r[k]]
            out[sp][k] = (st.median(vals), min(vals), max(vals), len(vals)) if vals else None
    return out


def idle_drift(rows):
    vals = [r["net"] for r in rows if r["mode"] == "IDLE" and not r.get("fell")]
    carry = [r["carry"] for r in rows if r["mode"] == "IDLE" and not r.get("fell")]
    return (st.median(vals) if vals else None, st.median(carry) if carry else None, len(vals))


# A bracketing pair this far apart is a STEP, not a slope: interpolating inside
# it invents a command that the gait does not actually deliver (it lands in the
# bistable transition). 3x is well above the 1.2-1.5x repeat scatter.
STEP_RATIO = 3.0


def invert(c, target):
    """A commanded speed that DELIVERS `target` m/s on this curve.

    Returns (speed, how). Linear interpolation between the bracketing measured
    points -- except inside a step, where interpolation is meaningless and the
    honest answer is the nearest measured point plus a warning.
    """
    pts = sorted((sp, m["net"][0]) for sp, m in c.items() if m["net"])
    if not pts:
        return None, "no data"
    if target <= pts[0][1]:
        return pts[0][0], f"at or below the slowest measured point ({pts[0][1]:.3f} m/s at {pts[0][0]})"
    if target >= pts[-1][1]:
        return pts[-1][0], f"ABOVE the fastest measured point ({pts[-1][1]:.3f} m/s at {pts[-1][0]}) -- unreachable"
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if v0 <= target <= v1:
            if v0 > 0 and v1 / v0 >= STEP_RATIO:
                near = s0 if abs(target - v0) <= abs(target - v1) else s1
                got = v0 if near == s0 else v1
                return near, (
                    f"IN A STEP ({s0}->{s1} jumps {v0:.3f}->{v1:.3f}, {v1 / v0:.1f}x): no command delivers "
                    f"{target:.3f}; nearest measured is {near} at {got:.3f} ({got / target:.1f}x target)"
                )
            f = 0.0 if v1 == v0 else (target - v0) / (v1 - v0)
            return round(s0 + f * (s1 - s0), 4), f"interpolated between {s0} ({v0:.3f}) and {s1} ({v1:.3f})"
    return None, "not bracketed (the curve is not monotone here)"


def step_edges(c):
    """Where a curve jumps by >= STEP_RATIO between adjacent commands."""
    pts = sorted((sp, m["net"][0]) for sp, m in c.items() if m["net"])
    return [(s0, s1, v0, v1) for (s0, v0), (s1, v1) in zip(pts, pts[1:]) if v0 > 0 and v1 / v0 >= STEP_RATIO]


def fmt(cell, scale=1.0, prec=3):
    if not cell:
        return "     --"
    med, lo, hi, n = cell
    return f"{med * scale:.{prec}f} [{lo * scale:.{prec}f},{hi * scale:.{prec}f}] n={n}"


def main():
    data = {a: load(a) for a in ARMS}
    report = {"stall_speed_m_s": STALL_SPEED, "arms": {}}
    for arm in ARMS:
        d = data[arm]
        if d is None:
            print(f"== {arm}: no legs.json yet")
            continue
        c = curve(d["rows"])
        drift, dcarry, dn = idle_drift(d["rows"])
        ran, labelled = d.get("policy_from_stack"), d.get("policy")
        note = f"deploy ran {ran}" if ran else "no stack.log"
        note += f", passes: {'+'.join(d.get('sources', []))}"
        if ran and labelled and ran != labelled:
            note += f"  [legs.json's own label says {labelled} -- WRONG, see load()]"
        print(f"\n== {arm}   ({note})")
        print(f"{'speed':>7} {'net m/s  [min,max]':>28} {'cm/step':>26} {'carry cm':>26}")
        for sp, m in c.items():
            print(f"{sp:>7} {fmt(m['net']):>28} {fmt(m['per_step'], 1, 1):>26} {fmt(m['carry'], 1, 1):>26}")
        print(
            f"IDLE control (no walk): net {drift if drift is None else round(drift, 4)} m/s, "
            f"3.5 s displacement {dcarry if dcarry is None else round(dcarry, 1)} cm, n={dn}"
        )
        report["arms"][arm] = {
            "policy_ran": d.get("policy_from_stack"),
            "policy_label_in_legs_json": d.get("policy"),
            "curve": {str(k): v for k, v in c.items()},
            "idle_drift_m_s": drift,
            "idle_3s5_cm": dcarry,
        }

    v11, rel = data.get("sonic_v1_1"), data.get("release")
    if not v11:
        return
    c11 = curve(v11["rows"])
    crel = curve(rel["rows"]) if rel else {}

    for arm, cc in (("release", crel), ("sonic_v1_1", c11)):
        for s0, s1, v0, v1 in step_edges(cc):
            print(f"== {arm}: STEP between {s0} and {s1}: {v0:.3f} -> {v1:.3f} m/s ({v1 / v0:.1f}x)")
    report["steps"] = {
        "release": step_edges(crel),
        "sonic_v1_1": step_edges(c11),
    }

    print("\n== the re-pick: which v1.1 command delivers what release's band delivered")
    print(f"{'band':>14} {'release cmd':>12} {'release m/s':>12} {'v1.1 cmd':>10}  note")
    picks = {}
    for name, (cmd, doc_v) in BANDS.items():
        # Prefer the release velocity measured in THIS session over the
        # documented one; the documented table is from 2026-08.
        tgt = crel.get(cmd, {}).get("net", (doc_v,))[0] if crel.get(cmd, {}).get("net") else doc_v
        sp, how = invert(c11, tgt)
        picks[name] = {"release_cmd": cmd, "target_m_s": tgt, "v11_cmd": sp, "how": how}
        print(f"{name:>14} {cmd:>12} {tgt:>12.3f} {str(sp):>10}  {how}")
    report["re_pick"] = picks

    print("\n== do the CURRENT constants still clear the stall rule on v1.1?")
    print(f"   stall fires below {STALL_SPEED:.4f} m/s (1.5 cm in 5 s)")
    stall = {}
    for name, cmd in list((k, v[0]) for k, v in BANDS.items()) + list(NAV.items()):
        cell = c11.get(cmd, {}).get("net")
        got = cell[0] if cell else None
        verdict = "no data" if got is None else ("STALLS" if got < STALL_SPEED else "ok")
        margin = "" if got is None else f"  ({got / STALL_SPEED:.1f}x the threshold)"
        stall[name] = {"cmd": cmd, "v11_m_s": got, "verdict": verdict}
        print(f"{name:>14} cmd {cmd:>5} -> {str(got if got is None else round(got, 4)):>7} m/s  {verdict}{margin}")
    report["stall_check"] = stall

    # -- the in-path time budget ------------------------------------------
    # Three phases, from the object depth the loop reads:
    #   coarse   spawn -> plan_r + MID_BAND          at APPR_COARSE
    #   mid      that  -> brake + DECEL_BAND         at APPR_MID
    #   decel    that  -> brake                      ramped MID -> SHUFFLE
    coarse_m = SPAWN_DEPTH_M - (PLAN_R_M + MID_BAND_M)
    mid_m = (PLAN_R_M + MID_BAND_M) - (BRAKE_DEPTH_M + DECEL_BAND_M)
    print(f"\n== the approach's own budget ({APPROACH_BUDGET_S:.0f} s), depth {SPAWN_DEPTH_M} -> {BRAKE_DEPTH_M} m")
    print(f"   coarse {coarse_m:.2f} m, mid {mid_m:.2f} m, decel {DECEL_BAND_M:.2f} m")
    budget = {}
    for label, cc, cmds in (
        ("release", crel, (0.30, 0.215, 0.20)),
        ("v1.1 as-is", c11, (0.30, 0.215, 0.20)),
        ("v1.1 re-picked", c11, (picks["APPR_COARSE"]["v11_cmd"], 0.215, 0.20)),
    ):

        def v(cmd, cc=cc):
            cell = cc.get(cmd, {}).get("net")
            if cell:
                return cell[0]
            got, _ = None, None
            # a re-picked command may be between measured points
            pts = sorted((sp, m["net"][0]) for sp, m in cc.items() if m["net"])
            for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
                if s0 <= cmd <= s1:
                    f = (cmd - s0) / (s1 - s0) if s1 > s0 else 0.0
                    got = v0 + f * (v1 - v0)
            return got

        vc, vm, vs = (v(x) for x in cmds)
        if None in (vc, vm, vs):
            continue
        t = coarse_m / vc + mid_m / vm + DECEL_BAND_M / ((vm + vs) / 2)
        verdict = "fits" if t < APPROACH_BUDGET_S else "OVER BUDGET"
        print(
            f"{label:>16} cmds {cmds} -> {coarse_m / vc:5.1f} + {mid_m / vm:5.1f} + "
            f"{DECEL_BAND_M / ((vm + vs) / 2):5.1f} = {t:5.1f} s   {verdict}"
        )
        budget[label] = {"cmds": list(cmds), "seconds": t, "verdict": verdict}
    report["budget"] = {
        "approach_budget_s": APPROACH_BUDGET_S,
        "phase_metres": {"coarse": coarse_m, "mid": mid_m, "decel": DECEL_BAND_M},
        "arms": budget,
    }

    out = DIAG / "report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
