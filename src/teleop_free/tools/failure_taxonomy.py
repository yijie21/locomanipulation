#!/usr/bin/env python3
"""Fine-grained closed-loop failure taxonomy, offline over eval traces.

    python failure_taxonomy.py outputs/eval/base100 [outputs/eval/clean100 ...] [--json]

One leaf per round (2026-09-03 scheme, agreed with the user):
  A1 stopped far        A2 walked past / off      A3 fell walking      A4 view gate (pre-VLA)
  A5 blind start: the target was off the ego frame at trace t=0 (prelude drift, not a policy failure)
  B1 stand too far      B2 arm never reached      B3 hover, no close   B4 knocked over on reach
  B5 dither             B6 stalled at table       B? other no-close
  C1 closed on air      C2 slipped after close    C3 knocked over at close  C4 close height off
  D  lifted then lost   E  fell (stage)           F  other
Old traces have no obj_tilt / touch / obj_z: B4 / C3 fall back to xy displacement, C4 is skipped.
"""
import argparse, collections, json, os, sys
from pathlib import Path
import numpy as np

CLOSE = 0.005      # grip command below this = close
NEAR = 0.15        # palm-link centre within this of the object = fingertips at it (the palm point sits ~12 cm behind the tips)
FAR_STAND = 0.36   # beyond this the arm cannot reach (reach-map band)
CLIFF = 0.45
KNOCK_XY = 0.05    # object xy displacement that counts as knocked
TILT = 30.0        # degrees from upright = toppled
LIFT = 0.06


def _arr(tr, k, default=np.nan):
    return np.array([(r.get(k) if r.get(k) is not None else default) for r in tr], dtype=float)


def classify(rec):
    why = rec.get("why", "") or ""
    tr = rec.get("trace") or []
    if rec.get("ok"):
        return "OK", ""
    if "[view]" in why or "[vla]" not in why and not tr:
        return "A4 view gate / pre-VLA", why[:60]
    # Blind start: the gym's ego gate runs at the spawn, BEFORE the prelude,
    # so a prelude that drifts the target off frame produces a round the
    # policy could never have solved. Recorded since 2026-09-04 as
    # `tstart_cam` (vla_round._tstart_view); older traces lack it and fall
    # through. In a paired comparison these are dropped from BOTH arms.
    cam0 = rec.get("tstart_cam", "absent")
    if cam0 != "absent" and (cam0 is None
                             or abs(cam0[0]) > 0.80 or abs(cam0[1]) > 0.90):
        return "A5 blind start (target off frame at t=0)", (
            "behind camera" if cam0 is None
            else f"cam [{cam0[0]:+.2f} {cam0[1]:+.2f}]")
    if not tr:
        return "F other", why[:60]
    t = _arr(tr, "t"); dist = _arr(tr, "dist"); grip = _arr(tr, "grip", 1.0)
    rise = _arr(tr, "rise", 0.0); po = _arr(tr, "palm_obj"); pz = _arr(tr, "palm_z")
    fallen = np.array([bool(r.get("fallen")) for r in tr])
    tilt = _arr(tr, "obj_tilt"); has_tilt = not np.all(np.isnan(tilt))
    oxy = np.array([r.get("obj_xy") or [np.nan, np.nan] for r in tr], dtype=float)
    pxy = np.array([r.get("pelvis_xy") or [np.nan, np.nan] for r in tr], dtype=float)
    disp = np.linalg.norm(oxy - oxy[0], axis=1) if not np.isnan(oxy).all() else np.zeros(len(tr))
    closed = np.where(grip < CLOSE)[0]
    ic = int(closed[0]) if len(closed) else None
    mind = float(np.nanmin(dist)); imin = int(np.nanargmin(dist))
    knocked = lambda sl: (has_tilt and np.nanmax(tilt[sl]) > TILT) or np.nanmax(disp[sl]) > KNOCK_XY

    if fallen.any():
        i = int(np.argmax(fallen))
        stage = "walking" if dist[i] >= CLIFF else ("after close" if ic is not None and i >= ic else "at table")
        return "E fell", stage
    if np.nanmax(rise) >= LIFT and rise[-1] < LIFT:
        return "D lifted then lost", f"peak rise {np.nanmax(rise) * 100:.0f} cm"
    if ic is None:
        # knocked first: a toppled bottle's centre moves ~10 cm AWAY from the
        # robot, which also inflates the final pelvis distance and would
        # otherwise read as "walked past" (base100 round_005: min 0.35 m at
        # 3.6 s, bottle tipped at 9.4 s, final 0.52 m).
        if knocked(slice(0, len(tr))):
            return "B4 knocked over on reach", (f"tilt {np.nanmax(tilt):.0f} deg" if has_tilt else f"moved {np.nanmax(disp) * 100:.0f} cm (topple proxy)")
        if mind >= CLIFF:
            tail = t > t[-1] - 10
            moved = np.linalg.norm(pxy[tail][-1] - pxy[tail][0]) if tail.sum() > 1 and not np.isnan(pxy).all() else np.nan
            if np.isnan(moved) or moved < 0.05:
                return "A1 stopped far", f"min {mind:.2f} m"
            return "A2 walked past / off", f"min {mind:.2f} m, still moving"
        if dist[-1] - mind > 0.15:
            return "A2 walked past / off", f"min {mind:.2f} m then {dist[-1]:.2f} m"
        minpo = float(np.nanmin(po)) if not np.isnan(po).all() else np.nan
        # dither: palm crossing the NEAR+0.04 line inward more than once
        crossings = 0
        if not np.isnan(po).all():
            near = po < NEAR + 0.04
            crossings = int(np.sum(near[1:] & ~near[:-1]))
        tail = t > t[-1] - 10
        pel_still = (not np.isnan(pxy).all()) and tail.sum() > 1 and np.linalg.norm(pxy[tail][-1] - pxy[tail][0]) < 0.02
        palm_still = (not np.isnan(pz).all()) and tail.sum() > 1 and (np.nanmax(pz[tail]) - np.nanmin(pz[tail])) < 0.02
        if crossings >= 2:
            return "B5 dither", f"{crossings} approaches, min palm-obj {minpo * 100:.0f} cm"
        if minpo <= NEAR:
            return "B3 hover, no close", f"min palm-obj {minpo * 100:.0f} cm"
        if mind >= FAR_STAND:
            return "B1 stand too far", f"stand {mind:.2f} m, min palm-obj {minpo * 100:.0f} cm"
        if pel_still and palm_still:
            return "B6 stalled at table", f"stand {mind:.2f} m, palm-obj {minpo * 100:.0f} cm"
        if not np.isnan(pz).all() and (np.nanmax(pz) - np.nanmin(pz)) < 0.05:
            return "B2 arm never reached", f"stand {mind:.2f} m, palm-obj {minpo * 100:.0f} cm"
        return "B? other no-close", f"stand {mind:.2f} m, palm-obj {minpo * 100:.0f} cm"
    # a close was commanded
    if knocked(slice(ic, len(tr))):
        return "C3 knocked over at close", (f"tilt {np.nanmax(tilt[ic:]):.0f} deg" if has_tilt else f"moved {np.nanmax(disp[ic:]) * 100:.0f} cm")
    po_c = float(po[ic]) if not np.isnan(po[ic]) else np.nan
    if not np.isnan(po_c) and po_c > 0.04:
        return "C1 closed on air", f"palm-obj {po_c * 100:.0f} cm at close"
    oz = _arr(tr, "obj_z")
    if not np.isnan(oz[ic]) and abs(pz[ic] - oz[ic]) > 0.05:
        return "C4 close height off", f"palm {(pz[ic] - oz[ic]) * 100:+.0f} cm vs object"
    if np.nanmax(rise[ic:]) >= 0.02:
        return "C2 slipped after close", f"peak rise {np.nanmax(rise[ic:]) * 100:.0f} cm"
    return "C2 slipped after close", "no rise after close"


def summarize_traces(trace_dir, *, max_rounds=None):
    """Return trace-level diagnoses separately from coarse timeout reasons.

    Pre-VLA failures have no trace; report coverage rather than inventing
    episode mappings for them. max_rounds excludes stale higher indices.
    """
    files=sorted(Path(trace_dir).glob('round_*.json'))
    if max_rounds is not None:
        files=[f for f in files if int(f.stem.split('_')[-1])<max_rounds]
    counts=collections.Counter();rows={}
    for f in files:
        rec=json.loads(f.read_text());leaf,note=classify(rec)
        counts[leaf]+=1
        rows[f.stem]=dict(leaf=leaf,note=note,why=rec.get('why',''))
    return dict(trace_count=len(files),counts=dict(counts),traces=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--json", action="store_true", help="also write taxonomy.json into each arm dir")
    a = ap.parse_args()
    for arm in a.arms:
        tdir = os.path.join(arm, "traces")
        files = sorted(f for f in os.listdir(tdir) if f.startswith("round_") and f.endswith(".json"))
        counts = collections.Counter(); rows = {}
        for f in files:
            rec = json.load(open(os.path.join(tdir, f)))
            leaf, note = classify(rec)
            counts[leaf] += 1; rows[f[:-5]] = dict(leaf=leaf, note=note, why=rec.get("why", ""))
        n = len(files); ok = counts.get("OK", 0)
        print(f"== {os.path.basename(arm.rstrip('/'))}: {ok}/{n} OK, {n - ok} failures")
        for leaf, c in sorted(counts.items()):
            if leaf == "OK":
                continue
            print(f"  {c:3d}  {c / max(1, n - ok) * 100:4.0f}%  {leaf}")
        if a.json:
            json.dump(rows, open(os.path.join(arm, "taxonomy.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
