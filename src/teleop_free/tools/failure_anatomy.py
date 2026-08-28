"""Split "at the table but never closed" failures into their two diseases.

    python src/teleop_free/tools/failure_anatomy.py <trace_dir> [<trace_dir>...]
        [--out anatomy.json]

Round 3 ended with 67/72 failures in one bucket: the robot stands at the table,
inside the band successes close from (min-dist p25/median/p75 = 0.23/0.28/0.32
vs successes' close-at 0.28/0.30/0.31), and never commands a close. That bucket
hides two different diseases with two different DAgger prescriptions:

  ARM-NEVER-UP    palm never rises above the table plane. The policy is not
                  initiating the reach at all -> collect the "standing at the
                  table -> raise" transition (round-2 style near-table spawns).
  UP-BUT-NO-FIRE  palm comes up, gets near the object, and the close never
                  fires -> collect dense "hand staged near object -> close"
                  coverage; the last centimetre is the hole.

Round 3's traces could not tell these apart -- they carried no arm state. The
palm_z / palm_obj fields were added for exactly this, so this tool only speaks
about traces that have them and says so when they do not.

Thresholds are taken from the successes IN THE SAME TRACE SET (their palm_z at
close, their palm_obj at close), not hardcoded: the same tool then works on any
future round without recalibration.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def q(xs, f):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(f * len(xs))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    succ, fail, legacy = [], [], 0
    for d in args.trace_dirs:
        for f in sorted(glob.glob(os.path.join(d, "round_*.json"))):
            t = json.load(open(f))
            tr = t.get("trace") or []
            if not tr:
                continue
            if "palm_z" not in tr[0]:
                legacy += 1
                continue
            rec = dict(file=f, ok=t["ok"], why=t.get("why", ""), trace=tr)
            (succ if t["ok"] else fail).append(rec)

    if legacy:
        print(f"[anatomy] {legacy} traces predate the palm fields -- skipped "
              f"(round 3's own traces cannot be decomposed; that is why the "
              f"fields were added)")
    if not succ:
        sys.exit("[anatomy] no successes with palm fields: cannot calibrate "
                 "thresholds. Run at least one sweep with the new tracer first.")

    # Calibrate from this set's own successes at their close moment.
    close_z, close_po = [], []
    for r in succ:
        rows = [x for x in r["trace"] if x.get("grip", 1.0) < 0.005]
        if rows:
            close_z.append(rows[0]["palm_z"])
            close_po.append(rows[0]["palm_obj"])
    z_up = q(close_z, 0.10)          # palm height successes close at (p10)
    po_near = q(close_po, 0.90)      # palm-object distance at close (p90)
    print(f"[anatomy] calibrated on {len(close_z)} successes: "
          f"arm-up when palm_z > {z_up:.3f} m, "
          f"staged when palm_obj < {po_near:.3f} m")

    # Rounds whose FIRST trace frame already reads grip < 0.005: priming
    # residue can leak into frame 0, and such a round is then marked
    # commanded_close at t=0 and can never land in "never commanded a
    # close" -- a systematic hole in the reach-class denominators (R4
    # measured 3/100). Counted and listed, not reclassified: the online
    # judge must stay comparable with the recorded ladder.
    first_closed = [r for r in succ + fail
                    if r["trace"][0].get("grip", 1.0) < 0.005]

    never_up, up_no_fire, staged_no_fire, other = [], [], [], []
    for r in fail:
        if "reach" not in r["why"]:
            other.append(r)
            continue
        zmax = max(x["palm_z"] for x in r["trace"])
        pomin = min(x["palm_obj"] for x in r["trace"])
        if zmax <= z_up:
            never_up.append(r)
        elif pomin <= po_near:
            staged_no_fire.append(r)   # hand got where successes close from
        else:
            up_no_fire.append(r)       # raised, never got near

    n = len(fail)
    res = dict(
        thresholds=dict(palm_z_up=z_up, palm_obj_near=po_near,
                        calibrated_on=len(close_z)),
        successes=len(succ), failures=n, legacy_skipped=legacy,
        arm_never_up=len(never_up),
        up_but_never_near=len(up_no_fire),
        staged_but_no_fire=len(staged_no_fire),
        non_reach=len(other),
        first_frame_closed=len(first_closed),
        first_frame_closed_files=[r["file"] for r in first_closed],
        examples=dict(
            arm_never_up=[r["file"] for r in never_up[:5]],
            up_but_never_near=[r["file"] for r in up_no_fire[:5]],
            staged_but_no_fire=[r["file"] for r in staged_no_fire[:5]]),
    )
    print(f"\n[anatomy] {n} reach-class failures decompose as:")
    print(f"  arm never up        {len(never_up):3d}  -> DAgger: near-table "
          f"spawns, cover the stand->raise transition")
    print(f"  up but never near   {len(up_no_fire):3d}  -> DAgger: reach "
          f"trajectory coverage (mid-reach states)")
    print(f"  staged, no fire     {len(staged_no_fire):3d}  -> DAgger: hand "
          f"staged near object -> close; the last centimetre")
    if other:
        print(f"  non-reach           {len(other):3d}")
    if first_closed:
        print(f"[anatomy] {len(first_closed)} rounds open with grip<0.005 at "
              f"frame 0 -- their commanded_close flag is untrustworthy")

    out = args.out or os.path.join(args.trace_dirs[0], "..", "anatomy.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n[anatomy] -> {out}")


if __name__ == "__main__":
    main()
