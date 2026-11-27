"""selfmodel_faildiag_20261126 question 3 (a), (b) over data3's in-place resets rerun in LML (fd_heft.py d3 -> fd_heft_an.py
json): per reach, which comes first after the command -- the pelvis 2 cm ahead of its feet with both feet still down,
a foot slipping, or a foot lifting -- for the reaches that got lost (the reset's lost frame inside the reach's window)
and the others; the forward shift before the first lift-off; the reference's torso pitch / pelvis height / legs.
Reproduction: each reset's record vs data3's own log (walked, fell, lost, why, steps, segments).

    python fd_heft_sum.py <an json> <fd_heft d3 dir> [--orig <data3 heft data dir>]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

FPS = 50


def q(a, p):
    a = [v for v in a if v is not None]
    return "nan" if not a else f"{np.percentile(a, p):.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("an")
    ap.add_argument("d3dir")
    ap.add_argument("--orig", default="/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/data")
    a = ap.parse_args()
    R = [r for r in json.loads(Path(a.an).read_text()) if r["src"] == "LML" and r["mode"] == "d3"]
    # reproduction vs data3's logs
    logs = {}
    for f in Path(a.orig).glob("w00[0-9]_00[0-9]_log.json"):
        for e in json.loads(f.read_text()):
            logs[e["reset"]] = e
    rep = Counter()
    metas = {}
    for f in sorted(Path(a.d3dir).glob("d3_*.npz")):
        m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        metas[f.name] = m
        o = logs.get(m["reset"])
        same_log = o is not None and all(o[k] == m[k] for k in ("walked", "fell", "lost", "why")) and o["steps"] == m["steps"] and o["n_segs"] == m["compare"].get("n", 0)
        one = m["compare"].get("same") is True or m["compare"].get("n", 0) == 0 or "both hand" in str(m["compare"].get("why", "")) or "segment count" in str(m["compare"].get("why", ""))
        rep[("log same" if same_log else "log differs", "arrays same" if m["compare"].get("same") else "arrays: " + str(m["compare"].get("why", m["compare"].get("diff"))))] += 1
    print(f"reproduction vs data3 (resets {len(metas)}): " + "; ".join(f"{k[0]}, {k[1]}: {v}" for k, v in rep.most_common()))
    inplace = [r for r in R if not r["walked"]]
    print(f"in-place resets {len({r['file'] for r in inplace})}, their reaches {len(inplace)} (walked resets left out: {len({r['file'] for r in R if r['walked']})})")
    rows = []
    for r in inplace:
        m = metas[r["file"]]
        rc = m["reaches"][r["reach"]]
        nxt = m["reaches"][r["reach"] + 1]["f_cmd"] if r["reach"] + 1 < len(m["reaches"]) else m["steps"]
        lost_here = m["lost"] is not None and rc["f_cmd"] <= m["lost"] <= nxt + 1
        ev = {"shift2": r["mag2cm_s"], "slip": r["slip_s"], "lift": r["lift_s"]}
        evs = sorted([(v, k) for k, v in ev.items() if v is not None])
        first = evs[0][1] if evs else "none"
        
        tgt = list(r["target_h"].values())[0]
        b = r.get("pose") or {}
        rows.append(dict(file=r["file"], reach=r["reach"], lost=lost_here, first=first, lift=r["lift_s"], slip=r["slip_s"], fwd2=r["fwd2cm_s"],
                         fwd_lift=r["at_lift"]["fwd_cm"], maxfwd=r["max_fwd_before_lift_cm"], maxmag=r["max_mag_before_lift_cm"], mag26=r["mag_2_6s_cm"],
                         maxalong=r.get("max_along_before_lift_cm"), along26=r.get("along_2_6s_cm"), cospk=r.get("cos_at_peak"), fwd26=r["at_2_6s"]["fwd_cm"], ref26=r["at_2_6s"]["ref_cm"],
                         ref_s=r["ref_end_s"], tx=tgt[0] if tgt else None, tz=tgt[2] if tgt else None, pz=r["pelvis_z"], hands=r["hands"],
                         dpitch=b.get("d_ref_torso_pitch_deg"), dz=b.get("d_ref_pelvis_z_cm"), dknee=None if not b else max(abs(b["d_ref_legs_rad"]["left_knee_joint"]), abs(b["d_ref_legs_rad"]["right_knee_joint"]))))  # fmt: skip
    for nm, S in (("lost in this reach", [x for x in rows if x["lost"]]), ("not lost", [x for x in rows if not x["lost"]])):
        if not S:
            continue
        print(f"== {nm}: {len(S)} reaches; first event after the command: {dict(Counter(x['first'] for x in S))}")
        print(f"   pelvis off the feet midpoint (any direction) before the first lift-off, max cm p10/50/90 {q([x['maxmag'] for x in S], 10)}/{q([x['maxmag'] for x in S], 50)}/{q([x['maxmag'] for x in S], 90)}; at 2.6 s p50 {q([x['mag26'] for x in S], 50)}; "
              f"along the reach direction max p50 {q([x['maxalong'] for x in S], 50)}, at 2.6 s p50 {q([x['along26'] for x in S], 50)}; shift's direction at its peak within 60 deg of the reach's (cos >= 0.5) {sum((x['cospk'] or -1) >= 0.5 for x in S)}/{len(S)}, opposite (cos <= -0.5) {sum((x['cospk'] or 1) <= -0.5 for x in S)}")  # fmt: skip
        print(f"   pelvis ahead of the feet midpoint before the first lift-off (max, cm) p10/50/90 {q([x['maxfwd'] for x in S], 10)}/{q([x['maxfwd'] for x in S], 50)}/{q([x['maxfwd'] for x in S], 90)}; "
              f"at 2.6 s fwd p50 {q([x['fwd26'] for x in S], 50)} cm (reference pelvis moved p50 {q([x['ref26'] for x in S], 50)} cm)")  # fmt: skip
        print(f"   first lift-off s p10/50/90 {q([x['lift'] for x in S], 10)}/{q([x['lift'] for x in S], 50)}/{q([x['lift'] for x in S], 90)} (none in {sum(x['lift'] is None for x in S)}); "
              f"reference length s p50 {q([x['ref_s'] for x in S], 50)}; lift-off after the reference ended {sum(x['lift'] is not None and x['ref_s'] is not None and x['lift'] > x['ref_s'] for x in S)}/{sum(x['lift'] is not None for x in S)}")  # fmt: skip
        print(f"   slip (a planted foot > 1 cm) before any lift-off: {sum(x['slip'] is not None for x in S)}; 2 cm off the feet (any direction) with both feet still down before a lift-off or slip: {sum(x['first'] == 'shift2' for x in S)}")
        print(f"   (b) reference torso pitch change (pelvis pitch + waist pitch, command -> reference end) deg p10/50/90 {q([x['dpitch'] for x in S], 10)}/{q([x['dpitch'] for x in S], 50)}/{q([x['dpitch'] for x in S], 90)}; "
              f"reference pelvis height change cm p50 {q([x['dz'] for x in S], 50)}; larger knee change rad p50 {q([x['dknee'] for x in S], 50)}; "
              f"target ahead x m p50 {q([x['tx'] for x in S], 50)}, height m p50 {q([x['tz'] for x in S], 50)}, crouched {sum(x['pz'] is not None for x in S)}/{len(S)}")  # fmt: skip
    # forward lean vs shift: does the shift follow the reference's lean?
    X = [(x["dpitch"], x["maxfwd"]) for x in rows if x["dpitch"] is not None]
    Xb = np.array([(x["dpitch"], x["fwd26"]) for x in rows if x["dpitch"] is not None])
    if len(Xb) > 3:
        for lo, hi in ((-180, -5), (-5, 5), (5, 20), (20, 180)):
            S = [x for x in rows if x["dpitch"] is not None and lo <= x["dpitch"] < hi]
            print(f"   reference torso pitch change {lo}..{hi} deg: n {len(S)}, pelvis ahead of the feet at 2.6 s p50 {q([x['fwd26'] for x in S], 50)} cm, lost {sum(x['lost'] for x in S)}")
    if len(X) > 3:
        Xa = np.array(X)
        print(f"   all in-place reaches: corr(reference torso pitch change, max forward shift before lift-off) {np.corrcoef(Xa[:, 0], Xa[:, 1])[0, 1]:.2f} (n {len(X)})")
    for lo, hi in ((0, 2.5), (2.5, 99)):
        S = [x for x in rows if x["ref_s"] is not None and lo <= x["ref_s"] < hi]
        print(f"   reference {lo}-{hi} s: lost {sum(x['lost'] for x in S)}/{len(S)}")
    Path(a.an).with_name("q3_inplace_rows.json").write_text(json.dumps(rows, indent=0))


if __name__ == "__main__":
    main()
