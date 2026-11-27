"""selfmodel_faildiag_20261126 revision 3 summary (measurement only).

3A (q3/lock3, fd_heft.py ha with --lock-waist / --lock legs|arm|root|waistlegs): the pelvis's forward shift from its feet
at 2.6 s (fd_heft_an.py json) per command and group, its ratio to the unlocked group's median, and the hand point
(wrist + R HAND_OFF) to the commanded target 1.2 s (60 frames) after the reference's end.
3B (q3/dur, fd_heft.py d3 --target-j --chain --durs): per selected reach and reference duration, lost inside that
reach's window (command frame .. next command + 1 or the reset's end), seconds from the command to the lost frame,
the hand point to the target at data3's label frame and 1.2 s after the motion's end (the worse hand of two), and
whether the original-duration run reproduced data3's record.

    python fd_rev3_an.py <out dir> [--an-lock <fd_heft_an json over q3/lock3>]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

FPS = 50
HAND_OFF = np.array([0.13, 0.0, 0.0])  # L.HAND_OFF (lml_selfmodel), the wrist-frame hand point
GROUPS = {None: "unlocked", "_lockwp": "waist", "_locklegs": "legs", "_lockarm": "arm only", "_lockroot": "root only", "_lockwaistlegs": "waist+legs"}


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def hand_err(rows, C, f, side, target):
    if f is None or f < 0 or f >= len(rows):
        return None
    k = "wl_" if side == "left" else "wr_"
    p = rows[f, [C[k + a] for a in "xyz"]]
    q = rows[f, [C[k + "q" + a] for a in "wxyz"]]
    return 100 * float(np.linalg.norm(p + qmat(q / np.linalg.norm(q)) @ HAND_OFF - np.asarray(target, float)))


def load(f):
    z = np.load(f, allow_pickle=True)
    return z["rows"], {str(c): i for i, c in enumerate(z["cols"])}, json.loads(str(z["meta"]))


def med(v):
    v = [x for x in v if x is not None]
    return None if not v else float(np.median(v))


def fmt(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def part_a(out: Path, an: Path) -> list[str]:
    R = json.loads(an.read_text())
    g = defaultdict(list)
    for r in R:
        m = re.match(r"ha_(ppbox_s29\d_r0)_c(\d)(?:_r(\d))?(_lockwp|_locklegs|_lockarm|_lockroot|_lockwaistlegs)?\.npz", r["file"])
        if not m:
            continue
        rows, C, meta = load(out / "q3" / "lock3" / r["file"])
        fe = meta["f_end"]
        he = hand_err(rows, C, None if fe is None else fe + 60, meta["side"], meta["target_w"])
        g[(f"{m.group(1)} c{m.group(2)}", GROUPS[m.group(4)])].append(dict(file=r["file"], fwd=r["at_2_6s"]["fwd_cm"], lift=r["lift_s"], he=he,
                                                                          he_end=hand_err(rows, C, fe, meta["side"], meta["target_w"])))  # fmt: skip
    lines = ["3A: pelvis ahead of the feet midpoint at 2.6 s (cm, 3 runs), ratio = group median / unlocked median; hand point to target 1.2 s after the reference end (cm)"]
    for cmd in sorted({k[0] for k in g}):
        base = med([x["fwd"] for x in g[(cmd, "unlocked")]])
        for grp in GROUPS.values():
            S = sorted(g.get((cmd, grp), []), key=lambda x: x["file"])
            if not S:
                continue
            lines.append(f"{cmd} {grp:10s} fwd {[round(x['fwd'], 1) for x in S]} ratio {med([x['fwd'] for x in S]) / base:.2f}; "
                         f"hand to target +1.2 s {[fmt(x['he']) for x in S]} (at the reference end {[fmt(x['he_end']) for x in S]}); lift {[x['lift'] for x in S]}")  # fmt: skip
    return lines


def part_b(out: Path) -> list[str]:
    tg = {(t["reset"], t["chain"]): t for t in json.loads((out / "q3" / "rev3_targets.json").read_text())}
    res = defaultdict(dict)
    for f in sorted((out / "q3" / "dur").glob("d3_w*_r*_c*_T*.npz")):
        m = re.match(r"d3_w(\d+)_r(\d+)_c(\d)_T(orig|[\d.]+)\.npz", f.name)
        reset, chain, T = int(m.group(2)), int(m.group(3)), m.group(4)
        rows, C, meta = load(f)
        rc = meta["reaches"][chain] if chain < len(meta["reaches"]) else None
        if rc is None:
            res[(reset, chain)][T] = dict(missing=True)
            continue
        nxt = meta["reaches"][chain + 1]["f_cmd"] if chain + 1 < len(meta["reaches"]) else meta["steps"]
        lost = meta["lost"]
        lost_here = lost is not None and rc["f_cmd"] <= lost <= nxt + 1
        mf = rc.get("motion_f") or (rc["f_end"] - rc["f_cmd"])
        f12 = rc["f_cmd"] + mf + 60
        fl = rc.get("f_label")

        def worst(fr):
            if fr is None or (lost is not None and fr >= lost):
                return None
            e = [hand_err(rows, C, fr, s, v["target_w"]) for s, v in rc["hands"].items()]
            return None if any(x is None for x in e) else max(e)

        res[(reset, chain)][T] = dict(lost=lost_here, lost_s=None if not lost_here else (lost - rc["f_cmd"]) / FPS, he_label=worst(fl), he_12=worst(f12),
                                      same=meta["compare"].get("same"), why=meta["compare"].get("why", meta["compare"].get("diff")), motion_s=rc.get("motion_s") or rc["reach_s"])  # fmt: skip
    order = ["orig", "2.5", "2.0", "1.5"]
    lines = ["3B: per reach (reset, chain, original reference s): lost here / s from the command to lost / hand to target at the label and 1.2 s after the motion (cm)"]
    for k in sorted(tg):
        v = res.get(k, {})
        cells = []
        for T in order:
            x = v.get(T)
            if x is None:
                cells.append(f"{T}: not run")
            elif x.get("missing"):
                cells.append(f"{T}: reach not reached")
            else:
                cells.append(f"{T}: {'LOST ' + fmt(x['lost_s']) + ' s' if x['lost'] else 'kept'} label {fmt(x['he_label'])} +1.2 {fmt(x['he_12'])}"
                             + (f" [orig reproduced {x['same']} {x['why'] or ''}]" if T == "orig" else ""))  # fmt: skip
        lines.append(f"{k[0]} c{k[1]} ({tg[k]['ref_s']:.2f} s) | " + " | ".join(cells))
    lines.append("totals:")
    for T in order:
        X = [res[k][T] for k in tg if T in res.get(k, {}) and not res[k][T].get("missing")]
        kept = [x for x in X if not x["lost"]]
        lines.append(f"  {T:4s}: run {len(X)}/12, lost here {sum(x['lost'] for x in X)}; lost s p50 {fmt(med([x['lost_s'] for x in X]))}; "
                     f"kept {len(kept)}: hand to target at the label p50 {fmt(med([x['he_label'] for x in kept]))} cm, +1.2 s p50 {fmt(med([x['he_12'] for x in kept]))} cm; "
                     f"all runs +1.2 s before any lost p50 {fmt(med([x['he_12'] for x in X]))} cm (n {sum(x['he_12'] is not None for x in X)})"
                     + (f"; orig reproduced {sum(x['same'] is True for x in X)}/{len(X)}" if T == "orig" else ""))  # fmt: skip
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--an-lock", default="")
    a = ap.parse_args()
    out = Path(a.out)
    lines = []
    if a.an_lock:
        lines += part_a(out, Path(a.an_lock))
    if (out / "q3" / "dur").exists():
        lines += part_b(out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
