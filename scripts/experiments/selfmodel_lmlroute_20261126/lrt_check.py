"""selfmodel_lmlroute_20261126's consistency checks (the entry's 一致性检查), one controller; read-only.

    $VPY lrt_check.py --backend scalebfm [--out OUT] [--ar AR] [--eps 30:120]

OUT: this round's records (C = route, CR = route_reachable; default $LC_OUT or loco_outputs/selfmodel_lmlroute_20261126);
AR: the records A (alone) and R (reachable) are paired with (revision 1: A and R run in this round on the same looks, so
the default is OUT; the smoke passes its own).
K1 every (task, ep) has C and CR records; worker errors listed.
K2 C without a route = A, CR without a route = R: every key of the episode record but the wall-clock ones (wall_s,
   plan_s, think_s, ...), the group's name and the lmlroute block; every array of segs / arm npz but reset and method_id.
K3 the same episode's C and CR: the same request (md5), the same answer file (md5 recorded = the file's md5 now), the
   same decision (routed or not); a routed episode's route command identical (reaches[0] and d3[0] but wall-clock
   fields; segment 0's arrays but reset / method_id).
K4 the answers: status counts, request md5 = the answer's, the planner's kinematics check (fk_hand_err_mm).
Ends with one line "CHECK ok" or "CHECK FAIL: ...".
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
TASKS = ("IThorBathroomLiftCandleFromFloor", "IThorKitchenApproachTableAndLiftApple", "IThorKitchenApproachTableAndLiftBowl",
         "IThorKitchenApproachTableAndLiftSaltShaker", "IThorKitchenTakeStatueFromCabinet",
         "IThorLivingRoomApproachTableAndLiftDumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan")  # fmt: skip
SHORT = dict(zip(TASKS, ("candle", "apple", "bowl", "salt", "statue_cab", "dumbbell", "wcan")))
CLOCK = {"wall_s", "plan_s", "t_wall", "wall", "elapsed_s", "think_s"}
SKIP_TOP = {"method", "lmlroute"}
SKIP_ARR = {"reset", "method_id"}
PAIR = (("route", "alone"), ("route_reachable", "reachable"))


def strip(x, top=False):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in CLOCK and not (top and k in SKIP_TOP)}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def diff(a, b, path=""):
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        out = [f"{path}.{k}: missing on one side" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (u, v) in enumerate(zip(a, b)):
            out += diff(u, v, f"{path}[{i}]")
        return out
    return [] if a == b or (a != a and b != b) else [f"{path}: {a!r} vs {b!r}"]


def npz(base: Path, be: str, task: str, g: str, ep: int, arm: bool = False):
    p = base / be / "segs" / task / g / ("arm/" if arm else "") / f"ep{ep:+05d}.npz"
    return np.load(p) if p.exists() else None


def arrays_equal(z1, z2, sl=None) -> list[str]:
    if (z1 is None) != (z2 is None):
        return ["npz exists on one side"]
    if z1 is None:
        return []
    bad = [] if set(z1.files) == set(z2.files) else ["keys differ"]
    for k in sorted(set(z1.files) & set(z2.files) - SKIP_ARR):
        a, b = z1[k], z2[k]
        if sl is not None:
            if k in ("radius", "link"):
                continue
            a, b = sl(k, a), sl(k, b)
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            bad.append(k)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", default=os.environ.get("LC_OUT", str(ROOT / "selfmodel_lmlroute_20261126")))
    ap.add_argument("--ar", default=None)  # revision 1: default = --out
    ap.add_argument("--eps", default="30:120")
    ap.add_argument("--ep-list", default=None)
    a = ap.parse_args()
    be, O = a.backend, Path(a.out)
    AR = Path(a.ar) if a.ar else O
    eps = [int(v) for v in a.ep_list.split(",")] if a.ep_list else list(range(*(int(v) for v in a.eps.split(":"))))
    fails = []
    load = lambda base, t, g, e: (lambda p: json.loads(p.read_text()) if p.exists() else None)(base / be / "episodes" / t / g / f"ep{e:+05d}.json")  # noqa: E731

    # K1
    miss, errs = [], []
    for t in TASKS:
        for e in eps:
            for g in ("route", "route_reachable"):
                r = load(O, t, g, e)
                if r is None:
                    miss.append((SHORT[t], g, e))
                elif "error" in r:
                    errs.append((SHORT[t], g, e, r["error"][:80]))
    print(f"K1 {be}: {len(TASKS) * len(eps) * 2} records expected; missing {len(miss)} {miss[:6]}; worker errors {len(errs)} {errs[:4]}")
    if miss or errs:
        fails.append(f"K1 missing {len(miss)} errors {len(errs)}")

    # K2, K3
    n2 = Counter()
    n3 = Counter()
    bad2, bad3 = [], []
    status = Counter()
    fk, reqok = [], Counter()
    for t in TASKS:
        for e in eps:
            rc = {g: load(O, t, g, e) for g in ("route", "route_reachable")}
            if any(v is None or "error" in v for v in rc.values()):
                continue
            for g, base in PAIR:
                r = rc[g]
                lr = r.get("lmlroute") or {}
                if lr.get("route_on"):
                    n2[f"{g} routed"] += 1
                    continue
                ref = load(AR, t, base, e)
                if ref is None:
                    bad2.append((SHORT[t], g, e, f"no {base} record"))
                    continue
                d = diff(strip(r, True), strip(ref, True))
                ar = arrays_equal(npz(O, be, t, g, e), npz(AR, be, t, base, e)) + [f"arm/{k}" for k in arrays_equal(npz(O, be, t, g, e, True), npz(AR, be, t, base, e, True))]
                if d or ar:
                    bad2.append((SHORT[t], g, e, d[:3], ar[:5]))
                    n2[f"{g} unrouted DIFF"] += 1
                else:
                    n2[f"{g} unrouted = {base}"] += 1
            ra, rb = ((rc[g].get("lmlroute") or {}).get("route") or {} for g in ("route", "route_reachable"))
            if ra:
                status[ra.get("status")] += 1
                if ra.get("fk_hand_err_mm") is not None:
                    fk.append(ra["fk_hand_err_mm"])
                reqok[bool(ra.get("req_match")) and bool(rb.get("req_match"))] += 1
                f = O / "routes" / "routes" / be / t / f"ep{e:+05d}.npz"
                fmd5 = hashlib.md5(f.read_bytes()).hexdigest() if f.exists() else None
                same = (ra.get("req_md5_here") == rb.get("req_md5_here") and ra.get("file_md5") == rb.get("file_md5") == fmd5
                        and ra.get("routed_here") == rb.get("routed_here"))  # fmt: skip
                if not same:
                    bad3.append((SHORT[t], e, "request / answer / decision differ"))
                    n3["DIFF"] += 1
                    continue
            elif rb:
                bad3.append((SHORT[t], e, "a route record in CR only"))
                n3["DIFF"] += 1
                continue
            on = [(rc[g].get("lmlroute") or {}).get("route_on") for g in ("route", "route_reachable")]
            if on[0] != on[1]:
                bad3.append((SHORT[t], e, f"route_on {on}"))
                n3["DIFF"] += 1
                continue
            if not on[0]:
                n3["no route, same answer"] += 1
                continue
            c, cr = rc["route"], rc["route_reachable"]
            d = diff(strip(c["reaches"][0]), strip(cr["reaches"][0])) + diff(strip(c["d3"][0]), strip(cr["d3"][0]))
            z1, z2 = npz(O, be, t, "route", e), npz(O, be, t, "route_reachable", e)
            n0, r0 = int(z1["n_steps"][0]), int(z1["n_ref"][0])
            nseg = len(z1["n_steps"])

            def sl(k, x, n0=n0, r0=r0, nseg=nseg):
                if k in ("RT", "RQ", "RP", "XR"):
                    return x[:r0]
                if len(x) == nseg and k not in ("S", "W", "J", "C", "HQ", "KP", "X"):
                    return x[:1]
                return x[:n0]

            ar = arrays_equal(z1, z2, sl) + [f"arm/{k}" for k in arrays_equal(npz(O, be, t, "route", e, True), npz(O, be, t, "route_reachable", e, True), sl)]
            if d or ar:
                bad3.append((SHORT[t], e, d[:3], ar[:5]))
                n3["route command DIFF"] += 1
            else:
                n3["route command same"] += 1
    print(f"K2 {be}: {dict(n2)}; differences {len(bad2)} {bad2[:4]}")
    print(f"K3 {be}: {dict(n3)}; differences {len(bad3)} {bad3[:4]}")
    if bad2:
        fails.append(f"K2 {len(bad2)}")
    if bad3:
        fails.append(f"K3 {len(bad3)}")
    fk = np.array(fk)
    print(f"K4 {be}: answers by status {dict(status)}; request md5 = answer's (C and CR) {dict(reqok)}; planner FK vs ours "
          f"{np.median(fk) if len(fk) else float('nan'):.3f} mm median, {fk.max() if len(fk) else float('nan'):.3f} max")  # fmt: skip
    if reqok.get(False):
        fails.append(f"K4 request md5 {reqok[False]}")
    if status.get("ERROR"):
        fails.append(f"K4 planner errors {status['ERROR']}")
    print("CHECK ok" if not fails else f"CHECK FAIL: {'; '.join(fails)}")


if __name__ == "__main__":
    main()
