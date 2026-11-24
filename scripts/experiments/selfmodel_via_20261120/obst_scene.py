"""selfmodel_via_20261120, sixth step (the plan entry's "第六步"): replays with the paths' real boxes standing in the simulation.

    python obst_scene.py <obst_routes.py output dir> <base run dir> [<run dir> ...] [--main 0,1,2,3,4,6] [--pool 2,3,4,6]
        [--at 4]

Each run is a REACH_VIA_SLOTS replay of the routes (obst_slots.py's slots.json; reset i is slot i % N in pass i // N) with
REACH_GAP_OBST=1: all.npz carries the contact record of every reach (gap_touch: wrist + palm + fingers, the right arm, any
robot body; gap_t0: a contact in the first step after the boxes were placed; gap_box: where the boxes stood). A run directory
named <run>_open beside a run is the same run on the open floor (the boxes parked) and is used for the last table.

Per reach (m: the margin its route was planned with, cm):
  CONTACT   the right arm (shoulder to hand), or any robot body, penetrates a box between the command and the settled end
  ARRIVES   the palm's settled end (the first palm sphere, the last 0.5 s) is within 2 cm of the wanted end -- the end of the
            original command's plan (comp_ref0 in a compensated run)
  SUCCESS   arrives, no robot body in contact, no fall
Reaches whose boxes already overlapped the robot when they were placed (gap_t0) are left out and counted.
NEEDED MARGIN m*: the smallest margin from which on every planned margin up to MAIN's largest has a contact rate of at most
5 %; "over" when the largest one does not. The main analysis uses the base scenes planned at every margin in MAIN (0-6 cm).
Intervals: base scenes resampled (2000 times, 95 %). Later runs are paired with the base run by (slot, pass).
--main: the margins a base scene must be planned at to count (MAIN); --pool: the margins whose contacts are pooled for the
difference between two runs; --at: the margin the success rates are compared at. The defaults are the plan entry's.
Checks printed first: the boxes' recorded placement against the route's boxes in the reach's heading frame.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_model as VM  # noqa: E402

ARGV = sys.argv[1:]


def opt(name: str, default: str) -> str:
    if name not in ARGV:
        return default
    k = ARGV.index(name)
    v = ARGV[k + 1]
    del ARGV[k : k + 2]
    return v


ARRIVE, RATE = 0.02, 0.05
MAIN = tuple(int(x) for x in opt("--main", "0,1,2,3,4,6").split(","))
POOL = tuple(int(x) for x in opt("--pool", "2,3,4,6").split(","))
AT = int(opt("--at", "4"))
NAME = re.compile(r"_s(\d+)_f\d+_m(\d+)$")


def needed(rate: dict) -> float:
    """rate {margin: contact rate} -> the smallest margin from which on every margin's rate is <= RATE; inf: none."""
    best = np.inf
    for m in sorted(rate, reverse=True):
        if not rate[m] <= RATE:
            break
        best = m
    return best


def load(R: Path, D: Path) -> dict:
    """One run's reaches: scene s, margin m, key (slot, pass), fell, start (overlap at placement), arm, body, end (m; inf
    after a fall), and the largest placement error of its boxes (m, rad)."""
    z, T = dict(np.load(D / "all.npz")), VM.load_table(D)
    names = json.load(open(D / "routes.json"))["names"]
    X = np.load(D / "arm.npz")["X"]
    slots = json.load(open(D / "asrun" / "slots.json"))["names"]
    assert "gap_touch" in z or D.name.endswith("_open"), f"{D}: no contact record (REACH_GAP_OBST=1?)"
    n, palm = len(z["reset"]), int(np.where(T["palm"])[0][0])
    end = np.full(n, np.inf)
    for j, seg in enumerate(T["seg"]):
        want = z["comp_ref0"][seg].astype(float) if "comp_ref0" in z else T["ref"][j].astype(float)
        ticks = X[T["a"][j] : T["b"][j]].astype(float)
        end[seg] = float(np.linalg.norm(ticks[-VM.TAIL :, palm].mean(0) - want[-1, palm]))
    sm = np.array([[int(x) for x in NAME.search(names[r]).groups()] for r in z["via_route"]])
    i = z["reset"] % 100000
    out = dict(
        s=sm[:, 0],
        m=sm[:, 1],
        slot=i % len(slots),
        pas=i // len(slots),
        fell=z["fell"].astype(bool),
        end=end,
        n_slot=len(slots),
    )
    assert all(slots[k] == names[r] for k, r in zip(out["slot"], z["via_route"])), "slot and route disagree"
    if "gap_touch" in z:
        out.update(
            start=z["gap_t0"].astype(bool),
            arm=z["gap_touch"][:, 1].astype(bool),
            body=z["gap_touch"][:, 2].astype(bool),
        )
        dp, dy = 0.0, 0.0
        for k, r in enumerate(z["via_route"]):  # the placement against the route's boxes, in the reach's heading frame
            pr = np.load(R / "prep" / f"prep_{names[r]}.npz")
            yaw_p = float(np.arctan2(pr["pelvis_R"][1, 0], pr["pelvis_R"][0, 0]))
            c, s = np.cos(yaw_p), np.sin(yaw_p)
            want = np.c_[(pr["boxes"][:, :2] - pr["pelvis_p"][:2]) @ np.array([[c, -s], [s, c]]), pr["boxes"][:, 2:]]
            p0, yaw = z["start"][k][:3].astype(float), float(z["start"][k][3])
            c, s = np.cos(yaw), np.sin(yaw)
            got = z["gap_box"][k][: len(want)].astype(float)
            assert np.isfinite(got).all() and np.isnan(z["gap_box"][k][len(want) :]).all(), (
                "boxes placed != the route's"
            )
            here = np.c_[(got[:, :2] - p0[:2]) @ np.array([[c, -s], [s, c]]), got[:, 2:3], got[:, 7:]]
            dp = max(dp, float(np.abs(here - want).max()))
            box_yaw = 2 * np.arctan2(got[:, 6], got[:, 3])  # the boxes turn about z only
            dy = max(dy, float(np.abs((box_yaw - (yaw - yaw_p) + np.pi) % (2 * np.pi) - np.pi).max()))
        out.update(place_m=dp, place_rad=dy)
    return out


def table(tag: str, r: dict, base: set) -> dict:
    """Prints a run's rows per margin; returns per-reach arrays restricted to the reaches that count."""
    keep = ~r["start"]
    print(f"{tag}: {len(keep)} reaches, {int(r['fell'].sum())} falls, {int(r['start'].sum())} left out (boxes placed onto the "
          f"robot); placement check: {1000 * r['place_m']:.3f} mm, {r['place_rad']:.5f} rad")  # fmt: skip
    print(
        "   m cm  scenes  reaches   contact arm   contact body   arrives   success      (main: scenes planned at every margin of --main)"
    )
    for m in sorted(set(r["m"].tolist())):
        for main in (False, True):
            i = keep & (r["m"] == m) & (np.isin(r["s"], list(base)) if main else True)
            if main and (m not in MAIN or not i.any()):
                continue
            ok = (r["end"][i] <= ARRIVE) & ~r["body"][i] & ~r["fell"][i]
            print(f"  {m:5d}  {len(set(r['s'][i].tolist())):6d}  {int(i.sum()):7d}   {r['arm'][i].mean():11.3f}   {r['body'][i].mean():12.3f}   "
                  f"{np.mean(r['end'][i] <= ARRIVE):7.3f}   {ok.mean():7.3f}      {'main' if main else 'all'}")  # fmt: skip
    return keep


def boot(scenes: np.ndarray, f, n: int = 2000) -> tuple:
    """f(rows) -> a number; rows grouped by scene, scenes resampled."""
    rng, u = np.random.default_rng(1), np.unique(scenes)
    g = [np.where(scenes == x)[0] for x in u]
    v = np.array([f(np.concatenate([g[q] for q in rng.integers(len(g), size=len(g))])) for _ in range(n)])
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5), v


def mstar(r: dict, rows: np.ndarray, part: str) -> float:
    return needed({m: r[part][rows][r["m"][rows] == m].mean() for m in MAIN if (r["m"][rows] == m).any()})


def main() -> None:
    R, dirs = Path(ARGV[0]), [Path(x) for x in ARGV[1:]]
    C = [json.load(open(p)) for p in sorted(R.glob("cands_*of*.json"))]
    ok = {(p["s"], p["m"]) for c in C for p in c["plans"] if p["ok"]}
    scenes = sorted({s for s, m in ok if m == 0})
    base = {s for s in scenes if all((s, m) in ok for m in MAIN)}
    margins = C[0]["margins"]
    print(f"{R.name}: {len(scenes)} base scenes; planned by margin: " + ", ".join(f"{m} cm {sum((s, m) in ok for s in scenes)}" for m in margins)
          + f"; planned at every margin of {MAIN}: {len(base)}")  # fmt: skip
    runs = [load(R, D) for D in dirs]
    keeps = [table(D.name, r, base) for D, r in zip(dirs, runs)]
    if not base:
        print(f"no base scene is planned at every margin of {MAIN}: nothing more to compare")
        return
    for D, r, keep in zip(dirs, runs, keeps):
        rows = np.where(keep & np.isin(r["s"], list(base)) & np.isin(r["m"], MAIN))[0]
        for part in ("arm", "body"):
            lo, hi, v = boot(r["s"][rows], lambda q: mstar(r, rows[q], part))
            print(f"{D.name}: needed margin by the {part}'s contacts {mstar(r, rows, part):g} cm ({lo:g} to {hi:g}; over {MAIN[-1]} cm in "
                  f"{np.mean(np.isinf(v)):.2f} of the resamples)")  # fmt: skip
    a, ka = runs[0], keeps[0]
    ia = {(s, p): k for k, (s, p) in enumerate(zip(a["slot"], a["pas"]))}
    for D, b, kb in zip(dirs[1:], runs[1:], keeps[1:]):
        pair = np.array([(ia[(s, p)], k) for k, (s, p) in enumerate(zip(b["slot"], b["pas"])) if (s, p) in ia])
        pair = pair[ka[pair[:, 0]] & kb[pair[:, 1]] & np.isin(a["s"][pair[:, 0]], list(base))]
        sc, mm = a["s"][pair[:, 0]], a["m"][pair[:, 0]]
        print(f"{D.name} against {dirs[0].name}: {len(pair)} pairs on {len(set(sc.tolist()))} main base scenes")
        for part in ("arm", "body"):
            q = np.isin(mm, POOL)
            d = lambda w: b[part][pair[q][w, 1]].mean() - a[part][pair[q][w, 0]].mean()  # noqa: E731
            lo, hi, _ = boot(sc[q], d)
            print(f"  contact by the {part}, margins {POOL} cm pooled: {a[part][pair[q, 0]].mean():.3f} there, {b[part][pair[q, 1]].mean():.3f} "
                  f"this run; difference {d(np.arange(q.sum())):+.3f} ({lo:+.3f} to {hi:+.3f})")  # fmt: skip
            ms = lambda w: mstar(a, pair[w, 0], part) - mstar(b, pair[w, 1], part)  # noqa: E731
            lo, hi, v = boot(sc, ms)
            print(f"  needed margin by the {part}: {mstar(a, pair[:, 0], part):g} cm there, {mstar(b, pair[:, 1], part):g} cm this run; there "
                  f"minus this run in the resamples: {lo:g} to {hi:g} cm, not a number (both over) in {np.mean(np.isnan(v)):.2f}")  # fmt: skip
        good = lambda r, w: (r["end"][w] <= ARRIVE) & ~r["body"][w] & ~r["fell"][w]  # noqa: E731
        q = mm == AT
        d = lambda w: good(b, pair[q][w, 1]).mean() - good(a, pair[q][w, 0]).mean()  # noqa: E731
        lo, hi, _ = boot(sc[q], d)
        print(f"  success at {AT} cm: {good(a, pair[q, 0]).mean():.3f} there, {good(b, pair[q, 1]).mean():.3f} this run; difference "
              f"{d(np.arange(q.sum())):+.3f} ({lo:+.3f} to {hi:+.3f}); falls {int(a['fell'].sum())} there, {int(b['fell'].sum())} this run")  # fmt: skip
    for D, r in zip(dirs, runs):  # the open-floor twin: overlap by the 24 spheres against the real contact
        O = D.with_name(D.name + "_open")
        if not (O / "scene_check.npz").exists():
            continue
        o, zo = dict(np.load(O / "scene_check.npz")), load(R, O)
        io = {int(x): k for k, x in enumerate(o["reset"])}
        ir = {(s, p): k for k, (s, p) in enumerate(zip(r["slot"], r["pas"]))}
        both = [(ir[(s, p)], io[s + p * zo["n_slot"]]) for s, p in zip(zo["slot"], zo["pas"])
                if (s, p) in ir and s + p * zo["n_slot"] in io]  # fmt: skip
        both = np.array([x for x in both if not r["start"][x[0]]])
        real, geo = r["arm"][both[:, 0]], o["actual"][both[:, 1]].min(1) < 0
        print(f"{D.name} against its open-floor twin, {len(both)} reaches: real contact by the arm {real.mean():.3f}, overlap of the 24 "
              f"spheres on the open floor {geo.mean():.3f}; both {np.mean(real & geo):.3f}, contact only {np.mean(real & ~geo):.3f}, "
              f"overlap only {np.mean(~real & geo):.3f}, agreement {np.mean(real == geo):.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
