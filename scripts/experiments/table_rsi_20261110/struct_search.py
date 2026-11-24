"""TODO 47.2: after a structural edit to one cell, the same search over the numbers it introduced and the proven ones.

Greedy, three stages: sweep every knob around the cell's current set, take the best plateau that beats what the cell
has by search.MARGIN, repeat. Rounds run the edited skills package on the cell's 16 training starts; the final set is
tested once on the cell's starts of pool_test2.

    struct_search.py run     # all stages, then the test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fast  # noqa: E402
import search as S  # noqa: E402
import table as T  # noqa: E402

TASK = "pick_WoodenSpoon"
D = T.OUT / "struct_spoon"
SKILLS = D / "skills"
KNOBS = {  # key: (default in the edited package, lo, hi); the SPOON_* are the edit's own, the rest proven or lift
    "pick.SPOON_TILT_DEG": (45.0, 25.0, 65.0),
    "pick.SPOON_PRE_M": (0.09, 0.05, 0.17),
    "pick.SPOON_FRAC": (0.5, 0.0, 0.7),
    "pick.SPOON_SIGN": (-1.0, -1.0, 1.0),
    "pick.SPOON_AIM_M": (0.033, 0.015, 0.051),
    "pick.TOP_MIN_D_M": (0.40, 0.30, 0.60),
    "pick.STANCE_CLEAR_M": (0.17, 0.12, 0.30),
    "pick.SLOW_LIFT_SPEED": (0.03, 0.01, 0.07),
    "pick.SLOW_DESCENT_SPEED": (0.025, 0.01, 0.07),
}
STAGES = 3


def values(key: str) -> list[float]:
    d, lo, hi = KNOBS[key]
    if key.endswith("SIGN"):
        return [v for v in (-1.0, 1.0) if v != d]
    vs = {round(lo + i * (hi - lo) / 6, 5) for i in range(7)}
    return sorted(v for v in vs if abs(v - d) > 1e-9)


def run_set(name: str, st: dict, pool: Path, labels: list, out: Path) -> dict:
    return dict(name=name, skills=str(SKILLS), set={**T.BASE, **st}, starts=labels, out=str(out))


def wins(d: Path) -> int | None:
    rs = list(T.rounds(d).values())
    return sum(r["ok"] for r in rs) if len(rs) == 16 else None


def main() -> None:
    labels = sorted(p.stem for p in T.TRAIN.glob(f"{TASK}_*.json"))
    cur, lines = {}, []
    job = D / "search_s0.json"
    job.write_text(json.dumps(dict(family="grasp_atomic", pool=str(T.TRAIN), memo=True, starts=labels,
                                   variants=[run_set("s0", {}, T.TRAIN, labels, D / "search/s0")])))  # fmt: skip
    fast.run(job, 8, "0,1")
    have = wins(D / "search/s0")
    lines.append(f"edited package as written: {have}/16")
    for stage in range(1, STAGES + 1):
        vs = []
        for k in KNOBS:
            if k in cur:
                continue
            for v in values(k):
                vs.append(run_set(f"s{stage}_{k.split('.')[1]}_{v:g}", {**cur, k: v}, T.TRAIN, labels,
                                  D / f"search/s{stage}/{k.split('.')[1]}/{v:g}"))  # fmt: skip
        job = D / f"search_s{stage}.json"
        job.write_text(
            json.dumps(dict(family="grasp_atomic", pool=str(T.TRAIN), memo=True, starts=labels, variants=vs))
        )
        fast.run(job, 8, "0,1")
        best = None
        for k in KNOBS:
            if k in cur:
                continue
            pts = {KNOBS[k][0]: have}
            for v in values(k):
                w = wins(D / f"search/s{stage}/{k.split('.')[1]}/{v:g}")
                if w is not None:
                    pts[v] = w
            x, s = S.plateau(pts)
            lines.append(
                f"  stage {stage} {k} given {cur}: "
                + " ".join(f"{a:g}:{pts[a]}" for a in sorted(pts))
                + f" -> {x:g} smoothed {s:.2f}"
            )
            if s >= have + S.MARGIN and (best is None or s > best[2]):
                best = (k, x, s, pts[x])
        if best is None:
            lines.append(f"stage {stage}: nothing beats {have} by {S.MARGIN}; stop")
            break
        cur[best[0]] = best[1]
        have = best[3]
        lines.append(f"stage {stage}: take {best[0]} = {best[1]:g} -> {have}/16 (smoothed {best[2]:.2f})")
    pool = T.OUT / "pool_test2"
    tl = sorted(p.stem for p in pool.glob(f"{TASK}_*.json"))
    job = D / "search_test.json"
    job.write_text(json.dumps(dict(family="grasp_atomic", pool=str(pool), memo=False, starts=tl, variants=[
        dict(name="edit", skills=str(SKILLS), set=dict(T.BASE), out=str(D / "test2/edit")),
        dict(name="edit_search", skills=str(SKILLS), set={**T.BASE, **cur}, out=str(D / "test2/edit_search"))])))  # fmt: skip
    fast.run(job, 8, "0,1")
    base = [r for r in T.rounds(T.OUT / "test2/base").values() if r["task"] == TASK]
    res = {n: sum(r["ok"] for r in T.rounds(D / "test2" / n).values()) for n in ("edit", "edit_search")}
    lines.append(f"final set {cur}; training {have}/16")
    lines.append(f"pool_test2, {len(base)} spoon starts: incumbent {sum(r['ok'] for r in base)}, edit as written {res['edit']}, "
                 f"edit + search {res['edit_search']}")  # fmt: skip
    (D / "struct_search.txt").write_text("\n".join(lines) + "\n")
    (D / "final_set.json").write_text(json.dumps(cur))
    print("\n".join(x for x in lines if not x.startswith("  ")))


if __name__ == "__main__":
    main()
