"""What the Composer is told by the imagined Motor (TODO 45.7.2, user 2026-11-09 "把虚拟结果给 Composer").

The Composer already has every draft's real rounds on the diagnostic starts, so on those starts the imagined result adds
nothing. What it does not have is any word about the starts outside them. Each draft and the incumbent are imagined with
the frozen arrival-error rule (arrival_v1/frozen.json), K draws per start, on the diagnostic starts and on a further pool
that no real round runs on. The view says which further starts look doubtful and why, and how far the same check agreed
with the real rounds of the diagnostic starts. It deliberately has no total and no ranking.

    RSI_IMAGINE=<this file> loop.py run grasp_floor     # the loop calls it, after the drafts' real rounds
    $VENV_SIM_PY composer_view.py --family F --diag-pool <pool_diag> --extra-pool <pool_imagine> --forbid <hidden pool>...
                 --candidates <list.json> --out <dir> --procs 6

Writes <out>/imagined.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRA_SEED = 730000  # above the family's diagnostic (430000), acceptance (530000) and sealed (630000) ranges
EXTRA_N = 40
K = 8  # draws per start, the frozen rule's
DOUBT = 0.25  # a start is doubtful below this share of draws that succeed, each weighted by the body table's odds
CHUNKS = 2
WHY_TOP = 2


def miss(r: dict) -> float:
    return max([float(s["pred_p_miss"]) for s in r.get("steps") or [] if s.get("pred_p_miss") is not None] or [0.0])


def start_row(draws: list[dict]) -> dict:
    """One start of one package: the K imagined rounds in one line."""
    why = Counter(str(d.get("why") or "?").split(": ", 1)[-1][:40] for d in draws if not d.get("ok"))
    return dict(
        task=draws[0]["task"],
        score=round(sum(float(d["ok"]) * (1 - miss(d)) for d in draws) / len(draws), 3),
        body_p_miss=round(sum(miss(d) for d in draws) / len(draws), 2),
        why=[f"{n} of {len(draws)}: {w}" for w, n in why.most_common(WHY_TOP)],
    )


def summarize(spec: list[dict], rows: dict, diag: list[str], extra: list[str], real: dict) -> dict:
    """`rows` {number: {label: start_row}}, `real` {number: {label: bool}} (empty for rounds that are not there). No word on
    the incumbent: on the 40 gate starts its imagined score ranks its rounds no better than chance (AUC 0.46), and "doubtful
    here, not doubtful for the incumbent" won as often as the starts both looked fine on (0.50 and 0.55)."""
    out = []
    for c in spec:
        if c["incumbent"]:
            continue
        n = c["number"]
        doubtful = sorted(x for x in extra if rows[n][x]["score"] < DOUBT)
        check = dict(doubtful=0, doubtful_real_wins=0, others=0, others_real_wins=0)
        for x in diag:
            if x in real.get(n, {}):
                k = "doubtful" if rows[n][x]["score"] < DOUBT else "others"
                check[k] += 1
                check[f"{k}_real_wins"] += int(real[n][x])
        out.append(
            dict(
                number=n,
                hash=c["hash"],
                doubtful=[dict(start=x, **rows[n][x]) for x in doubtful],
                per_task={
                    t: dict(
                        starts=sum(rows[n][x]["task"] == t for x in extra),
                        doubtful=sum(rows[n][x]["task"] == t for x in doubtful),
                    )
                    for t in sorted({rows[n][x]["task"] for x in extra})
                },
                check_on_diagnostic_starts=check,
            )
        )
    return dict(extra_starts=len(extra), draws_per_start=K, doubtful_below=DOUBT, candidates=out)


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--diag-pool", type=Path, required=True)
    ap.add_argument("--extra-pool", type=Path, required=True)
    ap.add_argument("--forbid", type=Path, nargs="*", default=[])
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args(argv)

    import multiprocessing as mp

    sys.path.insert(0, str(HERE))
    import arrival_run as AR
    import imagine as IM

    IM.setup_paths()
    import family as F

    if a.family != AR.FAMILY:
        raise SystemExit(f"the arrival errors were measured on {AR.FAMILY} only, not on {a.family}")
    frozen = json.loads((AR.BASE / "arrival_v1" / "frozen.json").read_text())
    now = dict(pool=digest(AR.POOL), arrival_motor=digest(HERE / "arrival_motor.py"), kin_motor=digest(HERE / "kin_motor.py"))  # fmt: skip
    if any(frozen[k] != v for k, v in now.items()):
        raise SystemExit(f"the rule is no longer the frozen one: {now}")
    os.chdir(IM.ROOT)  # the family file's paths are relative to the repository
    if not (a.extra_pool / "_pool.json").exists():
        F.draw(F.load(a.family), "diag", a.extra_pool, seed=EXTRA_SEED, n=EXTRA_N)
    diag, extra = F.labels(a.diag_pool), F.labels(a.extra_pool)
    for p in [a.diag_pool, *a.forbid]:
        clash = set(extra) & set(F.labels(p))
        if clash:
            raise SystemExit(f"the extra pool shares {len(clash)} starts with {p}")
    spec = json.loads(a.candidates.read_text())
    root = a.out / "rounds"
    jobs = [
        dict(id=f"{c['number']}_{name}", pkg=str(Path(c["dir"]) / "skills"), set=c["set"], pool=str(pool.resolve()), labels=labels)
        for c in spec
        for name, pool, labels in (("diag", a.diag_pool, diag), ("extra", a.extra_pool, extra))
    ]  # fmt: skip
    tasks = [(j, j["labels"][i::CHUNKS], str(root), K, True, True) for j in jobs for i in range(CHUNKS)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as p:
        p.map(AR.work, tasks, chunksize=1)
    rows, real = {}, {}
    for c in spec:
        n = c["number"]
        rows[n] = {}
        for name, labels in (("diag", diag), ("extra", extra)):
            for x in labels:
                draws = [json.loads((root / f"{n}_{name}" / f"k{i}" / f"{x}.json").read_text()) for i in range(K)]
                rows[n][x] = start_row(draws)
        real[n] = {x: bool(json.loads((Path(c["diag"]) / f"{x}.json").read_text())["ok"]) for x in diag if (Path(c["diag"]) / f"{x}.json").exists()}  # fmt: skip
    view = summarize(spec, rows, diag, extra, real)
    (a.out / "imagined.json").write_text(json.dumps(view, indent=1))
    (a.out / "provenance.json").write_text(json.dumps(dict(rule=now, wall_s=round(time.time() - t0, 1)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
