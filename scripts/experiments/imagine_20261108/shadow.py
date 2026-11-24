"""Shadow mode of the imagined Motor (TODO 45.7, user 2026-11-09): an iteration's drafts and its incumbent are imagined on
the diagnostic starts before their real rounds run, and the score goes to the hidden tree. No role reads it and nothing in
the loop acts on it; shadow_report.py sets it beside the real rounds afterwards, on drafts written after the rule froze.

    RSI_SHADOW=<this file> loop.py run <family>     # the loop calls it, between the Engineers and the Composer
    $VENV_SIM_PY shadow.py --family F --pool <pool_diag> --candidates <list.json> --out <dir> --procs 6

The score is the one measured on the recorded drafts, unchanged: imagined success x (1 - worst body-table odds of a miss),
summed over the starts. Writes <out>/shadow.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import imagine as IM  # noqa: E402
from contact_paired import miss  # noqa: E402

CHUNKS = 2  # shards of one candidate's starts; each one builds a world, so more of them only adds start-up time


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def rule() -> dict:
    """What the score depends on, so a later change to the rule cannot pass for the frozen one."""
    return dict(
        kin_motor=digest(HERE / "kin_motor.py"),
        body_table=digest(IM.ROOT / "outputs" / "scene_rsi_20261016" / "body" / "reach_knn_envelope.npz"),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args(argv)
    os.chdir(IM.ROOT)  # the family file's paths are relative to the repository
    spec = json.loads(a.candidates.read_text())
    labels = sorted(p.stem for p in a.pool.glob("*.json") if not p.stem.startswith("_"))
    jobs = []
    for c in spec:
        d = a.out / str(c["number"])
        d.mkdir(parents=True, exist_ok=True)
        jobs += [
            (a.family, str(Path(c["dir"]) / "skills"), c["set"], str(a.pool), str(d), labels[i::CHUNKS])
            for i in range(CHUNKS)
            if labels[i::CHUNKS]
        ]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as p:
        p.map(IM.work, jobs, chunksize=1)
    rows = []
    for c in spec:
        rounds = {}
        for l in labels:
            r = json.loads((a.out / str(c["number"]) / f"{l}.json").read_text())
            rounds[l] = dict(ok=bool(r["ok"]), pred_p_miss=miss(r), score=float(r["ok"]) * (1 - miss(r)))
        rows.append(
            dict(
                number=c["number"],
                hash=c["hash"],
                incumbent=bool(c["incumbent"]),
                score=sum(v["score"] for v in rounds.values()),
                rounds=rounds,
            )
        )
    (a.out / "shadow.json").write_text(
        json.dumps(
            dict(family=a.family, labels=labels, rule=rule(), wall_s=round(time.time() - t0, 1), candidates=rows),
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
