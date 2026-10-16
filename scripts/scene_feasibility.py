#!/usr/bin/env python3
"""Sort a scene's tasks into the three buckets and rank the gaps (TODO 19.7; scene_discovery.feasibility). Sim venv.

    $VENV_SIM_PY scripts/scene_feasibility.py --scene outputs/scene_discovery/adt_apartment/scene_meta.json \\
        [--tasks tasks.jsonl] [--capability <table.yaml>] [--approach-edges <overlay.json>] [--out <dir>]

Without `--tasks` it checks the template space (every in / on / held goal over the graspable objects): the
upper bound of pick-and-place work in the scene before Claude proposes anything. Writes `<out>/tasks.jsonl`
(one result per task: bucket under the strict and the pilot policy, gaps, blocks, every check) and
`<out>/summary.json` (bucket counts under both policies, blocked reasons, the gap ranking). `<out>` defaults
to `<scene dir>/feasibility`. ADT-derived output: it stays under outputs/. `--approach-edges` replaces the scene
file's standable edges with an overlay (scene_file.with_approach_edges; the mars + ScaleBFM one is
outputs/scene_rsi_20261016/capability/stance/adt_apartment_mars.json, TODO 27.3.1); it also changes the template
space, which offers put-on goals only for surfaces with an edge.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.scene_discovery import feasibility, scene_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tasks", help="jsonl of {id, goal: [predicate, ...]}; default: the template space")
    ap.add_argument("--capability", default=str(REPO / "configs/scene_discovery/capability.yaml"))
    ap.add_argument("--approach-edges", help="overlay of each surface's standable edges (TODO 27.3.1)")
    ap.add_argument("--out")
    args = ap.parse_args()
    meta = scene_file.load(args.scene)
    if args.approach_edges:
        meta = scene_file.with_approach_edges(meta, args.approach_edges)
    cap = yaml.safe_load(Path(args.capability).read_text())
    tasks = (
        [json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
        if args.tasks
        else feasibility.templates(meta)
    )
    out = Path(args.out) if args.out else Path(args.scene).parent / "feasibility"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    checker = feasibility.Checker(meta, cap)
    results = [checker.check(t) for t in tasks]
    with (out / "tasks.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), default=float) + "\n")
    blocked = collections.Counter(b.split(": ", 1)[1] for r in results for b in r.blocked)
    summary = {
        "scene": str(args.scene),
        "tasks": len(results),
        "source": "--tasks " + args.tasks if args.tasks else "templates",
        "capability": str(args.capability),
        "approach_edges": args.approach_edges or "scene file",
        "buckets_strict": dict(sorted(collections.Counter(r.bucket("strict") for r in results).items())),
        "buckets_pilot": dict(sorted(collections.Counter(r.bucket("pilot") for r in results).items())),
        "invalid": sum(1 for r in results if r.invalid),
        "blocked_reasons": dict(blocked.most_common()),
        "gap_ranking": feasibility.gap_ranking(results),
        "bucket_1": sorted(r.task["id"] for r in results if r.bucket("strict") == 1),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[feasibility] {len(results)} tasks in {summary['seconds']} s -> {out}")
    print(f"  strict buckets {summary['buckets_strict']}   pilot-policy buckets {summary['buckets_pilot']}")
    print("  blocked:", ", ".join(f"{k} x{v}" for k, v in blocked.most_common(8)))
    print("  gaps (unlocks alone / in tasks):")
    for g in summary["gap_ranking"][:12]:
        print(f"    {g['gap']:32s} {g['unlocks_alone']:5d} {g['in_tasks']:5d}")


if __name__ == "__main__":
    main()
