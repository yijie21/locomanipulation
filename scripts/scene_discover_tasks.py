#!/usr/bin/env python3
"""Claude proposes a scene's tasks, the feasibility checker answers, Claude revises (TODO 19.10; plan decision 8).

    $VENV_SIM_PY scripts/scene_discover_tasks.py --scene outputs/scene_discovery/adt_apartment/scene_meta.json \\
        [--model opus] [--iterations 2] [--per-room 30] [--out <dir>]

Writes into `<out>` (default `<scene dir>/discovery/<UTC stamp>`): `claude_io.jsonl` (every exchange),
`iteration_NN.jsonl` (the tasks and verdicts of each round), `tasks.jsonl` (the final list, task file format
§6, `plan` empty until decision 10's plan step) and `report.json` (buckets per iteration, overlap with the
template space in `<scene dir>/feasibility/tasks.jsonl`, the gap counts). ADT-derived output: it stays under
outputs/ (decision 17). Opus takes minutes per call; the property pass cost about 0.5 USD per call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.scene_discovery import properties, proposals  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--capability", default=str(REPO / "configs/scene_discovery/capability.yaml"))
    ap.add_argument("--model", default="opus")
    ap.add_argument("--iterations", type=int, default=2, help="revision rounds after the first proposal")
    ap.add_argument("--per-room", type=int, default=30)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    scene = Path(args.scene)
    meta = json.loads(scene.read_text())
    capability = yaml.safe_load(Path(args.capability).read_text())
    out = Path(args.out) if args.out else scene.parent / "discovery" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out.mkdir(parents=True, exist_ok=True)
    ask = partial(_ask, model=args.model, log_path=out / "claude_io.jsonl")
    t0 = time.time()
    rows, history = proposals.discover(
        meta, capability, ask, iterations=args.iterations, per_room=args.per_room, model=args.model
    )
    for h in history:
        with (out / f"iteration_{h['iteration']:02d}.jsonl").open("w") as f:
            for r in h["tasks"]:
                f.write(json.dumps(r, default=float) + "\n")
    with (out / "tasks.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    templates_path = scene.parent / "feasibility" / "tasks.jsonl"
    templates = (
        [json.loads(line) for line in templates_path.read_text().splitlines()] if templates_path.exists() else []
    )
    io = [json.loads(line) for line in (out / "claude_io.jsonl").read_text().splitlines()]
    report = {
        "scene": str(scene),
        "model": args.model,
        "prompt_sha": proposals.prompt_sha(),
        "seconds": round(time.time() - t0, 1),
        "claude_calls": len(io),
        "cost_usd": round(sum(e.get("cost_usd") or 0.0 for e in io), 2),
        "iterations": [{k: h[k] for k in ("iteration", "buckets", "changed", "notes")} for h in history],
        "final": proposals.report(rows, templates),
        "templates_from": str(templates_path) if templates else None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"[discover] {len(rows)} tasks, buckets {report['final']['buckets']}, {report['claude_calls']} calls, "
          f"{report['cost_usd']} USD, {report['seconds']} s -> {out}")  # fmt: skip
    return 0


def _ask(prompt, schema, system, *, model, log_path):
    return properties.ask(prompt, schema, system, model, log_path, timeout_s=1800.0)


if __name__ == "__main__":
    raise SystemExit(main())
