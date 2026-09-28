"""Scene capability sweep: which box-on-platform layouts the bimanual expert + SONIC grasp at least half the time.

    python -m locomani.tasks.crate.sweep cells  configs/sweeps/<sweep>.yaml               # list the cells, touch nothing
    python -m locomani.tasks.crate.sweep next   configs/sweeps/<sweep>.yaml --instance 0 --gpu 0  # claim one
    python -m locomani.tasks.crate.sweep report configs/sweeps/<sweep>.yaml               # table -> outputs/sweeps/<sweep>/

A sweep file names a base experiment, the values every cell overrides (`set`), a `centre` for every axis and the
cells: a `grid` (every combination, for the box geometry) and `chains` (one axis at a time, levels running outward
from the centre). A cell is an ordinary experiment file under `outputs/sweeps/<sweep>/cells/`, collected by
`run_overnight.sh <cell> --line bimanual --to collect` in a loop typed in the terminal (docs/scene_sweep.md), and its
`collect.stop_rule` stops it once the Wilson interval of its success rate clears the threshold. `next` is the only
place cells are handed out: it resumes the caller's own unfinished cell first, skips a chain level whose inner
level came out "below" (a harder level of a monotone axis cannot do better), waits while the container lacks the
headroom for a stack or the remaining cells wait on another instance's inner level, and builds the cell's scene
(`box_platform_scene`) the first time a geometry is needed. It prints the cell's experiment file and exits 0; 1 =
no cells left, 2 = a `STOP` file in the sweep directory (the loop ends between cells).

Axes: width / length / height (the box; one scene per combination) and start_edge / start_yaw_deg / start_lateral /
object_setback / object_yaw_deg (per-episode ranges of the visual expert, `task.visual.*`).
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import itertools
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from locomani.pipeline import batches

ROOT = Path(__file__).resolve().parents[4]
EXIT_CELL, EXIT_FINISHED, EXIT_STOP, EXIT_WAIT = 0, 1, 2, 3  # EXIT_WAIT stays inside `next`, which polls
WAIT_SECONDS = 60
MAX_ATTEMPTS = 3  # a cell whose stack or driver failed this often is left as failed, not retried forever
GEOMETRY = ("width", "length", "height")
RANGES = {  # axis -> the visual config field(s) its (low, high) range sets
    "start_edge": ("start_edge_min", "start_edge_max"),
    "start_yaw_deg": ("start_yaw_range_deg",),
    "start_lateral": ("start_lateral_range",),
    "object_setback": ("object_setback_range",),
    "object_yaw_deg": ("object_yaw_range_deg",),
}
# The expert's phase log (visual_NN.log), in order: the last one a failed round reached says where it failed.
PHASES = ("walk", "settle", "arm preparation", "reach", "align", "descend", "close", "lift", "stand", "hold")


def load(sweep_file: Path) -> dict:
    sweep = yaml.safe_load(Path(sweep_file).read_text())
    for axis in sweep["centre"]:
        if axis not in GEOMETRY and axis not in RANGES:
            raise SystemExit(f"{sweep_file}: unknown axis {axis!r}")
    return sweep


def sweep_dir(sweep: dict) -> Path:
    return ROOT / "outputs" / "sweeps" / sweep["name"]


def _label(axis: str, value) -> str:
    if isinstance(value, (list, tuple)):
        return f"{axis}{value[0]:+g}_{value[1]:+g}"
    return f"{axis}{value:g}"


def cells(sweep: dict) -> list[dict]:
    """Every cell in run order: {id, values (all axes), after (inner chain level's id or None)}."""
    out = []
    grid = sweep.get("grid") or {}
    for combo in itertools.product(*grid.values()):
        values = dict(sweep["centre"], **dict(zip(grid, combo)))
        out.append(dict(id="_".join(_label(a, v) for a, v in zip(grid, combo)), values=values, after=None))
    for axis, chains in (sweep.get("chains") or {}).items():
        for chain in chains:
            after = None
            for level in chain:
                cell_id = _label(axis, level)
                out.append(dict(id=cell_id, values=dict(sweep["centre"], **{axis: level}), after=after))
                after = cell_id
    ids = [c["id"] for c in out]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate cells: {sorted(i for i in ids if ids.count(i) > 1)}")
    return out


def scene_dir(sweep: dict, values: dict) -> Path:
    return sweep_dir(sweep) / "scenes" / "_".join(f"{a[0].upper()}{values[a]:g}" for a in GEOMETRY)


def run_name(sweep: dict, cell: dict) -> str:
    return f"{sweep['name']}_{cell['id']}"


def _set(cfg: dict, dotted: str, value) -> None:
    node = cfg
    *path, last = dotted.split(".")
    for key in path:
        node = node.setdefault(key, {})
    node[last] = value


def experiment(sweep: dict, cell: dict, instance: int) -> dict:
    """The cell's complete experiment: the base, the sweep's `set`, this cell's layout, its instance."""
    cfg = copy.deepcopy(yaml.safe_load((ROOT / sweep["base"]).read_text()))
    for dotted, value in (sweep.get("set") or {}).items():
        _set(cfg, dotted, copy.deepcopy(value))
    values = cell["values"]
    for axis, fields in RANGES.items():
        low, high = values[axis]
        if len(fields) == 2:
            _set(cfg, f"task.visual.{fields[0]}", float(low))
            _set(cfg, f"task.visual.{fields[1]}", float(high))
        else:
            _set(cfg, f"task.visual.{fields[0]}", [float(low), float(high)])
    if sweep.get("stand_forward_clearance") is not None:
        # visual.stand_forward_min is "the box's half depth plus clearance" (0.35 for the 0.32 m carton): follow the box
        _set(cfg, "task.visual.stand_forward_min", round(values["length"] / 2 + sweep["stand_forward_clearance"], 4))
    cfg["name"] = run_name(sweep, cell)
    cfg["scene"] = {"dir": str(scene_dir(sweep, values).relative_to(ROOT))}
    cfg["collect"]["instance"] = int(instance)
    cfg["collect"]["lanes"] = 1  # the sweep runs one loop per instance itself; a cell spreading into lanes takes theirs
    return cfg


def run_dir(sweep: dict, cell: dict) -> Path:
    return ROOT / "outputs" / "runs" / run_name(sweep, cell)


def manifests(sweep: dict, cell: dict) -> list[str]:
    return [str(p) for p in sorted(run_dir(sweep, cell).glob("manifest_*.jsonl"))]


def verdict(sweep: dict, cell: dict) -> dict | None:
    """The finished cell's tally and verdict (the same rule its collect stopped on), or None while it is not done."""
    summary = run_dir(sweep, cell) / "summary.json"
    if not summary.exists() or "collect" not in json.loads(summary.read_text()):
        return None
    from locomani.tasks.crate.collect import stop_tally

    rule = sweep["set"]["collect.stop_rule"]
    tally = stop_tally(manifests(sweep, cell), rule)
    decided = batches.stop_verdict(tally["successes"], tally["rounds"], rule) or "undecided"
    return dict(tally, verdict=decided)


def status(sweep: dict, all_cells: list[dict]) -> dict[str, str]:
    """done:<verdict> / pruned / failed / claimed / pending for every cell."""
    claims = sweep_dir(sweep) / "claims"
    out: dict[str, str] = {}
    for cell in all_cells:
        done = verdict(sweep, cell)
        claim = claims / f"{cell['id']}.json"
        after = out.get(cell["after"]) if cell["after"] else None
        if done:
            out[cell["id"]] = f"done:{done['verdict']}"
        elif after in ("pruned", "done:below"):
            out[cell["id"]] = "pruned"
        elif claim.exists():
            out[cell["id"]] = "failed" if json.loads(claim.read_text())["attempts"] > MAX_ATTEMPTS else "claimed"
        else:
            out[cell["id"]] = "pending"
    return out


def claim_next(sweep: dict, instance: int, gpu: str) -> tuple[int, Path | None]:
    """Hand this instance a cell (its own unfinished one first). Serialised by a lock file; returns (EXIT_*, path).

    Waits (EXIT_WAIT) while the container lacks the headroom a stack without exporter needs, so a refused start does
    not count against the cell's attempts."""
    base = sweep_dir(sweep)
    if (base / "STOP").exists():
        return EXIT_STOP, None
    from locomani.pipeline import run_context

    if (why := run_context.shortfall("collect", run_context.measure(gpu), record=False)) is not None:
        print(f"sweep: waiting for headroom -- {why}", file=sys.stderr)
        return EXIT_WAIT, None
    (base / "claims").mkdir(parents=True, exist_ok=True)
    with open(base / "queue.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        all_cells = cells(sweep)
        state = status(sweep, all_cells)
        mine = [
            c
            for c in all_cells
            if state[c["id"]] == "claimed"
            and json.loads((base / "claims" / f"{c['id']}.json").read_text())["instance"] == instance
        ]
        runnable = [
            c
            for c in all_cells
            if state[c["id"]] == "pending" and (c["after"] is None or state[c["after"]].startswith("done:"))
        ]
        chosen = (mine or runnable or [None])[0]
        if chosen is None:
            waiting = any(state[c["id"]] in ("pending", "claimed") for c in all_cells)
            return (EXIT_WAIT if waiting else EXIT_FINISHED), None
        claim = base / "claims" / f"{chosen['id']}.json"
        record = json.loads(claim.read_text()) if claim.exists() else dict(instance=instance, attempts=0)
        record.update(attempts=record["attempts"] + 1, last=time.strftime("%Y-%m-%d %H:%M:%S"))
        claim.write_text(json.dumps(record))
        scene = scene_dir(sweep, chosen["values"])
        if not (scene / "scene_meta.json").exists():
            from locomani.tasks.crate import box_platform_scene

            values = chosen["values"]
            box_platform_scene.build(
                scene, **{a: float(values[a]) for a in GEOMETRY}, platform_top=float(sweep["platform_top"])
            )
        path = base / "cells" / f"{chosen['id']}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(experiment(sweep, chosen, instance), sort_keys=False)
        if not path.exists():
            path.write_text(f"# generated by locomani.tasks.crate.sweep from {sweep['name']}; do not edit\n{text}")
        return EXIT_CELL, path


def _phases(run: Path) -> dict[str, str]:
    """label -> the last expert phase the round reached, from the run's visual_NN.log files."""
    reached: dict[str, str] = {}
    for log in sorted(run.glob("visual_*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("[visual_crate_") and " at " in line:
                label, rest = line[1:].split("] ", 1)
                phase = rest.rsplit(" at ", 1)[0]
                if phase in PHASES:
                    reached[label] = phase
    return reached


def report(sweep: dict) -> list[dict]:
    """One row per cell; writes outputs/sweeps/<sweep>/report.{json,md}."""
    all_cells = cells(sweep)
    state = status(sweep, all_cells)
    rule = sweep["set"]["collect.stop_rule"]
    from locomani.tasks.crate.collect import stop_tally

    rows = []
    for cell in all_cells:
        found = manifests(sweep, cell)
        tally = stop_tally(found, rule) if found else dict(successes=0, rounds=0, excluded=0)
        low, high = batches.wilson(tally["successes"], tally["rounds"])
        data = [json.loads(line) for m in found for line in Path(m).read_text().splitlines() if line.strip()]
        reached = _phases(run_dir(sweep, cell))
        failed = [r for r in data if not r.get("ok")]
        rtf = [r["real_time_factor"] for r in data if r.get("real_time_factor") is not None]
        depth = [r["stand_attempts"][0]["depth_m"] for r in data if r.get("stand_attempts")]
        lateral = [abs(r["stand_attempts"][0]["lateral_m"]) for r in data if r.get("stand_attempts")]
        rows.append(
            dict(
                cell=cell["id"],
                status=state[cell["id"]],
                **tally,
                rate=round(tally["successes"] / tally["rounds"], 3) if tally["rounds"] else None,
                ci=[round(low, 3), round(high, 3)],
                failed_at=dict(Counter(reached.get(r["label"], "none") for r in failed).most_common()),
                why=dict(Counter(str(r.get("why")) for r in failed).most_common(4)),
                rtf_median=round(statistics.median(rtf), 3) if rtf else None,
                rtf_min=round(min(rtf), 3) if rtf else None,
                stand_depth_median=round(statistics.median(depth), 3) if depth else None,
                stand_lateral_abs_median=round(statistics.median(lateral), 3) if lateral else None,
                values={a: cell["values"][a] for a in cell["values"]},
            )
        )
    base = sweep_dir(sweep)
    base.mkdir(parents=True, exist_ok=True)
    (base / "report.json").write_text(json.dumps(rows, indent=1))
    lines = [
        f"# {sweep['name']}",
        "",
        f"Success = the expert's `ok` (lift, upright, hold; no retreat). Threshold {rule['threshold']}, Wilson 95%, "
        f"{rule['min_rounds']}-{rule['max_rounds']} counted rounds; rounds under RTF {rule.get('min_rtf', 0)} or "
        "that measured the stack (collect.NOT_A_MEASUREMENT) are excluded.",
        "",
        "| cell | status | k/n | rate | 95% CI | excl | failed at (last phase reached) | stand depth / abs lateral | RTF med/min |",
        "|---|---|---:|---:|---|---:|---|---|---|",
    ]
    for r in rows:
        failed_at = ", ".join(f"{k} {v}" for k, v in r["failed_at"].items())
        rate = f"{r['rate']:.2f}" if r["rate"] is not None else "-"
        lines.append(
            f"| {r['cell']} | {r['status']} | {r['successes']}/{r['rounds']} | {rate} | "
            f"{r['ci'][0]:.2f}-{r['ci'][1]:.2f} | {r['excluded']} | {failed_at} | "
            f"{r['stand_depth_median']} / {r['stand_lateral_abs_median']} | {r['rtf_median']}/{r['rtf_min']} |"
        )
    (base / "report.md").write_text("\n".join(lines) + "\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("cells", "next", "report"))
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args(argv)
    sweep = load(args.sweep)
    if args.verb == "cells":
        all_cells = cells(sweep)
        state = status(sweep, all_cells)
        for cell in all_cells:
            print(f"{cell['id']:32s} {state[cell['id']]:14s} after={cell['after']}")
        return 0
    if args.verb == "report":
        print((sweep_dir(sweep) / "report.md").read_text() if report(sweep) else "no cells")
        return 0
    while (claimed := claim_next(sweep, args.instance, args.gpu))[0] == EXIT_WAIT:
        time.sleep(WAIT_SECONDS)
    code, path = claimed
    if path is not None:
        print(path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
