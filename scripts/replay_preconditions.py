#!/usr/bin/env python3
"""Replay the grasp admission band on recorded stances: which successes it would refuse, which failures it admits.

    $VENV_SIM_PY scripts/replay_preconditions.py --out outputs/analysis/precondition_replay_20260928
    $VENV_SIM_PY scripts/replay_preconditions.py --skills 'outputs/runs/skills_*' \\
        --bottle outputs/runs/bottle_simple_200_20260916 --warehouse 'outputs/runs/box_centre_*_20260927'

Three recorded sources give a (profile, depth, lateral, ok, why) row per round that reached its grasp:
`skills/agent_trace_NN.jsonl` of a plan run (the approach's produced `Stance` and the grasp entry's
outcome); a bottle line `collect/manifest_*.jsonl` (`park` = |rel| at the stop, lateral unknown; rows
whose failure is in the stand itself are excluded because they never reached the grasp); a crate line
`manifest_*.jsonl` (`stand_attempts[-1]` depth/lateral, the accepted stance). Each row goes through
`skills.session.stance_violations` with the profile's current `STANCE_BANDS` entry, and then a sweep of
candidate depth upper bounds and lateral bounds prints, per bound, how many successes it keeps and how many
failures it refuses. The report and the rows are written under `--out`; nothing else is touched.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.skills.profiles import STANCE_BANDS  # noqa: E402
from locomani.skills.session import Stance, stance_violations  # noqa: E402

GRASP_STEPS = ("grasp_single", "grasp_bimanual")


def rows_from_skills_run(run: Path) -> list[dict]:
    rows = []
    for trace in sorted((run / "skills").glob("agent_trace_*.jsonl")):
        entries = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        stance = None
        for e in entries:
            outcome = e.get("outcome")
            if e["primitive"] == "approach" and outcome and outcome.get("ok") and outcome.get("produced"):
                stance = outcome["produced"]
            elif e["primitive"] in GRASP_STEPS and stance is not None:
                rejected = e.get("rejected")
                ok = bool(outcome and outcome.get("ok"))
                rows.append(
                    dict(
                        source=f"{run.name}/{trace.name}",
                        profile=stance["profile"],
                        depth=stance["depth_m"],
                        lateral=stance["lateral_m"],
                        ok=ok,
                        why=(outcome or {}).get("reason", "") if not rejected else "rejected:" + json.dumps(rejected),
                        rejected=bool(rejected),
                    )
                )
                stance = None
    return rows


def rows_from_bottle_run(run: Path) -> list[dict]:
    rows = []
    for m in sorted(run.glob("collect/manifest_*.jsonl")) + sorted(run.glob("manifest_*.jsonl")):
        for line in m.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("park") is None:
                continue
            why = r.get("why") or ""
            if not r["ok"] and why.startswith("[stand]"):
                continue  # refused before the grasp: not a grasp admission question
            rows.append(
                dict(
                    source=f"{run.name}/{m.name}",
                    profile="bottle_simple",
                    depth=float(r["park"]),
                    lateral=float("nan"),
                    ok=bool(r["ok"]),
                    why=why,
                    rejected=False,
                )
            )
    return rows


def rows_from_warehouse_run(run: Path) -> list[dict]:
    rows = []
    for m in sorted(run.glob("manifest_*.jsonl")):
        for line in m.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            attempts = r.get("stand_attempts") or []
            if not attempts:
                continue
            last = attempts[-1]
            rows.append(
                dict(
                    source=f"{run.name}/{m.name}",
                    profile="warehouse_box",
                    depth=float(last["depth_m"]),
                    lateral=float(last.get("lateral_m", np.nan)),
                    ok=bool(r["ok"]),
                    why=str(r.get("control_reason") or r.get("why") or ""),
                    rejected=False,
                )
            )
    return rows


def evaluate(rows: list[dict], band: dict | None) -> dict:
    """Successes the band refuses, failures it admits (counts and rates)."""
    ok_rows = [r for r in rows if r["ok"]]
    bad_rows = [r for r in rows if not r["ok"]]

    def refused(r) -> bool:
        stance = Stance("", r["profile"], r["depth"], r["lateral"], 0.0, True)
        return bool(stance_violations(stance, band))

    false_reject = sum(refused(r) for r in ok_rows)
    admitted_fail = sum(not refused(r) for r in bad_rows)
    return dict(
        n=len(rows),
        ok=len(ok_rows),
        fail=len(bad_rows),
        false_reject=false_reject,
        false_reject_rate=false_reject / len(ok_rows) if ok_rows else None,
        admitted_fail=admitted_fail,
        admitted_fail_rate=admitted_fail / len(bad_rows) if bad_rows else None,
    )


def sweep(rows: list[dict], key: str, bounds: list[float], side: str) -> list[dict]:
    """For each candidate bound on `key` ("depth" | "lateral"), what an upper (or |lateral|) bound would do."""
    out = []
    for b in bounds:
        if key == "depth":
            band = {"depth": (None, b) if side == "hi" else (b, None)}
        else:
            band = {"lateral": (-b, b)}
        out.append(dict(bound=round(float(b), 3), **evaluate(rows, band)))
    return out


def report(rows: list[dict]) -> str:
    by_profile = defaultdict(list)
    for r in rows:
        by_profile[r["profile"]].append(r)
    lines = ["# Grasp admission band replay", ""]
    for profile, prows in sorted(by_profile.items()):
        band = STANCE_BANDS.get(profile) or {}
        cur = evaluate(prows, band)
        lines += [
            f"## {profile}: {cur['n']} rounds ({cur['ok']} ok, {cur['fail']} failed), current band {json.dumps(band)}",
            "",
            f"- current band: refuses {cur['false_reject']}/{cur['ok']} successes, admits {cur['admitted_fail']}/{cur['fail']} failures",
            "",
        ]
        depths = np.array([r["depth"] for r in prows if np.isfinite(r["depth"])])
        if depths.size:
            oks = np.array([r["depth"] for r in prows if r["ok"] and np.isfinite(r["depth"])])
            bads = np.array([r["depth"] for r in prows if not r["ok"] and np.isfinite(r["depth"])])
            lines.append(f"- depth: successes {_span(oks)}, failures {_span(bads)}")
            lines += ["", "| depth <= | kept ok | refused ok | refused fail | admitted fail |", "|---|---|---|---|---|"]
            for s in sweep(
                prows, "depth", sorted(set(np.round(np.quantile(depths, np.linspace(0.3, 1.0, 8)), 2))), "hi"
            ):
                lines.append(
                    f"| {s['bound']:.2f} | {s['ok'] - s['false_reject']} | {s['false_reject']} | {s['fail'] - s['admitted_fail']} | {s['admitted_fail']} |"
                )
            lines += ["", "| depth >= | kept ok | refused ok | refused fail | admitted fail |", "|---|---|---|---|---|"]
            for s in sweep(
                prows, "depth", sorted(set(np.round(np.quantile(depths, np.linspace(0.0, 0.7, 8)), 2))), "lo"
            ):
                lines.append(
                    f"| {s['bound']:.2f} | {s['ok'] - s['false_reject']} | {s['false_reject']} | {s['fail'] - s['admitted_fail']} | {s['admitted_fail']} |"
                )
        laterals = np.array([abs(r["lateral"]) for r in prows if np.isfinite(r["lateral"])])
        if laterals.size:
            oks = np.array([abs(r["lateral"]) for r in prows if r["ok"] and np.isfinite(r["lateral"])])
            bads = np.array([abs(r["lateral"]) for r in prows if not r["ok"] and np.isfinite(r["lateral"])])
            lines += ["", f"- |lateral|: successes {_span(oks)}, failures {_span(bads)}", ""]
            lines += [
                "| \\|lateral\\| <= | kept ok | refused ok | refused fail | admitted fail |",
                "|---|---|---|---|---|",
            ]
            for s in sweep(
                prows, "lateral", sorted(set(np.round(np.quantile(laterals, np.linspace(0.3, 1.0, 8)), 3))), "hi"
            ):
                lines.append(
                    f"| {s['bound']:.3f} | {s['ok'] - s['false_reject']} | {s['false_reject']} | {s['fail'] - s['admitted_fail']} | {s['admitted_fail']} |"
                )
        fails = defaultdict(int)
        for r in prows:
            if not r["ok"]:
                fails[r["why"].split(" ")[0] if r["why"] else "?"] += 1
        lines += [
            "",
            "- failure tags: " + ", ".join(f"{k} {v}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])),
            "",
        ]
    return "\n".join(lines) + "\n"


def _span(values: np.ndarray) -> str:
    if values.size == 0:
        return "none"
    return f"n={values.size} median {np.median(values):.3f} range {values.min():.3f}..{values.max():.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skills", default="outputs/runs/skills_*", help="glob of plan runs (skills/agent_trace_*.jsonl)")
    ap.add_argument("--bottle", default="outputs/runs/bottle_simple_200_20260916", help="glob of bottle collect runs")
    ap.add_argument("--warehouse", default="outputs/runs/box_centre_*_20260927", help="glob of crate-line runs")
    ap.add_argument("--out", type=Path, default=Path("outputs/analysis/precondition_replay"))
    args = ap.parse_args(argv)
    rows: list[dict] = []
    for pattern, reader in (
        (args.skills, rows_from_skills_run),
        (args.bottle, rows_from_bottle_run),
        (args.warehouse, rows_from_warehouse_run),
    ):
        for run in sorted(glob.glob(pattern)):
            rows += reader(Path(run))
    text = report(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(text)
    with (args.out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(text)
    print(f"[replay] {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
