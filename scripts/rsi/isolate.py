"""TODO 33.20 (plan section 1.4, D20): whose failure is it -- the motor layer's or the skill's. A failure class that
recurs gets its suspect primitive run again alone: the round's start (robot pose, objects placed, the same settle),
then only the one command that was running when the round failed, with the arguments it was given (the motor's call
log, `calls` in the round record since 33.20); none of the skill's other steps. It fails the same way alone: the
motor layer or the controller cannot carry out that command from there -- a constraint for the body model and an
escalation (rules.md 20); a skill may route around it, not change the layer below. It does not: the failure needs
what the skill did first.

    python3 scripts/rsi/isolate.py <family> --pool <starts> --rounds <round records> --out <dir> [--labels a,b] \\
        [--class fell_walk]

Writes <out>/<label>.json (the isolated round's record, the call it ran, the verdict) and <out>/summary.json. Holds one
of the machine's simulation slots (batch.SIM_SLOTS) while it runs, like a batch worker.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import batch as B  # noqa: E402
import family as F  # noqa: E402
from scene_round import make_world  # noqa: E402

from locomani.tasks.scene.episode import run_episode  # noqa: E402
from locomani.tasks.scene.start import Start  # noqa: E402

CLASSES = ("fell_walk",)  # the failure classes the driver attributes (the first use case: falls, 33.1.3's C probe)


def suspect(calls: list[dict], fell: bool) -> int | None:
    """The call that was running when the round failed: the one that ended in the fall, else the first that failed."""
    if fell:
        k = next((i for i, c in enumerate(calls) if str(c.get("why", "")).startswith("fell")), None)
        if k is not None:
            return k
    return next((i for i, c in enumerate(calls) if not c.get("ok", True)), None)


def verdict(original: dict, isolated: dict) -> str:
    """motor: the command alone fails the same way from the round's start; skill: alone it does not."""
    if original.get("fell"):
        return "motor" if isolated.get("fell") else "skill"
    return "motor" if not isolated.get("call_ok", True) else "skill"


def isolate_one(world, fam: dict, start_row: dict, rec: dict) -> dict:
    calls = rec.get("calls") or []
    k = suspect(calls, bool(rec.get("fell")))
    if k is None:
        return dict(label=rec["label"], verdict=None, why="no call log or no failed call (recorded before TODO 33.20?)")
    call = calls[k]
    task = F.task(fam, start_row["task"])
    start = Start.from_dict(start_row["start"])
    seen = {}

    def plan(ctx):
        out = getattr(ctx.motor, call["name"])(*call.get("args", []), **call.get("kwargs", {}))
        seen.update(ok=bool(out.ok), why=out.why)
        return out

    t0 = time.time()
    res, _ = run_episode(world, task, start, plan)
    ex = res.execution
    iso = dict(fell=bool(ex["fell"]), call_ok=seen.get("ok"), call_why=seen.get("why"), t_end=ex["t_end"],
               steps=ex["steps"], calls=ex.get("calls", []), wall_s=round(time.time() - t0, 1))  # fmt: skip
    orig = dict(fell=bool(rec.get("fell")), why=rec.get("why"), failure_class=rec.get("failure_class"))
    return dict(label=rec["label"], task=rec["task"], call_index=k, call=call, original=orig, isolated=iso,
                verdict=verdict(orig, iso), skills=rec.get("skills"), motor=rec.get("motor"))  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("--pool", type=Path, required=True, help="the start files")
    ap.add_argument("--rounds", type=Path, required=True, help="the round records to attribute")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--class", dest="classes", action="append", help=f"failure classes (default {', '.join(CLASSES)})")
    ap.add_argument("--controller", default="scalebfm")
    a = ap.parse_args(argv)
    fam = F.load(a.family)
    classes = set(a.classes or CLASSES)
    recs = [r for r in B.outcomes(a.rounds).values() if not r.get("ok")]
    if a.labels:
        recs = [r for r in recs if r["label"] in set(a.labels.split(","))]
    else:
        recs = [r for r in recs if r.get("failure_class") in classes]
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in recs if not (a.out / f"{r['label']}.json").exists() and (a.pool / f"{r['label']}.json").exists()]
    if todo:
        slot = None
        while B.SIM_SLOTS and (slot := B.take_slot()) is None:
            time.sleep(5)
        try:
            world = make_world(fam, a.controller)
            for r in todo:
                row = json.loads((a.pool / f"{r['label']}.json").read_text())
                (a.out / f"{r['label']}.json").write_text(json.dumps(isolate_one(world, fam, row, r), default=float))
        finally:
            if slot is not None:
                slot.close()
    done = [json.loads(p.read_text()) for p in sorted(a.out.glob("*.json")) if p.name != "summary.json"]
    summary = dict(n=len(done), motor=[d["label"] for d in done if d.get("verdict") == "motor"],
                   skill=[d["label"] for d in done if d.get("verdict") == "skill"],
                   undecided=[d["label"] for d in done if d.get("verdict") is None])  # fmt: skip
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
