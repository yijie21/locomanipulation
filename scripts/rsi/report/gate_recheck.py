"""TODO 33.10 (plan section 3.2, from Dream-RSI): the finished loops' gate calls judged again under the relaxed gate
(D6: b - c >= 3 and c <= b / 3, no p) and the candidate pool (D7: the best 2 rejected with b > c), read-only.

Only the first change of verdict in a run is meaningful: once a call would have been accepted, the later calls were
made against an incumbent the run never had. Rejections for drops, falls or stage 1 stay rejections (D6 keeps those
rules). A flipped candidate with sealed rounds (TODO 31's rejected ones in grasp_atomic_refs) is compared with the
incumbent of its time on the sealed pool, start by start.

    python3 scripts/rsi/report/gate_recheck.py [--root /workspace/code/rsi-scene] [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HIDDEN = "outputs/scene_rsi_20261016_hidden"
REFS = {"grasp-atomic-a": REPO / "outputs" / "scene_rsi_20261016" / "grasp_atomic_refs"}
STAT_ONLY = re.compile(r"^(net wins -?\d+ < 3|one-sided McNemar p [\d.e-]+ > 0\.05)$")


def iter_of(path: str) -> int | None:
    for part in Path(path).parts:
        if part.startswith("iter_") and part[5:].isdigit():
            return int(part[5:])
    return None


def relaxed(b: int, c: int) -> bool:
    return b - c >= 3 and 3 * c <= b


def stat_only(reason: str) -> bool:
    """The call was rejected only by the net-wins / p rule, the part D6 changes."""
    return all(STAT_ONLY.match(p.strip()) for p in reason.split(";"))


def outcomes(d: Path) -> dict[str, bool]:
    return {f.stem: json.loads(f.read_text())["ok"] for f in sorted(d.glob("*.json")) if not f.name.startswith("_")}


def sealed_pair(run: Path, family: str, call: dict, incumbent: dict | None) -> dict | None:
    refs = REFS.get(run.name)
    if refs is None or incumbent is None:
        return None
    cand = refs / f"rounds_rejected_{call['call']:02d}_{call['hash']}"
    inc = run / HIDDEN / family / "sealed" / f"{incumbent['call']:02d}_{incumbent['hash']}"
    if not cand.is_dir() or not inc.is_dir():
        return None
    a, b = outcomes(cand), outcomes(inc)
    keys = sorted(set(a) & set(b))
    return dict(n=len(keys), b=sum(a[k] and not b[k] for k in keys), c=sum(b[k] and not a[k] for k in keys),
                candidate_ok=sum(a[k] for k in keys), incumbent_ok=sum(b[k] for k in keys))  # fmt: skip


def recheck(run: Path) -> list[dict]:
    out = []
    for led in sorted((run / HIDDEN).glob("*/gate/ledger.jsonl")):
        family = led.parents[1].name
        calls = [json.loads(x) for x in led.read_text().splitlines() if x.strip()]
        meta = run / "outputs" / "scene_rsi_20261016" / family / "loop" / "meta.json"
        since = (json.loads(meta.read_text()).get("replay") or {}).get("iteration", 0) if meta.exists() else 0
        calls = [
            c for c in calls if (iter_of(c["candidate"]) or 0) >= since
        ]  # a replay's copied history is its source's
        incumbent, pool, diverged = None, [], None
        for c in calls:
            if c.get("dry_run"):
                continue
            r = c.get("result") or {}
            b, cc = r.get("b"), r.get("c")
            new = c["accepted"] or (b is not None and stat_only(c["reason"]) and relaxed(b, cc))
            row = dict(run=run.name, family=family, call=c["call"], hash=c["hash"], b=b, c=cc, old=c["accepted"],
                       new=bool(new), reason=c["reason"], after_divergence=diverged is not None)  # fmt: skip
            if new and not c["accepted"] and diverged is None:
                diverged = c["call"]
                row["first_flip"] = True
                row["sealed_vs_incumbent"] = sealed_pair(run, family, c, incumbent)
            if c["accepted"]:
                incumbent, pool = c, []
            elif b is not None and b > cc and stat_only(c["reason"]):
                pool = sorted(
                    pool + [dict(call=c["call"], hash=c["hash"], net=b - cc)], key=lambda p: (-p["net"], -p["call"])
                )[:2]
            row["pool_after"] = [f"{p['call']:02d}_{p['hash']} ({p['net']:+d})" for p in pool]
            out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/workspace/code/rsi-scene"))
    ap.add_argument(
        "--out", type=Path, default=REPO / "outputs" / "scene_rsi_20261016" / "gate_recheck" / "recheck.json"
    )
    a = ap.parse_args(argv)
    runs = [d for d in sorted(a.root.iterdir()) if d.is_dir() and not d.name.startswith(("ab-", "replay"))]
    rows = [r for d in runs for r in recheck(d)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for r in rows:
        mark = "FLIP" if r.get("first_flip") else ("    " if r["old"] == r["new"] else "flip")
        tail = f" sealed {r['sealed_vs_incumbent']}" if r.get("sealed_vs_incumbent") else ""
        print(f"{mark} {r['run']:24s} {r['call']:2d} {r['hash']} b{r['b']} c{r['c']} old {r['old']!s:5} new {r['new']!s:5}"
              f"{' (after the first flip)' if r['after_divergence'] else ''} pool {r['pool_after']}{tail}")  # fmt: skip
    flips = [r for r in rows if r.get("first_flip")]
    print(f"{len(flips)} runs change their first verdict under D6; {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
