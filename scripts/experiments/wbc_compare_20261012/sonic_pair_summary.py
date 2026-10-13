"""wbc_compare (TODO 24.2.2): two SONIC batches of the 65-object line paired by (object, round).
Used for `_wbc` (wide waist, the StreamedArm override) against `_wbc2` (grasp_planned.WAIST_LIMITS re-applied after
the takeover), and before that `_wbc` against `_wbc_nobias`. Same rows as compare_summary.py's SONIC side: lifted,
discordant cells with the exact McNemar p, failure classes, pre-grasp open-loop error, the three stability measures.
    python3 scripts/experiments/wbc_compare_20261012/sonic_pair_summary.py --a _wbc --b _wbc2 --rounds 2
        -> outputs/wbc_compare_20261012/compare/sonic_<a>_vs_<b>.md / .json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_summary import REPO, classify, mcnemar_exact, q, sonic_round, stability  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "experiments" / "graspnet_bottle_20261008"))


def objects() -> list[int]:
    return [
        o["id"]
        for o in json.loads((REPO / "outputs" / "graspnet_bottle_20261008" / "objects.json").read_text())["objects"]
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="tag of the first SONIC batch (outputs/runs/gnb_obj_XXX<tag>)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "wbc_compare_20261012" / "compare")
    args = ap.parse_args(argv)
    rows = []
    for oid in objects():
        for rnd in range(args.rounds):
            a, b = sonic_round(args.a, oid, rnd), sonic_round(args.b, oid, rnd)
            if a is None and b is None:
                continue
            rows.append(
                dict(
                    object=oid, round=rnd,
                    a=dict(ok=bool(a and a.get("ok")), cls=classify(a), tier=((a or {}).get("grasp") or {}).get("tier"),
                           pregrasp_ol_cm=100 * a["pregrasp_error_open_loop_m"] if a and "pregrasp_error_open_loop_m" in a else None,
                           **stability(a)),
                    b=dict(ok=bool(b and b.get("ok")), cls=classify(b), tier=((b or {}).get("grasp") or {}).get("tier"),
                           pregrasp_ol_cm=100 * b["pregrasp_error_open_loop_m"] if b and "pregrasp_error_open_loop_m" in b else None,
                           **stability(b)),
                )
            )  # fmt: skip
    both = sum(r["a"]["ok"] and r["b"]["ok"] for r in rows)
    only_a = sum(r["a"]["ok"] and not r["b"]["ok"] for r in rows)
    only_b = sum(r["b"]["ok"] and not r["a"]["ok"] for r in rows)
    neither = len(rows) - both - only_a - only_b
    p = mcnemar_exact(only_a, only_b)
    cls = {k: Counter(r[k]["cls"] for r in rows) for k in ("a", "b")}
    tiers = {k: Counter(r[k]["tier"] for r in rows if r[k]["tier"] is not None) for k in ("a", "b")}
    lines = [
        f"# SONIC {args.a} vs SONIC {args.b} on the 65-object line, paired by (object, round)",
        "",
        f"Rounds with a record on either side: {len(rows)}. Lifted {args.a} {both + only_a}, {args.b} {both + only_b}; "
        f"both {both} / only {args.a} {only_a} / only {args.b} {only_b} / neither {neither}; exact McNemar p = {p:.3g}.",
        "",
        f"| failure class | {args.a} | {args.b} |",
        "|---|---|---|",
    ]
    for name in sorted(set(cls["a"]) | set(cls["b"]), key=lambda n: -(cls["a"][n] + cls["b"][n])):
        lines.append(f"| {name} | {cls['a'][name]} | {cls['b'][name]} |")
    lines += ["", f"| chosen grasp tier | {args.a} | {args.b} |", "|---|---|---|"]
    for t in sorted(set(tiers["a"]) | set(tiers["b"])):
        lines.append(f"| {t} | {tiers['a'][t]} | {tiers['b'][t]} |")
    lines += ["", f"| median / p90 | {args.a} | {args.b} |", "|---|---|---|"]
    for key, label in (
        ("pregrasp_ol_cm", "pre-grasp open-loop error cm"),
        ("entry_knock_cm", "object moved before the close cm"),
        ("close_err_cm", "tool vs grasp at the close cm"),
        ("hold_slip_cm", "slip over lift + hold cm"),
    ):
        va, vb = [r["a"][key] for r in rows], [r["b"][key] for r in rows]
        lines.append(f"| {label} | {q(va, 0.5):.2f} / {q(va, 0.9):.2f} | {q(vb, 0.5):.2f} / {q(vb, 0.9):.2f} |")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"sonic{args.a}_vs{args.b}"
    (args.out / f"{stem}.md").write_text("\n".join(lines) + "\n")
    (args.out / f"{stem}.json").write_text(
        json.dumps(
            dict(
                a=args.a,
                b=args.b,
                rounds=len(rows),
                both=both,
                only_a=only_a,
                only_b=only_b,
                neither=neither,
                p=p,
                classes=cls,
                tiers={k: dict(v) for k, v in tiers.items()},
                rows=rows,
            ),
            indent=1,
            default=float,
        )  # fmt: skip
    )
    print("\n".join(lines[:3]))
    print(f"wrote {args.out / (stem + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
