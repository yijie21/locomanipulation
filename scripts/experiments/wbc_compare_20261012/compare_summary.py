"""wbc_compare step 5 (TODO 24.5): SONIC vs ScaleBFM on the 65-object line, paired by (object, round).

Reads the SONIC rounds (outputs/runs/gnb_obj_XXX<tag>/grasp_planned/round_RR_ik/record.json + expert_trace.npy) and
a ScaleBFM arm's rounds (outputs/wbc_compare_20261012/bfm_grasp/<arm>/obj_XXX/round_RR/record.json + trace.npz) and
writes <out>/<arm>.md and <arm>.json with:

- the paired `lifted` table (both / SONIC only / ScaleBFM only / neither) and the exact McNemar p of the discordant
  cells;
- the failure classes (cache_summary's lifted / no_candidate / no_path / crash / execution, the execution reasons
  split by the program's own strings: knocked in approach / settle / entry / retreat, no bilateral contact, lost
  contact in the lift, not lifted, entry timeout, fell, other);
- the three physical stability quantities (decision 3) from the traces, the same definition on both: the object's
  horizontal displacement from the start of the closed loop to the close (entry knock), the tool-to-grasp error at
  the close (`entry.error_m`), and the slip of the object relative to the tool over the hold (world frame, from the
  lift's end to the hold's end); plus the pre-grasp open-loop error and the settle / gate times.

    python3 scripts/experiments/wbc_compare_20261012/compare_summary.py --sonic _wbc --arm paired \
        --out outputs/wbc_compare_20261012/compare
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OBJECTS = REPO / "outputs" / "graspnet_bottle_20261008" / "objects.json"
BFM_ROOT = REPO / "outputs" / "wbc_compare_20261012" / "bfm_grasp"

REASONS = (
    ("knock_approach", r"knocked .*approach"),
    ("knock_settle", r"knocked .*settle"),
    ("knock_entry", r"knocked .*entry"),
    ("knock_retreat", r"knocked .*retreat"),
    ("no_contact_at_close", r"no bilateral pad contact"),
    ("lost_contact_in_lift", r"lost pad contact"),
    ("not_lifted", r"not lifted"),
    ("entry_timeout", r"entry timed out"),
    ("fell", r"fell"),
    ("no_pair", r"no SONIC round|SONIC round chose no grasp|SONIC's grasp"),
    ("no_deliverable_attitude", r"inside the envelope"),
    ("no_path", r"no path|found no path|clears the target"),
    ("no_candidate", r"no cached grasp fits|no graspnet candidate"),
)


def classify(rec: dict | None) -> str:
    if rec is None:
        return "crash"
    if rec.get("ok"):
        return "lifted"
    why = rec.get("why") or ""
    for name, pat in REASONS:
        if re.search(pat, why):
            return name
    return "other"


def sonic_round(tag: str, oid: int, rnd: int) -> dict | None:
    d = REPO / "outputs" / "runs" / f"gnb_obj_{oid:03d}{tag}" / "grasp_planned" / f"round_{rnd:02d}_ik"
    if not (d / "record.json").exists():
        return None
    rec = json.loads((d / "record.json").read_text())
    rec["_dir"] = d
    t = d / "expert_trace.npy"
    if t.exists():
        T = np.load(t)  # stamp, phase, target3, actual3, lean, obj3, closure, bilateral, f1, f2, ... (grasp_planned)
        rec["_phase"], rec["_tool"], rec["_obj"] = T[:, 1].astype(int), T[:, 5:8], T[:, 9:12]
        rec["_phases"] = ("settle", "entry", "close", "lift", "recover", "retreat")
    return rec


def bfm_round(arm: str, oid: int, rnd: int) -> dict | None:
    d = BFM_ROOT / arm / f"obj_{oid:03d}" / f"round_{rnd:02d}"
    if not (d / "record.json").exists():
        return None
    rec = json.loads((d / "record.json").read_text())
    rec["_dir"] = d
    t = d / "trace.npz"
    if t.exists():
        T = np.load(t)["trace"]  # t, phase, target3, corrected3, actual3, object3, closure, bilateral, f1, f2, ...
        rec["_phase"], rec["_tool"], rec["_obj"] = T[:, 1].astype(int), T[:, 8:11], T[:, 11:14]
        rec["_phases"] = ("settle", "entry", "close", "lift", "recover", "retreat")
    return rec


def stability(rec: dict | None) -> dict:
    """Entry knock, close residual, hold slip from the trace (nan where the phase was not reached)."""
    out = dict(entry_knock_cm=np.nan, close_err_cm=np.nan, hold_slip_cm=np.nan)
    if rec is None or "_phase" not in rec:
        return out
    ph, tool, obj, names = rec["_phase"], rec["_tool"], rec["_obj"], rec["_phases"]
    close = np.nonzero(ph == names.index("close"))[0]
    if len(close):
        out["entry_knock_cm"] = 100 * float(np.linalg.norm(obj[close[0], :2] - obj[0, :2]))
    if rec.get("entry"):
        out["close_err_cm"] = 100 * float(rec["entry"]["error_m"])
    rec_ph = np.nonzero(ph == names.index("recover"))[0]
    if len(rec_ph) > 1:
        rel = obj[rec_ph] - tool[rec_ph]
        out["hold_slip_cm"] = 100 * float(np.linalg.norm(rel[-1] - rel[0]))
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def q(vals, p):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    return float(np.quantile(v, p)) if len(v) else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sonic", default="_wbc", help="tag of the SONIC runs (outputs/runs/gnb_obj_XXX<tag>)")
    ap.add_argument("--arm", default="paired", help="bfm_grasp output directory name (paired, free)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "wbc_compare_20261012" / "compare")
    args = ap.parse_args(argv)
    ids = [int(o["id"]) for o in json.loads(OBJECTS.read_text())["objects"]]
    rows = []
    for oid in ids:
        for rnd in range(args.rounds):
            s, b = sonic_round(args.sonic, oid, rnd), bfm_round(args.arm, oid, rnd)
            if s is None and b is None:
                continue
            row = dict(
                object=oid,
                round=rnd,
                sonic=dict(
                    present=s is not None,
                    ok=bool(s and s.get("ok")),
                    cls=classify(s),
                    why=(s or {}).get("why", ""),
                    row=((s or {}).get("grasp") or {}).get("row"),
                    tier=((s or {}).get("grasp") or {}).get("tier"),
                    pregrasp_ol_cm=100 * s["pregrasp_error_open_loop_m"]
                    if s and "pregrasp_error_open_loop_m" in s
                    else np.nan,
                    settle_s=(s or {}).get("settle", {}).get("seconds", np.nan) if s else np.nan,
                    gate_held_s=(s or {}).get("entry", {}).get("gate_held_s", np.nan) if s else np.nan,
                    **stability(s),
                ),
                scalebfm=dict(
                    present=b is not None,
                    ok=bool(b and b.get("ok")),
                    cls=classify(b),
                    why=(b or {}).get("why", ""),
                    row=((b or {}).get("grasp") or {}).get("row"),
                    tier=((b or {}).get("grasp") or {}).get("tier"),
                    pregrasp_ol_cm=100 * b["pregrasp_error_open_loop_m"]
                    if b and "pregrasp_error_open_loop_m" in b
                    else np.nan,
                    settle_s=(b or {}).get("settle", {}).get("seconds", np.nan) if b else np.nan,
                    gate_held_s=(b or {}).get("entry", {}).get("gate_held_s", np.nan) if b else np.nan,
                    **stability(b),
                ),
            )
            rows.append(row)
    both = [r for r in rows if r["sonic"]["present"] and r["scalebfm"]["present"]]
    cells = Counter((r["sonic"]["ok"], r["scalebfm"]["ok"]) for r in both)
    n_both, n_sonic_only, n_bfm_only, n_neither = (
        cells[(True, True)], cells[(True, False)], cells[(False, True)], cells[(False, False)],
    )  # fmt: skip
    p = mcnemar_exact(n_sonic_only, n_bfm_only)
    lines = [f"# SONIC ({args.sonic}) vs ScaleBFM ({args.arm}) on the 65-object line, paired by (object, round)", ""]
    lines.append(
        f"- rounds with both controllers: {len(both)} (SONIC rounds {sum(r['sonic']['present'] for r in rows)}, "
        f"ScaleBFM rounds {sum(r['scalebfm']['present'] for r in rows)}); lifted SONIC "
        f"{sum(r['sonic']['ok'] for r in both)} / ScaleBFM {sum(r['scalebfm']['ok'] for r in both)}"
    )
    lines.append("")
    lines.append("| | ScaleBFM lifted | ScaleBFM failed |")
    lines.append("|---|---|---|")
    lines.append(f"| SONIC lifted | {n_both} | {n_sonic_only} |")
    lines.append(f"| SONIC failed | {n_bfm_only} | {n_neither} |")
    lines.append("")
    lines.append(f"Exact McNemar on the discordant cells {n_sonic_only} : {n_bfm_only}, p = {p:.3f}.")
    lines.append("")
    lines.append("## Failure classes")
    lines.append("")
    classes = sorted({r[k]["cls"] for r in both for k in ("sonic", "scalebfm")})
    lines.append("| class | SONIC | ScaleBFM |")
    lines.append("|---|---|---|")
    cs, cb = Counter(r["sonic"]["cls"] for r in both), Counter(r["scalebfm"]["cls"] for r in both)
    for c in classes:
        lines.append(f"| {c} | {cs.get(c, 0)} | {cb.get(c, 0)} |")
    lines.append("")
    lines.append("## Stability and delivery (rounds that reached the quantity; median / p90)")
    lines.append("")
    lines.append("| quantity | SONIC | ScaleBFM |")
    lines.append("|---|---|---|")
    for key, label in (
        ("pregrasp_ol_cm", "pre-grasp open-loop error cm"),
        ("settle_s", "settle seconds"),
        ("gate_held_s", "entry gate held s"),
        ("entry_knock_cm", "object moved before the close cm"),
        ("close_err_cm", "tool-to-grasp error at the close cm"),
        ("hold_slip_cm", "slip over the hold cm"),
    ):
        vs = [r["sonic"][key] for r in both]
        vb = [r["scalebfm"][key] for r in both]
        ns, nb = sum(np.isfinite(v) for v in vs), sum(np.isfinite(v) for v in vb)
        lines.append(
            f"| {label} | {q(vs, 0.5):.2f} / {q(vs, 0.9):.2f} (n {ns}) | {q(vb, 0.5):.2f} / {q(vb, 0.9):.2f} (n {nb}) |"
        )
    lines.append("")
    lines.append("## Per round")
    lines.append("")
    lines.append("| object | round | SONIC | ScaleBFM | SONIC why | ScaleBFM why | rows (S / B) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        s, b = r["sonic"], r["scalebfm"]
        lines.append(
            f"| {r['object']:03d} | {r['round']} | {'OK' if s['ok'] else s['cls'] if s['present'] else '-'} | "
            f"{'OK' if b['ok'] else b['cls'] if b['present'] else '-'} | {s['why'][:50]} | {b['why'][:50]} | {s['row']} / {b['row']} |"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.arm}.md").write_text("\n".join(lines) + "\n")
    (args.out / f"{args.arm}.json").write_text(
        json.dumps(
            dict(
                sonic_tag=args.sonic,
                arm=args.arm,
                cells=dict(both=n_both, sonic_only=n_sonic_only, bfm_only=n_bfm_only, neither=n_neither),
                p=p,
                rows=rows,
            ),
            default=lambda o: (
                None
                if isinstance(o, float) and not np.isfinite(o)
                else (o.tolist() if hasattr(o, "tolist") else str(o))
            ),
            indent=1,
        )
    )
    print("\n".join(lines[:14]))
    print(f"wrote {args.out / (args.arm + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
