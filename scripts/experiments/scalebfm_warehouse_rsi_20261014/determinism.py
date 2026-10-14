"""TODO 25.9.1 phase 0 (plan §6.3, §9): do two sets of ScaleBFM rounds agree round by round? The pairing in the gate and
every fork rest on the round being a function of (start state, spec) alone -- not of the process, the shard, the order
of rounds or a forked prefix.

Compares directories of round files (bfm_round.py's, or 25.7's facts-only JSON) by label: verdict, reason and end time
(to 0.01 s), and where both have arrays, the largest |qpos| difference over the common ticks.

    python3 scripts/experiments/scalebfm_warehouse_rsi_20261014/determinism.py A_dir B_dir [--out x.json]
Exit 0 when every common round agrees (qpos within 1e-6 where both have it), 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rounds(d: Path) -> dict:
    return {p.stem: p for p in sorted(d.glob("bfm_*.json"))}


def compare(a: Path, b: Path) -> dict:
    fa, fb = json.loads(a.read_text()), json.loads(b.read_text())
    out = dict(
        ok=(fa["ok"], fb["ok"]),
        reason=(fa.get("expert_reason"), fb.get("expert_reason")),
        end=(round(fa["events"]["end"], 2), round(fb["events"]["end"], 2)),
    )
    same = out["ok"][0] == out["ok"][1] and out["reason"][0] == out["reason"][1] and out["end"][0] == out["end"][1]
    za, zb = a.with_suffix(".npz"), b.with_suffix(".npz")
    if za.exists() and zb.exists():
        qa, qb = np.load(za)["qpos"], np.load(zb)["qpos"]
        n = min(len(qa), len(qb))
        out["qpos_max_diff"] = float(np.max(np.abs(qa[:n] - qb[:n]))) if n else None
        out["ticks"] = (len(qa), len(qb))
        same = same and len(qa) == len(qb) and out["qpos_max_diff"] <= 1e-6
    out["same"] = bool(same)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    ra, rb = rounds(a.a), rounds(a.b)
    common = sorted(set(ra) & set(rb))
    res = {k: compare(ra[k], rb[k]) for k in common}
    diff = [k for k, v in res.items() if not v["same"]]
    summary = dict(a=str(a.a), b=str(a.b), common=len(common), differ=diff, rounds=res)
    for k, v in res.items():
        q = f" qpos {v['qpos_max_diff']:.2e}" if v.get("qpos_max_diff") is not None else ""
        print(f"{k}: {'same' if v['same'] else 'DIFF'} ok {v['ok']} reason {v['reason']} end {v['end']}{q}")
    print(f"{len(common)} common rounds, {len(diff)} differ")
    if a.out:
        a.out.write_text(json.dumps(summary, indent=1))
    return 0 if common and not diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
