"""selfmodel_train3b_20261125: the registered predictions' and criteria's numbers, read from the final tables
(final_<split>.json from t3b_mx.sh's t3b_eval.py run; lml_real.json from t3b_lmleval.py) of one or more controllers.
Read-only; it only picks cells and prints their point values with the 95 % intervals the evaluation computed.
Ratios across two cells (T3 on data3b's entry / T3 on data3's val, M's real 第二下 / M's val, chained / first) are
point ratios without an interval (the two cells were resampled separately).

    python3 t3b_summary.py <train3b output dir> [<backend> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def find(sec: dict, group: str, dim: str, lab_start: str):
    for c in sec["cells"]:
        if c["group"] == group and c["dim"] == dim and c["cell"].startswith(lab_start):
            return c
    return None


def sec_of(j: dict, title_start: str):
    for s in j["sections"]:
        if s and s.get("title", "").startswith(title_start):
            return s
    return None


def v(c, k, d=2):
    if c is None or k not in c["point"]:
        return "—"
    lo, hi = c["ci"][k]
    return f"{c['point'][k]:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def seeds(c, arm):
    if c is None:
        return "—"
    ks = [f"{arm}{j}/M{j}" if arm != "M" else f"M{j}/T3" for j in range(3)]
    xs = [c["point"][k] for k in ks if k in c["point"]]
    return ", ".join(f"{x:.3f}" for x in xs) + (f" (all < 1: {all(x < 1 for x in xs)})" if xs else "")


def pt(c, k):
    return c["point"][k] if c is not None and k in c["point"] else float("nan")


def mean_m(c, pre=""):
    xs = [pt(c, f"{pre}M{j}") for j in range(3)]
    return sum(xs) / len(xs)


def report(root: Path, be: str) -> None:
    print(f"\n######## {be}")
    fin = {s: load(root / be / f"final_{s}.json") for s in ("test_iid", "val", "edge")}
    for split in ("test_iid", "val", "edge"):
        j = fin[split]
        if j is None:
            print(f"  {split}: no final table yet")
            continue
        print(f"\n  ==== {split}: models {[m[0] for m in j['models']]} + {j['mx']}")
        d3, d3b = sec_of(j, "data3 "), sec_of(j, "data3b ")
        d3v = sec_of(fin["val"], "data3 ") if fin["val"] else None
        for name, sec, dim, lab in (("data3 all", d3, "all", "all"), ("data3b all", d3b, "all", "all"),
                                    ("data3b entry", d3b, "kind", "entry"), ("data3b pre-grasp", d3b, "kind", "pre-grasp"),
                                    ("data3b rel < 15", d3b, "kind", "rel < 15"), ("data3b rel 15-30", d3b, "kind", "rel 15-30"),
                                    ("data3b rel >= 30", d3b, "kind", "rel >= 30"),
                                    ("data3b entry 8-17 cm, wrist < 20", d3b, "entry", "8-17")):  # fmt: skip
            if sec is None:
                continue
            for g in ("all", "first", "chained"):
                c = find(sec, g, dim, lab)
                if c is None:
                    continue
                print(f"  [{name} | {g}] n {c['n']} ({c['n_resets']} resets)")
                print(f"     pos median: direct {v(c, 'direct')} | T3 {v(c, 'T3')} | M0 {v(c, 'M0')} | D0 {v(c, 'D0')} | MX0 {v(c, 'MX0')} | MXa {v(c, 'MXa')}")
                print(f"     pos p90:    direct {v(c, 'p90_direct')} | T3 {v(c, 'p90_T3')} | M0 {v(c, 'p90_M0')} | D0 {v(c, 'p90_D0')} | MX0 {v(c, 'p90_MX0')}")
                print(f"     att median: direct {v(c, 'att_direct')} | T3 {v(c, 'att_T3')} | M0 {v(c, 'att_M0')} | D0 {v(c, 'att_D0')} | MX0 {v(c, 'att_MX0')}")
                print(f"     M/T3 {v(c, 'M/T3', 3)} seeds {seeds(c, 'M')} | att M/T3 {v(c, 'att_M/T3', 3)}")
                print(f"     D/M {v(c, 'D/M', 3)} seeds {seeds(c, 'D')} | att D/M {v(c, 'att_D/M', 3)}")
                print(f"     MX/M {v(c, 'MX/M', 3)} seeds {seeds(c, 'MX')} | att MX/M {v(c, 'att_MX/M', 3)} | MXa/M0 (副臂，只描述) {v(c, 'MXa/M0', 3)}")
                print(f"     curve 1.0/0.3 {v(c, 'curve_1/0.3', 3)} | 0.6/0.3 {v(c, 'curve_0.6/0.3', 3)}"
                      f" | M (seed mean) / direct {mean_m(c) / pt(c, 'direct'):.3f} | T3 / direct {pt(c, 'T3') / pt(c, 'direct'):.3f}")  # fmt: skip
        # cross-cell point ratios
        if d3b is not None and d3v is not None:
            e, v3 = find(d3b, "all", "kind", "entry"), find(d3v, "all", "all", "all")
            print(f"  point ratios: T3 entry ({split}) / T3 data3 val all = {pt(e, 'T3') / pt(v3, 'T3'):.3f};"
                  f" M (seed mean) entry / M data3 val all = {mean_m(e) / mean_m(v3):.3f}")  # fmt: skip
        for name, sec in (("data3", d3), ("data3b", d3b)):
            if sec is None:
                continue
            f_, c_ = find(sec, "first", "all", "all"), find(sec, "chained", "all", "all")
            print(f"  point ratios {name} chained / first: T3 {pt(c_, 'T3') / pt(f_, 'T3'):.3f}, M (seed mean) {mean_m(c_) / mean_m(f_):.3f}, direct {pt(c_, 'direct') / pt(f_, 'direct'):.3f}")
    lj = load(root / be / "lml_real.json")
    if lj is None:
        print("\n  lmlreal: no table yet")
        return
    print(f"\n  ==== lmlreal: {lj['n']} segments; models {lj['models']}; {lj['note']}")
    v3 = find(sec_of(fin["val"], "data3 "), "all", "all", "all") if fin["val"] else None
    v3b = find(sec_of(fin["val"], "data3b "), "all", "all", "all") if fin["val"] else None
    for c in lj["cells"]:
        print(f"  [{c['cell']} | {c['group']}] n {c['n']} ({c['n_resets']} episodes){'  (MX 留空: chain 0)' if c.get('mx_empty') else ''}")
        print(f"     pos median: direct {v(c, 'direct')} | T3 {v(c, 'T3')} | M0 {v(c, 'M0')} | D0 {v(c, 'D0')} | MX0 {v(c, 'MX0')} | MXa {v(c, 'MXa')}")
        print(f"     pos p90:    direct {v(c, 'p90_direct')} | T3 {v(c, 'p90_T3')} | M0 {v(c, 'p90_M0')} | D0 {v(c, 'p90_D0')} | MX0 {v(c, 'p90_MX0')}")
        print(f"     att median: direct {v(c, 'att_direct')} | T3 {v(c, 'att_T3')} | M0 {v(c, 'att_M0')} | D0 {v(c, 'att_D0')} | MX0 {v(c, 'att_MX0')}")
        print(f"     M/T3 {v(c, 'M/T3', 3)} seeds {seeds(c, 'M')} | D/M {v(c, 'D/M', 3)} | MX/M {v(c, 'MX/M', 3)} seeds {seeds(c, 'MX')} | MXa/M0 {v(c, 'MXa/M0', 3)}")
        if v3 is not None:
            print(f"     M (seed mean) here / M data3 val all = {mean_m(c) / mean_m(v3):.2f}; / M data3b val all = {mean_m(c) / mean_m(v3b):.2f}")


def main() -> None:
    root = Path(sys.argv[1])
    for be in sys.argv[2:] or ["scalebfm", "heft", "sonic"]:
        report(root, be)


if __name__ == "__main__":
    main()
