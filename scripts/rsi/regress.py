"""TODO 27.10.8 (plan docs/handoff/scene-rsi-plan-20261016.md §12.6): every written-back family's sealed pool re-run
under the catalog as it is now, paired by start with the family's last recorded rounds -- the regression after each
write-back and each scene change (wb3_regress.sh ... upright_regress.sh by hand until 2026-10-19).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/regress.py run <tag> [--families a,b] [--procs 12]
    PYTHONPATH=src $PY scripts/rsi/regress.py add <family> <sandbox> <kind> [--pool <dir>]

The registry, outputs/scene_rsi_20261016/outer/registry.json, lists per family its sandbox, its sealed pool (the
sandbox's, or a filtered copy), its kind for capability_write.py (pick / held / place / into / near / calibration /
none) and the tag of its last recorded rounds (`<sandbox>/<hidden>/<family>/sealed/<tag>/`). A calibration sample
(calibrate.py; user 2026-10-19, its tasks count as measured) is registered with its pool and `--dir` (its sample
dir: the first rounds are tag `rounds`), so a write-back re-runs it too. `run` re-runs them all into `<tag>`
(one batch; the machine-wide simulation slots share the processes), prints per family the recorded and new successes
with Wilson intervals, b, c and the one-sided McNemar p both ways, writes `<registry dir>/regress_<tag>.json`, and --
unless a family is worse at p <= 0.05 -- moves every family's `last` to `<tag>`. `capability_args` gives
capability_write.py's arguments from the registry.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import batch as B  # noqa: E402
from gate import mcnemar_one_sided, wilson  # noqa: E402

REGISTRY = REPO / "outputs" / "scene_rsi_20261016" / "outer" / "registry.json"
HIDDEN = "outputs/scene_rsi_20261016_hidden"
CATALOG = REPO / "src" / "locomani" / "tasks" / "scene" / "skills"
WORSE_P = 0.05
FLAG = {"held": "--held-loop", "place": "--place-loop", "into": "--into-loop", "near": "--near-loop"}


def load() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def rounds_dir(family: str, row: dict, tag: str) -> Path:
    """`<sandbox>/<hidden>/<family>/sealed/<tag>`; a calibration sample's `<dir>/<tag>` (its first rounds: `rounds`)."""
    return Path(row["dir"]) / tag if row.get("dir") else Path(row["sandbox"]) / HIDDEN / family / "sealed" / tag


def pool_of(family: str, row: dict) -> Path:
    return Path(row.get("pool") or Path(row["sandbox"]) / HIDDEN / family / "pool_sealed")


def compare(old: dict, new: dict) -> dict:
    keys = sorted(set(old) & set(new))
    b = sum(new[k]["ok"] and not old[k]["ok"] for k in keys)
    c = sum(old[k]["ok"] and not new[k]["ok"] for k in keys)
    per = collections.defaultdict(lambda: [0, 0, 0])
    for k in keys:
        t = per[new[k]["task"]]
        t[0] += bool(old[k]["ok"])
        t[1] += bool(new[k]["ok"])
        t[2] += 1
    ok_old, ok_new = sum(old[k]["ok"] for k in keys), sum(new[k]["ok"] for k in keys)
    return dict(n=len(keys), old=ok_old, new=ok_new, old_ci=wilson(ok_old, len(keys)), new_ci=wilson(ok_new, len(keys)),
                b=b, c=c, p_worse=round(mcnemar_one_sided(c, b), 4), p_better=round(mcnemar_one_sided(b, c), 4),
                fell=sum(bool(new[k].get("fell")) for k in keys), dropped=sum(bool(new[k].get("dropped")) for k in keys),
                per_task={t: f"{a}/{n} -> {z}/{n}" for t, (a, z, n) in sorted(per.items())})  # fmt: skip


def run(tag: str, families: list[str] | None, procs: int, gpus: str, catalog: Path) -> dict:
    reg = load()
    names = families or sorted(reg)
    for f in names:  # one family per batch call: run_jobs takes one family's pool
        row = reg[f]
        B.run_jobs([B.Job(catalog, rounds_dir(f, row, tag), {})], f, pool_of(f, row), procs,
                   rounds_dir(f, row, tag) / "_regress.log", gpus)  # fmt: skip
    report = {f: compare(B.outcomes(rounds_dir(f, reg[f], reg[f]["last"])), B.outcomes(rounds_dir(f, reg[f], tag)))
              for f in names}  # fmt: skip
    worse = sorted(f for f, r in report.items() if r["p_worse"] <= WORSE_P)
    out = dict(tag=tag, catalog=str(catalog), families=report, worse=worse,
               total=dict(b=sum(r["b"] for r in report.values()), c=sum(r["c"] for r in report.values())))  # fmt: skip
    (REGISTRY.parent / f"regress_{tag}.json").write_text(json.dumps(out, indent=1))
    for f, r in report.items():
        print(f"{f:24s} {r['old']:3d} -> {r['new']:3d} / {r['n']}  {r['old_ci']} -> {r['new_ci']}  b {r['b']} c {r['c']}  "
              f"p(worse) {r['p_worse']} p(better) {r['p_better']}  fell {r['fell']} dropped {r['dropped']}")  # fmt: skip
    print(f"total b {out['total']['b']} c {out['total']['c']}; worse at p <= {WORSE_P}: {worse or 'none'}")
    if not worse:
        for f in names:
            reg[f]["last"] = tag
        REGISTRY.write_text(json.dumps(reg, indent=1))
    return out


def capability_args(reg: dict | None = None) -> list[str]:
    """capability_write.py's arguments for every registered family at its last recorded rounds."""
    reg = reg if reg is not None else load()
    args = []
    for f, row in sorted(reg.items()):
        d = str(rounds_dir(f, row, row["last"]))
        if row["kind"] == "pick":
            args += ["--loop-root", row["sandbox"], "--pick-rounds", d]
        elif row["kind"] in FLAG:
            args += [FLAG[row["kind"]], f"{row['sandbox']}:{f}:{d}"]
        elif row["kind"] == "calibration":
            args += ["--calibration", f"{row['sandbox']}:{f}:{d}:{pool_of(f, row)}"]
    return args


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("tag")
    r.add_argument("--families")
    r.add_argument("--procs", type=int, default=12)
    r.add_argument("--gpus", default="0,1")
    r.add_argument("--catalog", type=Path, default=CATALOG)
    a_ = sub.add_parser("add")
    a_.add_argument("family")
    a_.add_argument("sandbox")
    a_.add_argument("kind", choices=("pick", *FLAG, "calibration", "none"))
    a_.add_argument("--pool")
    a_.add_argument("--last", default="seed")
    a_.add_argument("--dir", help="where the tags' rounds dirs are, when not <sandbox>/<hidden>/<family>/sealed")
    sub.add_parser("capability-args")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.tag, a.families.split(",") if a.families else None, a.procs, a.gpus, a.catalog)
    elif a.cmd == "add":
        reg = load()
        reg[a.family] = dict(
            sandbox=a.sandbox, kind=a.kind, pool=a.pool, last=a.last, **({"dir": a.dir} if a.dir else {})
        )
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY.write_text(json.dumps(reg, indent=1))
    else:
        print(" ".join(capability_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
