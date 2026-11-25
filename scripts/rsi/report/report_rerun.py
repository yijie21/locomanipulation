"""TODO 27.14: the progress report's video rounds run again in the corrected scene (user 2026-10-20: "台面改完后在新场景里
重跑再渲染"). Every round a report video shows -- pick_single's four steps (report_unreal.PS), the accepted changes
(report_media.STEPS) and the untrained tasks (report_media.SOLOS) -- runs once more from the same start, with the same
skills package (the round file's content hash checked against the snapshot found) and the same overrides, and records
the measured fingers (motor trace `fingers`). The steps' versions come from their sandboxes (`incumbents/<hash>/skills`,
or a gate's `cand/skills`), the untrained tasks run on today's catalog as before.

Outputs mirror the sources under outputs/scene_rsi_20261016/report/rounds_20261020/: `pick/<version>/`,
`diag/<step family>/<version>/`, `untrained/<sample>/rounds/`; report_media.py and report_unreal.py read them with
`--rounds rounds_20261020`. `check` lists each round's outcome then and now.

    PYTHONPATH=src $PY scripts/rsi/report/report_rerun.py run [--procs 8]
    PYTHONPATH=src $PY scripts/rsi/report/report_rerun.py check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # scripts/rsi/report/ since TODO 36.8
sys.path[:0] = [str(HERE), str(HERE.parent), str(REPO / "src")]

import batch as B  # noqa: E402
import regress as RG  # noqa: E402
import report_media as RM  # noqa: E402
import report_unreal as RU  # noqa: E402
from scene_round import skills_hash  # noqa: E402

OUT = REPO / "outputs/scene_rsi_20261016/report/rounds_20261020"


def family_dir(diag: Path) -> tuple[Path, str]:
    """(sandbox, family) of a loop's diag directory `<sandbox>/outputs/scene_rsi_20261016/<family>/diag`."""
    return diag.parents[3], diag.parent.name


def skills_of(sandbox: Path, family: str, version: str, want: str) -> Path:
    """The skills package of `version` ("incumbent_<hash>" / "candidate_<hash>") whose content hash is `want`."""
    h = version.split("_", 1)[1]
    found = [
        *(sandbox / "outputs/scene_rsi_20261016").glob(f"*/incumbents/{h}/skills"),
        *(sandbox / "outputs/scene_rsi_20261016_hidden").glob(f"*/gate/*_{h}/cand/skills"),
    ]
    for d in found:
        if skills_hash(d) == want:
            return d
    raise SystemExit(f"{family} {version}: no snapshot with content hash {want} (looked at {[str(d) for d in found]})")


def jobs() -> list[dict]:
    """One entry per (source round): where it was, where its rerun goes, family, pool, skills, overrides."""
    out = []

    def add(src: Path, dst: Path, family: str, pool: Path, skills: Path | None, sandbox: Path | None):
        rec = json.loads(src.with_suffix(".json").read_text())
        if skills is None:
            skills = skills_of(sandbox, family, src.parent.name, rec["skills"])
        out.append(dict(src=src, dst=dst, family=family, pool=pool, skills=skills, set=rec.get("set") or {},
                        label=src.name, ok=bool(rec["ok"])))  # fmt: skip

    sand, fam = family_dir(RU.PICK)
    for before, after, label in RU.PS.values():
        for v in (before, after):
            add(RU.PICK / v / label, OUT / "pick" / v / label, fam, RU.PICK.parent / "pool_diag", None, sand)
    for key, (fk, before, after, label, *_rest) in RM.STEPS.items():
        sand, fam = family_dir(RM.DIAG[fk])
        for v in [x for x in (before, after) if x]:
            add(
                RM.DIAG[fk] / v / label,
                OUT / "diag" / fk / v / label,
                fam,
                RM.DIAG[fk].parent / "pool_diag",
                None,
                sand,
            )
    reg = RG.load()
    for sample, labels, *_rest in RM.SOLOS.values():
        for label in labels:
            add(RM.UNTRAINED / sample / "rounds" / label, OUT / "untrained" / sample / "rounds" / label, sample,
                RG.pool_of(sample, reg[sample]), RG.CATALOG, None)  # fmt: skip
    seen, uniq = set(), []
    for j in out:  # pick_single's steps share versions and starts
        if str(j["dst"]) not in seen:
            seen.add(str(j["dst"]))
            uniq.append(j)
    return uniq


def run(procs: int, gpus: str) -> None:
    groups: dict = {}
    for j in jobs():
        key = (
            j["family"],
            str(j["pool"]),
            str(j["skills"]),
            json.dumps(j["set"], sort_keys=True),
            str(j["dst"].parent),
        )
        groups.setdefault(key, []).append(j["label"])
    print(f"{sum(len(v) for v in groups.values())} rounds in {len(groups)} jobs", flush=True)
    for (family, pool, skills, overrides, dst), labels in groups.items():
        job = B.Job(Path(skills), Path(dst), json.loads(overrides), tuple(sorted(labels)), npz=True)
        B.run_jobs([job], family, Path(pool), procs, Path(dst) / "_batch.log", gpus)
        print(family, Path(dst).relative_to(OUT), len(labels), flush=True)


def check() -> None:
    flips = 0
    for j in jobs():
        f = j["dst"].with_suffix(".json")
        now = json.loads(f.read_text()) if f.exists() else None
        state = "missing" if now is None else ("ok" if now["ok"] else "fail: " + (now.get("why") or "")[:70])
        flips += now is not None and bool(now["ok"]) != j["ok"]
        mark = "" if now is None or bool(now["ok"]) == j["ok"] else "  <-- changed"
        seat = f" seated {now.get('seated')}" if now and now.get("seated") else ""
        print(f"{str(j['dst'].relative_to(OUT)):78s} then {'ok' if j['ok'] else 'fail'} | now {state}{seat}{mark}")
    print("changed outcomes:", flips)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "check", "list"))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.procs, a.gpus)
    elif a.cmd == "check":
        check()
    else:
        for j in jobs():
            print(j["family"], j["dst"].relative_to(OUT), j["skills"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
