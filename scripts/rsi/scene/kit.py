"""The scene RSI reproduction kit (user 2026-10-20: another developer must be able to re-run the evolution and check
its recorded results). `rsi_kit/` holds what cannot be regenerated: model outputs, and starts drawn on the scene of
their day. That is the ADT source scene file with Claude's property estimates, the current scene file, the three
discoveries' task lists, the calibration samples, the stance overlay, the outer rounds' selections, and per
registered family its three start pools and its last recorded outcome per start.

These are ADT derivatives. Plan decision 17 kept those out of git; the user overrode it for this kit (DECISIONS
2026-10-20). `rsi_kit` is export-ignore (.gitattributes), so a loop sandbox (`git archive HEAD`, sandbox.py) does not
carry the hidden pools, and loop.py's transcript audit flags any read of it.

    PYTHONPATH=src $PY scripts/rsi/kit.py pack   # maintainer, sim python: rebuild rsi_kit/ from outputs/ and the registry
    python3 scripts/rsi/kit.py restore    # reproducer: rsi_kit/ back under outputs/, and the registry
    python3 scripts/rsi/kit.py seed       # the seed skills catalog (before the first write-back) into the repository

`restore` never overwrites: a file already there with other content is kept (printed `kept`), and an existing
registry is left alone. The families' pools and outcomes land in outputs/scene_rsi_20261016/kit/<family>/; the
registry it writes points regress.py there, with the recorded outcomes as tag `recorded`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KIT = REPO / "rsi_kit"
SEED_COMMIT = "801a28c^"  # the catalog before pick_single's write-back: the seed pick_single started from on 2026-10-16
CATALOG = "src/locomani/tasks/scene/skills"
# repository-relative files copied as they are
FILES = (
    "outputs/scene_discovery/adt_apartment/scene_meta.json",  # adt_to_scene geometry + estimate (Claude's properties)
    "outputs/scene_discovery/adt_apartment_mars/scene_meta.json",  # the scene as the loop runs it now
    "outputs/scene_discovery/adt_apartment/discovery/opus_20261005b/tasks.jsonl",  # pick_single's tasks
    "outputs/scene_discovery/adt_apartment_mars/discovery/rsi_round1/tasks.jsonl",
    "outputs/scene_discovery/adt_apartment_mars/discovery/rsi_round1/report.json",
    "outputs/scene_discovery/adt_apartment_mars/discovery/rsi_round2/tasks.jsonl",
    "outputs/scene_discovery/adt_apartment_mars/discovery/rsi_round2/report.json",
    "outputs/scene_rsi_20261016/capability/stance/adt_apartment_mars.json",
    "outputs/scene_rsi_20261016/outer/round1/selection.json",
    "outputs/scene_rsi_20261016/outer/round2/selection.json",
    *(f"outputs/scene_rsi_20261016_hidden/calibration/{c}/{f}" for c in ("cal_r2", "cal_r3", "cal_r4")
      for f in ("tasks.jsonl", "sample.json")),
)  # fmt: skip
ROUNDS = {"round1": "rsi_round1", "round2": "rsi_round2"}  # outer round dir -> its discovery
KEEP = ("label", "task", "ok", "why", "fell", "dropped", "t_end", "seated")  # a recorded round, as regress.py reads it


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pools(family: str, row: dict) -> dict[str, Path]:
    import regress

    if row["kind"] == "calibration":
        return {"pool_sealed": regress.pool_of(family, row)}
    sb = Path(row["sandbox"])
    return {
        "pool_diag": sb / "outputs/scene_rsi_20261016" / family / "pool_diag",
        "pool_accept": sb / regress.HIDDEN / family / "pool_accept",
        "pool_sealed": regress.pool_of(family, row),
    }


def pack() -> int:
    sys.path.insert(0, str(HERE))
    import regress

    reg = regress.load()
    if KIT.exists():
        shutil.rmtree(KIT)
    (KIT / "files").mkdir(parents=True)
    for f in FILES:
        dst = KIT / "files" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, dst)
    families = {}
    for fam, row in sorted(reg.items()):
        d = KIT / "families" / fam
        for name, src in pools(fam, row).items():
            (d / name).mkdir(parents=True)
            for p in sorted(src.glob("*.json")):
                shutil.copy2(p, d / name / p.name)
        rec = regress.rounds_dir(fam, row, row["last"])
        (d / "recorded").mkdir()
        for label, r in regress.B.outcomes(rec).items():
            (d / "recorded" / f"{label}.json").write_text(json.dumps({k: r[k] for k in KEEP if k in r}) + "\n")
        families[fam] = dict(kind=row["kind"], recorded_tag=row["last"])
        print(f"{fam:22s}", {p.name: len(list(p.glob("*.json"))) for p in sorted(d.iterdir())})
    # the scene files and selections name absolute paths under the packing repository; restore rewrites them
    root = str((REPO / "outputs").resolve().parent)
    files = {str(p.relative_to(KIT)): sha(p) for p in sorted(KIT.rglob("*")) if p.is_file()}
    manifest = dict(root=root, families=families, files=files)
    (KIT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    size = sum(p.stat().st_size for p in KIT.rglob("*") if p.is_file())
    print(f"{len(manifest['files'])} files, {size / 1e6:.1f} MB in {KIT.relative_to(REPO)}")
    return 0


def put(src: Path, dst: Path, counts: dict, swap: tuple[str, str] | None = None) -> None:
    data = src.read_bytes()
    if swap and swap[0] != swap[1]:  # the packing repository's absolute paths -> this one's
        data = data.replace(swap[0].encode() + b"/", swap[1].encode() + b"/")
    if dst.exists():
        same = dst.read_bytes() == data
        counts["same" if same else "kept"] += 1
        if not same:
            print("kept", dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    counts["new"] += 1


def restore(root: Path) -> int:
    manifest = json.loads((KIT / "MANIFEST.json").read_text())
    bad = [f for f, h in manifest["files"].items() if sha(KIT / f) != h]
    if bad:
        raise SystemExit(f"rsi_kit/ does not match its MANIFEST.json: {bad[:5]}")
    counts = dict(new=0, same=0, kept=0)
    for f in FILES:
        put(KIT / "files" / f, root / f, counts, (manifest["root"], str(root)))
    for rnd, disc in ROUNDS.items():  # outer.py reads <round>/discovery/tasks.jsonl
        link = root / "outputs/scene_rsi_20261016/outer" / rnd / "discovery"
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path("../../../scene_discovery/adt_apartment_mars/discovery") / disc)
    print(f"files: {counts['new']} new, {counts['same']} already there, {counts['kept']} kept")
    base = root / "outputs/scene_rsi_20261016/kit"
    reg = {}
    for fam, info in manifest["families"].items():
        c = dict(new=0, same=0, kept=0)
        for p in sorted((KIT / "families" / fam).rglob("*.json")):
            put(p, base / fam / p.relative_to(KIT / "families" / fam), c)
        d = (base / fam).resolve()
        reg[fam] = dict(sandbox=str(d), kind=info["kind"], pool=str(d / "pool_sealed"), last="recorded", dir=str(d))
        print(f"{fam:22s} pools and recorded outcomes ({info['recorded_tag']}): {c['new']} new, {c['same']} already there, {c['kept']} kept")  # fmt: skip
    registry = root / "outputs/scene_rsi_20261016/outer/registry.json"
    if registry.exists():
        print("registry kept:", registry)
    else:
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(json.dumps(reg, indent=1))
        print(f"registry written: {registry} ({len(reg)} families, tag `recorded`)")
    return 0


def seed() -> int:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", CATALOG], cwd=REPO, capture_output=True, text=True, check=True
    )
    if dirty.stdout.strip():
        raise SystemExit(f"{CATALOG} has uncommitted changes; commit or stash them first")
    tar = subprocess.run(["git", "archive", SEED_COMMIT, CATALOG], cwd=REPO, capture_output=True, check=True).stdout
    shutil.rmtree(REPO / CATALOG)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        try:
            t.extractall(REPO, filter="data")
        except TypeError:  # Python before 3.10.12 has no extraction filters; the archive is our own commit
            t.extractall(REPO)
    n = sum(1 for _ in (REPO / CATALOG).rglob("*.py"))
    print(f"{CATALOG}: the seed catalog ({SEED_COMMIT}), {n} Python files; commit it before building a sandbox")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("pack", "restore", "seed"))
    ap.add_argument("--root", type=Path, default=REPO, help="restore: the repository to restore into")
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        return pack()
    if a.cmd == "restore":
        return restore(a.root.resolve())
    return seed()


if __name__ == "__main__":
    raise SystemExit(main())
