"""TODO 26.3.3 (docs/handoff/scalebfm-bottle-rsi-20261015.md §5): build the bottle loop's sandbox -- `git archive HEAD` with
the warehouse RSI line taken out (its plan, scripts, demo videos and tests carry the warehouse fixes: the bottle loop
starts cold and reads only rules.md and its own plan), STATE / TODO / DECISIONS as one-line stubs, the data copied (the
diagnostic and acceptance pools, the bottle scene), third_party linked, and a one-commit git repository so the driver
can tell the Engineer's edits apart.

    python3 scripts/experiments/scalebfm_bottle_rsi_20261015/sandbox.py /workspace/code/rsi-phase4/bottle-1
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOOP = Path("outputs/scalebfm_bottle_rsi_20261015")
HIDDEN = Path("outputs/scalebfm_bottle_rsi_20261015_hidden")
SCENE = Path("outputs/scene/sim_lowcom")
DROP = (
    "docs/handoff/scalebfm-warehouse-rsi-20261014.md",
    "scripts/experiments/scalebfm_warehouse_rsi_20261014",
    "scripts/experiments/rsi_demo_20261015",
    "tests/test_scalebfm_rsi_gate.py",
    "tests/test_scalebfm_rsi_phase4.py",
)
# what the warehouse loop found, by name: none may be left in the sandbox
MARKERS = re.compile(
    r"pelvis_floor|base_anchor: reference|71743845d6b5|8eb038e21ad0|phase1_best|h1_lift_reseat|封存集 68|scalebfm-warehouse-rsi"
)
STUB = "# {name}\n\nThis checkout is the sandbox of the bottle RSI loop (its plan §5). The project's {name} is not part of it.\n"


def git(dest: Path, *args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True, text=True, **kw)


def build(dest: Path) -> None:
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    if re.search(r"[^A-Za-z0-9/-]", str(dest)):
        raise SystemExit("use letters, digits and '-' only: the transcript audit maps the path to ~/.claude/projects")
    if not (REPO / HIDDEN / "pool_accept" / "states").is_dir():
        raise SystemExit("no acceptance pool yet (sonic_pool.sh accept, prep_pools.py accept)")
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "HEAD"], cwd=REPO, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    for rel in DROP:
        p = dest / rel
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    for name in ("STATE.md", "TODO.md", "DECISIONS.md"):
        (dest / name).write_text(STUB.format(name=name.removesuffix(".md")))
    readme = dest / "docs" / "README.md"
    readme.write_text("".join(x for x in readme.read_text().splitlines(True) if "scalebfm-warehouse-rsi" not in x))
    hits = []
    for p in sorted(dest.rglob("*")):
        rel = p.relative_to(dest).as_posix()
        if p.is_file() and not p.is_symlink() and not rel.startswith((".git/", "outputs/")):
            try:
                hits += [f"{rel}: {m.group(0)}" for m in MARKERS.finditer(p.read_text())]
            except (UnicodeDecodeError, OSError):
                pass
    if hits:
        raise SystemExit("warehouse-loop markers left:\n  " + "\n  ".join(hits))
    (dest / "third_party").symlink_to(REPO / "third_party")
    shutil.copytree(REPO / SCENE, dest / SCENE)
    shutil.copytree(REPO / LOOP / "pool_diag" / "states", dest / LOOP / "pool_diag" / "states")
    shutil.copytree(REPO / HIDDEN / "pool_accept" / "states", dest / HIDDEN / "pool_accept" / "states")
    git(dest, "init", "-q")
    files = git(dest, "ls-files", "-o", "--exclude-standard", "-z").stdout
    if any(f.startswith(("third_party", "outputs")) for f in files.split("\0")):
        raise SystemExit("third_party or outputs would be tracked; check .gitignore")
    git(dest, "add", "--pathspec-from-file=-", "--pathspec-file-nul", input=files)
    git(
        dest,
        "-c",
        "user.name=rsi-bottle",
        "-c",
        "user.email=rsi-bottle@localhost",
        "commit",
        "-q",
        "-m",
        "bottle RSI sandbox",
    )
    print(f"{dest}: {len(files.split(chr(0))) - 1} files tracked")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(Path(sys.argv[1]).resolve())
