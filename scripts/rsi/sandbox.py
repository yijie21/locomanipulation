"""TODO 27.2.4: a sandbox checkout for one scene-loop run -- the repository at HEAD (git archive), third_party and the
ADT scene data linked to the origin (read-only), its own git history (the driver diffs the Engineer's edits), and the
family's diagnostic and acceptance pools drawn inside it (the acceptance pool in the hidden tree). The loop then runs
from the sandbox: `<sandbox>/scripts/rsi/loop.py run <family>`.

    python3 scripts/rsi/sandbox.py /workspace/code/rsi-scene/<name> pick_single
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
LINKS = ("third_party", "outputs/scene_discovery")


def git(dest: Path, *args, **kw):
    return subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True, text=True, **kw)


def build(dest: Path, family: str) -> None:
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "HEAD"], cwd=REPO, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    for rel in LINKS:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).symlink_to(REPO / rel)
    git(dest, "init", "-q")
    files = git(dest, "ls-files", "-o", "--exclude-standard", "-z").stdout
    if any(f.startswith(("third_party", "outputs")) for f in files.split("\0")):
        raise SystemExit("third_party or outputs would be tracked; check .gitignore")
    git(dest, "add", "--pathspec-from-file=-", "--pathspec-file-nul", input=files)
    git(
        dest,
        "-c",
        "user.name=rsi-scene",
        "-c",
        "user.email=rsi-scene@localhost",
        "commit",
        "-q",
        "-m",
        "scene RSI sandbox",
    )
    loop = dest / "outputs" / "scene_rsi_20261016" / family
    hidden = dest / "outputs" / "scene_rsi_20261016_hidden" / family
    fam = dest / "scripts" / "rsi" / "family.py"
    env = {"PYTHONPATH": "src", "TMPDIR": "/workspace/tmp", "PATH": "/usr/bin:/bin"}
    for pool, out in (("diag", loop / "pool_diag"), ("accept", hidden / "pool_accept")):
        r = subprocess.run(
            [str(SIM_PY), str(fam), "draw", family, pool, str(out)], cwd=dest, env=env, capture_output=True, text=True
        )
        if r.returncode:
            raise SystemExit(r.stderr[-2000:])
        print(r.stdout.strip())
    print(f"{dest}: sandbox for {family}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", type=Path)
    ap.add_argument("family")
    a = ap.parse_args(argv)
    build(a.dest.resolve(), a.family)
    return 0


if __name__ == "__main__":
    sys.exit(main())
