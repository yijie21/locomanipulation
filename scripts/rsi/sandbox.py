"""TODO 27.2.4: a sandbox checkout for one scene-loop run -- the repository at HEAD (git archive), third_party and the
ADT scene data linked to the origin (read-only), its own git history (the driver diffs the Engineer's edits), and the
family's diagnostic and acceptance pools drawn inside it (the acceptance pool in the hidden tree). The loop then runs
from the sandbox: `<sandbox>/scripts/rsi/loop.py run <family>`.

    python3 scripts/rsi/sandbox.py /workspace/code/rsi-scene/<name> pick_single

A family file's `sandbox` block (TODO 31, user 2026-10-20: the loop starts from the motor layer and must not find the
earlier skill catalog, the scene RSI records or the earlier grasp programs) changes what the sandbox holds:
- `seed`: a directory that replaces the skills package;
- `keep`: the only paths kept outside src/ (files or directories); everything else in the archive is removed;
- `closure`: modules whose imports (traced with `python -X importtime`) decide which src/ modules stay; the skills
  package and the data files beside a kept module stay too;
- `strip`: paths removed from what `keep` kept;
- `scene_exclude`: entries of the scene directory not linked (outputs/scene_discovery is then a directory of links to
  the scene's other entries and its physics copy, instead of one link).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
LINKS = ("third_party", "outputs/scene_discovery")


def git(dest: Path, *args, **kw):
    return subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True, text=True, **kw)


def closure(dest: Path, entries: list[str]) -> set[str]:
    """The locomani modules importing `entries` loads in the sandbox (scripts/rsi and src on the path)."""
    code = "import sys; sys.path[:0] = ['scripts/rsi', 'src']\n" + "".join(f"import {e}\n" for e in entries)
    env = {"PYTHONPATH": "src", "TMPDIR": "/workspace/tmp", "PATH": "/usr/bin:/bin"}
    r = subprocess.run([str(SIM_PY), "-X", "importtime", "-c", code], cwd=dest, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"closure import failed:\n{r.stderr[-2000:]}")
    names = {line.rsplit("|", 1)[-1].strip() for line in r.stderr.splitlines() if "|" in line}
    return {n for n in names if n == "locomani" or n.startswith("locomani.")}


def apply_spec(dest: Path, family: str) -> dict:
    """The family file's `sandbox` block (module docstring) applied to the extracted archive; returns the block."""
    spec = json.loads((dest / "scripts" / "rsi" / "families" / f"{family}.json").read_text()).get("sandbox") or {}
    if not spec:
        return spec
    skills = dest / "src" / "locomani" / "tasks" / "scene" / "skills"
    shutil.rmtree(skills)
    shutil.copytree(dest / spec["seed"], skills, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    mods = closure(dest, spec["closure"])
    src = dest / "src"
    kept_py = {src / (m.replace(".", "/") + ".py") for m in mods} | {
        src / m.replace(".", "/") / "__init__.py" for m in mods
    }
    kept_py = {f for f in kept_py if f.exists()}
    keep = [dest / k for k in spec["keep"]]
    for f in sorted(p for p in dest.rglob("*") if p.is_file() or p.is_symlink()):
        if f.is_relative_to(src):
            ok = (
                f in kept_py
                or f.is_relative_to(skills)
                or (f.suffix != ".py" and any(k.parent == f.parent for k in kept_py))
            )
        else:
            ok = any(f == k or f.is_relative_to(k) for k in keep)
        if not ok or "__pycache__" in f.parts:
            f.unlink()
    for rel in spec.get("strip", []):
        t = dest / rel
        if t.is_dir():
            shutil.rmtree(t)
        elif t.exists():
            t.unlink()
    for d in sorted((p for p in dest.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()
    print(f"sandbox spec: seed {spec['seed']}, {len(kept_py)} src modules kept")
    return spec


def build(dest: Path, family: str, ref: str = "HEAD") -> None:
    """`ref`: the commit exported (replays launched over hours pin one, so later commits do not leak in)."""
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", ref], cwd=REPO, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    spec = apply_spec(dest, family)
    links = [(rel, REPO / rel) for rel in LINKS]
    if spec.get("scene_exclude") is not None:
        scene = json.loads((dest / "scripts" / "rsi" / "families" / f"{family}.json").read_text())["scene"]
        links = [x for x in links if x[0] != "outputs/scene_discovery"]
        links += [
            (f"{scene}/{e.name}", e) for e in sorted((REPO / scene).iterdir()) if e.name not in spec["scene_exclude"]
        ]
        links.append((f"{scene}_physics", REPO / f"{scene}_physics"))
    for rel, target in links:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).symlink_to(target)
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
