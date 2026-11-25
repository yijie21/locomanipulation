"""TODO 27.2.4: a sandbox checkout for one scene-loop run -- the repository at HEAD (git archive), third_party and the
ADT scene data linked to the origin (read-only), its own git history (the driver diffs the Engineer's edits), and the
family's diagnostic and acceptance pools drawn inside it (the acceptance pool in the hidden tree). The loop then runs
from the sandbox: `<sandbox>/scripts/rsi/loop.py run <family>`.

    python3 scripts/rsi/sandbox.py /workspace/code/rsi-scene/<name> pick_single [--fresh]

`--fresh` (user 2026-11-18): the sandbox holds neither the knowledge library (scripts/rsi/knowledge) nor the
experience library (scripts/rsi/experience): the run inherits no earlier family's knowledge. scripts/rsi/alternate.py
builds its two sandboxes this way and installs the shared library itself.

A family file's `sandbox` block (TODO 31, user 2026-10-20: the loop starts from the motor layer and must not find the
earlier skill catalog, the scene RSI records or the earlier grasp programs) changes what the sandbox holds:
- `seed`: a directory that replaces the skills package;
- `keep`: the only paths kept outside src/ (files or directories); everything else in the archive is removed;
- `closure`: modules whose imports (traced with `python -X importtime`) decide which src/ modules stay; the skills
  package and the data files beside a kept module stay too;
- `strip`: paths removed from what `keep` kept;
- `scene_exclude`: entries of the scene directory not linked (outputs/scene_discovery is then a directory of links to
  the scene's other entries and its physics copy, instead of one link).

The experience library (scripts/rsi/experience, TODO 32.10) goes in without the family's own entry, and its index is
rebuilt: a re-run does not read how its family's earlier run ended (leave-one-out, as TODO 32.12 ran; the default
since TODO 32.15, user 2026-10-22). For a family with a `sandbox` block it also goes in without any entry's
final.patch (the old grasp code; rule 27 -- in TODO 32.12 an Engineer read experience/pick_single/final.patch) and
without the entries of the family file's `library_exclude` (a list of family names). TODO 36 (review 2026-10-22).

Every sandbox, block or not, is built without CLAUDE.md, AGENTS.md and scripts/rsi/report: the role sessions load the
sandbox's CLAUDE.md / AGENTS.md, which send them to STATE, `make check` and commits, and report/ holds the Reviewer
tools turned off in TODO 32.6 (moved there from scripts/rsi/tools in TODO 36.8, with the report scripts; the 10-19
library scripts/rsi/library was deleted then; both are still stripped from a sandbox built from an older commit).

The acceptance and sealed pools' seeds are not in the sandbox's copy of the family file: build writes them to
outputs/scene_rsi_20261016_hidden/<family>/seeds.json and blanks them in the file before the sandbox's git commit, so no
role can regenerate the hidden starts from family.py draw (family.load reads them back from the hidden file). The
diagnostic seed stays. TODO 36 (review 2026-10-22).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src")]

from locomani import paths  # noqa: E402

SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
LINKS = ("third_party", "outputs/scene_discovery")
HIDDEN = "outputs/scene_rsi_20261016_hidden"
# TODO 36 (review 2026-10-22); scripts/rsi/report since TODO 36.8 (the human analysis and report scripts, the old
# tools/ among them); tools/ and library/ stay listed for `build --ref` of a commit from before 36.8
DEFAULT_STRIP = ("CLAUDE.md", "AGENTS.md", "scripts/rsi/report", "scripts/rsi/tools", "scripts/rsi/library")
# What the sandbox's own python runs see: the workspace (so paths.py resolves the same tmp and models there) and
# the usual tmp; nothing else of this shell.
SUB_ENV = {"PYTHONPATH": "src", "TMPDIR": paths.TMP, "LOCOMANI_WORKSPACE": paths.WORKSPACE, "PATH": "/usr/bin:/bin"}


def git(dest: Path, *args, **kw):
    return subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True, text=True, **kw)


def closure(dest: Path, entries: list[str]) -> set[str]:
    """The locomani modules importing `entries` loads in the sandbox (scripts/rsi and src on the path)."""
    code = "import sys; sys.path[:0] = ['scripts/rsi', 'src']\n" + "".join(f"import {e}\n" for e in entries)
    r = subprocess.run(
        [str(SIM_PY), "-X", "importtime", "-c", code], cwd=dest, env=SUB_ENV, capture_output=True, text=True
    )
    if r.returncode:
        raise SystemExit(f"closure import failed:\n{r.stderr[-2000:]}")
    names = {line.rsplit("|", 1)[-1].strip() for line in r.stderr.splitlines() if "|" in line}
    return {n for n in names if n == "locomani" or n.startswith("locomani.")}


def seed_skills(dest: Path, family: str, seed: str) -> None:
    """The skills package of the extracted archive replaced by the directory `seed` (repository-relative, committed at
    the exported ref), and the sandbox's family file told (`sandbox.seed`), so the run's record says where its package
    came from. `--seed` (user 2026-11-24: a run from scratch, not from what an earlier run wrote back) overrides the
    family file's own `sandbox.seed`; null there means the main library as it is (the pack families, 2026-11-05)."""
    skills = dest / "src" / "locomani" / "tasks" / "scene" / "skills"
    if not (dest / seed / "__init__.py").is_file():
        raise SystemExit(f"{seed}: not a skills package in the exported commit (no __init__.py)")
    shutil.rmtree(skills)
    shutil.copytree(dest / seed, skills, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    fam_file = dest / "scripts" / "rsi" / "families" / f"{family}.json"
    fam = json.loads(fam_file.read_text())
    if fam.get("sandbox", {}).get("seed") != seed:
        fam.setdefault("sandbox", {})["seed"] = seed
        fam_file.write_text(json.dumps(fam, indent=1) + "\n")


def apply_spec(dest: Path, family: str, seed: str | None = None) -> dict:
    """The family file's `sandbox` block (module docstring) applied to the extracted archive; returns the block.
    `seed` overrides the block's `seed` (seed_skills)."""
    spec = json.loads((dest / "scripts" / "rsi" / "families" / f"{family}.json").read_text()).get("sandbox") or {}
    if not spec:
        if seed:
            raise SystemExit(f"{family} has no `sandbox` block in its family file: --seed needs one")
        return spec
    skills = dest / "src" / "locomani" / "tasks" / "scene" / "skills"
    if seed or spec.get("seed"):
        seed_skills(dest, family, seed or spec["seed"])
        spec = dict(spec, seed=seed or spec["seed"])
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
    print(f"sandbox spec: seed {spec.get('seed') or 'the main library'}, {len(kept_py)} src modules kept")
    return spec


def default_strip(dest: Path) -> None:
    """DEFAULT_STRIP (module docstring) removed from every sandbox. TODO 36 (review 2026-10-22)."""
    for rel in DEFAULT_STRIP:
        t = dest / rel
        if t.is_dir() and not t.is_symlink():
            shutil.rmtree(t)
        elif t.exists() or t.is_symlink():
            t.unlink()


def hide_seeds(dest: Path, family: str) -> dict:
    """The acceptance and sealed seeds out of the sandbox's family file into the hidden seeds.json (module docstring);
    returns what was written. Run before the sandbox's git commit. TODO 36 (review 2026-10-22)."""
    f = dest / "scripts" / "rsi" / "families" / f"{family}.json"
    fam = json.loads(f.read_text())
    hidden = {k: v["seed"] for k, v in fam.get("pools", {}).items() if k != "diag" and v.get("seed") is not None}
    if hidden:
        out = dest / HIDDEN / family / "seeds.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(hidden, indent=1))
        for k in hidden:
            fam["pools"][k]["seed"] = None
        f.write_text(json.dumps(fam, indent=1, ensure_ascii=False) + "\n")
    return hidden


def library_without(dest: Path, family: str) -> None:
    """The sandbox's experience library without `family`'s entry (module docstring); nothing when it holds none. With a
    `sandbox` block in the family file: also no entry's final.patch nor the line pointing at it, and no entry of the
    file's `library_exclude`. TODO 36 (review 2026-10-22)."""
    lib = dest / "scripts" / "rsi" / "experience"
    if not lib.is_dir():
        return
    fam_file = dest / "scripts" / "rsi" / "families" / f"{family}.json"
    fam = json.loads(fam_file.read_text()) if fam_file.exists() else {}
    for name in [family, *fam.get("library_exclude", [])]:
        if (lib / name).exists():
            shutil.rmtree(lib / name)
    if fam.get("sandbox"):
        for d in sorted(p for p in lib.iterdir() if p.is_dir()):
            (d / "final.patch").unlink(missing_ok=True)
            md = d / "EXPERIENCE.md"
            if md.exists():
                md.write_text("".join(x for x in md.read_text().splitlines(keepends=True) if "final.patch" not in x))
    mod = importlib.util.spec_from_file_location("rsi_reflect", HERE / "reflect.py")
    reflect = importlib.util.module_from_spec(mod)
    mod.loader.exec_module(reflect)
    reflect.index(lib)


def _knowledge_module():
    mod = importlib.util.spec_from_file_location("rsi_knowledge", HERE / "knowledge.py")
    kn = importlib.util.module_from_spec(mod)
    mod.loader.exec_module(kn)
    return kn


def knowledge_without(dest: Path, family: str) -> None:
    """The sandbox's knowledge library (scripts/rsi/knowledge, plan docs/handoff/shared-knowledge-plan-20261111.md)
    without the entries built from `family` and from the families of the family file's `library_exclude` (by each
    entry's `source.family` in knowledge.jsonl); nothing when it holds none. KNOWLEDGE.md is written by the loop before
    each Reviewer (knowledge.write_hits)."""
    lib = dest / "scripts" / "rsi" / "knowledge"
    if not lib.is_dir():
        return
    fam_file = dest / "scripts" / "rsi" / "families" / f"{family}.json"
    fam = json.loads(fam_file.read_text()) if fam_file.exists() else {}
    drop = {family, *fam.get("library_exclude", [])}
    kn = _knowledge_module()
    kn.save_entries(lib, [e for e in kn.entries(lib) if kn.source_family(e) not in drop])
    for f in ("KNOWLEDGE.md", "HITS.md"):
        (lib / f).unlink(missing_ok=True)
    kn.index(lib)


def without_libraries(dest: Path) -> None:
    """--fresh: neither the knowledge library nor the experience library in the sandbox (user 2026-11-18)."""
    for rel in ("scripts/rsi/knowledge", "scripts/rsi/experience"):
        if (dest / rel).exists():
            shutil.rmtree(dest / rel)


def physics_target(repo: Path, scene: str) -> Path:
    """`<scene>_physics/` in the repository, the directory a sandbox links its scene's physics copy to (world.
    physics_scene writes it on the first round). Created empty when missing: a link to nothing makes the round's
    `mkdir(exist_ok=True)` raise FileExistsError -- the state right after a scene directory and its physics copy are
    deleted and rebuilt (2026-11-17, the hand-picked gso scenes' smoke check)."""
    target = repo / f"{scene}_physics"
    target.mkdir(parents=True, exist_ok=True)
    return target


def build(dest: Path, family: str, ref: str = "HEAD", fresh: bool = False, seed: str | None = None) -> None:
    """`ref`: the commit exported (replays launched over hours pin one, so later commits do not leak in). `fresh`:
    no knowledge and no experience library (without_libraries). `seed`: a skills package directory that replaces the
    family file's `sandbox.seed` (seed_skills)."""
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", ref], cwd=REPO, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    spec = apply_spec(dest, family, seed)
    default_strip(dest)
    if fresh:
        without_libraries(dest)
    library_without(dest, family)
    knowledge_without(dest, family)
    hide_seeds(dest, family)
    links = [(rel, REPO / rel) for rel in LINKS]
    if spec.get("scene_exclude") is not None:
        scene = json.loads((dest / "scripts" / "rsi" / "families" / f"{family}.json").read_text())["scene"]
        links = [x for x in links if x[0] != "outputs/scene_discovery"]
        links += [
            (f"{scene}/{e.name}", e) for e in sorted((REPO / scene).iterdir()) if e.name not in spec["scene_exclude"]
        ]
        links.append((f"{scene}_physics", physics_target(REPO, scene)))
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
    hidden = dest / HIDDEN / family
    fam = dest / "scripts" / "rsi" / "family.py"
    for pool, out in (("diag", loop / "pool_diag"), ("accept", hidden / "pool_accept")):
        r = subprocess.run(
            [str(SIM_PY), str(fam), "draw", family, pool, str(out)],
            cwd=dest,
            env=SUB_ENV,
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise SystemExit(r.stderr[-2000:])
        print(r.stdout.strip())
    lib = REPO / "outputs" / "scene_rsi_20261016" / "records" / "library.jsonl"  # TODO 33.15.1: the record library
    if lib.exists():
        (loop.parent / "records").mkdir(parents=True, exist_ok=True)
        shutil.copy(lib, loop.parent / "records" / lib.name)
    body = REPO / "outputs" / "scene_rsi_20261016" / "body"  # TODO 33.1.4: the body model's reach table (runtime data)
    if body.exists():
        (loop.parent / "body").mkdir(parents=True, exist_ok=True)
        for p in sorted(body.glob("*.npz")):
            shutil.copy(p, loop.parent / "body" / p.name)
    print(f"{dest}: sandbox for {family}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", type=Path)
    ap.add_argument("family")
    ap.add_argument("--fresh", action="store_true", help="no knowledge and no experience library in the sandbox")
    ap.add_argument("--seed", default=None, metavar="DIR", help="a skills package directory, repository-relative and "
                    "committed (e.g. scripts/rsi/seeds/gso_picked_start), that replaces the family file's sandbox.seed: "
                    "the run starts from it instead of the main library")  # fmt: skip
    a = ap.parse_args(argv)
    families = sorted(f.stem for f in (HERE / "families").glob("*.json"))
    if a.family not in families:  # before anything is written: a typo used to leave a half-extracted tree behind
        raise SystemExit(f"no family {a.family!r} in scripts/rsi/families; known: {', '.join(families)}")
    build(a.dest.resolve(), a.family, fresh=a.fresh, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
