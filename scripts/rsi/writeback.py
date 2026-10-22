"""TODO 27.10.8 (plan docs/handoff/scene-rsi-plan-20261016.md §12.6): write a finished family's final incumbent back into
the skill catalog, as write-backs 1-7 did by hand. The family's changes are its final incumbent's skills against the
seed it started from (the gate's `incumbent_seed`); they merge three-way into the catalog as it is now (`git
merge-file`), so a family that ran beside another keeps the other's write-back. The final incumbent's parameter
overrides (`module.CONSTANT`, or `module.DICT.key` for an entry of a dict literal) are then written into the
constants; a package-level name without a module is a conflict. Conflicts -- a file both changed in the same
place, a constant not found, a file the family deleted that the catalog changed -- are listed and nothing is
written; so is a family whose final is its seed.

    python3 scripts/rsi/writeback.py <sandbox> <family> [--catalog <skills dir>] [--dry-run]

Writes `<sandbox>/outputs/scene_rsi_20261016_hidden/<family>/writeback.json` (files merged, copied, deleted; constants
folded; conflicts). The regression of every family's sealed pool under the new catalog (regress.py) comes next.
"""

from __future__ import annotations

import argparse
import ast
import filecmp
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CATALOG = REPO / "src" / "locomani" / "tasks" / "scene" / "skills"
SKIP = re.compile(r"(__pycache__|\.pyc$)")
RUFF = Path("/venv/main/bin/ruff")


def files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not SKIP.search(str(p))}


def fold(src: str, const: str, value, family: str) -> tuple[str, bool]:
    """`CONST = <old>  # comment` -> `CONST = <value>  # (<old> until <family>'s set) comment` at module level, as the
    hand write-backs recorded it; (text, found)."""
    pat = re.compile(rf"^({re.escape(const)}\s*(?::[^=\n]+)?=\s*)([^#\n]+?)(\s*)(?:(#\s?)(.*))?$", re.M)
    m = pat.search(src)
    if m is None:
        return src, False
    old = m.group(2).strip()
    try:
        ast.literal_eval(old)
    except (ValueError, SyntaxError):  # an expression, not a literal: leave it to a person
        return src, False
    note = f"# ({old} until {family}'s set) {m.group(5) or ''}".rstrip()
    return src[: m.start(2)] + f"{value!r}  {note}" + src[m.end() :], True


def fold_entry(src: str, const: str, path: list[str], value, family: str) -> tuple[str, bool]:
    """TODO 36 (review 2026-10-22): `CONST = {..., "key": <old>, ...}` -> the entry `path` (one or more keys down nested
    dict literals) set to `value`, with `# (<old> until <family>'s set)` on the entry's last line (skills.apply_set's
    `module.CONST.key`); (text, found). Found only for a module-level dict literal whose entry is a literal."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, False
    node = None
    for st in tree.body:
        if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == const for t in st.targets):
            node = st.value
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == const:
            node = st.value
    for key in path:
        if not isinstance(node, ast.Dict):
            return src, False
        hit = [v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant) and k.value == key]
        node = hit[-1] if hit else None
    if node is None or isinstance(node, ast.Dict):
        return src, False
    try:
        old = repr(ast.literal_eval(node))
    except ValueError:
        return src, False
    lines = src.splitlines(keepends=True)
    first, last = node.lineno - 1, node.end_lineno - 1
    # ast columns are UTF-8 byte offsets
    head = lines[first].encode()[: node.col_offset].decode()
    tail = lines[last].encode()[node.end_col_offset :].decode()
    note = f"(was {old} until {family}'s set)"
    if "#" in tail:  # keep the line's own comment after ours
        i = tail.index("#")
        tail = tail[:i] + f"# {note} " + tail[i + 1 :].lstrip()
    else:
        tail = tail.rstrip("\n").rstrip() + f"  # {note}\n"
    lines[first : last + 1] = [head + repr(value) + tail]
    return "".join(lines), True


def plan(sandbox: Path, family: str, catalog: Path) -> dict:
    gate = sandbox / "outputs" / "scene_rsi_20261016_hidden" / family / "gate"
    inc = json.loads((gate / "incumbent.json").read_text())
    final, seed = Path(inc["dir"]), gate / "incumbent_seed"
    overrides = json.loads((final / "candidate.json").read_text()).get("set") or {}
    out = dict(family=family, final=str(final), seed=str(seed), merged=[], copied=[], deleted=[], folded={},
               conflicts=[], unchanged=final == seed)  # fmt: skip
    if final == seed:
        return out
    f_skills, s_skills = final / "skills", seed / "skills"
    texts: dict[str, str | None] = {}
    for rel in sorted(files(f_skills) | files(s_skills)):
        f, s, c = f_skills / rel, s_skills / rel, catalog / rel
        if f.exists() and s.exists() and filecmp.cmp(f, s, shallow=False):
            continue
        if not s.exists():  # the family added it
            if c.exists() and not filecmp.cmp(f, c, shallow=False):
                out["conflicts"].append(f"{rel}: added by the family and, differently, in the catalog")
            else:
                texts[rel] = f.read_text()
                out["copied"].append(rel)
        elif not f.exists():  # the family deleted it
            if c.exists() and not filecmp.cmp(s, c, shallow=False):
                out["conflicts"].append(f"{rel}: deleted by the family, changed in the catalog since")
            else:
                texts[rel] = None
                out["deleted"].append(rel)
        elif not c.exists():
            out["conflicts"].append(f"{rel}: changed by the family, gone from the catalog")
        else:
            with tempfile.TemporaryDirectory() as d:
                ours = Path(d) / "ours"
                ours.write_bytes(c.read_bytes())
                r = subprocess.run(["git", "merge-file", "-p", "-L", "catalog", "-L", "seed", "-L", family, str(ours),
                                    str(s), str(f)], capture_output=True, text=True)  # fmt: skip
            if r.returncode:
                out["conflicts"].append(f"{rel}: {r.returncode} conflicting hunk(s) with the catalog")
            else:
                texts[rel] = r.stdout
                out["merged"].append(rel)
    for key, value in sorted(overrides.items()):
        parts = key.split(".")
        if len(parts) < 2:  # a package-level NAME (skills/__init__.py): no module to fold into
            out["conflicts"].append(f"override {key}: a package-level parameter has no module; fold it by hand")
            continue
        # module.CONST, or module.CONST.key[.key...] (an entry of a dict literal, skills.apply_set); the module is the
        # longest prefix that is a file
        for i in range(len(parts) - 1, 0, -1):
            rel = "/".join(parts[:i]) + ".py"
            if rel in texts or (catalog / rel).exists():
                break
        else:
            rel, i = "/".join(parts[:-1]) + ".py", len(parts) - 1
        const, entry = parts[i], parts[i + 1 :]
        src = texts.get(rel) if rel in texts else (catalog / rel).read_text() if (catalog / rel).exists() else None
        if src is None:
            new, found = None, False
        elif entry:
            new, found = fold_entry(src, const, entry, value, family)
        else:
            new, found = fold(src, const, value, family)
        if not found:
            what = (
                f"no literal dict entry {'.'.join([const, *entry])}" if entry else f"no literal module constant {const}"
            )
            out["conflicts"].append(f"override {key}: {what} in {rel}")
            continue
        texts[rel] = new
        out["folded"][key] = value
    out["texts"] = texts
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sandbox", type=Path)
    ap.add_argument("family")
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    p = plan(a.sandbox.resolve(), a.family, a.catalog.resolve())
    texts = p.pop("texts", {})
    if not p["conflicts"] and not a.dry_run:
        for rel, text in texts.items():
            dst = a.catalog / rel
            if text is None:
                dst.unlink()
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(text)
    p["written"] = bool(texts) and not p["conflicts"] and not a.dry_run
    written = [str(a.catalog / rel) for rel, t in texts.items() if t is not None]
    if p["written"] and written and RUFF.exists():  # the repository's formatter, as the commit hook checks it
        subprocess.run([str(RUFF), "format", "-q", *written], check=False)
    out = a.sandbox / "outputs" / "scene_rsi_20261016_hidden" / a.family / "writeback.json"
    out.write_text(json.dumps(p, indent=1, default=str))
    print(json.dumps({k: v for k, v in p.items() if k not in ("final", "seed")}, indent=1, default=str))
    return 1 if p["conflicts"] else 0


if __name__ == "__main__":
    sys.exit(main())
