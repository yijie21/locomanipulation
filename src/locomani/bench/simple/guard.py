"""Static guard for engineer edits to the SIMPLE skill package (moved from escalation_20261116/guard_simple.py, TODO 49.2).
The package is now one folder for all nine tasks (seed/): skill.py, pick.py, camera_view.py, handover.py editable, __init__.py
frozen. The simulator-side classes (TruthView, Belief, the odometry, the services) are fixed modules of locomani and are not in
the package; a draft may not import the bed's run / client / server / episodes / session / feed / config / guard modules or
perception's truth and services (rule R8): it gets a View, a body and the env facade, and must not reach the simulator.
Stdlib only (the loop runs it as `python -I <this file>`).

Compares a candidate directory with a base directory (source text only) and
refuses edits that reach for simulator truth, episode identity, files, the
network, or that change locked code.

Usage:
    python -I guard.py <base_dir> <cand_dir>   -> JSON, exit 0 if ok else 1
    python -I guard.py --selftest              -> "selftest ok" or failures
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Files the engineer may edit; nothing else may appear in the candidate directory.
EDITABLE = ("skill.py", "handover.py", "pick.py", "camera_view.py")
# The package's protocol file (parameters, apply_set): not part of the skill, must stay byte-identical.
FROZEN = ("__init__.py",)
# Top-level classes, functions or assigned names whose source must not change.
# (TruthView, Belief and body_verts moved out of the package into fixed modules; the entry keeps guarding a base that still has them.)
LOCKED = {"skill.py": ("TruthView", "Belief", "body_verts", "LIB")}
# Modules (dotted prefixes) a package may not import, whatever its base imports: the simulator side of the bed.
FORBIDDEN_MODULES = (
    "locomani.bench.simple.run",
    "locomani.bench.simple.client",
    "locomani.bench.simple.server",
    "locomani.bench.simple.episodes",
    "locomani.bench.simple.session",
    "locomani.bench.simple.feed",
    "locomani.bench.simple.config",
    "locomani.bench.simple.guard",
    "locomani.bench.simple.judge",
    "locomani.perception.truth",
    "locomani.perception.services",
    "locomani.perception.odometry",
    "locomani.perception.grasps",
)
# Identifiers that may not appear more often than in the base file.
FORBIDDEN = (
    "d", "xpos", "xquat", "xmat", "qpos", "qvel", "cvel", "contact", "ncon",
    "geom_xpos", "site_xpos", "truth", "_truth", "TruthView", "Belief", "body_verts",
    "mujoco", "episode_id", "episode", "seed", "EXAM", "EXAM_OPEN",
    "environ", "getenv", "open", "load", "subprocess", "socket",
    "requests", "urllib", "eval", "exec", "compile", "__import__",
    "importlib", "glob", "Path", "listdir", "walk", "system", "popen",
)  # fmt: skip
# Any identifier starting with one of these is counted under the prefix.
FORBIDDEN_PREFIX = ("mj_", "mju_")
# Top-level modules a candidate may import in addition to the base's imports.
NEW_IMPORTS_OK = ("numpy", "math", "dataclasses", "typing", "functools", "itertools", "collections")
# Number of lines a file may grow by.
MAX_GROWTH = 400
# Calls that reach attributes by string and so get round the name check.
DYNAMIC_CALLS = ("getattr", "setattr", "vars", "globals", "locals")


def _key(name: str) -> str | None:
    """Return the FORBIDDEN key a name counts under, or None."""
    for p in FORBIDDEN_PREFIX:
        if name.startswith(p):
            return p
    return name if name in FORBIDDEN else None


def count_forbidden(tree: ast.AST) -> Counter:
    """Count forbidden identifiers in the AST (never strings or comments)."""
    names: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.append(n.id)
        elif isinstance(n, ast.Attribute):
            names.append(n.attr)
        elif isinstance(n, ast.alias):
            names += n.name.split(".")
            if n.asname:
                names += n.asname.split(".")
        elif isinstance(n, ast.arg):
            names.append(n.arg)
        elif isinstance(n, ast.keyword) and n.arg:
            names.append(n.arg)
    return Counter(k for k in map(_key, names) if k)


def imported_modules(tree: ast.AST) -> set[str]:
    """Top-level module names imported anywhere in the tree."""
    mods: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
    return mods


def absolute_imports(tree: ast.AST) -> set[str]:
    """Every dotted module path an absolute import names: `import a.b` -> a.b; `from a.b import c` -> a.b and a.b.c."""
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def count_dynamic(tree: ast.AST) -> int:
    """Count getattr-style calls and __dict__ accesses."""
    c = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id in DYNAMIC_CALLS) or (
                isinstance(f, ast.Attribute) and f.attr in DYNAMIC_CALLS
            ):
                c += 1
        elif isinstance(n, ast.Attribute) and n.attr == "__dict__":
            c += 1
        elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "__dict__":
            c += 1
    return c


def top_level_defs(tree: ast.Module) -> dict[str, list[str]]:
    """Map each top-level defined name to the ast.dump of its definitions."""
    out: dict[str, list[str]] = {}
    for n in tree.body:
        names: list[str] = []
        if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names = [n.name]
        elif isinstance(n, ast.Assign):
            names = [t.id for tg in n.targets for t in ast.walk(tg) if isinstance(t, ast.Name)]
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names = [n.target.id]
        for name in names:
            out.setdefault(name, []).append(ast.dump(n))
    return out


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return None


def _lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines()) if path.is_file() else 0


def _listing(d: Path) -> set[str]:
    """Names in a directory, ignoring bytecode caches."""
    return {
        p.name for p in d.iterdir() if p.name != "__pycache__" and p.suffix != ".pyc" and not p.name.startswith(".")
    }


def check(base: Path, cand: Path) -> dict:
    """Run rules R1 to R8 and return the report dictionary."""
    viol: list[dict] = []

    def add(rule: str, file: str, detail: str) -> None:
        viol.append({"rule": rule, "file": file, "detail": detail})

    bnames, cnames = _listing(base), _listing(cand)
    for f in sorted(cnames - set(EDITABLE) - set(FROZEN)):
        add("R1", f, "file is not editable")
    for f in FROZEN:  # on both sides and identical, or on neither
        if (f in bnames) != (f in cnames) or (f in bnames and (base / f).read_bytes() != (cand / f).read_bytes()):
            add("R1", f, "frozen file was added, removed or changed")
    for f in EDITABLE:
        if f in bnames and f not in cnames:
            add("R1", f, "file present in base is missing from candidate")

    files: dict[str, dict] = {}
    for f in EDITABLE:
        bp, cp = base / f, cand / f
        if not cp.is_file():
            continue
        # A file the candidate adds (a new flow, for example the hand-over) is checked against an empty base:
        # it may hold no forbidden name at all and is bounded by MAX_GROWTH.
        new = not bp.is_file()
        lb, lc = (0 if new else _lines(bp)), _lines(cp)
        files[f] = {
            "changed": new or bp.read_bytes() != cp.read_bytes(),
            "lines_base": lb,
            "lines_cand": lc,
        }
        if lc - lb > MAX_GROWTH:
            add("R7", f, f"grew by {lc - lb} lines (max {MAX_GROWTH})")
        ct = _parse(cp)
        if ct is None:
            add("R2", f, "candidate does not parse")
            continue
        bt = ast.parse("") if new else _parse(bp)
        if bt is None:
            add("R2", f, "base does not parse; R3 to R6 skipped")
            continue
        bd, cd = top_level_defs(bt), top_level_defs(ct)
        for name in LOCKED.get(f, ()):
            if name not in bd:
                continue
            if name not in cd:
                add("R3", f, f"locked name {name} is missing")
            elif bd[name] != cd[name]:
                add("R3", f, f"locked name {name} was changed")
        bc, cc = count_forbidden(bt), count_forbidden(ct)
        for k in sorted(cc):
            if cc[k] > bc[k]:
                add("R4", f, f"forbidden name {k!r}: {bc[k]} in base, {cc[k]} in candidate")
        extra = imported_modules(ct) - imported_modules(bt) - set(NEW_IMPORTS_OK) - {Path(e).stem for e in EDITABLE}
        if extra:
            add("R5", f, f"new imports: {sorted(extra)}")
        for mod in sorted(absolute_imports(ct)):
            if any(mod == p or mod.startswith(p + ".") for p in FORBIDDEN_MODULES):
                add("R8", f, f"imports {mod}: the simulator side of the bed")
        db, dc = count_dynamic(bt), count_dynamic(ct)
        if dc > db:
            add("R6", f, f"dynamic access count {db} in base, {dc} in candidate")
    return {"ok": not viol, "violations": viol, "files": files}


# ---------------------------------------------------------------- self-test

_BASE = """\
import numpy as np

class TruthView:
    scale = 1.0
    def get(self):
        return 1.0

LIB = {"a": 1}

def run(sim):
    x = sim.d
    return np.zeros(3)
"""


def _rules(base: dict[str, str], cand: dict[str, str]) -> set[str]:
    """Run the guard on synthetic directories and return the violated rules."""
    with tempfile.TemporaryDirectory() as t:
        b, c = Path(t) / "b", Path(t) / "c"
        for d, src in ((b, base), (c, cand)):
            d.mkdir()
            for name, text in src.items():
                (d / name).write_text(text)
        return {v["rule"] for v in check(b, c)["violations"]}


def selftest() -> list[str]:
    """Return one line per failing test."""
    S = "skill.py"
    base = {S: _BASE}
    grow = _BASE + "\n" * (MAX_GROWTH + 5)
    cases: list[tuple[str, dict, dict, set[str]]] = [
        ("identical copy passes", base, {S: _BASE}, set()),
        ("R1 extra file fails", base, {S: _BASE, "evil.py": "x = 1\n"}, {"R1"}),
        ("R1 pycache ignored", base, {S: _BASE}, set()),
        ("R1 dot files ignored", base, {S: _BASE, ".note": "x\n"}, set()),
        ("R1 missing file fails", base, {}, {"R1"}),
        (
            "R1 new editable file passes when clean",
            base,
            {S: _BASE, "pick.py": "import numpy as np\nfrom . import skill\nx = 1\n"},
            set(),
        ),
        (
            "R4 new editable file reading sim.d fails",
            base,
            {S: _BASE, "pick.py": "def f(sim):\n    return sim.d.qpos\n"},
            {"R4"},
        ),
        ("R5 new editable file importing os fails", base, {S: _BASE, "pick.py": "import os\n"}, {"R5"}),
        (
            "R1 frozen file identical passes",
            {S: _BASE, "__init__.py": "a = 1\n"},
            {S: _BASE, "__init__.py": "a = 1\n"},
            set(),
        ),
        (
            "R1 frozen file changed fails",
            {S: _BASE, "__init__.py": "a = 1\n"},
            {S: _BASE, "__init__.py": "a = 2\n"},
            {"R1"},
        ),
        ("R1 frozen file added fails", base, {S: _BASE, "__init__.py": "a = 1\n"}, {"R1"}),
        ("R2 syntax error fails", base, {S: "def (:\n"}, {"R2"}),
        ("R2 valid edit passes", base, {S: _BASE + "\ny = 2\n"}, set()),
        ("R3 reformat+comments pass", base, {S: _BASE.replace("scale = 1.0", "# note\n    scale  =  1.0  # c")}, set()),
        ("R3 number change fails", base, {S: _BASE.replace("1.0", "2.0")}, {"R3"}),
        ("R3 locked removed fails", base, {S: _BASE.replace('LIB = {"a": 1}', "")}, {"R3"}),
        ("R3 locked assign change fails", base, {S: _BASE.replace('"a": 1', '"a": 2')}, {"R3"}),
        ("R4 add sim.d.qpos fails", base, {S: _BASE + "\ndef g(sim):\n    return sim.d.qpos\n"}, {"R4"}),
        (
            "R4 move sim.d passes",
            base,
            {S: _BASE.replace("    x = sim.d\n", "\n    x = (\n        sim.d\n    )\n")},
            set(),
        ),
        ("R4 string and comment not counted", base, {S: _BASE + '\nz = "qpos"  # qpos\n'}, set()),
        ("R4 prefix counted", base, {S: _BASE + "\ndef g(m):\n    return m.mj_step\n"}, {"R4"}),
        ("R4 keyword arg counted", base, {S: _BASE + "\nw = f(seed=1)\n"}, {"R4"}),
        ("R5 import os fails", base, {S: "import os\n" + _BASE}, {"R5"}),
        ("R5 import math passes", base, {S: "import math\n" + _BASE}, set()),
        ("R5 import inside function fails", base, {S: _BASE + "\ndef g():\n    import json\n"}, {"R5"}),
        ("R6 getattr fails", base, {S: _BASE + '\nq = getattr(sim, "d")\n'}, {"R6"}),
        ("R6 __dict__ fails", base, {S: _BASE + "\nq = o.__dict__\n"}, {"R6"}),
        ("R6 plain call passes", base, {S: _BASE + "\nq = len([1])\n"}, set()),
        ("R7 big growth fails", base, {S: grow}, {"R7"}),
        (
            "R8 import of the bed's run fails",
            base,
            {S: "from locomani.bench.simple import run\n" + _BASE},
            {"R5", "R8"},
        ),
        (
            "R8 import of client by dotted path fails",
            base,
            {S: "import locomani.bench.simple.client\n" + _BASE},
            {"R5", "R8"},
        ),
        (
            "R8 from server import fails",
            base,
            {S: "from locomani.bench.simple.server import x\n" + _BASE},
            {"R5", "R8"},
        ),
        ("R8 episodes fails", base, {S: "from locomani.bench.simple import episodes as E\n" + _BASE}, {"R5", "R8"}),
        (
            "R8 perception truth fails",
            base,
            {S: "from locomani.perception.truth import TruthView as T\n" + _BASE},
            {"R4", "R5", "R8"},
        ),
        (
            "R8 body and head pass",
            {S: "from locomani.bench.simple import body\n" + _BASE},
            {S: "from locomani.bench.simple import body\nfrom locomani.perception import head\n" + _BASE},
            set(),
        ),
        ("R4 Belief name fails", base, {S: _BASE + "\nb = Belief\n"}, {"R4"}),
        ("R7 small growth passes", base, {S: _BASE + "\n" * 50}, set()),
    ]
    bad = []
    for name, b, c, want in cases:
        got = _rules(b, c)
        if got != want:
            bad.append(f"FAIL {name}: expected {sorted(want)}, got {sorted(got)}")
    return bad


def main(argv: list[str]) -> int:
    """Command line entry point."""
    if argv == ["--selftest"]:
        bad = selftest()
        print("\n".join(bad) if bad else "selftest ok")
        return 1 if bad else 0
    if len(argv) != 2:
        print(__doc__)
        return 2
    report = check(Path(argv[0]), Path(argv[1]))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
