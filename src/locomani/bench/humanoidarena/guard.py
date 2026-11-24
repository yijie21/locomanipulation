"""Static guard for engineer edits to the HumanoidArena skill package (TODO 49.5): the rules of the SIMPLE and LocoManiLab guards that
apply to a two-file package.

Compares a candidate package directory with a base directory (source text only) and refuses edits that reach for the simulator, the
episode's identity (its seed), files, the network, or that change locked code.

  R1  the files: only skill.py is editable, __init__.py is byte-identical, none is missing, nothing else is in the package
  R2  every editable file parses
  R3  locked definitions are unchanged (compared as syntax trees) -- none yet in this package
  R4  forbidden names: none as an identifier, attribute, argument, keyword or imported name
  R5  no new imports (beyond NEW_IMPORTS_OK)
  R6  no dynamic attribute access (getattr, vars, __dict__ ...)
  R7  a file grows by at most MAX_GROWTH lines
  R8  no import of the simulator side of the bed (client, server, run, session, episodes, privileged, truth, judge, record, ...):
      a package gets the facade (`env`), never the link or the Privileged

What a package may use: `env.K`, `env.task`, `env.instruction`, `env.t`, `env.me` (the robot's own readings and arm solutions),
`env.view.goal()`, `env.drive`, `env.mark`, `env.cameras`.

Usage:
    python -I guard.py <base_dir> <cand_dir>   -> JSON {ok, violations, files}, exit 0 if ok else 1
    python -I guard.py --selftest              -> "selftest ok" or the failures
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
import tempfile
from pathlib import Path

EDITABLE = ("skill.py",)
FROZEN = ("__init__.py",)
LOCKED: dict[str, tuple[str, ...]] = {}
FORBIDDEN_MODULES = tuple(
    f"locomani.bench.humanoidarena.{m}"
    for m in (
        "client",
        "server",
        "run",
        "session",
        "episodes",
        "config",
        "guard",
        "judge",
        "record",
        "privileged",
        "truth",
    )
) + ("locomani.perception.truth", "locomani.perception.services")
FORBIDDEN_ANY = (
    # the link, the episode's identity and the privileged state
    "_ep", "link", "ServerLink", "Episode", "ask", "priv", "Privileged", "HumanoidArenaPrivileged", "TruthView", "facts",
    "episode_id", "seed", "outcome", "contacts", "verdict", "objects", "obj_move",
    # the simulator
    "isaaclab", "isaacsim", "omni", "torch", "carb", "sim", "model", "data", "root_state_w", "body_pos_w", "contact", "sensor",
    # files, the environment, other processes, code from strings
    "os", "sys", "pathlib", "shutil", "pickle", "json", "ctypes", "threading", "multiprocessing", "environ", "getenv",
    "open", "load", "loadtxt", "genfromtxt", "fromfile", "memmap", "save", "savez", "read_text", "read_bytes", "write_text",
    "write_bytes", "subprocess", "socket", "requests", "urllib", "eval", "exec", "compile", "__import__", "importlib", "glob",
    "Path", "listdir", "system", "popen", "__file__",
)  # fmt: skip
FORBIDDEN_PREFIX = ("isaac", "omni", "carb")
NEW_IMPORTS_OK = ("numpy", "math", "dataclasses", "typing", "functools", "itertools", "collections", "heapq", "scipy")
MAX_GROWTH = 300
DYNAMIC = (
    "getattr", "setattr", "delattr", "vars", "globals", "locals", "__dict__", "__getattribute__", "__globals__",
    "__builtins__", "__class__", "__subclasses__", "__mro__", "__bases__", "__code__", "__closure__", "__self__",
    "__func__", "__module__",
)  # fmt: skip


def _hit(name: str) -> str | None:
    if name in FORBIDDEN_ANY:
        return name
    return next((p for p in FORBIDDEN_PREFIX if name.startswith(p)), None)


def forbidden_names(tree: ast.AST) -> list[str]:
    out = []
    for n in ast.walk(tree):
        found: list[str] = []
        if isinstance(n, ast.Name):
            found = [n.id]
        elif isinstance(n, ast.Attribute):
            found = [n.attr]
        elif isinstance(n, ast.alias):
            found = n.name.split(".") + (n.asname.split(".") if n.asname else [])
        elif isinstance(n, ast.arg):
            found = [n.arg]
        elif isinstance(n, ast.keyword) and n.arg:
            found = [n.arg]
        out += [h for h in (_hit(x) for x in found) if h]
    return out


def absolute_imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def count_dynamic(tree: ast.AST) -> int:
    c = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            c += (isinstance(f, ast.Name) and f.id in DYNAMIC) or (isinstance(f, ast.Attribute) and f.attr in DYNAMIC)
        elif isinstance(n, ast.Attribute) and n.attr in DYNAMIC:
            c += 1
        elif isinstance(n, ast.Name) and n.id in DYNAMIC and not isinstance(n.ctx, ast.Load):
            c += 1
    return c


def _locked_dump(tree: ast.Module, names: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for n in tree.body:
        if isinstance(n, ast.ClassDef | ast.FunctionDef) and n.name in names:
            out[n.name] = ast.dump(n)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = ast.dump(n)
    return out


def check(base: Path, cand: Path) -> dict:
    base, cand = Path(base), Path(cand)
    violations: list[str] = []
    files = sorted(p.name for p in cand.iterdir() if p.is_file() and p.suffix == ".py")
    extra = [f for f in files if f not in EDITABLE + FROZEN]
    if extra:
        violations.append(f"R1 files not allowed: {extra}")
    for f in EDITABLE + FROZEN:
        if not (cand / f).is_file():
            violations.append(f"R1 {f} is missing")
    for f in FROZEN:
        if (cand / f).is_file() and (base / f).is_file() and (cand / f).read_bytes() != (base / f).read_bytes():
            violations.append(f"R1 {f} is frozen and differs from the base")
    for f in EDITABLE:
        if not ((cand / f).is_file() and (base / f).is_file()):
            continue
        try:
            ctree = ast.parse((cand / f).read_text())
        except SyntaxError as e:
            violations.append(f"R2 {f} does not parse: {e}")
            continue
        btree = ast.parse((base / f).read_text())
        for name, dump in _locked_dump(btree, LOCKED.get(f, ())).items():
            if _locked_dump(ctree, (name,)).get(name) != dump:
                violations.append(f"R3 {f}: locked definition {name} changed")
        bad = sorted(set(forbidden_names(ctree)))
        if bad:
            violations.append(f"R4 {f}: forbidden names {bad}")
        new = sorted(
            m for m in absolute_imports(ctree) - absolute_imports(btree) if m.split(".")[0] not in NEW_IMPORTS_OK
        )
        if new:
            violations.append(f"R5 {f}: new imports {new}")
        sim = sorted(
            m for m in absolute_imports(ctree) if any(m == x or m.startswith(x + ".") for x in FORBIDDEN_MODULES)
        )
        if sim:
            violations.append(f"R8 {f}: imports the simulator side of the bed: {sim}")
        if count_dynamic(ctree) > count_dynamic(btree):
            violations.append(f"R6 {f}: more dynamic attribute access than the base")
        growth = len((cand / f).read_text().splitlines()) - len((base / f).read_text().splitlines())
        if growth > MAX_GROWTH:
            violations.append(f"R7 {f}: grew by {growth} lines (limit {MAX_GROWTH})")
    return dict(ok=not violations, violations=violations, files=files)


def selftest() -> str:
    seed = Path(__file__).resolve().parent / "seed"
    fails = []
    if not check(seed, seed)["ok"]:
        fails.append(f"the seed package fails its own guard: {check(seed, seed)['violations']}")
    cases = {
        "R4 seed": "x = env.seed\n",
        "R4 link": "y = env._ep.link\n",
        "R4 isaaclab": "import isaaclab\n",
        "R5 import": "import scipy.sparse\nimport requests_toolbelt\n",
        "R6 getattr": "z = getattr(env, 'me')\n",
        "R8 client": "from locomani.bench.humanoidarena import client\n",
    }
    for name, line in cases.items():
        with tempfile.TemporaryDirectory() as t:
            shutil.copytree(seed, t, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
            (Path(t) / "skill.py").write_text((seed / "skill.py").read_text() + "\n" + line)
            res = check(seed, Path(t))
            if res["ok"] or not any(v.startswith(name.split()[0]) for v in res["violations"]):
                fails.append(f"{name}: not refused ({res['violations']})")
    return "selftest ok" if not fails else "\n".join(fails)


def main(argv: list[str]) -> int:
    if argv == ["--selftest"]:
        out = selftest()
        print(out)
        return 0 if out == "selftest ok" else 1
    if len(argv) != 2:
        print(__doc__)
        return 2
    res = check(Path(argv[0]), Path(argv[1]))
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
