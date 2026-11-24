"""Static guard for engineer edits to the LocoManiLab camera skill package (moved from guard_lml.py, TODO 49.4: the seven rules of
guard_simple.py, the three beds' lists of forbidden names in one file, and rule R8, the forbidden imports).

Compares a candidate package directory with a base directory (source text only) and refuses edits that reach for
simulator truth, the episode's identity, files, the network, or that change locked code.

  R1  the files: only the editable ones may be in the package, the frozen one is byte-identical, none is missing
  R2  every editable file parses
  R3  locked definitions are unchanged (compared as syntax trees: formatting and comments may change)
  R4  forbidden names: none outside the locked definitions, in the files the seed was written without them
      (flows.py, camera_view.py); in the two files carried over from SIMPLE (skill.py, pick.py), which read some of
      them (frame bridge, the measured palm for the record), no more than the base has, counted per top-level definition
  R5  no new imports (a locked definition's own imports do not count as the file's)
  R6  no dynamic attribute access (getattr, vars, __dict__ ...): none in the first files, no more per definition
      than the base in the other two
  R7  a file grows by at most MAX_GROWTH lines
  R8  no import of the simulator side of the bed (the bed's run / client / server / episodes / session / record / truth ...,
      perception's truth, services, odometry ...): a package gets the episode facade, a view and a Belief, never the simulator

Why a count that may not grow is not enough for the first files: an edit that keeps the count can turn a read made
for the record into one a decision uses. The simulator's truth is therefore taken out of those files altogether (the
bed's recorder reads it); what the package needs of it comes through narrow accessors: `env.hand` (finger commands and
readings), the head camera's image through the Belief, the cloud name the view is given. `sim` itself is not a name a package
may use: the facade (client.PackageEnv) does not hold it.

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
from collections import Counter
from pathlib import Path

EDITABLE = ("skill.py", "pick.py", "flows.py", "camera_view.py")
FROZEN = ("__init__.py",)
# Files carried over from SIMPLE: they read a few of the forbidden names; R4 and R6 compare them with the base.
CARRIED = ("skill.py", "pick.py")
# Top-level classes, functions or assigned names whose source must not change (these are also left out of the counts):
# the frame bridge and the robot's own kinematics, the segmentation client, the grasp-file loader, the cloud name.
LOCKED = {
    "pick.py": ("T", "cont"),
    "camera_view.py": ("cloud_name", "have_grasps", "load_frame", "load_grasps"),
}  # fmt: skip
# Modules (dotted prefixes) a package may not import, whatever its base imports: the simulator side of the bed.
FORBIDDEN_MODULES = (
    "locomani.bench.locomanilab.run",
    "locomani.bench.locomanilab.client",
    "locomani.bench.locomanilab.server",
    "locomani.bench.locomanilab.episodes",
    "locomani.bench.locomanilab.session",
    "locomani.bench.locomanilab.config",
    "locomani.bench.locomanilab.guard",
    "locomani.bench.locomanilab.comply",
    "locomani.bench.locomanilab.loss",
    "locomani.bench.locomanilab.record",
    "locomani.bench.locomanilab.truth",
    "locomani.bench.locomanilab.scene",
    "locomani.bench.locomanilab.body",
    "locomani.bench.locomanilab.families",
    "locomani.bench.simple.run",
    "locomani.bench.simple.client",
    "locomani.bench.simple.server",
    "locomani.bench.simple.episodes",
    "locomani.bench.simple.session",
    "locomani.bench.simple.feed",
    "locomani.bench.simple.config",
    "locomani.bench.simple.guard",
    "locomani.bench.simple.judge",
    "locomani.bench.simple.privileged",
    "locomani.perception.truth",
    "locomani.perception.rgbd",
    "locomani.perception.noise",
    "locomani.perception.services",
    "locomani.perception.odometry",
    "locomani.perception.grasps",
)
# Names that may not appear as any identifier, attribute, argument, keyword or imported name.
FORBIDDEN_ANY = (
    # the task's definition and the check's target
    "spec", "spec_of", "specs", "bodies", "snapshot", "marks",
    # (fork, 2026-11-21) the door tasks' truth: the joint, its address, the fixture and the handle point the check reads
    "target_joint", "target_joint_pos", "target_fixture", "target_face_pos", "qpos_adr",
    # (bed3) the drawer's joint state, range and ends, and what the recorder keeps of it
    "joint_range", "closed_qpos", "open_qpos", "joint_type", "hand_drives", "moved_by", "pelvis_drop_cm",
    # the truth view, the receptacle from the task's region, bodies' points
    "TruthView", "LmlTruthView", "truth", "_truth", "receptacle", "body_points", "subtree", "body_verts",
    # the frame bridge to the simulator and the true pelvis
    "to_sim", "to_sim_R", "from_sim", "_pelvis_sim", "jaw_centre",
    # the server: the camera reply with the true pose, the reset reply, the socket
    "eyes", "ask", "LmlSim", "ServerLink", "Recorder", "ArtRecorder",
    # the episode's identity and seed
    "episode_id", "seed", "EXAM", "EXAM_OPEN",
    # files, the environment, other processes, code from strings
    "os", "sys", "pathlib", "shutil", "pickle", "json", "ctypes", "threading", "multiprocessing",
    "environ", "getenv", "open", "load", "loadtxt", "genfromtxt", "fromfile", "memmap", "save", "savez",
    "read_text", "read_bytes", "write_text", "write_bytes", "subprocess", "socket", "requests", "urllib",
    "eval", "exec", "compile", "__import__", "importlib", "glob", "Path", "listdir", "system", "popen", "__file__",
    "mujoco",
)  # fmt: skip
# Names that may not appear as an attribute (a local variable of that name is no harm).
FORBIDDEN_ATTR = (
    "d", "m", "xpos", "xquat", "xmat", "qpos", "qvel", "qacc", "cvel", "contact", "ncon", "geom_xpos", "geom_xmat",
    "site_xpos", "cam_xpos", "cam_xmat", "gq", "qadr", "vadr",
    "first", "link", "over", "success", "truncated", "task", "info", "pt", "yt", "feed", "body", "rec",
    "sim", "_sim", "_env", "_clock", "_view", "_on_grasp",  # the simulator and the facade's insides
)  # fmt: skip
FORBIDDEN_PREFIX = ("mj_", "mju_")
# (fork) Pieces of the fridge door's and the low drawer's own names: forbidden in any identifier and in any string a package holds
# (the bodies "refrigerator_<id>_1_6_0" (door) and "..._1_7_0" (handle) are in the scene's file).
FORBIDDEN_PIECES = (
    "refrigerator_",
    "_1_6_0",
    "_1_7_0",
    "chestofdrawers",
    "DrawerMesh",
    "Receptacle0",
    "Dresser_209",
    "FloorPlan209",
)  # fmt: skip  (bed3: the lower-left drawer's joint / body / handle / receptacle-site / fixture names)
NEW_IMPORTS_OK = ("numpy", "math", "dataclasses", "typing", "functools", "itertools", "collections", "heapq", "scipy")
MAX_GROWTH = 400
DYNAMIC = (
    "getattr", "setattr", "delattr", "vars", "globals", "locals", "__dict__", "__getattribute__", "__globals__",
    "__builtins__", "__class__", "__subclasses__", "__mro__", "__bases__", "__code__", "__closure__", "__self__",
    "__func__", "__module__",
)  # fmt: skip


def _hit(name: str, attr: bool) -> str | None:
    """The forbidden key a name counts under, or None."""
    if name in FORBIDDEN_ANY or (attr and name in FORBIDDEN_ATTR):
        return name
    piece = next((p for p in FORBIDDEN_PIECES if p in name), None)
    if piece:
        return piece
    return next((p for p in FORBIDDEN_PREFIX if name.startswith(p)), None)


def count_forbidden(tree: ast.AST) -> Counter:
    """Count forbidden identifiers in the AST (never strings or comments)."""
    out: Counter = Counter()
    for n in ast.walk(tree):
        found: list[tuple[str, bool]] = []
        if isinstance(n, ast.Name):
            found = [(n.id, False)]
        elif isinstance(n, ast.Attribute):
            found = [(n.attr, True)]
        elif isinstance(n, ast.alias):
            found = [(x, False) for x in n.name.split(".") + (n.asname.split(".") if n.asname else [])]
        elif isinstance(n, ast.arg):
            found = [(n.arg, False)]
        elif isinstance(n, ast.keyword) and n.arg:
            found = [(n.arg, False)]
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):  # (fork) a string naming the door's bodies
            out.update(p for p in FORBIDDEN_PIECES if p in n.value)
        out.update(k for k in (_hit(x, a) for x, a in found) if k)
    return out


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
    """Count getattr-style calls and dunder-attribute accesses."""
    c = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            c += (isinstance(f, ast.Name) and f.id in DYNAMIC) or (isinstance(f, ast.Attribute) and f.attr in DYNAMIC)
        elif isinstance(n, ast.Attribute) and n.attr in DYNAMIC:
            c += 1
        elif isinstance(n, ast.Name) and n.id in DYNAMIC and not isinstance(n.ctx, ast.Load):
            c += 1
        elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in DYNAMIC:
            c += 1
    return c


def _names_of(n: ast.AST) -> list[str]:
    if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return [n.name]
    if isinstance(n, ast.Assign):
        return [t.id for tg in n.targets for t in ast.walk(tg) if isinstance(t, ast.Name)]
    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
        return [n.target.id]
    return []


def top_level_defs(tree: ast.Module) -> dict[str, list[str]]:
    """Map each top-level defined name to the ast.dump of its definitions."""
    out: dict[str, list[str]] = {}
    for n in tree.body:
        for name in _names_of(n):
            out.setdefault(name, []).append(ast.dump(n))
    return out


def units(tree: ast.Module, locked: tuple[str, ...]) -> dict[str, list[ast.AST]]:
    """The top-level nodes that are not locked, grouped by definition name ("<module>" for the rest)."""
    out: dict[str, list[ast.AST]] = {}
    for n in tree.body:
        names = _names_of(n)
        if names and all(x in locked for x in names):
            continue
        out.setdefault(names[0] if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) else "<module>", []).append(n)  # fmt: skip
    return out


def imported_modules(nodes: list[ast.AST]) -> set[str]:
    """Top-level module names imported anywhere in the nodes."""
    mods: set[str] = set()
    for root in nodes:
        for n in ast.walk(root):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods.add(n.module.split(".")[0])
    return mods


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


def _count(nodes: list[ast.AST]) -> tuple[Counter, int]:
    c, dyn = Counter(), 0
    for n in nodes:
        c += count_forbidden(n)
        dyn += count_dynamic(n)
    return c, dyn


def check(base: Path, cand: Path) -> dict:
    """Run rules R1 to R8 and return the report dictionary."""
    viol: list[dict] = []

    def add(rule: str, file: str, detail: str) -> None:
        viol.append({"rule": rule, "file": file, "detail": detail})

    bnames, cnames = _listing(base), _listing(cand)
    for f in sorted(cnames - set(EDITABLE) - set(FROZEN)):
        add("R1", f, "file is not editable")
    for f in FROZEN:
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
        new = not bp.is_file()
        lb, lc = (0 if new else _lines(bp)), _lines(cp)
        files[f] = {"changed": new or bp.read_bytes() != cp.read_bytes(), "lines_base": lb, "lines_cand": lc}
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
        locked = LOCKED.get(f, ())
        bd, cd = top_level_defs(bt), top_level_defs(ct)
        for name in locked:
            if name not in bd:
                continue
            if name not in cd:
                add("R3", f, f"locked name {name} is missing")
            elif bd[name] != cd[name]:
                add("R3", f, f"locked name {name} was changed")
        bu, cu = units(bt, locked), units(ct, locked)
        for mod in sorted(absolute_imports(ct)):
            if any(mod == p or mod.startswith(p + ".") for p in FORBIDDEN_MODULES):
                add("R8", f, f"imports {mod}: the simulator side of the bed")
        extra = (
            imported_modules([n for v in cu.values() for n in v])
            - imported_modules([n for v in bu.values() for n in v])
            - set(NEW_IMPORTS_OK)
            - {Path(e).stem for e in EDITABLE}
        )
        if extra:
            add("R5", f, f"new imports: {sorted(extra)}")
        for key, nodes in sorted(cu.items()):
            cc, cdyn = _count(nodes)
            bc, bdyn = _count(bu.get(key, [])) if f in CARRIED else (Counter(), 0)
            where = "" if key == "<module>" else f" in {key}"
            for k in sorted(cc):
                if cc[k] > bc[k]:
                    add("R4", f, f"forbidden name {k!r}{where}: {bc[k]} allowed, {cc[k]} in candidate")
            if cdyn > bdyn:
                add("R6", f, f"dynamic access{where}: {bdyn} allowed, {cdyn} in candidate")
    return {"ok": not viol, "violations": viol, "files": files}


# ---------------------------------------------------------------- self-test

SEED = Path(__file__).resolve().parent / "seed"


def _rules(edit: dict[str, tuple[str, str] | str | None], extra: dict[str, str] | None = None) -> set[str]:
    """Copy the seed package, apply `edit` ({file: (old, new)} or {file: full text} or None to delete), add `extra`
    files, run the guard against the seed and return the violated rules."""
    with tempfile.TemporaryDirectory() as t:
        c = Path(t) / "c"
        shutil.copytree(SEED, c, ignore=shutil.ignore_patterns("__pycache__"))
        for f, e in edit.items():
            if e is None:
                (c / f).unlink()
            elif isinstance(e, tuple):
                text = (c / f).read_text()
                assert e[0] in text, f"selftest anchor missing in {f}: {e[0][:40]!r}"
                (c / f).write_text(text.replace(e[0], e[1], 1))
            else:
                (c / f).write_text(e)
        for f, text in (extra or {}).items():
            (c / f).write_text(text)
        return {v["rule"] for v in check(SEED, c)["violations"]}


def _append(f: str, code: str) -> dict:
    return {f: ((c := (SEED / f).read_text()), c + "\n\n" + code + "\n")}


def selftest() -> list[str]:
    """Return one line per failing test."""
    CAM, VIEW, SIMPLE, PICK = "flows.py", "camera_view.py", "skill.py", "pick.py"
    bad: list[str] = []

    def case(name: str, got: set[str], want: set[str]) -> None:
        if got != want:
            bad.append(f"FAIL {name}: expected {sorted(want)}, got {sorted(got)}")

    rep = check(SEED, SEED)
    if not rep["ok"]:
        bad.append(f"FAIL seed against seed: {rep['violations'][:3]}")

    # one candidate per kind of privileged read (each added to the camera skill: zero names allowed there)
    reads = {
        "1 simulator state (sim.d)": "def f(env):\n    return env.sim.d.qpos[:3]",
        "1 simulator model (sim.m)": "def f(env):\n    return env.sim.m.nbody",
        "1 simulator verdict": "def f(env):\n    return env.sim.success",
        "2 env.spec": "def f(env):\n    return env.spec['target']",
        "2 spec_of": "def f(task):\n    return spec_of(task)",
        "2 specs.json": "def f(task):\n    return open('specs.json').read()",
        "3 env.info target": "def f(env):\n    return env.info['target']",
        "3 env.bodies": "def f(env):\n    return env.bodies",
        "3 env.snapshot": "def f(env):\n    return env.snapshot()['p']",
        "3 env.marks": "def f(env):\n    return env.marks['start']",
        "4 truth view": "def f(env):\n    return LmlTruthView(env, None)",
        "4 receptacle": "def f(env):\n    return receptacle(env)",
        "4 body_points": "def f(env):\n    return body_points(env.sim, env.bodies)",
        "5 to_sim": "def f(env, p):\n    return env.bel.to_sim(p)",
        "5 from_sim": "def f(env, p):\n    return env.bel.from_sim(p)",
        "5 _truth": "def f(env):\n    return env.bel._truth()",
        "5 true pelvis tie": "def f(env):\n    return env.bel.pt",
        "5 measured hand": "def f(env):\n    return env.mo.jaw_centre('right')[0]",
        "6 camera reply with pose": "def f(env):\n    return env.sim.eyes()",
        "7 reset reply": "def f(env):\n    return env.sim.first['spec']",
        "8 server link": "def f(env):\n    return env.sim.link.send(('x',))",
        "8 ask": "def f():\n    return ask('state')",
        "9 episode id": "def f(episode_id):\n    return episode_id",
        "9 env.task": "def f(env):\n    return env.task",
        "10 open": "def f():\n    return open('/etc/passwd').read()",
        "10 np.load": "def f(np):\n    return np.load('x.npz')",
        "10 Path": "def f(p):\n    return Path(p).read_text()",
        "10 environ": "def f(os):\n    return os.environ['HOME']",
        "10 import os": "import os\n\n\ndef f():\n    return os.getcwd()",
        "10 import subprocess": "import subprocess",
        "10 import sys": "import sys",
        "dynamic getattr": "def f(sim):\n    return getattr(sim, 'e' + 'yes')()",
        "dynamic vars": "def f(env):\n    return vars(env)",
        "dynamic __dict__": "def f(env):\n    return env.__dict__",
        "dynamic globals": "def f():\n    return globals()",
        "mujoco call": "def f(m, d):\n    return mj_forward(m, d)",
        "11 target_joint_pos": "def f(env):\n    return env.target_joint_pos",
        "11 target_joint": "def f(target_joint):\n    return target_joint",
        "11 target_fixture": "def f(env):\n    return env.target_fixture",
        "11 target_face_pos": "def f(env):\n    return env.target_face_pos",
        "11 qpos_adr": "def f(j):\n    return j['qpos_adr']\n\n\ndef g(qpos_adr):\n    return qpos_adr",
        "11 door body string": "def f():\n    return 'refrigerator_14a808d19515703c7e4cc20a5049143a_1_6_0'",
        "11 handle body string": "def f():\n    return '_1_7_0'",
        "11 door body name": "def f():\n    refrigerator_x = 1\n    return refrigerator_x",
        "11 drawer joint string": "def f():\n    return 'chestofdrawers_663f11089f8df7c4a9a159fdc6b749ac_1_6_0_DrawerMesh021_3'",
        "11 drawer site string": "def f():\n    return 'DrawerMesh02Receptacle0_3'",
        "11 drawer fixture string": "def f():\n    return 'FloorPlan209:Dresser_209_1_4'",
        "11 drawer name": "def f():\n    chestofdrawers_x = 1\n    return chestofdrawers_x",
        "11 drawer range": "def f(env):\n    return env.spec['joint_range']",
        "hidden recorder": "def f(env):\n    return env.rec.dest_truth",
        "12 the facade's env": "def f(env):\n    return env._env.sim",
        "12 the hand's sim": "def f(env):\n    return env.hand._sim",
        "12 the clock's env": "def f(env):\n    return env.clock._clock.env",
        "12 mo.sim": "def f(env):\n    return env.mo.sim",
        "13 import the bed's run": "from locomani.bench.locomanilab import run",
        "13 import the bed's client": "import locomani.bench.locomanilab.client",
        "13 import the truth side": "from locomani.bench.locomanilab.truth import receptacle",
        "13 import SIMPLE's privileged": "from locomani.bench.simple.privileged import SimplePrivileged",
        "13 import perception truth": "from locomani.perception.truth import TruthView",
        "13 import odometry": "from locomani.perception import odometry",
    }
    for kind, code in reads.items():
        for f in (CAM, VIEW):
            got = _rules(_append(f, code))
            if not got & {"R4", "R5", "R6", "R8"}:
                bad.append(f"FAIL read '{kind}' in {f} passes the guard")
    # the same reads in the carried-over files: new ones fail
    for kind in ("1 simulator state (sim.d)", "5 to_sim", "9 env.task", "10 open", "dynamic getattr"):
        for f in (SIMPLE, PICK):
            if not _rules(_append(f, reads[kind])) & {"R4", "R6"}:
                bad.append(f"FAIL new read '{kind}' in {f} passes the guard")
    # a read moved from the function that has it to another one, count unchanged, fails in a carried-over file
    text = (SEED / SIMPLE).read_text()
    k = "    p, Rm = mo.jaw_centre(SIDE)  # record: the true palm against the grasp\n"
    assert k in text, "selftest anchor missing"
    moved = text.replace(k, "    p, Rm = Tg[:3, 3], Tg[:3, :3]\n").replace(
        "def one_step(mo, K: dict) -> None:\n", "def one_step(mo, K: dict) -> None:\n    _ = mo.jaw_centre(SIDE)\n"
    )
    case("count-preserving move of the measured palm fails", _rules({SIMPLE: moved}), {"R4"})
    # in the camera skill a one-for-one swap of a harmless expression for a read fails
    case(
        "swap in camera skill fails",
        _rules({CAM: ("    return float(np.abs(env.hand.q(S.SIDE)", "    return float(np.abs(env.sim.d.qpos[:3]")}),
        {"R4"},
    )
    # locked definitions
    case(
        "R3 cloud_name changed",
        _rules({VIEW: ('return f"{task}_L{kind}_e{n}"', 'return f"{task}_{kind}_{n}"')}),
        {"R3"},
    )
    case(
        "R3 loader changed",
        _rules({VIEW: ('np.load(folder / f"{name}.npz")["p_o"]', 'np.load(folder / f"{name}.npz")["p_o"] + 0.01')}),
        {"R3"},
    )
    case("R3 locked removed", _rules({VIEW: ("def have_grasps(", "def have_grasps2(")}), {"R3"})
    case("R3 T changed", _rules({PICK: ("    M = np.eye(4)\n", "    M = np.eye(4) * 1\n")}), {"R3"})
    case("R3 cont changed", _rules({PICK: ("ik_from_ref=True", "ik_from_ref=False")}), {"R3"})
    # files
    case("R1 extra file", _rules({}, {"evil.py": "x = 1\n"}), {"R1"})
    case("R1 frozen changed", _rules({"__init__.py": (SEED / "__init__.py").read_text() + "\nx = 1\n"}), {"R1"})
    case("R1 frozen removed", _rules({"__init__.py": None}), {"R1"})
    case("R1 editable removed", _rules({PICK: None}), {"R1"})
    case("R1 dot and cache files ignored", _rules({}, {".note": "x\n"}), set())
    case("R2 syntax error", _rules({CAM: "def (:\n"}), {"R2"})
    case("R7 growth", _rules({CAM: (SEED / CAM).read_text() + "\n" * (MAX_GROWTH + 5)}), {"R7"})
    case("R5 new import of a project module", _rules(_append(CAM, "import skill_lml")), {"R5"})
    case("R5 new stdlib import", _rules(_append(CAM, "import random")), {"R5"})
    # edits that must pass
    case("passes: comment and format", _rules({CAM: ("SEG = 2.5", "SEG = 2.5  # a note")}), set())
    case(
        "passes: new pure function",
        _rules(_append(CAM, "def g(a):\n    d = np.linalg.norm(a)\n    return d * 2")),
        set(),
    )
    case(
        "passes: local named like a state attribute",
        _rules(_append(CAM, "def g(first, over, task):\n    return first + over")),
        set(),
    )
    case("passes: new numpy import", _rules({CAM: ("import numpy as np", "import math\n\nimport numpy as np")}), set())
    case("passes: number change in editable code", _rules({CAM: ("MIN_PX_NAV = 25", "MIN_PX_NAV = 30")}), set())
    case("passes: parameter in skill_simple", _rules({SIMPLE: ("press=0.3,", "press=0.35,")}), set())
    case(
        "passes: hand accessor use",
        _rules(_append(CAM, "def g(env):\n    return env.hand.q('right')")),
        set(),
    )
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
