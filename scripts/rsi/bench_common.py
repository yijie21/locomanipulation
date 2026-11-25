"""What every benchmark adapter of backend_bench.py shares (TODO 54 phase 3, one benchmark backend; 2026-11-25).

Not about one adapter: the paths and the default caps, the family file (loaded with the hidden split seeds), the seed package
copy and its write guard, the skill package's parameters and `set` keys, the scratch directory, the process handling, the
work units, the files of a stored round, the render and memo switches of the adapter's call, and the sandbox walk and its
audit. The pools are bench_pools.py, rows -> rounds and cards are bench_rows.py (shared) and bench_<adapter>.py (the row
schema), the adapter table, the signature and run_jobs are backend_bench.py. Nothing here starts a simulator itself."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HIDDEN = "outputs/scene_rsi_20261016_hidden"
SUCCESS_RULE_FALLBACK = 2
SCAN_MAX_BYTES = 2_000_000
# Level 2: a stance is given as `set` keys with this prefix and reaches the adapter as its --stance argument.
STANCE = "stance_"

# The pipeline switches of a benchmark family (the loop reads them through backend.layout(family)["caps"]); a family file's
# `caps` override single keys. A scene family has none of these on (backend.SCENE_CAPS).
DEFAULT_CAPS = {
    "claims": False,
    "design_probes": False,
    "diag_redraw": False,
    "sentinel_exact": False,
    "min_net": 3,
    "p_max": 0.05,
    "sentinel_flips": 1,
    "probe_scope": True,
    "check_half": True,
    "check_min_net": 4,
    "stage1_first": True,
    "role_rerun": True,
    "sealed_at_stop": True,
    "alpha_task": 0.05,
    "alpha_task_check": 0.10,
    "manager": "program",
    "prober": False,
    "measure_claims": False,
    "compose": "program",
    "stop_no_progress": 2,
}


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def split_set(setting: dict) -> tuple[dict, dict | None]:
    """(the skill's parameter overrides, the stance override or None) of a job's `set`."""
    skill = {k: v for k, v in setting.items() if not k.startswith(STANCE)}
    stance = {k[len(STANCE) :]: v for k, v in setting.items() if k.startswith(STANCE)}
    return skill, (stance or None)


def bench(fam: dict) -> dict | None:
    """The family's "bench" block."""
    return fam.get("bench") or None


def bed_root(fam: dict) -> Path:
    return Path(bench(fam)["root"])


def bed_fixed(fam: dict) -> dict:
    """The family's fixed bed switches (`bed_fixed`, [bed-fixed]): corrections of the bed's interface to the benchmark,
    found by people, that every round of every role runs with (diagnostic rounds, probes, try.py, drafts, the gate,
    the sealed set). run_jobs lays them over each job's set; no set may name them. Keys starting with "_" are notes."""
    return {k: v for k, v in (fam.get("bed_fixed") or {}).items() if not k.startswith("_")}


# ------------------------------------------------------------------------------------------------ the family file
def hidden_family_dir(name: str) -> Path:
    """The family's hidden directory (seeds.json, practice/)."""
    return REPO / HIDDEN / name


def hidden_seeds(name: str) -> dict:
    """The seeds the sandbox keeps in the hidden tree; {} when there is none or the tree cannot be read (a role runs
    with the hidden tree at mode 000)."""
    p = hidden_family_dir(name) / "seeds.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (PermissionError, ValueError):
        return {}


def load_family(name: str) -> dict:
    """The family dict from families/<name>.json, blank pool seeds filled from the hidden tree when readable, and
    `rows` = {task id: tasks_file row}."""
    fam = json.loads((HERE / "families" / f"{name}.json").read_text())
    hid = hidden_seeds(fam["name"])
    for pool, spec in fam.get(
        "pools", {}
    ).items():  # one convention: seeds.json[pool] is the split seed, or {"ids": [...]}
        if spec.get("seed") is None and not spec.get("ids") and pool in hid:
            h = hid[pool]
            fam["pools"][pool] = dict(spec, **h) if isinstance(h, dict) else dict(spec, seed=h)
    rows = {}
    for x in _abs(fam["tasks_file"]).read_text().splitlines():
        if x.strip():
            row = json.loads(x)
            rows[row["id"]] = row
    fam["rows"] = {t: rows[t] for t in fam["tasks"]}
    return fam


# ------------------------------------------------------------------------------------- the seed package, write guard
PROTECTED: set[Path] = (
    set()
)  # the seed packages of the families this process has seen (backend_bench.base_dir registers them)


class BaseWriteRefused(PermissionError):
    """Something was about to write or delete inside the seed package."""


def guard_write(path) -> None:
    """Raise BaseWriteRefused when `path` is a protected seed package, inside one, or a parent of one."""
    p = Path(path).resolve()
    for b in PROTECTED:
        if p == b or b in p.parents or p in b.parents:
            raise BaseWriteRefused(f"refused: {path} is, or contains, the seed package {b} (read-only source)")


def package_root(fam: dict) -> Path:
    """The loop's working package: under the loop's own state directory for the family (<outputs>/scene_rsi_20261016/<family>/
    package; RSI_PACKAGE_ROOT replaces <outputs>/scene_rsi_20261016 in a test). put_skills and the Engineers write here,
    never in the base."""
    root = Path(os.environ.get("RSI_PACKAGE_ROOT") or REPO / "outputs" / "scene_rsi_20261016")
    return root / fam["name"] / "package"


def ensure_package(base: Path, dst: Path) -> Path:
    """The working package `dst`, copied from the base on first use."""
    if not dst.exists():
        guard_write(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dst


# ------------------------------------------------------------------------------------------ the skill package
_PACKAGE_KEYS = ("K_ORIG", "K_CUT", "PLACING")  # the dict literals of skill.py that make up the seed's parameters
_ART_KEYS = ("K_ART",)  # also: the articulated tasks' `art_*` keys, K_ART's and ART_TASK's


def literal(node):
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)


def package_parameters(root: Path) -> dict:
    """The package's parameter dict (skill.defaults) read from the source text: the keyword arguments of the dict(...)
    calls K_ORIG, K_CUT, PLACING and K_ART of skill.py, and every keyword of ART_TASK's per-task dict(...) not already
    there (its first value). The package imports the bench, so the loop's process never imports it. {} without a skill.py."""
    p = Path(root) / "skill.py"
    if not p.is_file():
        return {}
    tree = ast.parse(p.read_text())
    out: dict = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & set(_PACKAGE_KEYS) or names & set(_ART_KEYS):
            for kw in node.value.keywords:
                if kw.arg:
                    out[kw.arg] = literal(kw.value)
    for node in tree.body:  # ART_TASK = dict(Task=dict(art_obj=..., ...), ...): the keys no task shares with K_ART
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ART_TASK" for t in node.targets):
            if isinstance(node.value, ast.Call):
                for task in node.value.keywords:
                    if isinstance(task.value, ast.Call):
                        for kw in task.value.keywords:
                            if kw.arg:
                                out.setdefault(kw.arg, literal(kw.value))
    return out


def extra_parameters(root: Path) -> dict:
    """The keywords of `EXTRA = dict(...)` in the package's __init__.py (keys the flows read that skill.py does not hold)."""
    init, out = Path(root) / "__init__.py", {}
    if init.exists():
        for node in ast.parse(init.read_text()).body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and any(isinstance(t, ast.Name) and t.id == "EXTRA" for t in node.targets)
            ):
                for kw in node.value.keywords:
                    if kw.arg:
                        out[kw.arg] = literal(kw.value)
    return out


def k_parameters(root: Path) -> dict:
    """The keywords of `K = dict(...)` in skill.py (the seed-pool adapters' packages)."""
    out: dict = {}
    p = Path(root) / "skill.py"
    if not p.is_file():
        return out
    for node in ast.parse(p.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and any(isinstance(t, ast.Name) and t.id == "K" for t in node.targets)
        ):
            for kw in node.value.keywords:
                if kw.arg:
                    out[kw.arg] = literal(kw.value)
    return out


class SkillPackage:
    """The protocol's package object: `apply_set(pkg, overrides)` validates the keys and `parameters(pkg)` lists them, both
    without importing the package. `defaults` is the adapter's reading of the package source (`params`)."""

    def __init__(self, root: Path, bed_keys=(), fixed=(), params=package_parameters):
        self.root = Path(root)
        self.defaults = params(self.root)
        self.bed_keys = tuple(bed_keys)
        self.fixed = tuple(fixed)
        self.current: dict = {}

    def apply_set(self, _pkg, overrides: dict | None = None) -> dict:
        pinned = sorted(k for k in (overrides or {}) if k in self.fixed)
        if pinned:
            raise KeyError(
                f"parameters {pinned} are fixed by the bed for every round ([bed-fixed]); a set may not name them"
            )
        ok = set(self.defaults) | set(self.bed_keys)
        bad = sorted(k for k in (overrides or {}) if k not in ok and not k.startswith(STANCE))
        if bad:
            raise KeyError(f"unknown parameters {bad}")
        self.current = dict(overrides or {})
        return self.current

    def parameters(self, _pkg=None) -> dict:
        return {**self.defaults, **{k: v for k, v in self.current.items() if k in self.defaults}}


def skills_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ stored rounds
def success_rule() -> int:
    try:
        sys.path[:0] = [str(HERE)]
        import batch

        return int(batch.SUCCESS_RULE)
    except Exception:
        return SUCCESS_RULE_FALLBACK
    finally:
        if str(HERE) in sys.path:
            sys.path.remove(str(HERE))


def current(f: Path, sig: str) -> bool:
    """True when the round file `f` was made under this success rule and this signature."""
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError):
        return False
    return r.get("rule") == success_rule() and r.get("motor") == sig


def write_round(f: Path, rec: dict) -> None:
    guard_write(f)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    tmp.rename(f)  # a killed process never leaves a half-written round


def ids_by_task(pool: Path, labels: list[str]) -> dict[str, list[tuple[str, str]]]:
    by: dict[str, list[tuple[str, str]]] = {}
    for x in labels:
        d = json.loads((pool / f"{x}.json").read_text())
        by.setdefault(d["task"], []).append((x, d["episode_id"]))
    return by


# --------------------------------------------------------------------------------- scratch, units and processes
def scratch() -> Path:
    """Where a call's run directories go (built scenes, server logs, rows): TMPDIR, else the workspace's tmp
    (scripts/paths.sh's LOCOMANI_TMP; the machine's /tmp is on a small root disk), else the system's."""
    if os.environ.get("TMPDIR"):
        return Path(os.environ["TMPDIR"])
    ws = Path(os.environ.get("LOCOMANI_TMP") or Path(os.environ.get("LOCOMANI_WORKSPACE") or "/workspace") / "tmp")
    return ws if ws.is_dir() else Path(tempfile.gettempdir())


def units(by: dict[str, list], slots: int) -> list[tuple[str, list]]:
    """The work units of one job, (task, items) each: every task's episodes cut into equal chunks of about a quarter of a
    slot's share, in round-robin order over the tasks. The adapter takes one of the machine's slots per call and waits when
    none is free, so `slots` units run at a time and stay busy to the end; neighbours in the order are different tasks,
    whose look passes do not wait for each other."""
    total = sum(len(v) for v in by.values())
    per = max(4, -(-total // (4 * max(1, slots))))
    cut: dict[str, list[list]] = {}
    for task, items in by.items():
        n = -(-len(items) // per)
        size = -(-len(items) // n)
        cut[task] = [items[i : i + size] for i in range(0, len(items), size)]
    out, k = [], 0
    while any(len(c) > k for c in cut.values()):
        out += [(task, c[k]) for task, c in cut.items() if len(c) > k]
        k += 1
    return out


def _tree(pid: int) -> list[int]:
    """`pid` and its live descendants, parents before children (read from /proc: the sim python has no psutil)."""
    kids: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(d.name))
    order, todo = [], [pid]
    while todo:
        x = todo.pop()
        order.append(x)
        todo += kids.get(x, [])
    return order


def _running(pid: int) -> bool:
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _stop_tree(p: subprocess.Popen, grace: float = 15.0) -> None:
    """Stop a runner and everything it started (its servers, the bed): SIGTERM to the whole tree, the runner first (it
    stops what it started when its foreground command ends), then SIGKILL to what is left. The children stay in the loop's
    process group, so a group kill of the loop takes them too."""
    pids = _tree(p.pid)
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
        for x in pids:
            if _running(x):
                try:
                    os.kill(x, sig)
                except OSError:
                    pass
        end = time.time() + wait
        while time.time() < end and any(_running(x) for x in pids):
            time.sleep(0.2)


def run_command(
    argv: list[str], env: dict, log, stop: threading.Event | None, cwd: Path | None, tree: bool = False
) -> int:
    """Run a command to its end; a set `stop` kills it (`tree`: with every process it started)."""
    p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=cwd)
    while True:
        try:
            return p.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            if stop is not None and stop.is_set():
                if tree:
                    _stop_tree(p)
                p.kill()
                p.wait()
                return -9


# --------------------------------------------------------------------------------- the adapter call's switches
LAZY_WARM = 48
RENDER_MODES = ("lazy", "every_step")
MEMO_GB = 20  # size cap of a family's disk store, in GB (bench.memo_gb overrides)


def render_mode(fam: dict) -> str:
    return (bench(fam) or {}).get("render", "lazy")


def render_sets(fam: dict) -> list[str]:
    """The adapter's --set values for how the cameras render: lazily with warm-up renders (default), or every step."""
    if render_mode(fam) == "every_step":
        return ["lazy_render=false"]
    return ["lazy_render=true", f"lazy_warm={LAZY_WARM}"]


def render_tag(fam: dict) -> str:
    """What enters the signature: two render modes never share a pool or a gate signature."""
    return "every_step" if render_mode(fam) == "every_step" else f"lazy{LAZY_WARM}"


def memo_on(fam: dict, full: bool) -> bool:
    """`bench.memo = true` turns on replay from saved states (memo.py) in the diagnostic, probe, sweep and draft rounds; the gate
    and the sealed set (`full`) always run every episode in full: they neither read nor write the store."""
    return bool((bench(fam) or {}).get("memo")) and not full


def memo_store_dir(fam: dict) -> Path:
    """The family's disk store of saved states (TODO 53): in the scratch directory (the loop's own, like the run directories),
    so outside every sandbox a role can read and outside the hidden tree; never under the repository."""
    root = scratch().resolve()
    origin = (REPO / "third_party").resolve().parent  # a sandbox links third_party and outputs to the origin checkout
    for bad in {REPO.resolve(), (REPO / HIDDEN).resolve(), origin, (origin / HIDDEN).resolve()}:
        if root == bad or bad in root.parents:
            raise ValueError(f"the memo store would be inside {bad}: set TMPDIR / LOCOMANI_TMP to a scratch folder")
    return root / f"simple_{fam['name']}_memo"


def memo_sets(fam: dict, full: bool, sig: str) -> list[str]:
    """The adapter's --set values for the memo: on (diagnostic, probe, sweep and draft rounds), its disk store in the scratch
    directory, the signature `sig` (the store's sub-folder: a changed bed or switch set reaches no old state) and the size cap."""
    if not memo_on(fam, full):
        return []
    gb = (bench(fam) or {}).get("memo_gb", MEMO_GB)
    return ["memo=true", f"memo_dir={memo_store_dir(fam)}", f"memo_sig={sig}", f"memo_gb={gb}"]


def setting_arg(v) -> str:
    """A --set / --bench-set value the way config_base casts it back: bools lower case, lists comma separated."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


# ------------------------------------------------------------------------------------------------ sandbox walk
def _copy(src: Path, dst: Path) -> None:
    guard_write(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    elif src.exists():
        shutil.copy2(src, dst)


def _cut(dest: Path, rel: str) -> None:
    """Remove `rel` (globs allowed) below `dest` without touching what a symlink points to: a symlinked directory on
    the way is replaced by a real directory of links to its entries first."""
    for hit in sorted(dest.glob(rel)) if "*" in rel else [dest / rel]:
        guard_write(hit)
        parts = hit.relative_to(dest).parts
        cur = dest
        for name in parts[:-1]:
            cur = cur / name
            if cur.is_symlink():
                target = cur.resolve()
                cur.unlink()
                cur.mkdir()
                for e in target.iterdir():
                    (cur / e.name).symlink_to(e)
            if not cur.exists():
                break
        else:
            if hit.is_symlink() or hit.is_file():
                hit.unlink()
            elif hit.is_dir():
                shutil.rmtree(hit)


def sandbox(dest, fam: dict, *, base: Path, base_rel: str, files: list[str], forbidden) -> None:
    """Add what the Engineer needs to the sandbox `dest` and take away what it must not see (before sandbox.py's git commit):
    under `<dest>/<sandbox.dir>` (default bed_readonly_copy; FOR READING ONLY, nothing runs from it) the seed package `base`,
    the adapter's `files` (below the bench root), the task files and the practice files with the hidden rows blanked; the full
    practice files go to the hidden tree. The bench root stays where the family file says, so a round in the sandbox runs on
    the real adapter; only `practice_dir` is rewritten, to the blanked copy. `forbidden` paths are cut out of the linked
    third_party and any other copy."""
    import bench_pools as practice  # (bench_pools imports this module)

    dest = Path(dest)
    spec = fam.get("sandbox", {})
    rel = spec.get("dir", "bed_readonly_copy")
    nest = dest / rel
    root = bed_root(fam)
    hidden = practice.hidden_ids(fam)
    _copy(base, nest / base_rel)
    for entry in files:
        _copy(root / entry, nest / entry)
        if (nest / entry).is_file():
            (nest / entry).chmod(0o444)
    for t in spec.get("task_files", []):
        _copy(_abs(t), nest / "tasks" / Path(t).name)
    src = practice.practice_dir(fam, True)
    vis, hid = nest / "practice", dest / HIDDEN / fam["name"] / "practice"
    vis.mkdir(parents=True, exist_ok=True)
    for task in fam["tasks"]:
        f = src / f"{task}.jsonl"
        if not f.exists():
            continue
        lines = f.read_text().splitlines()
        (vis / f.name).write_text(
            "".join(("{}" if f"{task}/Q/{i}" in hidden else x) + "\n" for i, x in enumerate(lines))
        )
        if src.resolve() != hid.resolve():
            hid.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, hid / f.name)
    if (dest / "third_party").exists() and not (nest / "third_party").exists():
        (nest / "third_party").symlink_to(dest / "third_party")
    for pat in forbidden:
        _cut(dest, pat)
    ff = dest / "scripts" / "rsi" / "families" / f"{fam['name']}.json"
    if ff.exists():
        d = json.loads(ff.read_text())
        d.update(practice_dir=f"{rel}/practice")  # the bench root stays: the sandbox's rounds run on the real adapter
        ff.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")


def audit(dest, *, forbidden, leak, nest_name: str, allowed: set[str]) -> list[str]:
    """What in the sandbox `dest` the Engineer must not see: forbidden paths that exist, the practice rows of hidden ids
    that are not blank, and leak patterns in text files (the hidden tree, .git, links and the files copied on purpose, `allowed`,
    are skipped). [] when clean."""
    import bench_pools as practice

    dest = Path(dest)
    bad: list[str] = []
    for pat in forbidden:
        for p in [dest / pat] if "*" not in pat else sorted(dest.glob(pat)):
            if os.path.lexists(p) and (p.exists() or p.is_symlink()):
                bad.append(f"path {p.relative_to(dest)}")
    nest = dest / nest_name
    for f in sorted((nest / "practice").glob("*.jsonl")) if (nest / "practice").is_dir() else []:
        seen = [0, 0, 0]
        for i, row in enumerate(practice.rows(f)):
            lv = row["level"]
            if lv is not None:
                seen[lv] += 1
                if seen[lv] > practice.DIAG_PER_LEVEL:
                    bad.append(f"practice row not blank: {f.relative_to(dest)} line {i}")
    pats = [re.compile(x) for x in leak]
    for cur, dirs, files in os.walk(nest):
        dirs[:] = [d for d in dirs if not (Path(cur) / d).is_symlink()]
        for name in files:
            p = Path(cur) / name
            if p.is_symlink() or str(p) in allowed or p.stat().st_size > SCAN_MAX_BYTES:
                continue
            try:
                text = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for pat in pats:
                m = pat.search(text)
                if m:
                    bad.append(f"text {p.relative_to(dest)}: {m.group(0)}")
                    break
    return bad
