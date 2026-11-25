"""The loop's ONE backend for every benchmark adapter (TODO 49.6b; 2026-11-24; the only one since TODO 54 phase 3).

A family file with `"backend": "bench"` and a `"bench"` block runs every round (diagnostic, probe, draft, check half, gate,
sealed) through the adapter's own command, `python -m locomani.bench.<name> run <Task> ...`, from the bench root:

    "bench": {"name": "simple" | "locomanilab" | "bigym2" | "humanoidarena", "root": <a bench worktree>, "options": {BenchOptions overrides},
              "config": {the adapter's config fields, --set k=v; e.g. slots}, "python": optional, "files": optional globs,
              "render": "every_step" optional, "memo": true optional}

What is written once here: the runner's argv, the signature (the adapter's files, the resolved switches), the adapter's static
guard, the package layout, the sandbox copy and audit, the work units and their slots, and the ONE `run_jobs`. The modules it
stands on:

- bench_common.py: paths, default caps, the family file, the seed package and its write guard, the process handling, the
  files of a stored round, the render and memo switches, the sandbox walk;
- bench_pools.py: the pools (the anti-leaderboard wall: the hidden split seeds and ids);
- bench_rows.py: the round's shape, the steps, the cards file; bench_simple.py, bench_lml.py, bench_seeded.py: what an
  adapter's record means (the row schema, the card text, the leak patterns, the forbidden paths, the failure classes).

What is adapter specific is ONE table below (ADAPTERS): the module name, the seed and guard paths, which files are read and
hashed, the config keys the backend sets itself (practice files, kinds, run dir, the look cache) and the render flag, and the
module that knows the adapter's row schema (`rows`). A new adapter is one more entry. The adapters whose episodes are seed
pools (`pool_mode` "seeds": BiGym 2.0, HumanoidArena) have no practice files (bench_pools.py). Nothing here starts a
simulator itself.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bench_common as C  # noqa: E402
import bench_pools as P  # noqa: E402
import bench_rows as R  # noqa: E402
from bench_common import DEFAULT_CAPS, STANCE, bed_fixed, skills_hash  # noqa: E402,F401
from bench_pools import GATE_LETTERS, PROBE_LETTERS  # noqa: E402,F401

REPO = C.REPO
HIDDEN = C.HIDDEN
PY = "/venv/main/bin/python"  # runs the adapter's `run` (the bed itself starts under the sim venv, the adapter's own config)

_SHARED = (  # code every adapter's rounds depend on: the switches, the perception modules and the robot's controller
    "src/locomani/bench/options.py",
    "src/locomani/perception/*.py",
    "src/locomani/backends/scalebfm/*.py",
)

ADAPTERS: dict[str, dict] = {
    "simple": dict(
        module="locomani.bench.simple",
        seed="src/locomani/bench/simple/seed",
        guard="src/locomani/bench/simple/guard.py",
        files=(
            "src/locomani/bench/options.py",
            "src/locomani/bench/simple/*.py",
            *_SHARED[1:],
        ),  # read by roles and hashed
        hash_files=(),  # hashed only
        kinds="Q",
        practice_keys=("practice_dir",),  # config fields that name the practice files (one per task group)
        render_key="lazy_render",  # lazy with warm-up renders in every round (bench_common.render_sets); bench.render = "every_step" for official reporting
        cache_key=None,  # a config field for the look and grasp cache the backend places in its scratch directory
        own=(
            "practice_dir",
            "run_dir",
            "lazy_render",
            "lazy_warm",
            "lazy_subframes",
            "memo",
            "memo_dir",
            "memo_sig",
            "memo_gb",
            "kinds",
        ),  # config keys the backend sets itself
        strip_env=("SIMPLE_",),
        tmpdir=False,
        units=3,  # work units side by side when the family names no official_slots and no `slots`
        benchmark=None,  # a path below the bench root whose git commit enters the signature
        rows="bench_simple",
    ),
    "locomanilab": dict(
        module="locomani.bench.locomanilab",
        seed="src/locomani/bench/locomanilab/seed",
        guard="src/locomani/bench/locomanilab/guard.py",
        files=(  # the bed's judges, the package's facade and runner, the shared fixed modules: read-only copies for roles
            "src/locomani/bench/options.py",
            "src/locomani/bench/locomanilab/run.py",
            "src/locomani/bench/locomanilab/client.py",
            "src/locomani/bench/locomanilab/comply.py",
            "src/locomani/bench/locomanilab/loss.py",
            "src/locomani/bench/locomanilab/guard.py",
            "src/locomani/bench/simple/body.py",
            *_SHARED[1:],
        ),
        hash_files=(  # in the signature, not shown to roles: the server, the truth side, the config, the session
            "src/locomani/bench/locomanilab/*.py",
            "src/locomani/bench/simple/*.py",
        ),
        kinds="Q",
        practice_keys=("practice_dir", "practice_dir_fridge", "practice_dir_drawer"),
        render_key=None,
        cache_key="cache_dir",
        own=("practice_dir", "practice_dir_fridge", "practice_dir_drawer", "run_dir", "cache_dir", "kinds"),
        strip_env=("LML_", "SIMPLE_"),
        tmpdir=True,  # the machine's /tmp is on a small root disk: sockets and scenes go to the scratch directory
        units=2,
        benchmark="third_party/LocoManiLab",
        rows="bench_lml",
    ),
    "bigym2": dict(
        module="locomani.bench.bigym2",
        seed="src/locomani/bench/bigym2/seed",
        guard="src/locomani/bench/bigym2/guard.py",
        files=(  # read-only copies for roles: the facade and runner, the judge and row, the guard, the switches, the frame type
            "src/locomani/bench/options.py",
            "src/locomani/bench/bigym2/run.py",
            "src/locomani/bench/bigym2/client.py",
            "src/locomani/bench/bigym2/judge.py",
            "src/locomani/bench/bigym2/record.py",
            "src/locomani/bench/bigym2/guard.py",
            "src/locomani/perception/sensor.py",
        ),
        hash_files=(  # in the signature, not shown to roles: the server (truth side, contact record), the pools, the session
            "src/locomani/bench/bigym2/*.py",
            "src/locomani/bench/spec.py",
            "src/locomani/config_base.py",
            "configs/bench/bigym2.yaml",
        ),  # no ScaleBFM, no perception beyond the frame type: BiGym keeps its own GR00T-WBC controller and runs on truth
        kinds=None,  # no kinds key: the pools are letters of its `pools` config field
        practice_keys=(),  # no practice files: the episodes are `<task>/<P|A|S>/<n>`, seeds of the adapter's pools
        render_key=None,  # no lazy render: the head camera is not drawn unless `depth` is set
        cache_key=None,
        own=("pools", "run_dir", "lock_file"),
        strip_env=("BIGYM", "SIMPLE_", "LML_"),
        tmpdir=True,  # sockets and scene builds go to the scratch directory
        units=1,  # one MuJoCo server at a time (the family pins its cores); the lock file is the scratch directory's
        benchmark="third_party/BiGym2",
        rows="bench_seeded",
        stub_file="skill.py",  # the package is skill.py (editable) and a frozen __init__.py
        pool_mode="seeds",
        pool_sizes=dict(P=9000, A=1000, S=10000),  # episodes per pool (config.POOLS); E, [620000, 630000), is no letter
        tasks=(
            "reach_target_single",
            "reach_target_multi_modal",
            "reach_target_dual",
            "move_plate",
            "move_two_plates",
            "flip_cup",
            "flip_cutlery",
            "stack_blocks",
            "dishwasher_close",
            "dishwasher_load_cups",
            "dishwasher_load_cutlery",
            "dishwasher_load_plates",
            "drawer_top_open",
            "drawer_top_close",
            "pick_box",
            "put_cups",
            "saucepan_to_hob",
            "sandwich_remove",
            "wall_cupboard_open",
            "wall_cupboard_close",
        ),  # fmt: skip
    ),
    "humanoidarena": dict(
        module="locomani.bench.humanoidarena",
        seed="src/locomani/bench/humanoidarena/seed",
        guard="src/locomani/bench/humanoidarena/guard.py",
        files=(
            "src/locomani/bench/options.py",
            "src/locomani/bench/humanoidarena/run.py",
            "src/locomani/bench/humanoidarena/client.py",
            "src/locomani/bench/humanoidarena/judge.py",
            "src/locomani/bench/humanoidarena/record.py",
            "src/locomani/bench/humanoidarena/guard.py",
            "src/locomani/perception/sensor.py",
        ),
        hash_files=(  # the server, the truth side, the pools, the session, the gait the native tracker follows
            "src/locomani/bench/humanoidarena/*.py",
            "src/locomani/bench/humanoidarena/gait_cycle.json",
            "src/locomani/bench/spec.py",
            "src/locomani/config_base.py",
            "configs/bench/humanoidarena.yaml",
        ),
        kinds=None,
        practice_keys=(),
        render_key=None,
        cache_key=None,
        own=("pools", "run_dir", "lock_file"),
        strip_env=("ISAAC", "OMNI_", "HA_", "SIMPLE_", "LML_"),
        tmpdir=True,
        units=1,  # one Isaac server at a time (GPU 1, four cores): the lock file is the scratch directory's
        benchmark="third_party/HumanoidArena/repo",
        rows="bench_seeded",
        stub_file="skill.py",
        pool_mode="seeds",
        pool_sizes=dict(
            P=20000, A=20000, S=20000
        ),  # 1000 groups of 20 each (config.POOLS); the groups 0-2 are no letter
        tasks=("pp_box", "double_desk", "open_door"),
    ),
}


# ------------------------------------------------------------------------------------------------ the family
def bench(fam: dict) -> dict | None:
    return fam.get("bench") or None


def adapter(fam: dict) -> dict:
    name = (bench(fam) or {}).get("name", "simple")
    if name not in ADAPTERS:
        raise ValueError(f"{fam.get('name')}: bench.name {name!r} is not ported (known: {', '.join(sorted(ADAPTERS))})")
    return ADAPTERS[name]


def rows_module(fam: dict):
    """The module that knows the adapter's rows: to_round, card_text, the leak patterns and the failure classes."""
    return importlib.import_module(adapter(fam)["rows"])


def seeded(fam: dict) -> bool:
    """True for an adapter whose episodes are seed pools (BiGym 2.0, HumanoidArena), not practice files."""
    return bool(bench(fam)) and P.seeded(adapter(fam))


def check_bench(fam: dict) -> None:
    """A bench family runs every round on the adapter (there is no mirror bed), and the backend owns some of the adapter's
    --set keys."""
    b = bench(fam)
    if not b:
        raise ValueError(f'{fam.get("name")}: backend "bench" needs a "bench" block')
    ad = adapter(fam)
    if not b.get("root"):
        raise ValueError(f"{fam.get('name')}: bench.root is missing")
    own = sorted(set(b.get("config") or {}) & set(ad["own"]))
    if own:
        raise ValueError(f"{fam.get('name')}: bench.config may not set {own} (the backend sets them)")
    if ad["render_key"] and C.render_mode(fam) not in C.RENDER_MODES:
        raise ValueError(f"{fam.get('name')}: bench.render {C.render_mode(fam)!r} is not one of {C.RENDER_MODES}")
    if P.seeded(ad):
        P.check_seeded(fam, ad)


def load(name: str) -> dict:
    fam = C.load_family(name)
    if fam.get("bench") and P.seeded(
        adapter(fam)
    ):  # no practice files: the sandbox and practice_dir read a directory that holds none
        fam.setdefault("practice_dir", "outputs/_no_practice_files")
    check_bench(fam)
    return fam


def _fam(family) -> dict:
    return load(family) if isinstance(family, str) else family


def id_refusal(fam: dict, eid: str, letters: str = GATE_LETTERS) -> str | None:
    """Why an episode id is not one this family may name or this call may run (None: fine; bench_pools.id_refusal)."""
    return P.id_refusal(adapter(fam), eid, letters)


def draw(fam: dict, pool: str, out, n: int | None = None, tasks: list[str] | None = None) -> dict:
    """The pool's files for the loop (bench_pools.draw)."""
    fam = _fam(fam)
    return P.draw(fam, adapter(fam), pool, out, n, tasks)


practice_dir = P.practice_dir
hidden_ids = P.hidden_ids
BaseWriteRefused = C.BaseWriteRefused


def bed_root(fam: dict) -> Path:
    return C.bed_root(fam)


def base_rel(fam: dict) -> str:
    return adapter(fam)["seed"]


def base_dir(fam: dict) -> Path:
    """The seed package: a read-only source. Nothing writes or deletes inside it (guard_write); the working package is a
    loop-owned copy (package_dir)."""
    b = (bed_root(fam) / base_rel(fam)).resolve()
    C.PROTECTED.add(b)
    return b


def package_dir(fam: dict) -> Path:
    base_dir(fam)
    return C.package_root(fam)


def ensure_package(fam: dict) -> Path:
    return C.ensure_package(base_dir(fam), package_dir(fam))


def guard_write(family, path) -> None:
    """Raise BaseWriteRefused when `path` is the family's base directory or inside it (or is a parent of it)."""
    if family is not None:
        base_dir(_fam(family))
    C.guard_write(path)


# --------------------------------------------------------------------------------------------- switches, files
def bench_options_args(fam: dict) -> list[str]:
    return [f"{k}={C.setting_arg(v)}" for k, v in (bench(fam).get("options") or {}).items()]


def bench_config_args(fam: dict) -> list[str]:
    return [f"{k}={C.setting_arg(v)}" for k, v in (bench(fam).get("config") or {}).items()]


_RESOLVED: dict = {}


def bench_record(fam: dict) -> dict:
    """The resolved switches of the family (`BenchOptions` record: options, signature, ...), asked of the adapter itself
    (`python -m locomani.bench show --bench <name>`, so the loop's python needs no locomani.bench), kept per process."""
    b = bench(fam)
    name = b.get("name", "simple")
    key = (b["root"], b.get("python") or PY, name, tuple(bench_options_args(fam)))
    if key not in _RESOLVED:
        argv = [key[1], "-m", "locomani.bench", "show", "--bench", name]
        for kv in key[3]:
            argv += ["--bench-set", kv]
        r = subprocess.run(argv, capture_output=True, text=True, cwd=b["root"], env=dict(os.environ, PYTHONPATH="src"))
        if r.returncode:
            raise ValueError(
                f"the bench switches {list(key[3])} do not resolve: {(r.stderr or r.stdout).strip()[-300:]}"
            )
        text = r.stdout
        _RESOLVED[key] = json.JSONDecoder().raw_decode(text[text.index("{") :])[0]
    return _RESOLVED[key]


def bench_signature(fam: dict) -> str:
    return bench_record(fam)["signature"]


def _globs(fam: dict, globs) -> list[str]:
    root = Path(bench(fam)["root"])
    out: list[str] = []
    for g in globs:
        out += sorted(p.relative_to(root).as_posix() for p in root.glob(g) if p.is_file())
    return list(dict.fromkeys(out))


def bench_files(fam: dict) -> list[str]:
    """Names (below bench.root) of the adapter's files roles may read: the table's `files` and the block's own `files`."""
    return _globs(fam, [*adapter(fam)["files"], *(bench(fam).get("files") or [])])


def hash_only_files(fam: dict) -> list[str]:
    """What the signature hashes and roles do not see (the adapter's server, truth side, config)."""
    seen = set(bench_files(fam))
    return [f for f in _globs(fam, adapter(fam)["hash_files"]) if f not in seen]


bed_files = bench_files  # the files copied into a sandbox and hashed into the signature (roles may read them)


_COMMIT: dict = {}


def bench_commit(fam: dict) -> str:
    """The benchmark's git commit (the table's `benchmark` path below the bench root); "" when the adapter names none."""
    rel = adapter(fam).get("benchmark")
    if not rel:
        return ""
    p = str(Path(bench(fam)["root"]) / rel)
    if p not in _COMMIT:
        try:
            r = subprocess.run(["git", "-C", p, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
            _COMMIT[p] = r.stdout.strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            _COMMIT[p] = "unknown"
    return _COMMIT[p]


def signature(family, where: str = "probe") -> str:
    """sha1 over the adapter's files (read and hashed-only), the track, its fixed bed switches, the resolved switches
    (`BenchOptions.signature()`), the render marker and the benchmark's commit. Two switch sets never share a pool or a gate
    signature. (The empty second part is the old `base` slot: a bench family has none, and keeping it keeps every signature
    the loops already stored.)"""
    fam = _fam(family)
    check_bench(fam)
    ad = adapter(fam)
    h = hashlib.sha1()
    root = Path(bench(fam)["root"])
    for rel in [*bed_files(fam), *hash_only_files(fam)]:
        p = root / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    for part in (fam.get("track", ""), "", "official"):
        h.update(b"\0" + part.encode())
    if bed_fixed(fam):
        h.update(b"\0fixed" + json.dumps(bed_fixed(fam), sort_keys=True).encode())
    h.update(b"\0bench" + bench_signature(fam).encode())
    if ad["render_key"]:  # TODO 52: lazy with warm-up everywhere (gate and sealed included), or every step
        h.update(b"\0render_" + C.render_tag(fam).encode())
        if C.memo_on(fam, where == "gate"):
            h.update(b"\0memo")
    if ad.get("benchmark"):
        h.update(b"\0benchmark" + bench_commit(fam).encode())
    return h.hexdigest()[:10]


# ------------------------------------------------------------------------------------------ package, guard
def load_skills(root, name: str = "scene_skills", family=None):
    """The package at `root` checked for the keys a `set` may hold, by the adapter's rows module (LocoManiLab refuses a
    stance key outside its list)."""
    fam = _fam(family) if family is not None else {}
    mod = rows_module(fam) if bench(fam) else importlib.import_module("bench_simple")
    return mod.load_skills(root, name, family=fam if fam else None)


def run_guard(fam: dict, base: Path, cand: Path) -> str | None:
    """None when the adapter's static guard passes (or the candidate is the base); else its first violation."""
    if Path(base).resolve() == Path(cand).resolve():
        return None
    guard = Path(bench(fam)["root"]) / adapter(fam)["guard"]
    r = subprocess.run([sys.executable, "-I", str(guard), str(base), str(cand)], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except ValueError:
        return f"the guard gave no JSON (exit {r.returncode}): {(r.stderr or r.stdout)[-200:]}"
    if d.get("ok"):
        return None
    v0 = (d.get("violations") or ["unspecified"])[0]
    if isinstance(v0, dict):  # the bench guards' violations are {rule, file, detail}
        return f"{v0.get('rule')} {v0.get('file')}: {v0.get('detail')}"
    return str(v0)


def layout(family) -> dict:
    fam = _fam(family)
    rm = rows_module(fam)
    return dict(
        skills=ensure_package(fam),
        rules=C._abs(fam["rules"]),
        prompts=C._abs(fam["prompts"]),
        plan=None,
        leak=list(rm.LEAK),
        stub_file=adapter(fam).get(
            "stub_file", "pick.py"
        ),  # __init__.py is frozen for the guard: a stub draft appends its comment to an editable file
        class_suggest=dict(rm.CLASS_SUGGEST),
        caps={**DEFAULT_CAPS, **fam.get("caps", {})},
    )


# ----------------------------------------------------------------------------------------- the runner call
def scratch_cache(fam: dict) -> Path | None:
    """The look and grasp cache of the family (adapters that make one): in the scratch directory, outside every sandbox.
    The adapter keys the folders below it by its package files and switches."""
    if not adapter(fam)["cache_key"]:
        return None
    return C.scratch() / f"{bench(fam).get('name', 'simple')}_{fam['name']}_cache"


def prune_cache(root: Path | None, keep: int = 6) -> None:
    """Keep the `keep` newest key folders of the cache (a changed package makes a new one)."""
    if root is None or not root.is_dir():
        return
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in dirs[keep:]:
        shutil.rmtree(p, ignore_errors=True)


def bed_env(fam: dict, package: Path, run_dir: Path, full: bool = False) -> dict:
    """The environment of an adapter call: ours, without EXAM_OPEN and the old beds' variables (the adapter takes everything
    as arguments), plus the loop's fixed ones."""
    ad = adapter(fam)
    env = {k: v for k, v in os.environ.items() if k != "EXAM_OPEN" and not k.startswith(tuple(ad["strip_env"]))}
    env.update(NUMPY_MADVISE_HUGEPAGE="0", OMP_NUM_THREADS="2", MUJOCO_GL="egl", PYTHONPATH="src")
    if ad["tmpdir"]:
        env["TMPDIR"] = str(C.scratch())
    return env


def bench_sets(fam: dict, run_dir, full: bool) -> list[str]:
    """The `--set` values of a call: the backend's own keys first (kinds, the practice files, the run directory, the render
    flag, the look cache), then the family's `bench.config`. `full`: the gate and the sealed set read the full practice
    files; the render keys are the same in every round (lazy with warm-up renders, or every step by `bench.render`)."""
    ad = adapter(fam)
    pdir = practice_dir(fam, full)
    sets = [f"kinds={ad['kinds']}"] if ad["kinds"] else []
    sets += [*(f"{k}={pdir}" for k in ad["practice_keys"]), f"run_dir={run_dir}"]
    if seeded(fam):  # the pools this call opens (a diagnostic round: the practice pool only), one simulator at a time
        sets += [
            f"pools={GATE_LETTERS if full else PROBE_LETTERS}",
            f"lock_file={C.scratch()}/{bench(fam)['name']}.lock",
        ]
    if ad["render_key"]:
        sets += [
            *C.render_sets(fam),
            *C.memo_sets(fam, full, signature(fam, "probe") if C.memo_on(fam, full) else ""),
        ]
    if ad["cache_key"]:
        sets.append(f"{ad['cache_key']}={scratch_cache(fam)}")
    return [*sets, *bench_config_args(fam)]


def bench_argv(fam: dict, task: str, ids_file, skill: dict, stance, out, package, run_dir, full: bool) -> list[str]:
    """The adapter's command for one unit: `python -m <module> run <Task> ...` (run in the bench root with PYTHONPATH=src)."""
    argv = [bench(fam).get("python") or PY, "-m", adapter(fam)["module"], "run", task, "--ids", str(ids_file),
            "--setting", json.dumps(skill), "--out", str(out), "--skills", str(Path(package).resolve())]  # fmt: skip
    if stance:
        argv += ["--stance", json.dumps(stance)]
    for kv in bench_options_args(fam):
        argv += ["--bench-set", kv]
    for kv in bench_sets(fam, run_dir, full):
        argv += ["--set", kv]
    return argv


def run_unit(task: str, ids: list[str], package, setting: dict, out, family=None, env=None, log=None,
             stop=None, full: bool = False) -> int:  # fmt: skip
    """The one place where the adapter's episodes are run; rows go to the file `out`, the unit's folder (its parent) is the
    adapter's run_dir. The adapter takes a slot, makes the services, looks and grasps, runs the batch and stops what it
    started; a set `stop` takes its whole process tree down."""
    fam = _fam(family) if family is not None else {}
    env = env if env is not None else bed_env(fam, Path(package), Path(out).parent)
    ids_file = Path(out).with_name(f"ids_{task}.txt")
    ids_file.write_text("\n".join(ids) + "\n")
    skill, stance = C.split_set(setting or {})
    argv = bench_argv(fam, task, ids_file, skill, stance, out, package, Path(out).parent, full)
    return C.run_command(argv, env, log or subprocess.DEVNULL, stop, Path(bench(fam)["root"]), tree=True)


def slots_of(fam: dict) -> int:
    """Work units side by side: the family's `official_slots`, else as many as the config's `slots`, else the adapter's."""
    if fam.get("official_slots"):
        return int(fam["official_slots"])
    cfg = (bench(fam).get("config") or {}).get("slots")
    if cfg:
        return len(cfg) if isinstance(cfg, (list, tuple)) else len(str(cfg).split(","))
    return int(adapter(fam)["units"])


# ------------------------------------------------------------------------------------------------- run_jobs
def to_round(rec: dict, label: str, skills_hash: str, set_: dict, signature: str, fam: dict | None = None) -> dict:
    """A bed record as the loop's round dict, by the adapter's rows module (the record's raw form stays under the module's
    own key), plus the row's resolved switches (`bench`)."""
    mod = rows_module(fam) if fam else importlib.import_module("bench_simple")
    out = mod.to_round(rec, label, skills_hash, set_, signature)
    if isinstance(rec.get("bench"), dict) and rec["bench"].get("signature"):
        out["bench"] = rec["bench"]["signature"]
    return out


def _blank(fam: dict, label: str, why: str, sh: str, setting: dict, sig: str, cls: str = "guard") -> dict:
    mod = rows_module(fam)
    return R.blank(label, why, sh, setting, sig, cls, mod.ROW_KEY, mod.CONTROLLER)


def run_jobs(jobs, family, pool, procs, log, gpus, stop=None, where: str = "probe") -> dict:
    """Run the jobs' missing rounds; returns {job index: labels still missing} as batch.run_jobs does. A job has `skills`
    (the package directory), `out`, `overrides` (the setting) and `labels` (None = the whole pool). Each job is guarded
    first; one that fails gets ok False rounds ("guard: <violation>") and runs nothing. One adapter serves probes and the
    gate; `where` selects the practice files (the full ones for "gate"); the render mode is the family's in every round."""
    fam = _fam(family)
    check_bench(fam)
    ad = adapter(fam)
    pool, log = Path(pool), Path(log)
    sig = signature(fam, where)
    every = sorted(p.stem for p in pool.glob("*.json") if not p.name.startswith("_"))
    log.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    base = package_dir(fam)
    tmp = C.scratch()
    fixed = bed_fixed(fam)
    slots = slots_of(fam)
    prune_cache(scratch_cache(fam))
    with log.open("a") as fh:
        for k, job in enumerate(jobs):
            out, setting = Path(job.out), dict(job.overrides or {})
            ran = {**setting, **fixed}  # what the bed runs: the fixed switches over the job's set ([bed-fixed])
            out.mkdir(parents=True, exist_ok=True)
            todo = [x for x in (job.labels or every) if not C.current(out / f"{x}.json", sig)]
            for x in todo:  # a round made under another signature is kept aside, not read again
                f = out / f"{x}.json"
                if f.exists():
                    f.rename(f.with_name(f"_old_{x}.json"))
            if not todo or (stop is not None and stop.is_set()):
                result[k] = todo
                continue
            sh = skills_hash(Path(job.skills))
            bad = run_guard(fam, base, Path(job.skills))
            if bad is not None:
                for x in todo:
                    C.write_round(out / f"{x}.json", _blank(fam, x, f"guard: {bad}", sh, setting, sig))
                result[k] = []
                continue
            by = C.ids_by_task(pool, todo)
            pdir = practice_dir(fam, where == "gate")  # the practice files this call is given (a sandbox: blanked rows)
            if P.seeded(ad):  # no practice files: the pools this call opens (id_refusal)
                letters = GATE_LETTERS if where == "gate" else PROBE_LETTERS
                gone = [x for rows in by.values() for x, eid in rows if P.id_refusal(ad, eid, letters)]
            else:
                gone = [x for rows in by.values() for x, eid in rows if not P.in_practice(pdir, eid)]
            for x in gone:  # an id that is blank or missing there is not an episode this call may run
                C.write_round(
                    out / f"{x}.json", _blank(fam, x, "not a diagnostic episode", sh, setting, sig, "refused")
                )
            by = {t: [(x, e) for x, e in rows if x not in gone] for t, rows in by.items()}
            by = {t: rows for t, rows in by.items() if rows}
            if not by:
                result[k] = []
                continue
            tmp.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix=f"bench_{fam['name']}_{k}_", dir=tmp))
            env = bed_env(fam, Path(job.skills), run_dir, full=where == "gate")
            by_id = {eid: x for rows in by.values() for x, eid in rows}

            def unit(u, job=job, setting=ran, env=env, run_dir=run_dir):
                i, (task, items) = u
                d = run_dir / f"u{i:02d}"  # a run directory per unit
                d.mkdir()
                return run_unit(task, [e for _, e in items], job.skills, setting, d / f"rows_{task}.jsonl",
                                family=fam, env=env, log=fh, stop=stop, full=where == "gate")  # fmt: skip

            with ThreadPoolExecutor(max_workers=slots) as ex:
                list(ex.map(unit, enumerate(C.units(by, slots))))
            for f in sorted(run_dir.glob("u*/rows_*.jsonl")):
                for line in f.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    x = by_id.get(
                        rec.get("episode_id") or rec.get("id")
                    )  # SIMPLE and LocoManiLab rows: episode_id; seed pools: id
                    if x is not None:
                        C.write_round(out / f"{x}.json", to_round(rec, x, sh, setting, sig, fam))
            result[k] = [x for x in todo if not C.current(out / f"{x}.json", sig)]
            if not result[k]:
                shutil.rmtree(run_dir, ignore_errors=True)  # (the cache is outside it)
    return result


# ----------------------------------------------------------------------------------------------------- cards
def rows_module_of(rounds: list[dict]) -> str:
    """The rows module a round list belongs to: a LocoManiLab round keeps its raw record under `lml`, a seed-pool adapter's
    under `seeded`, SIMPLE's under `simple` (what old ledgers hold)."""
    if any("seeded" in r for r in rounds):
        return "bench_seeded"
    return "bench_lml" if any("lml" in r for r in rounds) else "bench_simple"


def card_text(r: dict) -> list[str]:
    lines = importlib.import_module(rows_module_of([r])).card_text(r)
    if r.get("bench") and not any(x.startswith("bench:") for x in lines):
        lines.insert(1, f"bench: {r['bench']}")
    return lines


def cards(rounds: list[dict], out) -> Path:
    """Failure cards: cards.md with the resolved switches in its header (and on every card) when the rounds carry them."""
    return R.cards(rounds, out, importlib.import_module(rows_module_of(rounds)).card_text)


# --------------------------------------------------------------------------------------------------- sandbox
def sandbox(dest, family) -> None:
    """Under `<dest>/<sandbox.dir>` the seed package, the adapter's files roles may read (read only), and the practice files
    with the hidden rows blanked; the full ones go to the hidden tree. The bench root stays where the family file says: a round
    in the sandbox runs on the real adapter. The adapter's forbidden paths are cut out of the linked third_party."""
    fam = _fam(family)
    C.sandbox(dest, fam, base=base_dir(fam), base_rel=base_rel(fam), files=bed_files(fam),
              forbidden=rows_module(fam).FORBIDDEN_PATHS)  # fmt: skip


def _sandbox_family(dest: Path) -> dict:
    for ff in sorted((dest / "scripts" / "rsi" / "families").glob("*.json")):
        try:
            d = json.loads(ff.read_text())
        except ValueError:
            continue
        if d.get("backend") == "bench":
            return d
    return {}


def audit(dest) -> list[str]:
    """What in the sandbox `dest` a role must not see (forbidden paths, practice rows of hidden ids that are not blank, leak
    patterns in text files). [] when clean. The family is the sandbox's own family file."""
    dest = Path(dest)
    d = _sandbox_family(dest)
    rm = rows_module(d)
    nest = dest / d.get("sandbox", {}).get("dir", "bed_readonly_copy")
    # the adapter's files and the seed package are the code the Engineer starts from; they name released ids in their own code
    allowed = (
        {str(nest / b) for b in bed_files(d)} | {str(p) for p in (nest / base_rel(d)).rglob("*.py")}
        if d.get("bench")
        else set()
    )
    return C.audit(dest, forbidden=rm.FORBIDDEN_PATHS, leak=rm.LEAK, nest_name=nest.name, allowed=allowed)
