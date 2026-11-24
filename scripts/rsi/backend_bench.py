"""The loop's ONE backend for every benchmark adapter (TODO 49.6b; 2026-11-24).

A family file with `"backend": "bench"` and a `"bench"` block runs every round (diagnostic, probe, draft, check half, gate,
sealed) through the adapter's own command, `python -m locomani.bench.<name> run <Task> ...`, from the bench root:

    "bench": {"name": "simple" | "locomanilab", "root": <a bench worktree>, "options": {BenchOptions overrides},
              "config": {the adapter's config fields, --set k=v; e.g. slots}, "python": optional, "files": optional globs}

What is generic here (written once): the runner's argv, the signature (the adapter's files, the resolved switches), the
adapter's static guard, the package layout and its parameters, the sandbox copy and audit, the practice files with blanked
rows, the work units and their slots, rows -> rounds, cards, and the render flags of the gate and the sealed set.

What is adapter specific is ONE table below (ADAPTERS): the module name, the seed and guard paths, which files are read and
hashed, the config keys the backend sets itself (practice files, kinds, run dir, the look cache) and the render flag, and the
module that knows the adapter's row schema (`flavor`: SIMPLE's rows, LocoManiLab's allow-listed rows). A new adapter is
one more entry. The sibling modules are reused for what is not about the adapter: backend_simple.py holds the pools,
the process handling and the sandbox walk; backend_lml.py (only a family of that flavor loads it) the LocoManiLab rows and
cards. Families without a `bench` block stay on backend_simple / backend_lml as before.
"""

from __future__ import annotations

import ast
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

import backend_simple as B  # noqa: E402

REPO = B.REPO
HIDDEN = B.HIDDEN
STANCE = B.STANCE
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
        render_key="lazy_render",  # the gate and the sealed set render every step: this is false there, else true
        cache_key=None,  # a config field for the look and grasp cache the backend places in its scratch directory
        own=("practice_dir", "run_dir", "lazy_render", "kinds"),  # config keys the backend sets itself
        strip_env=("SIMPLE_",),
        tmpdir=False,
        units=3,  # work units side by side when the family names no official_slots and no `slots`
        benchmark=None,  # a path below the bench root whose git commit enters the signature
        flavor="backend_simple",
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
        flavor="backend_lml",
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


def flavor(fam: dict):
    """The module that knows the adapter's rows: to_round, cards, the leak patterns and the failure classes."""
    return importlib.import_module(adapter(fam)["flavor"])


def check_bench(fam: dict) -> None:
    """A bench family runs every round on the adapter: a non-official probe bed is not ported, and the backend owns some of
    the adapter's --set keys."""
    b = bench(fam)
    if not b:
        raise ValueError(f'{fam.get("name")}: backend "bench" needs a "bench" block')
    if fam.get("probe_bed", "mirror") != "official":
        raise ValueError(f'{fam.get("name")}: the bench adapter has no mirror bed; set "probe_bed": "official"')
    ad = adapter(fam)
    if not b.get("root"):
        raise ValueError(f"{fam.get('name')}: bench.root is missing")
    own = sorted(set(b.get("config") or {}) & set(ad["own"]))
    if own:
        raise ValueError(f"{fam.get('name')}: bench.config may not set {own} (the backend sets them)")


def load(name: str) -> dict:
    fam = B.load(name)
    check_bench(fam)
    return fam


def _fam(family) -> dict:
    return load(family) if isinstance(family, str) else family


bed_root = B.bed_root
base_dir = B.base_dir
package_dir = B.package_dir
ensure_package = B.ensure_package
guard_write = B.guard_write
practice_dir = B.practice_dir
draw = B.draw
skills_hash = B.skills_hash
bed_fixed = B.bed_fixed
gate_env = B.gate_env


def base_rel(fam: dict) -> str:
    return adapter(fam)["seed"]


# --------------------------------------------------------------------------------------------- switches, files
def _setting_arg(v) -> str:
    return B._setting_arg(v)


def bench_options_args(fam: dict) -> list[str]:
    return [f"{k}={_setting_arg(v)}" for k, v in (bench(fam).get("options") or {}).items()]


def bench_config_args(fam: dict) -> list[str]:
    return [f"{k}={_setting_arg(v)}" for k, v in (bench(fam).get("config") or {}).items()]


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


def bed_files(fam: dict) -> list[str]:
    return [*bench_files(fam), *fam.get("bed_files", [])]


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


def on_official(fam: dict, where: str) -> bool:
    return True  # a bench family has no mirror (check_bench)


def signature(family, where: str = "probe") -> str:
    """sha1 over the adapter's files (read and hashed-only), the track, the base, the family's own `bed_env`, its fixed bed
    switches, the gate's `bed_env_gate`, the resolved switches (`BenchOptions.signature()`), the gate's render marker and the
    benchmark's commit. Two switch sets never share a pool or a gate signature. For SIMPLE this is backend_simple's, byte
    for byte (the family's `bed_files`, track and base come first, as there)."""
    fam = _fam(family)
    check_bench(fam)
    ad = adapter(fam)
    h = hashlib.sha1()
    root = Path(bench(fam)["root"])
    for rel in [*bed_files(fam), *hash_only_files(fam)]:
        p = root / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    for part in (fam.get("track", ""), str(fam.get("base", "")), "official"):
        h.update(b"\0" + part.encode())
    if fam.get("bed_env"):
        h.update(b"\0" + json.dumps(fam["bed_env"], sort_keys=True).encode())
    if bed_fixed(fam):
        h.update(b"\0fixed" + json.dumps(bed_fixed(fam), sort_keys=True).encode())
    if where == "gate" and gate_env(fam):
        h.update(b"\0gate_env" + json.dumps(gate_env(fam), sort_keys=True).encode())
    h.update(b"\0bench" + bench_signature(fam).encode())
    if where == "gate" and ad["render_key"]:  # the gate and the sealed set render every step, the rest lazily
        h.update(b"\0gate_render")
    if ad.get("benchmark"):
        h.update(b"\0benchmark" + bench_commit(fam).encode())
    return h.hexdigest()[:10]


# ------------------------------------------------------------------------------------------ package, guard
def _extra(root: Path) -> dict:
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
                        out[kw.arg] = B._literal(kw.value)
    return out


def package_parameters(root: Path) -> dict:
    """The package's parameters read from the source text (the loop's process never imports a package): skill.py's K_ORIG,
    K_CUT, PLACING, K_ART and ART_TASK keys (backend_simple.package_parameters) and `EXTRA` of __init__.py."""
    return {**B.package_parameters(root), **_extra(root)}


def load_skills(root, name: str = "scene_skills", family=None):
    """The package at `root` checked for the keys a `set` may hold, by the adapter's flavor (LocoManiLab refuses a
    stance key outside its list)."""
    fam = _fam(family) if family is not None else {}
    mod = flavor(fam) if bench(fam) else B
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
    fl = flavor(fam)
    plan = fam.get("plan")
    return dict(
        skills=ensure_package(fam),
        rules=B._abs(fam["rules"]),
        prompts=B._abs(fam["prompts"]),
        plan=B._abs(plan) if plan else None,
        leak=list(fl.LEAK),
        stub_file="pick.py",  # __init__.py is frozen for the guard, so a stub draft appends its comment to an editable file
        class_suggest=dict(fl.CLASS_SUGGEST),
        caps={**B.FALLBACK_CAPS, **fam.get("caps", {})},
    )


# ----------------------------------------------------------------------------------------- the runner call
def scratch_cache(fam: dict) -> Path | None:
    """The look and grasp cache of the family (adapters that make one): in the scratch directory, outside every sandbox.
    The adapter keys the folders below it by its package files and switches."""
    if not adapter(fam)["cache_key"]:
        return None
    return B._scratch() / f"{bench(fam).get('name', 'simple')}_{fam['name']}_cache"


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
        env["TMPDIR"] = str(B._scratch())
    return env


def bench_sets(fam: dict, run_dir, full: bool) -> list[str]:
    """The `--set` values of a call: the backend's own keys first (kinds, the practice files, the run directory, the render
    flag, the look cache), then the family's `bench.config`. `full`: the gate and the sealed set read the full practice
    files and render every step; everything else the family's visible files, lazily."""
    ad = adapter(fam)
    pdir = practice_dir(fam, full)
    sets = [f"kinds={ad['kinds']}", *(f"{k}={pdir}" for k in ad["practice_keys"]), f"run_dir={run_dir}"]
    if ad["render_key"]:
        sets.append(f"{ad['render_key']}={'false' if full else 'true'}")
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


def official_runner(task: str, ids: list[str], package, setting: dict, out, family=None, env=None, log=None,
                    stop=None, procs: int = 1, gpus: str = "0", full: bool = False) -> int:  # fmt: skip
    """The one place where the adapter's episodes are run; rows go to the file `out`, the unit's folder (its parent) is the
    adapter's run_dir. The adapter takes a slot, makes the services, looks and grasps, runs the batch and stops what it
    started; a set `stop` takes its whole process tree down."""
    fam = _fam(family) if family is not None else {}
    env = env if env is not None else bed_env(fam, Path(package), Path(out).parent)
    ids_file = Path(out).with_name(f"ids_{task}.txt")
    ids_file.write_text("\n".join(ids) + "\n")
    skill, stance = B.split_set(setting or {})
    argv = bench_argv(fam, task, ids_file, skill, stance, out, package, Path(out).parent, full)
    return B._exec(argv, env, log or subprocess.DEVNULL, stop, Path(bench(fam)["root"]), tree=True)


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
    """A bed record as the loop's round dict, by the adapter's flavor (the record's raw form stays under the flavor's own
    key), plus the row's resolved switches (`bench`)."""
    mod = flavor(fam) if fam else B
    out = mod.to_round(rec, label, skills_hash, set_, signature)
    if isinstance(rec.get("bench"), dict) and rec["bench"].get("signature"):
        out["bench"] = rec["bench"]["signature"]
    return out


def _blank(fam: dict, label: str, why: str, sh: str, setting: dict, sig: str, cls: str = "guard") -> dict:
    mod = flavor(fam)
    return getattr(mod, "_blank", B._blank)(label, why, sh, setting, sig, cls)


def run_jobs(jobs, family, pool, procs, log, gpus, stop=None, where: str = "probe") -> dict:
    """Run the jobs' missing rounds; returns {job index: labels still missing} as batch.run_jobs does. A job has `skills`
    (the package directory), `out`, `overrides` (the setting) and `labels` (None = the whole pool). Each job is guarded
    first; one that fails gets ok False rounds ("guard: <violation>") and runs nothing. One adapter serves probes and the
    gate; `where` selects the practice files and the render flag (the full files and every step for "gate")."""
    fam = _fam(family)
    check_bench(fam)
    pool, log = Path(pool), Path(log)
    sig = signature(fam, where)
    every = sorted(p.stem for p in pool.glob("*.json") if not p.name.startswith("_"))
    log.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    base = package_dir(fam)
    tmp = B._scratch()
    fixed = bed_fixed(fam)
    slots = slots_of(fam)
    prune_cache(scratch_cache(fam))
    with log.open("a") as fh:
        for k, job in enumerate(jobs):
            out, setting = Path(job.out), dict(job.overrides or {})
            ran = {**setting, **fixed}  # what the bed runs: the fixed switches over the job's set (rule 46)
            out.mkdir(parents=True, exist_ok=True)
            todo = [x for x in (job.labels or every) if not B._current(out / f"{x}.json", sig)]
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
                    B._write(out / f"{x}.json", _blank(fam, x, f"guard: {bad}", sh, setting, sig))
                result[k] = []
                continue
            by = B._ids_by_task(pool, todo)
            pdir = practice_dir(fam, where == "gate")  # the practice files this call is given (a sandbox: blanked rows)
            gone = [x for rows in by.values() for x, eid in rows if not B._in_practice(pdir, eid)]
            for x in gone:  # an id that is blank or missing there is not an episode this call may run
                B._write(out / f"{x}.json", _blank(fam, x, "not a diagnostic episode", sh, setting, sig, "refused"))
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
                return official_runner(task, [e for _, e in items], job.skills, setting, d / f"rows_{task}.jsonl",
                                       family=fam, env=env, log=fh, stop=stop, procs=procs, gpus=gpus,
                                       full=where == "gate")  # fmt: skip

            with ThreadPoolExecutor(max_workers=slots) as ex:
                list(ex.map(unit, enumerate(B._units(by, slots))))
            for f in sorted(run_dir.glob("u*/rows_*.jsonl")):
                for line in f.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    x = by_id.get(rec.get("episode_id"))
                    if x is not None:
                        B._write(out / f"{x}.json", to_round(rec, x, sh, setting, sig, fam))
            result[k] = [x for x in todo if not B._current(out / f"{x}.json", sig)]
            if not result[k]:
                shutil.rmtree(run_dir, ignore_errors=True)  # (the cache is outside it)
    return result


# ----------------------------------------------------------------------------------------------------- cards
def _fam_of_rounds(rounds: list[dict]) -> str:
    """The flavor a round list belongs to: a LocoManiLab round keeps its raw record under `lml`, SIMPLE's under `simple`."""
    return "backend_lml" if any("lml" in r for r in rounds) else "backend_simple"


def card_text(r: dict) -> list[str]:
    mod = importlib.import_module("backend_lml" if "lml" in r else "backend_simple")
    lines = mod.card_text(r)
    if r.get("bench") and not any(x.startswith("bench:") for x in lines):
        lines.insert(1, f"bench: {r['bench']}")
    return lines


def cards(rounds: list[dict], out) -> Path:
    """Failure cards: the flavor's cards.md, with the resolved switches in its header (and on every card) when the rounds
    carry them."""
    f = importlib.import_module(_fam_of_rounds(rounds)).cards(rounds, out)
    benches = sorted({r["bench"] for r in rounds if r.get("bench")})
    text = f.read_text().splitlines()
    if benches and not any(x.startswith("bench:") for x in text[:4]):
        text.insert(2, "bench: " + "; ".join(benches))
        f.write_text("\n".join(text))
    return f


# --------------------------------------------------------------------------------------------------- sandbox
def sandbox(dest, family) -> None:
    """backend_simple.sandbox with the flavor's forbidden paths: under `<dest>/<sandbox.dir>` the seed package, the adapter's
    files roles may read (read only), and the practice files with the hidden rows blanked; the full ones go to the hidden
    tree. The bench root stays where the family file says: a round in the sandbox runs on the real adapter."""
    fam = _fam(family)
    B.sandbox(dest, fam, forbidden=flavor(fam).FORBIDDEN_PATHS)


def audit(dest) -> list[str]:
    """What in the sandbox `dest` a role must not see (forbidden paths, practice rows of hidden ids that are not blank, leak
    patterns in text files). [] when clean. The family is the sandbox's own family file."""
    fl = flavor(B._sandbox_family(Path(dest), "bench"))
    return B.audit(dest, forbidden=fl.FORBIDDEN_PATHS, leak=fl.LEAK, backend="bench")
