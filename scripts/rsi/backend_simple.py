"""The SIMPLE-benchmark backend of the RSI loop (seam plan B, 2026-11-19).

A family file with `"backend": "simple"` makes the loop run the SIMPLE beds (the other worktree's
scripts/experiments/escalation_20261116) instead of the scene runner. This module holds everything that differs:

- pools: practice episodes `<Task>/Q/<line>` (line = 0-based position in outputs/.../practice2/<Task>.jsonl, level in
  the row's `src.level`). The diagnostic pool is fixed and visible: the first 10 lines of each level. The remaining
  lines (all of them hidden) are split by a hidden seed into "accept" and "sealed", the same number per level; the
  seed is `pools.accept.seed` in the family file, which sandbox.py moves to the hidden seeds.json;
- run_jobs: one bed process per job (mirror bed for every probe, official bed for the gate), guarded first;
- to_round: a bed record as the loop's round dict (plan table C); cards: failure cards for the Reviewer;
- layout, sandbox, audit: what the Engineer may edit and read, and a check that nothing hidden is in the sandbox.

Pool files are `<Task>_s<n>.json` = {"task", "episode_id"} with n the practice line, so the loop's label parsing works
unchanged. Nothing here starts a simulator itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HIDDEN = "outputs/scene_rsi_20261016_hidden"
LEVELS = (0, 1, 2)
DIAG_PER_LEVEL = 10
PER_LEVEL = 50
SUCCESS_RULE_FALLBACK = 2
CLASSES = ("error", "timeout", "fall", "perceive", "deliver", "grasp", "short", "collide", "place", "comply")
# the phase a failure class is blamed on (plan table C); a class not listed, or a phase the record lacks: the last one
BLAME = {"deliver": "reach_grasp", "grasp": "lift", "place": "release", "collide": "close", "perceive": "look"}
# the articulated tasks (family simple_art): the record has `art`; their phases are walk, reach, close, move, release
BLAME_ART = {"deliver": "reach", "grasp": "close", "short": "move", "collide": "move"}
FALLBACK_CAPS = {"claims": False, "design_probes": False, "diag_redraw": False, "sentinel_exact": False,
                 "min_net": 3, "repeat": 1, "p_max": 0.05, "sentinel_flips": 1, "probe_scope": True,
                 # TODO 48.17 (2026-11-23, loop.py's block after diag_incumbent; rules 3, 25, 33, 47)
                 "check_half": True, "check_min_net": 1, "check_stop": 2, "stage1_first": True,
                 "probe_parallel": True, "tune_max": 2, "role_rerun": True,
                 # TODO 48.19 (user 2026-11-23, "三条都加上"; rules 25, 47, 49)
                 "task_regress": 2, "sealed_at_stop": True}  # fmt: skip
CLASS_SUGGEST = {
    "error": "the skill or the bed raised: read the error text, fix the code path before tuning anything",
    "timeout": "the check passed after the step limit: shorten the slow phases (walk, settle, search), not the grasp",
    "fall": "the robot fell: slower or shorter moves, a wider stance, less arm reach while the feet move",
    "perceive": "the target or the basket was not found: look from another place or angle before giving up",
    "deliver": "the palm ended far from the grasp: check the reach target, the grasp choice and the arm's error",
    "grasp": "the fingers closed and the object did not come up or was lost: grip width, force, palm position",
    "short": "the part moved but not far enough: keep the hand on it and push or pull through to the threshold",
    "collide": "the target was pushed or tipped before the fingers closed: approach path, clearance, entry speed",
    "place": "carried and released, but it did not stay in the basket: release point and height, hand open timing",
    "comply": "the environment's check passed but not what the instruction says: do what the sentence asks",
}
LEAK = [
    r"\bL[0-9]/e[0-9]+",  # a released evaluation episode id
    r"\bLhard\b",
    r"EXAM_OPEN",
    r"pool_(?:accept|sealed)",  # the hidden pools' directory names
    r"scene_rsi_20261016_hidden",
    r"outputs/simple_eval_20261112|simple_eval_20261112/x\b|dr-level-(?:[0-9]|hard)",  # the released episodes, not the code folder of that name
    r"escalation_20261116/official/(?:exam|rehearsal|exam_aborted|ho|q_seed|videos)",  # exam and hand-tuning outputs of the bed tree
    r"escalation_20261116/(?:a|b|practice|comply)/",
    r"locomanipulation-esc/outputs",  # the bed tree's outputs by absolute path: the full practice files, clouds of hidden episodes
    r"best_official|seed_pkg_removed|teleop_wbc",
    r"SIMPLE/data/(?:teleop_|mp_full|evals_|replay_)",  # the benchmark's evaluation-style and demonstration sets
    r"handoff/(?:escalation|benchmarks)",
]
# Level 2: a stance is given as `set` keys with this prefix and reaches the bed as its --stance argument
# (d, t, yaw for the picking stance, d2, t2, yaw2 for the placing stance, here: no walk; see skill_simple.py's header).
STANCE = "stance_"
PROTECTED: set[Path] = set()  # the seed packages of the families this process has seen (base_dir registers them)


def split_set(setting: dict) -> tuple[dict, dict | None]:
    """(the skill's parameter overrides, the stance override or None) of a job's `set`."""
    skill = {k: v for k, v in setting.items() if not k.startswith(STANCE)}
    stance = {k[len(STANCE) :]: v for k, v in setting.items() if k.startswith(STANCE)}
    return skill, (stance or None)


# What an Engineer-visible tree must not hold (relative to the sandbox root; `*` globs)
FORBIDDEN_PATHS = (
    "outputs/simple_eval_20261112",
    "outputs/escalation_20261116/official/exam",
    "outputs/escalation_20261116/official/rehearsal",
    "outputs/escalation_20261116/official/exam_aborted_0519",
    "outputs/escalation_20261116/a",
    "outputs/escalation_20261116/b",
    "outputs/escalation_20261116/practice",
    "third_party/SIMPLE/data/teleop_wbc",
    "third_party/SIMPLE/data/teleop_full",  # the benchmark's demonstration sets (100 episodes a task) and replays
    "third_party/SIMPLE/data/teleop_fix",
    "third_party/SIMPLE/data/mp_full",
    "third_party/SIMPLE/data/evals_wbc_replay",
    "third_party/SIMPLE/data/replay_decoupled_wbc",
    "docs/handoff/escalation-*.md",
    "docs/handoff/benchmarks-*.md",
)
SCAN_MAX_BYTES = 2_000_000


# ---------------------------------------------------------------------------------------------------- family
def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _fam(family) -> dict:
    return load(family) if isinstance(family, str) else family


# ------------------------------------------------------------------------------------------ bench adapter
# A family file with a "bench" block runs on the bench adapter (`python -m locomani.bench.simple run`, TODO 49.6a) instead of
# the esc bed's scripts: {"name": "simple", "root": <the bench worktree>, "options": {BenchOptions overrides},
# "config": {SimpleConfig --set overrides}}. Without the block nothing below is used.
BENCH_PY = "/venv/main/bin/python"  # runs the adapter's `run` (the bed itself starts under the sim venv, the adapter's own config)
BENCH_SEED = "src/locomani/bench/simple/seed"
BENCH_GUARD = "src/locomani/bench/simple/guard.py"
BENCH_FILES = (  # the adapter's code a round depends on (globs below the bench root): the signature and the read-only copy
    "src/locomani/bench/options.py",
    "src/locomani/bench/simple/*.py",
    "src/locomani/perception/*.py",
    "src/locomani/backends/scalebfm/*.py",
)
BENCH_OWN = (
    "practice_dir",
    "run_dir",
    "lazy_render",
    "lazy_warm",
    "lazy_subframes",
    "memo",
    "kinds",
)  # --set keys the backend sets itself

# TODO 52 (2026-11-25): every round renders the cameras lazily with LAZY_WARM warm-up renders per look, the gate and the sealed set
# included (outputs/fast_rounds_20261125: same success rates and per-start agreement as rendering every step, 2.5 x cheaper).
# `bench.render = "every_step"` in the family is the explicit every-step mode for official benchmark / exam reporting.
LAZY_WARM = 48
RENDER_MODES = ("lazy", "every_step")


def render_mode(fam: dict) -> str:
    return (bench(fam) or {}).get("render", "lazy")


def render_sets(fam: dict) -> list[str]:
    """The adapter's --set values for how the cameras render: lazily with warm-up renders (default), or every step."""
    if render_mode(fam) == "every_step":
        return ["lazy_render=false"]
    return ["lazy_render=true", f"lazy_warm={LAZY_WARM}"]


def memo_sets(fam: dict, full: bool) -> list[str]:
    """TODO 52 mechanism 2: `bench.memo = true` turns on replay from saved states (memo.py) in the diagnostic, probe, sweep and draft
    rounds; the gate and the sealed set (`full`) always run every episode in full. Off by default: a round of one setting never
    visits a state twice, and the saving needs the variants of a start to run in one process (see docs/handoff/fast-rounds-20261125.md)."""
    return ["memo=true"] if (bench(fam) or {}).get("memo") and not full else []


def render_tag(fam: dict) -> str:
    """What enters the bed signature: two render modes never share a pool or a gate signature."""
    return "every_step" if render_mode(fam) == "every_step" else f"lazy{LAZY_WARM}"


def bench(fam: dict) -> dict | None:
    """The family's "bench" block, or None (the esc bed)."""
    return fam.get("bench") or None


def check_bench(fam: dict) -> None:
    """A bench family runs every round on the adapter: a non-official probe bed is not ported, and the backend owns four
    of the adapter's --set keys."""
    b = bench(fam)
    if not b:
        return
    if fam.get("probe_bed", "mirror") != "official":
        raise ValueError(f'{fam.get("name")}: the bench adapter has no mirror bed; set "probe_bed": "official"')
    if b.get("name", "simple") != "simple":
        raise ValueError(f'{fam.get("name")}: bench.name {b.get("name")!r} is not ported (only "simple")')
    if not b.get("root"):
        raise ValueError(f"{fam.get('name')}: bench.root is missing")
    own = sorted(set(b.get("config") or {}) & set(BENCH_OWN))
    if own:
        raise ValueError(f"{fam.get('name')}: bench.config may not set {own} (the backend sets them)")
    if render_mode(fam) not in RENDER_MODES:
        raise ValueError(f"{fam.get('name')}: bench.render {render_mode(fam)!r} is not one of {RENDER_MODES}")


def _setting_arg(v) -> str:
    """A --set / --bench-set value the way config_base casts it back: bools lower case, lists comma separated."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def bench_options_args(fam: dict) -> list[str]:
    return [f"{k}={_setting_arg(v)}" for k, v in ((bench(fam) or {}).get("options") or {}).items()]


def bench_config_args(fam: dict) -> list[str]:
    return [f"{k}={_setting_arg(v)}" for k, v in ((bench(fam) or {}).get("config") or {}).items()]


_RESOLVED: dict = {}


def bench_record(fam: dict) -> dict:
    """The resolved switches of the family (`BenchOptions` record: options, signature, profile, ...), asked of the adapter
    itself (`python -m locomani.bench show`, so the loop's python needs no locomani.bench) and kept per process."""
    b = bench(fam)
    key = (b["root"], b.get("python") or BENCH_PY, tuple(bench_options_args(fam)))
    if key not in _RESOLVED:
        argv = [key[1], "-m", "locomani.bench", "show", "--bench", b.get("name", "simple")]
        for kv in key[2]:
            argv += ["--bench-set", kv]
        r = subprocess.run(argv, capture_output=True, text=True, cwd=b["root"], env=dict(os.environ, PYTHONPATH="src"))
        if r.returncode:
            raise ValueError(
                f"the bench switches {list(key[2])} do not resolve: {(r.stderr or r.stdout).strip()[-300:]}"
            )
        text = r.stdout
        _RESOLVED[key] = json.JSONDecoder().raw_decode(text[text.index("{") :])[0]
    return _RESOLVED[key]


def bench_signature(fam: dict) -> str:
    return bench_record(fam)["signature"]


def _generic(fam: dict):
    """backend_bench when the family names it (the one backend for every bench adapter, TODO 49.6b), else None: this
    module's own bench path below stays what it was for a family that says "backend": "simple"."""
    if fam.get("backend") == "bench":
        import backend_bench

        return backend_bench
    return None


def bench_files(fam: dict) -> list[str]:
    """Relative names (below bench.root) of the adapter's files: BENCH_FILES' globs, plus the block's own `files` globs."""
    if g := _generic(fam):
        return g.bench_files(fam)
    root = Path(bench(fam)["root"])
    out: list[str] = []
    for g in [*BENCH_FILES, *(bench(fam).get("files") or [])]:
        out += sorted(p.relative_to(root).as_posix() for p in root.glob(g) if p.is_file())
    return list(dict.fromkeys(out))


def bed_files(fam: dict) -> list[str]:
    """The family's bed files: the adapter's modules for a bench family, else the `bed_files` list."""
    return [*bench_files(fam), *fam.get("bed_files", [])] if bench(fam) else list(fam.get("bed_files", []))


def base_rel(fam: dict) -> str:
    if bench(fam) and (g := _generic(fam)):
        return g.base_rel(fam)
    return BENCH_SEED if bench(fam) else fam["base"]


def hidden_family_dir(name: str) -> Path:
    """The family's hidden directory (seeds.json, practice/). RSI_HIDDEN_FAMILY names a private copy of the two: the
    speculative gate (loop.gate_candidate) runs while the real directory is at mode 000."""
    e = os.environ.get("RSI_HIDDEN_FAMILY")
    return Path(e) if e else REPO / HIDDEN / name


def hidden_seeds(name: str) -> dict:
    """The seeds the sandbox keeps in the hidden tree; {} when there is none or the tree cannot be read (a role runs
    with the hidden tree at mode 000)."""
    p = hidden_family_dir(name) / "seeds.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (PermissionError, ValueError):
        return {}


def load(name: str) -> dict:
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
    if fam.get("backend") != "bench":  # backend_bench checks its own blocks
        check_bench(fam)
    return fam


def bed_root(fam: dict) -> Path:
    return Path(bench(fam)["root"]) if bench(fam) else _abs(fam["bed_root"])


def bed_dir(fam: dict) -> Path:
    return bed_root(fam) / fam["bed_dir"]


def base_dir(fam: dict) -> Path:
    """The seed package (family key `base`): a read-only source. Nothing in this module writes or deletes inside it
    (guard_write); the working package is a loop-owned copy (package_dir)."""
    b = (bed_root(fam) / base_rel(fam)).resolve()
    PROTECTED.add(b)
    return b


def package_dir(fam: dict) -> Path:
    """The loop's working package: a copy of the base under the loop's own state directory for the family
    (<outputs>/scene_rsi_20261016/<family>/package; RSI_PACKAGE_ROOT replaces <outputs>/scene_rsi_20261016 in a test).
    put_skills and the Engineers write here, never in the base."""
    root = Path(os.environ.get("RSI_PACKAGE_ROOT") or REPO / "outputs" / "scene_rsi_20261016")
    base_dir(fam)
    return root / fam["name"] / "package"


class BaseWriteRefused(PermissionError):
    """Something was about to write or delete inside the seed package."""


def guard_write(family, path) -> None:
    """Raise BaseWriteRefused when `path` is the family's base directory or inside it (or is a parent of it)."""
    if family is not None:
        base_dir(_fam(family))
    p = Path(path).resolve()
    for b in PROTECTED:
        if p == b or b in p.parents or p in b.parents:
            raise BaseWriteRefused(f"refused: {path} is, or contains, the seed package {b} (read-only source)")


def ensure_package(fam: dict) -> Path:
    """The working package, copied from the base on first use."""
    dst = package_dir(fam)
    if not dst.exists():
        guard_write(None, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base_dir(fam), dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dst


# ----------------------------------------------------------------------------------------------------- pools
def practice_dir(fam: dict, full: bool = True) -> Path:
    """The directory of the practice files. `full`: the hidden tree's copy when it can be listed (a sandbox), else
    the family file's `practice_dir` (which, in a sandbox, holds blanked rows)."""
    if full:
        h = hidden_family_dir(fam["name"]) / "practice"
        try:
            if h.is_dir() and any(h.iterdir()):
                return h
        except PermissionError:
            pass
    return _abs(fam["practice_dir"])


def _rows(path: Path) -> list[dict]:
    """The practice file's rows as {level (None for a blanked line)} in file order."""
    out = []
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line) if line.strip() else {}
        except ValueError:
            d = {}
        lv = (d.get("src") or {}).get("level") if isinstance(d, dict) else None
        out.append(dict(level=lv if lv in LEVELS else None))
    return out


def levels_of(fam: dict, task: str, full: bool = True) -> list[list[int]]:
    """For each level 0, 1, 2 the line numbers of that level in file order."""
    f = practice_dir(fam, full) / f"{task}.jsonl"
    by: list[list[int]] = [[], [], []]
    if f.exists():
        for i, r in enumerate(_rows(f)):
            if r["level"] is not None:
                by[r["level"]].append(i)
    return by


def diag_lines(fam: dict, task: str) -> list[int]:
    """The diagnostic pool's lines: the first 10 of each level in file order."""
    return sorted(i for lv in levels_of(fam, task) for i in lv[:DIAG_PER_LEVEL])


def split(fam: dict, task: str, seed: int) -> dict[str, list[int]]:
    """{"accept": lines, "sealed": lines}: each level's lines after the first 10, shuffled by the seed, half each.
    A pool's `per_level` in the family file keeps that many of its half per level (an official round costs minutes:
    the gate's pool is 10 per level; the lines left over stay hidden and unused, and can be added later)."""
    acc: list[int] = []
    sea: list[int] = []
    keep = {p: (fam.get("pools", {}).get(p) or {}).get("per_level") for p in ("accept", "sealed")}
    for k, lines in enumerate(levels_of(fam, task)):
        rest = lines[DIAG_PER_LEVEL:]
        random.Random(f"{seed}:{task}:{k}").shuffle(rest)
        h = len(rest) // 2
        acc += rest[:h][: keep["accept"]]
        sea += rest[h:][: keep["sealed"]]
    return dict(accept=sorted(acc), sealed=sorted(sea))


def hidden_ids(fam: dict) -> set[str]:
    """Every episode id outside the diagnostic pool (accept and sealed together), as `<Task>/Q/<line>`. Needs only the
    levels in the file, so it works on the full file and on the blanked one."""
    out = set()
    for task in fam["tasks"]:
        f = practice_dir(fam, True) / f"{task}.jsonl"
        if not f.exists():
            continue
        diag = set(diag_lines(fam, task))
        out |= {f"{task}/Q/{i}" for i in range(len(_rows(f))) if i not in diag}
    return out


def _pool_seed(fam: dict) -> int | None:
    a = fam["pools"].get("accept", {}).get("seed")
    s = fam["pools"].get("sealed", {}).get("seed")
    if a is not None and s is not None and a != s:
        raise SystemExit(f"{fam['name']}: accept and sealed must share one split seed ({a} != {s})")
    return a if a is not None else s


def draw(fam: dict, pool: str, out, n: int | None = None, tasks: list[str] | None = None) -> dict:
    """Write the pool's files `<Task>_s<line>.json` into `out`; returns {written, skipped, unavailable, next_seed}.
    `n` caps the total (tasks taken in turn); `tasks` restricts the tasks. A task whose file lacks the lines the pool
    needs (diag: 10 per level; accept / sealed: 50 per level, from the full file) is `unavailable` with the reason."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed = None
    ids = fam["pools"].get(pool, {}).get("ids") if pool != "diag" else None
    if pool != "diag" and not ids:
        seed = _pool_seed(fam)
        if seed is None:
            raise SystemExit(f"{fam['name']}: the {pool} pool's seed is blank and the hidden seeds.json has none")
    per_task: dict[str, list[int]] = {}
    unavailable: dict[str, str] = {}
    for task in fam["tasks"] if tasks is None else tasks:
        if ids:  # the pool is a list of episode ids (a hidden seeds.json entry {"ids": [...]})
            per_task[task] = sorted(int(x.rsplit("/", 1)[1]) for x in ids if x.startswith(f"{task}/Q/"))
            continue
        lv = levels_of(fam, task)
        need = DIAG_PER_LEVEL if pool == "diag" else PER_LEVEL
        short = [k for k in LEVELS if len(lv[k]) < need]
        if short:
            unavailable[task] = f"level {short[0]} has {len(lv[short[0]])} lines, the {pool} pool needs {need}"
            continue
        per_task[task] = diag_lines(fam, task) if pool == "diag" else split(fam, task, seed)[pool]
    order, k = [], 0  # one line of each task in turn, until `n` or nothing is left
    while any(per_task.values()) and (n is None or len(order) < n):
        t = list(per_task)[k % len(per_task)]
        k += 1
        if per_task[t]:
            order.append((t, per_task[t].pop(0)))
    for t, i in order:
        (out / f"{t}_s{i}.json").write_text(json.dumps(dict(task=t, episode_id=f"{t}/Q/{i}"), indent=1))
    summary = dict(family=fam["name"], pool=pool, written=len(order), skipped=[], unavailable=unavailable,
                   next_seed=(seed if seed is not None else 0))  # fmt: skip
    (out / "_pool.json").write_text(json.dumps(summary, indent=1))
    return summary


# ----------------------------------------------------------------------------------------------- records
def _sf(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def to_round(rec: dict, label: str, skills_hash: str, set_: dict, signature: str) -> dict:
    """A bed record as the loop's round dict (seam plan table C). Pure; the raw record is kept under `simple`."""
    comply = rec.get("comply")
    comply_ok = bool(comply.get("ok")) if isinstance(comply, dict) else False  # a missing check counts as false
    success = bool(rec.get("success"))
    ok = success and comply_ok
    loss = rec.get("loss") or ("error" if rec.get("error") else None)
    if ok:
        cls = None
    elif loss:
        cls = loss
    elif success:
        cls = "comply"
    else:
        cls = "unclassified"
    text = rec.get("why") or rec.get("error")
    if cls == "comply" and not text:
        text = (comply or {}).get("why") if isinstance(comply, dict) else None
        text = text or "no compliance check in the record"
    why = "" if ok else (f"{cls}: {text}" if text else str(cls))
    marks = rec.get("marks") or {}
    end = marks.get("end") or marks.get("before_open") or marks.get("after_lift") or {}
    rise = _sf(end.get("rise_cm"))
    plan_text = rec.get("why") or rec.get("error") or ""
    steps = _steps(rec, marks, cls, ok)
    task = rec.get("episode_id", label).split("/")[0] if rec.get("episode_id") else label.rsplit("_s", 1)[0]
    out = dict(
        label=label,
        task=task,
        ok=ok,
        why=why,
        fell=rec.get("fell") is not None,
        dropped=bool(not ok and rise is not None and rise < -15),
        plan_ok=not plan_text,
        plan_why=str(plan_text),
        error=rec.get("error"),
        steps=steps,
        t_end=rec.get("t_end"),
        failure_class=cls or "",
        skills=skills_hash,
        set=dict(set_ or {}),
        wall_s=rec.get("wall_s"),
        rule=_success_rule(),
        controller="scalebfm",
        motor=signature,
        simple=rec,
    )
    if isinstance(rec.get("bench"), dict) and rec["bench"].get("signature"):  # a bench adapter's row: its switches
        out["bench"] = rec["bench"]["signature"]
    return out


def _steps(rec: dict, marks: dict, cls: str | None, ok: bool) -> list[dict]:
    """One step per phase in order, t0 / t1 from the summed seconds; the phase the class names has ok False."""
    phases = rec.get("phases") or {}
    names = list(phases)
    blame = None
    if not ok and names:
        want = (BLAME_ART if isinstance(rec.get("art"), dict) else BLAME).get(cls or "")
        blame = want if want in phases else names[-1]
    facts = {
        "reach_grasp": dict(palm_err_cm=_sf(rec.get("palm_err_cm")), reach_ok=rec.get("reach_ok")),
        "close": {k: marks.get("before_close", {}).get(k) for k in ("moved_cm", "tilt_deg")},
        "lift": {k: marks.get("after_lift", {}).get(k) for k in ("rise_cm", "held")},
        "release": {k: marks.get("before_open", {}).get(k) for k in ("basket_cm", "held")},
    }
    art, touch = rec.get("art"), rec.get("art_touch")
    if isinstance(art, dict):  # articulated record: the facts come from art / art_touch, not from the pick marks
        touch = touch if isinstance(touch, dict) else {}
        facts = {
            "reach": dict(t_hand=_sf(touch.get("t_hand"))),
            "close": dict(t_hand=_sf(touch.get("t_hand"))),
            "move": dict(start_deg=_sf(art.get("start_deg")), best_deg=_sf(art.get("best_deg")),
                         need_deg=_sf(art.get("need_deg"))),
        }  # fmt: skip
    steps, t = [], 0.0
    for name in names:
        dt = float(_sf(phases[name]) or 0.0)
        s = dict(step=name, ok=name != blame, why="", t0=round(t, 3), t1=round(t + dt, 3))
        if name == blame:
            s["why"] = cls or ""
        s.update({k: v for k, v in facts.get(name, {}).items() if v is not None})
        steps.append(s)
        t += dt
    return steps


def _success_rule() -> int:
    try:
        sys.path[:0] = [str(HERE)]
        import batch

        return int(batch.SUCCESS_RULE)
    except Exception:
        return SUCCESS_RULE_FALLBACK
    finally:
        if str(HERE) in sys.path:
            sys.path.remove(str(HERE))


# ------------------------------------------------------------------------------------------------- cards
def _goal(task: str) -> str:
    for name in ("simple_q_tasks.jsonl", "simple_art_tasks.jsonl"):
        try:
            for x in (HERE / "families" / name).read_text().splitlines():
                if x.strip() and json.loads(x)["id"] == task:
                    return str(json.loads(x)["goal"])
        except (OSError, ValueError, KeyError):
            pass
    return ""


def _f(x, nd: int = 1) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def art_lines(raw: dict) -> list[str]:
    """An articulated record's joint story in plain words (one or two lines); [] for a record without `art`."""
    art = raw.get("art")
    if not isinstance(art, dict):
        return []
    touch = raw.get("art_touch") if isinstance(raw.get("art_touch"), dict) else {}
    sign = art.get("sign")
    way = "toward success" if sign not in (1, -1) else ("up" if sign > 0 else "down")
    sd, bd, nd = art.get("start_deg"), art.get("best_deg"), art.get("need_deg")
    line = (f"joint {art.get('joint')}: started at {_f(sd)} deg, furthest {_f(bd)} deg, needed {_f(nd)} deg "
            f"(success lies {way}), ended at {_f(art.get('end_deg'))} deg")  # fmt: skip
    if art.get("t_move") is not None:
        line += f"; it first moved 3 deg at {_f(art['t_move'])} s"
    out = [line]
    th, tb = touch.get("t_hand"), touch.get("t_body")
    parts = ["a hand first touched the moving part at " + f"{_f(th)} s" if th is not None
             else "no hand ever touched the moving part"]  # fmt: skip
    if th is not None and touch.get("hand_deg") is not None:
        parts.append(f"{_f(touch['hand_deg'])} deg were made with a hand on it")
    if tb is not None or touch.get("body_links"):
        links = ", ".join(touch.get("body_links") or []) or "a link"
        parts.append(f"a non-hand link touched it ({links})" + (f" from {_f(tb)} s" if tb is not None else ""))
    out.append("; ".join(parts))
    return out


def card_text(r: dict) -> list[str]:
    """One failed round as plain text (about a dozen lines)."""
    raw = r.get("simple") or {}
    instr = raw.get("instruction") or r.get("instruction") or _goal(r.get("task", ""))
    lines = [
        f"## {r['label']}  ({r.get('failure_class') or 'unclassified'})",
        f"task: {r.get('task')}",
        f"instruction: {instr}",
    ]
    if r.get("bench"):
        lines.append(f"bench: {r['bench']}")
    lines.append(f"outcome: {r.get('why') or 'failed'}; t_end {_f(r.get('t_end'))} s; fell {bool(r.get('fell'))}; dropped {bool(r.get('dropped'))}")  # fmt: skip
    bad = [s for s in r.get("steps") or [] if not s.get("ok", True)]
    if bad:
        s = bad[0]
        facts = ", ".join(f"{k} {_f(v)}" for k, v in s.items() if k not in ("step", "ok", "why", "t0", "t1"))
        lines.append(
            f"failed phase: {s['step']} from {_f(s['t0'])} to {_f(s['t1'])} s" + (f" ({facts})" if facts else "")
        )
    seq = " ".join(f"{s['step']}:{_f(s['t1'] - s['t0'])}" for s in r.get("steps") or [])
    if seq:
        lines.append(f"phase seconds: {seq}")
    lines += art_lines(raw)
    if raw.get("palm_err_cm") is not None:
        lines.append(f"palm error: {_f(raw['palm_err_cm'])} cm; reach_ok {raw.get('reach_ok')}")
    gd = raw.get("grasp_desc")
    if (
        isinstance(gd, list) and len(gd) >= 6
    ):  # rule 17: where on the object the chosen grasp sits, as the camera saw it
        lines.append(
            f"grasp chosen: palm {_f(100 * gd[0])} cm ahead, {_f(100 * gd[1])} cm left, {_f(100 * gd[2])} cm up of the"
            f" middle of the seen points; palm normal ({_f(gd[3], 2)}, {_f(gd[4], 2)}, {_f(gd[5], 2)}) in the same axes"
            + (f"; model score {_f(raw['score'], 2)}" if isinstance(raw.get("score"), (int, float)) else "")
        )
    for k, m in (raw.get("marks") or {}).items():
        if isinstance(m, dict):
            lines.append(f"mark {k}: " + ", ".join(f"{a} {_f(b)}" for a, b in m.items()))
    comply = raw.get("comply") if isinstance(raw.get("comply"), dict) else {}
    if comply.get("why") or raw.get("t_check") is not None:
        cf = raw.get("comply_facts") or {}
        keys = ("t_hand", "t_carry", "best_held_s", "t_right", "t_left_ok", "best_left_s", "t_container")
        facts = ", ".join(f"{k} {_f(cf[k])}" for k in keys if cf.get(k) is not None)
        first = raw.get("t_check") if raw.get("t_check") is not None else raw.get("t_success")
        lines.append(
            f"comply: {comply.get('ok')} -- {comply.get('why') or 'ok'}"
            + (f"; the environment's check first held at {_f(first)} s" if raw.get("success") else "")
            + (f"; contacts: {facts}" if facts else "")
        )
    st = r.get("set") or {}
    lines.append("settings that differ from the default: " + (", ".join(f"{k}={st[k]}" for k in sorted(st)) or "none"))
    return lines[:60]


def segments_module():
    """scripts/rsi/segments.py (TODO 48.17.6), whatever the importer's sys.path holds."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import segments

    return segments


def segment_header(rounds: list[dict]) -> list[str]:
    """The cards' segment table (rules 48): per task with losses, its steps in order with the losses blamed on each
    over the rounds that reached it, the hot step marked, the stable ones named."""
    sg = segments_module()
    body = sg.table_lines(sg.segment_table(rounds), rounds)
    if not body:
        return []
    return ["## Segments: losses blamed on each step / rounds that reached it (HOT: most of the task's losses; "
            "stable: reached often, never failed)", *body, ""]  # fmt: skip


def cards(rounds: list[dict], out) -> Path:
    """Failure cards for the Reviewer: <out>/cards.md (or `out` itself when it ends in .md). A two-line header with
    the wins per task, the segment table (rules 48), then one card per failed round."""
    out = Path(out)
    f = out if out.suffix == ".md" else out / "cards.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted({r["task"] for r in rounds})
    wins = {
        t: (sum(bool(r["ok"]) for r in rounds if r["task"] == t), sum(r["task"] == t for r in rounds)) for t in tasks
    }
    fails = sorted((r for r in rounds if not r["ok"]), key=lambda r: r["label"])
    lines = [f"# Failure cards: {sum(not r['ok'] for r in rounds)} of {len(rounds)} rounds failed",
             "wins per task: " + ("; ".join(f"{t} {a}/{b}" for t, (a, b) in wins.items()) or "none"), ""]  # fmt: skip
    benches = sorted({r["bench"] for r in rounds if r.get("bench")})
    if benches:
        lines.insert(2, "bench: " + "; ".join(benches))
    lines += segment_header(rounds)
    for r in fails:
        lines += card_text(r) + [""]
    f.write_text("\n".join(lines))
    return f


# ------------------------------------------------------------------------------------------ skill package
_PACKAGE_KEYS = (
    "K_ORIG",
    "K_CUT",
    "PLACING",
)  # the dict literals of skill_simple.py that make up the seed's parameters
_ART_KEYS = ("K_ART",)  # skill.py (the bench seed) also: the articulated tasks' `art_*` keys, K_ART's and ART_TASK's


def _literal(node):
    import ast

    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)


def package_parameters(root: Path) -> dict:
    """The package's parameter dict (skill.defaults) read from the source text: the keyword arguments of the dict(...)
    calls K_ORIG, K_CUT and PLACING. The bench seed's skill.py (the esc bed's skill_simple.py when there is none) also
    gives the articulated tasks' keys: K_ART's keywords, and every keyword of ART_TASK's per-task dict(...) not already
    there (its first value). The package imports the bed tree, so the loop's process never imports it."""
    import ast

    root = Path(root)
    new = (root / "skill.py").is_file()
    tree = ast.parse((root / ("skill.py" if new else "skill_simple.py")).read_text())
    out: dict = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & set(_PACKAGE_KEYS) or (new and names & set(_ART_KEYS)):
            for kw in node.value.keywords:
                if kw.arg:
                    out[kw.arg] = _literal(kw.value)
    if new:  # ART_TASK = dict(Task=dict(art_obj=..., ...), ...): the keys no task shares with K_ART come from here
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ART_TASK" for t in node.targets
            ):
                if isinstance(node.value, ast.Call):
                    for task in node.value.keywords:
                        if isinstance(task.value, ast.Call):
                            for kw in task.value.keywords:
                                if kw.arg:
                                    out.setdefault(kw.arg, _literal(kw.value))
    return out


class SkillPackage:
    """The scene protocol's package object for a SIMPLE package: `apply_set(pkg, overrides)` validates the keys and
    `parameters(pkg)` lists them, both without importing the package (scene_round.load_skills is for the scene's)."""

    def __init__(self, root: Path, bed_keys=(), fixed=()):
        self.root = Path(root)
        self.defaults = package_parameters(self.root)
        self.bed_keys = tuple(bed_keys)
        self.fixed = tuple(fixed)
        self.current: dict = {}

    def apply_set(self, _pkg, overrides: dict | None = None) -> dict:
        pinned = sorted(k for k in (overrides or {}) if k in self.fixed)
        if pinned:
            raise KeyError(
                f"parameters {pinned} are fixed by the bed for every round (rule 46); a set may not name them"
            )
        ok = set(self.defaults) | set(self.bed_keys)
        bad = sorted(k for k in (overrides or {}) if k not in ok and not k.startswith(STANCE))
        if bad:
            raise KeyError(f"unknown parameters {bad}")
        self.current = dict(overrides or {})
        return self.current

    def parameters(self, _pkg=None) -> dict:
        return {**self.defaults, **{k: v for k, v in self.current.items() if k in self.defaults}}


def load_skills(root, name: str = "scene_skills", family=None) -> SkillPackage:
    """The backend's `scene_round.load_skills`: the package at `root`, checked for the keys a `set` may hold (the
    package's, the `stance_` ones, and the family's `bed_keys`). `name` is accepted for the same call shape."""
    fam = _fam(family) if family is not None else {}
    return SkillPackage(Path(root), fam.get("bed_keys", []), bed_fixed(fam))


def bed_fixed(fam: dict) -> dict:
    """The family's fixed bed switches (`bed_fixed`, rule 46): corrections of the bed's interface to the benchmark,
    found by people, that every round of every role runs with (diagnostic rounds, probes, try.py, drafts, the gate,
    the sealed set). run_jobs lays them over each job's set; no set may name them. Keys starting with "_" are notes."""
    return {k: v for k, v in (fam.get("bed_fixed") or {}).items() if not k.startswith("_")}


# ----------------------------------------------------------------------------------------------- signature
def signature(family, where: str) -> str:
    """sha1 over the listed bed files, the track, the base, the family's own bed environment (`bed_env`; for the gate also
    `bed_env_gate`), its fixed
    bed switches (`bed_fixed`) and where (`gate`: the official bed, else the mirror)."""
    fam = _fam(family)
    check_bench(fam)
    h = hashlib.sha1()
    root = bed_root(fam)
    for rel in bed_files(fam):
        p = root / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    for part in (fam.get("track", ""), str(fam.get("base", "")), "official" if on_official(fam, where) else "mirror"):
        h.update(b"\0" + part.encode())
    if fam.get("bed_env"):  # absent: the hash of before the key existed
        h.update(b"\0" + json.dumps(fam["bed_env"], sort_keys=True).encode())
    if bed_fixed(fam):  # absent: the hash of before the key existed; present, no round of the old bed is reused
        h.update(b"\0fixed" + json.dumps(bed_fixed(fam), sort_keys=True).encode())
    if where == "gate" and gate_env(fam):  # the gate's own environment: its incumbent rounds run again
        h.update(b"\0gate_env" + json.dumps(gate_env(fam), sort_keys=True).encode())
    if bench(fam):  # the resolved switches: two switch sets never share a pool or a gate signature
        h.update(b"\0bench" + bench_signature(fam).encode())
        h.update(b"\0render_" + render_tag(fam).encode())  # TODO 52: lazy with warm-up everywhere, or every step
        if memo_sets(fam, where == "gate"):
            h.update(b"\0memo")  # saved-state replay shares perception between variants: its rounds are their own
    return h.hexdigest()[:10]


def gate_env(fam: dict) -> dict:
    """`bed_env_gate`: environment variables over `bed_env` for the gate and the sealed evaluation (user 2026-11-23:
    cameras rendered every step there). Keys starting with "_" are notes."""
    return {k: str(v) for k, v in (fam.get("bed_env_gate") or {}).items() if not k.startswith("_")}


# -------------------------------------------------------------------------------------------------- layout
def layout(family) -> dict:
    fam = _fam(family)
    plan = fam.get("plan")
    return dict(
        skills=ensure_package(fam),
        rules=_abs(fam["rules"]),
        prompts=_abs(fam["prompts"]),
        plan=_abs(plan) if plan else None,
        leak=list(LEAK),
        stub_file="pick.py",  # __init__.py is frozen for the guard, so a stub draft appends its comment to an editable file
        class_suggest=dict(CLASS_SUGGEST),
        caps={**FALLBACK_CAPS, **fam.get("caps", {})},
    )


# ------------------------------------------------------------------------------------------------ run_jobs
def skills_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _current(f: Path, sig: str) -> bool:
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError):
        return False
    return r.get("rule") == _success_rule() and r.get("motor") == sig


def _write(f: Path, rec: dict) -> None:
    guard_write(None, f)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    tmp.rename(f)  # a killed process never leaves a half-written round


def run_guard(fam: dict, base: Path, cand: Path) -> str | None:
    """None when the static guard passes (or the candidate is the base); else its first violation."""
    if Path(base).resolve() == Path(cand).resolve():
        return None
    guard = bed_root(fam) / (BENCH_GUARD if bench(fam) else f"{fam['bed_dir']}/{fam.get('guard', 'guard_simple.py')}")
    r = subprocess.run([sys.executable, "-I", str(guard), str(base), str(cand)], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except ValueError:
        return f"the guard gave no JSON (exit {r.returncode}): {(r.stderr or r.stdout)[-200:]}"
    if d.get("ok"):
        return None
    v = d.get("violations") or ["unspecified"]
    v0 = v[0]
    if isinstance(v0, dict):  # the bench guard's violations are {rule, file, detail}
        return f"{v0.get('rule')} {v0.get('file')}: {v0.get('detail')}"
    return str(v0)


def bed_env(fam: dict, package: Path, run_dir: Path, full: bool = False) -> dict:
    """The environment of a bed process: ours, without EXAM_OPEN, plus the loop's variables. SIMPLE_PRACTICE is the
    family's practice directory (in a sandbox the blanked copy); only the gate (`full`) gets the full files."""
    env = {k: v for k, v in os.environ.items() if k != "EXAM_OPEN"}
    if bench(fam):  # the adapter takes everything as arguments (--set ...): no SIMPLE_* variable, ours or inherited
        env = {k: v for k, v in env.items() if not k.startswith("SIMPLE_")}
        env.update(NUMPY_MADVISE_HUGEPAGE="0", OMP_NUM_THREADS="2", MUJOCO_GL="egl", PYTHONPATH="src")
        return env
    env.update(
        NUMPY_MADVISE_HUGEPAGE="0",
        OMP_NUM_THREADS="2",
        BFM_ROOT="true",
        MUJOCO_GL="egl",
        PYTHONPATH="src",
        SIMPLE_SKILLS=str(Path(package).resolve()),
        SIMPLE_LOOP="1",
        SIMPLE_PRACTICE=str(practice_dir(fam, full)),
        SIMPLE_RUN_DIR=str(run_dir),
    )
    env.update({k: str(v) for k, v in (fam.get("bed_env") or {}).items()})  # the family's switches of the bed
    if full:  # the gate and the sealed evaluation (user 2026-11-23: render every step there, SIMPLE_LAZY_RENDER=0)
        env.update(gate_env(fam))
    env.pop("EXAM_OPEN", None)
    return env


def _bed_argv(fam: dict, script: str, args: list[str]) -> list[str]:
    """The command: the family's `bed_cmd` (a fake bed in a test) plus the arguments, else bash that enters the bed
    root, sources scripts/paths.sh and runs the sim python on the bed script."""
    if fam.get("bed_cmd"):
        return [*fam["bed_cmd"], *args]
    prog = (
        'cd "$1" && . scripts/paths.sh && export TMPDIR="${TMPDIR:-$LOCOMANI_TMP}" && shift && exec "$VENV_SIM_PY" "$@"'
    )
    return ["bash", "-c", prog, "bash", str(bed_root(fam)), str(bed_dir(fam) / script), *args]


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
    """Stop a runner script and everything it started (its Isaac and depth servers, the bed): SIGTERM to the whole
    tree, the script first (it stops what it started when its foreground command ends), then SIGKILL to what is left.
    The children stay in the loop's process group, so a group kill of the loop takes them too."""
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


def _exec(argv: list[str], env: dict, log, stop: threading.Event | None, cwd: Path | None, tree: bool = False) -> int:
    """Run a bed command to its end; a set `stop` kills it (`tree`: with every process it started)."""
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


def bench_argv(fam: dict, task: str, ids_file, skill: dict, stance, out, package, run_dir, full: bool) -> list[str]:
    """The adapter's command for one unit: `python -m locomani.bench.simple run <Task> ...` (run in the bench root with
    PYTHONPATH=src). --set kinds, practice_dir, run_dir and the render keys are the backend's: the gate and the sealed set
    (`full`) read the full practice files; every round renders lazily with warm-up renders unless the family says bench.render =
    "every_step" (render_sets)."""
    argv = [fam["bench"].get("python") or BENCH_PY, "-m", "locomani.bench.simple", "run", task, "--ids", str(ids_file),
            "--setting", json.dumps(skill), "--out", str(out), "--skills", str(Path(package).resolve())]  # fmt: skip
    if stance:
        argv += ["--stance", json.dumps(stance)]
    for kv in bench_options_args(fam):
        argv += ["--bench-set", kv]
    for kv in ("kinds=Q", f"practice_dir={practice_dir(fam, full)}", f"run_dir={run_dir}",
               *render_sets(fam), *memo_sets(fam, full), *bench_config_args(fam)):  # fmt: skip
        argv += ["--set", kv]
    return argv


def official_runner(task: str, ids: list[str], package, setting: dict, out, family=None, env=None, log=None,
                    stop=None, procs: int = 1, gpus: str = "0", full: bool = False) -> int:  # fmt: skip
    """The one place where the official environment's episodes are run (a gate call); rows go to the file `out`.

    First implementation: if the family file names an "official_runner" script, it is run (in the bed root, with the
    bed environment) as `<script> <task> --ids <file> --setting <json> --out <out>` and must bring up what the task
    needs and write the rows; otherwise `bed_simple_official.py batch` runs directly against servers assumed up.
    The script (run_official_q.sh in the bed tree) takes one of the machine's three slots, starts its own Isaac and
    depth servers, makes the missing looks and grasps, runs the batch and stops what it started; run_jobs calls this
    for several units side by side, and a set `stop` takes the script's whole process tree down."""
    fam = _fam(family) if family is not None else {}
    env = env if env is not None else bed_env(fam, Path(package), Path(out).parent)
    ids_file = Path(out).with_name(f"ids_{task}.txt")
    ids_file.write_text("\n".join(ids) + "\n")
    skill, stance = split_set(setting or {})
    args = ["--ids", str(ids_file), "--setting", json.dumps(skill), "--out", str(out)]
    args += ["--stance", json.dumps(stance)] if stance else []
    log = log or subprocess.DEVNULL
    if bench(
        fam
    ):  # the bench adapter: its own slot, services, looks and grasps (session.py); the unit's folder is run_dir
        argv = bench_argv(fam, task, ids_file, skill, stance, out, package, Path(out).parent, full)
        return _exec(argv, env, log, stop, bed_root(fam), tree=True)
    script = fam.get("official_runner")
    if script:
        argv = ["bash", "-c", 'cd "$1" && . scripts/paths.sh && export TMPDIR="${TMPDIR:-$LOCOMANI_TMP}" && shift && '
                'exec "$@"', "bash", str(bed_root(fam)), *shlex.split(script), task, *args]  # fmt: skip
        return _exec(argv, env, log, stop, None, tree=True)
    argv = _bed_argv(fam, "bed_simple_official.py", ["batch", *args])
    return _exec(argv, env, log, stop, bed_root(fam) if fam.get("bed_cmd") and bed_root(fam).is_dir() else None)


def _units(by: dict[str, list], slots: int) -> list[tuple[str, list]]:
    """The official environment's work units of one job, (task, items) each: every task's episodes cut into equal
    chunks of about a quarter of a slot's share, in round-robin order over the tasks. The runner script takes one of the
    machine's slots (an Isaac server each) per call and waits when none is free, so `slots` units run at a time and
    stay busy to the end; neighbours in the order are different tasks, whose look passes do not wait for each other."""
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


def _scratch() -> Path:
    """Where a call's run directories go (built scenes, server logs, rows): TMPDIR, else the workspace's tmp
    (scripts/paths.sh's LOCOMANI_TMP; the machine's /tmp is on a small root disk), else the system's."""
    if os.environ.get("TMPDIR"):
        return Path(os.environ["TMPDIR"])
    ws = Path(os.environ.get("LOCOMANI_TMP") or Path(os.environ.get("LOCOMANI_WORKSPACE") or "/workspace") / "tmp")
    return ws if ws.is_dir() else Path(tempfile.gettempdir())


def _ids_by_task(pool: Path, labels: list[str]) -> dict[str, list[tuple[str, str]]]:
    by: dict[str, list[tuple[str, str]]] = {}
    for x in labels:
        d = json.loads((pool / f"{x}.json").read_text())
        by.setdefault(d["task"], []).append((x, d["episode_id"]))
    return by


def on_official(fam: dict, where: str) -> bool:
    """Whether a call runs in the official environment: always for the gate; for diagnostic rounds and probes when
    the family says "probe_bed": "official" (the mirror does not show the official environment's perception and
    grasp failures, so cards and probes made there would steer the loop to the wrong problems)."""
    return where == "gate" or fam.get("probe_bed", "mirror") == "official"


def run_jobs(jobs, family, pool, procs, log, gpus, stop=None, where: str = "probe") -> dict:
    """Run the jobs' missing rounds; returns {job index: labels still missing} as batch.run_jobs does. A job has
    `skills` (the package directory), `out`, `overrides` (the setting) and `labels` (None = the whole pool). Each job
    is guarded first; a job that fails the guard gets ok False rounds ("guard: <violation>") and runs nothing."""
    fam = _fam(family)
    check_bench(fam)
    pool, log = Path(pool), Path(log)
    sig = signature(fam, where)
    every = sorted(p.stem for p in pool.glob("*.json") if not p.name.startswith("_"))
    log.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    base = package_dir(fam)
    tmp = _scratch()
    fixed = bed_fixed(fam)
    with log.open("a") as fh:
        for k, job in enumerate(jobs):
            out, setting = Path(job.out), dict(job.overrides or {})
            ran = {**setting, **fixed}  # what the bed runs: the fixed switches over the job's set (rule 46)
            out.mkdir(parents=True, exist_ok=True)
            todo = [x for x in (job.labels or every) if not _current(out / f"{x}.json", sig)]
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
                    _write(out / f"{x}.json", _blank(x, f"guard: {bad}", sh, setting, sig))
                result[k] = []
                continue
            by = _ids_by_task(pool, todo)
            pdir = practice_dir(fam, where == "gate")  # the practice files this call is given (a sandbox: blanked rows)
            gone = [x for rows in by.values() for x, eid in rows if not _in_practice(pdir, eid)]
            for x in gone:  # an id that is blank or missing there is not an episode this call may run
                _write(out / f"{x}.json", _blank(x, "not a diagnostic episode", sh, setting, sig, "refused"))
            by = {t: [(x, e) for x, e in rows if x not in gone] for t, rows in by.items()}
            by = {t: rows for t, rows in by.items() if rows}
            if not by:
                result[k] = []
                continue
            tmp.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix=f"simple_{fam['name']}_{k}_", dir=tmp))
            env = bed_env(fam, Path(job.skills), run_dir, full=where == "gate")
            by_id = {eid: x for rows in by.values() for x, eid in rows}
            rows = run_dir / "rows.jsonl"
            if on_official(fam, where):
                slots = int(fam.get("official_slots", 3))

                def unit(u, job=job, setting=ran, env=env, run_dir=run_dir):
                    i, (task, items) = u
                    d = run_dir / f"u{i:02d}"  # a run directory per unit: the script's pid files and logs are per task
                    d.mkdir()
                    more = (
                        dict(full=where == "gate") if bench(fam) else {}
                    )  # the adapter's lazy_render and practice files
                    return official_runner(task, [e for _, e in items], job.skills, setting, d / f"rows_{task}.jsonl",
                                           family=fam, env=env if bench(fam) else dict(env, SIMPLE_RUN_DIR=str(d)),
                                           log=fh, stop=stop, procs=procs, gpus=gpus, **more)  # fmt: skip

                with ThreadPoolExecutor(max_workers=slots) as pool_x:
                    list(pool_x.map(unit, enumerate(_units(by, slots))))
                files = sorted(run_dir.glob("u*/rows_*.jsonl"))
            else:
                ids = run_dir / "ids.txt"
                ids.write_text("\n".join(by_id) + "\n")
                skill, stance = split_set(ran)
                # the mirror runs the family's track too (the gate's official bed is camera only): "official" = head camera
                args = ["batch", "--ids", str(ids), "--out", str(rows), "--track", fam.get("track", "truth"),
                        "--setting", json.dumps(skill),
                        "--procs", str(procs), "--gpus", *[g for g in str(gpus).split(",") if g]]  # fmt: skip
                args += ["--stance", json.dumps(stance)] if stance else []
                _exec(_bed_argv(fam, "bed_simple.py", args), env, fh, stop,
                      bed_root(fam) if fam.get("bed_cmd") and bed_root(fam).is_dir() else None)  # fmt: skip
                files = [rows]
            for f in files:
                for line in f.read_text().splitlines() if f.exists() else []:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    x = by_id.get(rec.get("episode_id"))
                    if x is not None:
                        _write(out / f"{x}.json", to_round(rec, x, sh, setting, sig))
            result[k] = [x for x in todo if not _current(out / f"{x}.json", sig)]
            if not result[k]:
                shutil.rmtree(run_dir, ignore_errors=True)
    return result


def _in_practice(pdir: Path, eid: str) -> bool:
    """True when line n of <pdir>/<Task>.jsonl, for the id `<Task>/Q/<n>`, is a real (not blank) row."""
    try:
        task, _, n = eid.split("/")
        rows = _rows(pdir / f"{task}.jsonl")
        return rows[int(n)]["level"] is not None
    except (ValueError, IndexError, OSError):
        return False


def _blank(label: str, why: str, sh: str, setting: dict, sig: str, cls: str = "guard") -> dict:
    """A round that never ran (the guard refused the package, or the episode is not one this call may run)."""
    return dict(label=label, task=label.rsplit("_s", 1)[0], ok=False, why=why, fell=False, dropped=False,
                plan_ok=False, plan_why=why, error=None, steps=[], t_end=None, failure_class=cls, skills=sh,
                set=setting, wall_s=0.0, rule=_success_rule(), controller="scalebfm", motor=sig, simple={})  # fmt: skip


# ------------------------------------------------------------------------------------------------ sandbox
def _copy(src: Path, dst: Path) -> None:
    guard_write(None, dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    elif src.exists():
        shutil.copy2(src, dst)


def _cut(dest: Path, rel: str) -> None:
    """Remove `rel` (globs allowed) below `dest` without touching what a symlink points to: a symlinked directory on
    the way is replaced by a real directory of links to its entries first."""
    for hit in sorted(dest.glob(rel)) if "*" in rel else [dest / rel]:
        guard_write(None, hit)
        parts = hit.relative_to(dest).parts
        cur = dest
        for i, name in enumerate(parts[:-1]):
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


def sandbox(dest, family, forbidden=None) -> None:
    """Add what the Engineer needs to the sandbox `dest` and take away what it must not see (before sandbox.py's git
    commit): under `<dest>/<sandbox.dir>` (default bed_readonly_copy; FOR READING ONLY, nothing runs from it) the seed
    package, the bed files, the five task files and the practice files with the hidden rows blanked; the full practice
    files go to the hidden tree. The bed itself stays where the family file says (`bed_root`: the real bed tree), so a
    round in the sandbox runs on the complete bed; only `practice_dir` is rewritten, to the blanked copy. The forbidden
    paths (`forbidden`, default FORBIDDEN_PATHS) are cut out of the linked third_party and any other copy."""
    dest = Path(dest)
    fam = _fam(family)
    spec = fam.get("sandbox", {})
    rel = spec.get("dir", "bed_readonly_copy")
    nest = dest / rel
    root = bed_root(fam)
    hidden = hidden_ids(fam)
    _copy(base_dir(fam), nest / base_rel(fam))
    for entry in bed_files(fam):
        _copy(root / entry, nest / entry)
        if (nest / entry).is_file():
            (nest / entry).chmod(0o444)
    for t in spec.get("task_files", []):
        _copy(_abs(t), nest / "tasks" / Path(t).name)
    src = practice_dir(fam, True)
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
    for pat in FORBIDDEN_PATHS if forbidden is None else forbidden:
        _cut(dest, pat)
    ff = dest / "scripts" / "rsi" / "families" / f"{fam['name']}.json"
    if ff.exists():
        d = json.loads(ff.read_text())
        d.update(practice_dir=f"{rel}/practice")  # bed_root stays: the sandbox's rounds run on the real bed
        ff.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")


def audit(dest, forbidden=None, leak=None, backend: str = "simple") -> list[str]:
    """What in the sandbox `dest` the Engineer must not see: forbidden paths that exist, the practice rows of hidden
    ids that are not blank, and leak patterns in text files (the hidden tree, .git, links and the bed files copied on
    purpose are skipped). [] when clean. `forbidden`, `leak` and `backend` default to this backend's own (another
    backend's audit passes its lists and its name, which finds its family file in the sandbox)."""
    dest = Path(dest)
    bad: list[str] = []
    for pat in FORBIDDEN_PATHS if forbidden is None else forbidden:
        for p in [dest / pat] if "*" not in pat else sorted(dest.glob(pat)):
            if os.path.lexists(p) and (p.exists() or p.is_symlink()):
                bad.append(f"path {p.relative_to(dest)}")
    nest = dest / _nest_name(dest, backend)
    for f in sorted((nest / "practice").glob("*.jsonl")) if (nest / "practice").is_dir() else []:
        seen = [0, 0, 0]
        for i, row in enumerate(_rows(f)):
            lv = row["level"]
            if lv is not None:
                seen[lv] += 1
                if seen[lv] > DIAG_PER_LEVEL:
                    bad.append(f"practice row not blank: {f.relative_to(dest)} line {i}")
    pats = [re.compile(x) for x in (LEAK if leak is None else leak)]
    allowed = _allowed(dest, backend)
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


def _sandbox_family(dest: Path, backend: str = "simple") -> dict:
    for ff in sorted(
        (dest / "scripts" / "rsi" / "families").glob("*.json" if backend == "bench" else f"{backend}*.json")
    ):
        try:
            d = json.loads(ff.read_text())
        except ValueError:
            continue
        if d.get("backend") == backend:
            return d
    return {}


def _nest_name(dest: Path, backend: str = "simple") -> str:
    return _sandbox_family(dest, backend).get("sandbox", {}).get("dir", "bed_readonly_copy")


def _allowed(dest: Path, backend: str = "simple") -> set[str]:
    """The bed files and the seed package copied on purpose (they name the released ids and EXAM_OPEN in their own code)."""
    d = _sandbox_family(dest, backend)
    nest = dest / _nest_name(dest, backend)
    out = {str(nest / b) for b in (bed_files(d) if d.get("bench") else d.get("bed_files", []))}
    # the seed package is the code the Engineer starts from; it names the older evaluation scripts' directory in its own
    # sys.path lines, which is code (the guard bars an edit from opening paths), not the data
    out |= {str(p) for p in (nest / base_rel(d)).rglob("*.py")} if d.get("base") or d.get("bench") else set()
    return out
