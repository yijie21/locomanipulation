"""TODO 27.2.4 (plan docs/handoff/scene-rsi-plan-20261016.md D6, D7): run rounds of a task family from a pool under one
candidate -- a skills package directory (a snapshot of src/locomani/tasks/scene/skills, loaded under the name
`scene_skills`) plus parameter overrides (`--set`, skills.apply_set keys) -- and write one JSON (and, with --npz, the
motor trace) per start into --out. One process loads the policy and the scene once; batch.py runs many of these.

Round record: label, task, ok, why (the goal's), fell, dropped (the goal missed and a task object ended more than DROP_M
below where it started), plan_ok / plan_why, steps (every skill step's outcome), skills (the package's content hash), set, wall_s,
rule (batch.SUCCESS_RULE: which `held` judged it), motor (batch.motor_signature: the motor settings it ran under),
failure_class and events (failure_class.py, TODO 33.15: computed from the trace, every round), others_moved (the
non-task free objects the round moved more than 2 cm or tilted more than 10 deg, episode.others_moved; TODO 36.9).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src $PY scripts/rsi/scene_round.py pick_single \\
        --pool <pool dir> --skills src/locomani/tasks/scene/skills --out <dir> [--labels a,b] [--set '{"k": v}'] [--npz]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import batch as B  # noqa: E402
import failure_class as FC  # noqa: E402
import family as F  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.tasks.scene.episode import run_episode  # noqa: E402
from locomani.tasks.scene.start import Start  # noqa: E402
from locomani.tasks.scene.world import SceneWorld  # noqa: E402

DROP_M = 0.15


def skills_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def load_skills(root: Path, name: str = "scene_skills"):
    """The skills package at `root` as module `name` (its modules import each other relatively)."""
    for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_set(text: str | None) -> dict:
    """Inline JSON (what batch.py passes) or a JSON file. Inline first: a long override set is not a file name, and
    asking the filesystem about one over 255 bytes raises ENAMETOOLONG (iteration 0's probes all died on it,
    2026-10-16)."""
    if not text:
        return {}
    if text.lstrip().startswith("{"):
        return json.loads(text)
    return json.loads(Path(text).read_text())


def make_world(fam: dict, controller: str) -> SceneWorld:
    if controller == "amo":  # TODO 28
        from locomani.backends.amo.runtime import AmoPolicy

        policy = AmoPolicy("cuda")
    elif controller.startswith("gr00t_wbc"):  # TODO 34
        from locomani.backends.gr00t_wbc.runtime import Gr00tPolicy

        policy = Gr00tPolicy()
    else:
        policy = R.Policy("cuda", ScaleBfmBackendConfig())
    world = SceneWorld(fam["scene_dir"], policy, controller=controller)
    if B.motor_config().reach_compensation:  # TODO 33.1.4 (2): RSI_REACH_COMPENSATION=1
        world.compensate_reaches()
    return world


def world_signature(world: SceneWorld, controller: str) -> str:
    """TODO 36 (review 2026-10-22): `batch.motor_signature` of the settings this world actually holds."""
    pol = getattr(world.motor, "pol", None)
    cuda_graph = getattr(getattr(pol, "cfg", None), "cuda_graph", None)  # amo / gr00t policies: the default
    return B.motor_signature(controller, world.motor.cfg, world.sim.cfg.sleep, cuda_graph)


def run_round(
    fam: dict, world: SceneWorld, skills, h: str, pool: Path, out: Path, label: str, overrides: dict, npz: bool,
    controller: str,
) -> None:  # fmt: skip
    f = out / f"{label}.json"
    motor = world_signature(world, controller)
    if B.done(f, motor):
        return
    B.retire(f)
    row = json.loads((pool / f"{label}.json").read_text())
    task = F.task(fam, row["task"])
    start = Start.from_dict(row["start"])
    t0 = time.time()
    arrays = None
    try:
        plan = skills.plan_for(task)
        plan_err = ""
    except Exception as e:  # TODO 36 (review 2026-10-22): a plan_for bug is the round's failure, not the worker's death
        plan, plan_err = None, f"plan_for raised: {type(e).__name__}: {e}"
    if plan_err:
        rec = dict(label=label, task=task.id, ok=False, why=plan_err, fell=False, dropped=False, plan_ok=False,
                   plan_why=plan_err, error=plan_err)  # fmt: skip
    elif plan is None:
        rec = dict(label=label, task=task.id, ok=False, why="no skill covers this goal", fell=False, dropped=False)
    else:
        res, arrays = run_episode(world, task, start, plan)
        ex = res.execution
        # a round that reaches its goal is no drop: a put-down onto a surface more than DROP_M below the start
        # (the clock onto the console table) counted as one (r1_on_single, human intervention 2, 2026-10-16)
        dropped = not res.ok and any(
            float(world.object_pose(k)[0][2]) < float(start.placements[k][0][2]) - DROP_M for k in task.objects
        )
        rec = dict(label=label, task=task.id, ok=res.ok, why=res.why, fell=ex["fell"], dropped=bool(dropped),
                   plan_ok=ex["plan_ok"], plan_why=ex["plan_why"], error=ex["error"], steps=ex["steps"],
                   t_end=ex["t_end"], seated=ex["seated"], calls=ex.get("calls", []),
                   others_moved=ex.get("others_moved", {}))  # fmt: skip
        if npz:
            np.savez_compressed(out / f"{label}.npz", **arrays)
    try:  # TODO 33.15: the round's failure class and events (failure_class.py); a classifier bug never costs a round
        rec["failure_class"], rec["events"] = FC.classify(rec, arrays, world, task.objects[0] if task.objects else None)
    except Exception as e:
        rec["failure_class"], rec["events"] = "unclassified", dict(error=f"{type(e).__name__}: {e}"[:300])
    rec.update(
        skills=h,
        set=overrides,
        wall_s=round(time.time() - t0, 1),
        rule=B.SUCCESS_RULE,
        controller=controller,
        motor=motor,
    )
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    tmp.rename(f)  # a killed process never leaves a half-written round


def run(
    fam_name: str,
    pool: Path,
    skills_dir: Path,
    out: Path,
    labels: list[str],
    overrides: dict,
    npz: bool,
    controller: str = "scalebfm",
) -> None:
    fam = F.load(fam_name)
    skills = load_skills(skills_dir)
    skills.apply_set(skills, overrides)
    h = skills_hash(skills_dir)
    world = make_world(fam, controller)
    out.mkdir(parents=True, exist_ok=True)
    for label in labels:
        run_round(fam, world, skills, h, pool, out, label, overrides, npz, controller)


def work(fam_name: str, pool: Path, q: Path) -> None:
    """A batch.py worker (TODO 32.2): take rounds from the queue `q` until it is empty. The scene and the policy load
    once; a job's skills package loads when the worker takes its first round of that job (fresh, as a process per job
    did), its overrides applied."""
    fam = F.load(fam_name)
    jobs = json.loads((q / "jobs.json").read_text())
    world, cur, skills, h = None, None, None, ""
    while (t := B.claim_round(q)) is not None:
        job = jobs[t["job"]]
        if t["job"] != cur:
            skills = load_skills(Path(job["skills"]))
            skills.apply_set(skills, job["overrides"])
            h, cur = skills_hash(Path(job["skills"])), t["job"]
        if world is None or world.controller != job["controller"]:
            world = make_world(fam, job["controller"])
        out = Path(job["out"])
        out.mkdir(parents=True, exist_ok=True)
        run_round(fam, world, skills, h, pool, out, t["label"], job["overrides"], job["npz"], job["controller"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--skills", type=Path, help="a skills package directory")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--queue", type=Path, help="a batch.py queue directory: work through it (--skills/--out unused)")
    ap.add_argument("--labels", default="", help="comma-separated start labels (default: the whole pool)")
    ap.add_argument("--set", default="", help="parameter overrides: inline JSON or a JSON file")
    ap.add_argument("--npz", action="store_true", help="also save each round's motor trace")
    ap.add_argument(
        "--controller",
        choices=("scalebfm", "amo", "gr00t_wbc", "gr00t_wbc_h066_p012"),
        default="scalebfm",
        help="whole-body controller (TODO 28, 34)",
    )
    a = ap.parse_args(argv)
    if a.queue is not None:
        work(a.family, a.pool, a.queue)
        return 0
    if a.skills is None or a.out is None:
        ap.error("--skills and --out are required without --queue")
    labels = [x for x in a.labels.split(",") if x] or F.labels(a.pool)
    run(a.family, a.pool, a.skills, a.out, labels, parse_set(a.set), a.npz, a.controller)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
