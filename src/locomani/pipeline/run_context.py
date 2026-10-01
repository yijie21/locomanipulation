"""What every stage entry does before it does its own work: read the experiment, claim the run
directory, refuse to start without headroom, and -- with --dry-run -- say what it would execute.

    cfg, run_dir = open_run(args.experiment, args.name)
    require_headroom("collect", gpu=args.gpu)
    require_deploy_patches(cfg)

Moved here from `scripts/run_common.py` on 2026-09-21: the stage entries became python modules
called by one flat bash script each, so their shared preamble belongs in the package rather than
beside the scripts (docs/handoff/refactor-flat-bash-20260921.md).

The experiment file is copied verbatim to `<run>/config.yaml`. That copy is the run's provenance
and is what a later stage reads, so a stage never depends on a file that may have changed since.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from locomani import paths
from locomani.sim import robot_look
from locomani.sim.mirror import MIRRORED_SOURCES

ROOT = Path(paths.REPO)
RUNS = ROOT / "outputs" / "runs"
SCENES = ROOT / "configs" / "scenes.yaml"
PID_MARGIN = 200  # a stack needs room to fork after the check passes
NEEDS = {  # measured, not theoretical: an exporter holds ~1800 pids and the container's ceiling is 3840
    "collect": dict(cores=8, pids=1800, gpu_free_mib=6000),
    "convert": dict(cores=4, pids=200, gpu_free_mib=0),
    "train": dict(cores=4, pids=300, gpu_free_mib=28000),
    "eval": dict(cores=8, pids=1800, gpu_free_mib=12000),
}
# What a stack costs WITHOUT the exporter. Measured on 2026-09-21: bringing sim + deploy up on
# instance 0 moved pids.current 2269 -> 2441 and peaked at 2445, and stopping it returned to 2266.
# So the whole 1800 above is the exporter, and a run that records nothing -- every evaluation, and
# the dry pilots that sample and score rounds without writing episodes -- was being refused on a
# budget it does not spend. Kept at more than twice the measurement because the agent process and
# a heavier scene are not in it.
NO_EXPORTER_PIDS = 400
# What the DRIVER costs when the stack is already up: the expert process and its ZMQ clients,
# measured 2026-09-22 at 1-1.5 cores while a sim + deploy stack held ~5. Charging the attaching
# driver the whole stage budget refused it whenever anything else on the box held ~3 cores.
ATTACH_CORES = 2
# A stack WITHOUT the exporter: the same 2026-09-22 measurement's ~5 cores for sim + deploy. The
# 8 above is the recording stack's; like NO_EXPORTER_PIDS, a run that records nothing is not
# charged for it (2026-09-23: a render-offline pilot refused at 9.2 in use + 8 > 15.36).
NO_EXPORTER_CORES = 5


def open_run(experiment: Path | str, name: str | None = None, *, write: bool = True) -> tuple[dict, Path]:
    """Read the experiment file and copy it verbatim into the run directory it names.

    `--name` chooses the DIRECTORY only. The copy stays byte-for-byte, so under a `--name` the
    `name:` inside it still reads as the experiment's own -- that is the file this run came from,
    which is the question provenance has to answer, and it is what lets the guard below compare
    the two by bytes. Nothing downstream reads `cfg["name"]`; the run directory is passed along.

    `write=False` is what --dry-run passes: a dry run answers "what would this do" and must leave
    nothing behind, not even the directory it would have claimed.
    """
    experiment = Path(experiment)
    cfg = resolve_scene(migrate_images(yaml.safe_load(experiment.read_text())))
    unreal_support(cfg)
    cfg["name"] = name or cfg["name"]
    run_dir = RUNS / cfg["name"]
    if not write:
        return cfg, run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    copy = run_dir / "config.yaml"
    if copy.exists() and copy.read_text() != experiment.read_text():
        raise SystemExit(f"{copy} differs from {experiment}; use a new name rather than changing a run in flight")
    shutil.copyfile(experiment, copy)
    return cfg, run_dir


def load_run_config(run_dir: Path) -> dict:
    """The copy this run started from. A later stage reads this, never the file in configs/."""
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        raise SystemExit(f"{path} is missing; this directory was not opened by a stage script")
    return unreal_support(resolve_scene(migrate_images(yaml.safe_load(path.read_text()))))


IMAGE_SOURCES = ("mujoco", *MIRRORED_SOURCES)


def eval_render(source: str) -> str:
    """The eval sim's `backend.render` for an image source: the policy reads MuJoCo frames only from a sim that
    renders them; with Isaac or Unreal the mirror draws them and the sim renders nothing."""
    return "live" if source == "mujoco" else "offline"


def migrate_images(cfg: dict) -> dict:
    """Bring the experiment's image source to the current format and check it; returns `cfg`, changed in place.

    The current format (2026-09-27) says once, at run level, which renderer draws the policy's pictures:
    `images: {source: mujoco | isaac | unreal}`. It decides the dataset cameras (collection records MuJoCo frames or
    leaves placeholders for a re-render, and convert checks which renderer filled them), the eval sim's
    rendering and where the closed loop's images come from. The historical format has no `images` block and
    said it as `eval.image_source`, although collection and convert read it too; such a file -- every run
    copy before that date -- is read as that source, and its eval sim renders what the source needs.

    Refused here, where every stage reads its configuration, so no stage starts on a pairing none can serve:
    `eval.image_source` beside an `images` block; `isaac` or `unreal` with a live collection (it records MuJoCo
    frames, the re-render leaves them in place and convert's check never runs, so training and evaluation see
    different renderers); and, in the current format, an `eval.backend.render` other than the source's.
    """
    evaluation = cfg.get("eval") or {}
    historical = "images" not in cfg
    if historical:
        cfg["images"] = {"source": evaluation.pop("image_source", None) or "mujoco"}
    elif "image_source" in evaluation:
        raise SystemExit("eval.image_source is the old spelling of images.source; give the source only in `images`")
    images = cfg["images"]
    if isinstance(images, dict):
        # `images.third_person` (4144452, 2026-09-30, pushed) switched the fill's filming; the fill no longer films
        # (scripts/render_third_person.sh does, on request), so either value means what no key means now
        images.pop("third_person", None)
    if not isinstance(images, dict) or set(images) != {"source"} or images["source"] not in IMAGE_SOURCES:
        raise SystemExit(f"images must be {{source: {' | '.join(IMAGE_SOURCES)}}}, not {images!r}")
    source = images["source"]
    if source != "mujoco" and (cfg.get("backend") or {}).get("render", "live") != "offline":
        raise SystemExit(
            f"images.source {source} needs backend.render: offline -- a live collection records MuJoCo frames, which "
            f"the {source} re-render leaves in place, so the policy would train on MuJoCo and be evaluated on {source}"
        )
    rendered = (evaluation.get("backend") or {}).get("render")
    if historical and rendered is not None:
        evaluation["backend"]["render"] = eval_render(source)  # the migrated file is a current-format one
    elif rendered not in (None, eval_render(source)):
        raise SystemExit(
            f"eval.backend.render {rendered!r} contradicts images.source {source!r}, which needs "
            f"{eval_render(source)!r}: leave it out or give that"
        )
    return cfg


def evaluation_of(cfg: dict):
    """The run's EvalConfig as its evaluator runs it: the eval block with the image source and the eval sim's
    rendering that `images.source` decides. Every stage that evaluates builds it here, not from the bare block."""
    from locomani.pipeline.eval_config import EvalConfig

    source = migrate_images(cfg)["images"]["source"]
    block = dict(cfg["eval"])
    backend = {**(block.get("backend") or {}), "render": eval_render(source)}
    return EvalConfig.from_dict({**block, "backend": backend, "image_source": source})


def require_images(cfg: dict) -> None:
    """Before a stack starts: an Isaac run needs its scene's Isaac stage and the render venv on this machine, an
    Unreal run its scene's built Unreal level and the URLab bridge venv."""
    source = cfg["images"]["source"]
    if source == "isaac":
        isaac_scene(cfg)
        if not os.access(paths.RENDER_PY, os.X_OK):
            raise SystemExit(
                f"images.source isaac renders in the render venv, and {paths.RENDER_PY} is not there: "
                "docs/setup.md step 7 (scripts/setup_render_runtime.sh)"
            )
    elif source == "unreal":
        unreal_scene(cfg)
        if not os.access(paths.BRIDGE_PY, os.X_OK):
            raise SystemExit(
                f"images.source unreal talks to the editor from the URLab bridge venv, and {paths.BRIDGE_PY} is not "
                "there: docs/setup.md, the optional Unreal step (scripts/setup_unreal.sh)"
            )


def resolve_scene(cfg: dict, registry: Path | None = None) -> dict:
    """Turn `scene: {name: warehouse}` into the `scene.dir` every consumer reads; returns `cfg`.

    Resolved here, where both the experiment and a run's copy are read, so no consumer knows names
    exist. The copy keeps the name; `configs/scenes.yaml` promises a name never moves.
    """
    scene = cfg.get("scene") or {}
    if "name" not in scene:
        return cfg
    if scene.get("dir"):
        raise SystemExit(f"scene gives both name {scene['name']!r} and dir {scene['dir']!r}; give one")
    names = yaml.safe_load(Path(registry or SCENES).read_text())
    if scene["name"] not in names:
        raise SystemExit(f"no scene called {scene['name']!r}; {registry or SCENES} names {', '.join(sorted(names))}")
    scene["dir"] = names[scene["name"]]["dir"]
    return cfg


def isaac_scene(cfg: dict, registry: Path | None = None) -> dict:
    """The run's scene as Isaac renders it: `{"dir": <Isaac stage dir>, "rtx_settings": [...], "supersample": N}`.

    From the scene's `isaac` entry in `configs/scenes.yaml`, so every Isaac renderer of a scene --
    the offline re-render and the eval mirror -- loads the same stage with the same look.
    """
    name = (cfg.get("scene") or {}).get("name")
    if not name:
        raise SystemExit("an Isaac render needs the run's scene by name (scene: {name: ...}, configs/scenes.yaml)")
    entry = (yaml.safe_load(Path(registry or SCENES).read_text()).get(name) or {}).get("isaac")
    if not entry:
        raise SystemExit(f"scene {name!r} has no Isaac stage: add an `isaac` entry for it in {registry or SCENES}")
    stage = Path(paths.ISAAC_ASSETS) / "scene" / entry["dir"]
    if not (stage / "scene.usda").is_file():
        build = entry.get("build") or "its builder"
        raise SystemExit(f"{stage / 'scene.usda'} is missing; build the Isaac stage of scene {name!r}: {build}")
    # A stage generated from the MuJoCo scene records which one; a rebuilt scene (other box, other platform)
    # would otherwise be rendered with the old geometry while physics runs the new one.
    source = stage / "source.json"
    if source.is_file() and cfg["scene"].get("dir"):
        built = json.loads(source.read_text()).get("scene_xml_sha256")
        xml = ROOT / cfg["scene"]["dir"] / "scene_locomani.xml"
        if xml.is_file() and built != hashlib.sha256(xml.read_bytes()).hexdigest():
            raise SystemExit(f"{stage} was built from another {xml}; rebuild it: {entry.get('build')}")
    supersample = entry.get("supersample", 1)
    if not isinstance(supersample, int) or isinstance(supersample, bool) or supersample < 1:
        raise SystemExit(f"scene {name!r}: isaac.supersample must be a whole number >= 1, not {supersample!r}")
    return {"dir": str(stage), "rtx_settings": list(entry.get("rtx_settings") or []), "supersample": supersample}


def unreal_entry(name: str, registry: Path | None = None) -> dict:
    """Scene `name`'s `unreal` entry in the registry; refuses a scene with none or an unsupported one."""
    entry = (yaml.safe_load(Path(registry or SCENES).read_text()).get(name) or {}).get("unreal")
    if not entry:
        raise SystemExit(f"scene {name!r} has no Unreal level: add an `unreal` entry for it in {registry or SCENES}")
    if "unsupported" in entry:
        raise SystemExit(f"scene {name!r} cannot use images.source unreal: {entry['unsupported']}")
    return entry


def _unreal_entry(cfg: dict, registry: Path | None) -> tuple[str, dict]:
    name = (cfg.get("scene") or {}).get("name")
    if not name:
        raise SystemExit("an Unreal render needs the run's scene by name (scene: {name: ...}, configs/scenes.yaml)")
    return name, unreal_entry(name, registry)


def unreal_support(cfg: dict, registry: Path | None = None) -> dict:
    """At load: an `images.source: unreal` run's scene must have an Unreal level in the registry; returns `cfg`.

    Only the registry is read here -- whether the level is built on this machine is `unreal_scene`'s question,
    asked before anything renders (`require_images`, the mirror, the re-render)."""
    if cfg["images"]["source"] == "unreal":
        _unreal_entry(cfg, registry)
    return cfg


def unreal_scene(cfg: dict, registry: Path | None = None) -> dict:
    """The run's scene as Unreal renders it: `{"level": <name>, "level_path": /Game/Levels/<name>, "repeats": N}`.

    From the scene's `unreal` entry in `configs/scenes.yaml`, so the dataset re-render and the eval mirror load the
    same level and capture it the same way. `repeats` is `urlab.FreshCaptureRepeats`, the captures each request
    accumulates (Lumen and TSR converge over frames). The level must be built in the host project ($UE_PROJECT)
    from this very scene and with the robot look Isaac binds (`locomani.sim.robot_look`): its builder records the
    scene XML's hash and the look's name in `$UNREAL_ASSETS/<level>/source.json` (default `$UNREAL_ROOT/levels`: one
    place per machine, whichever checkout built the level).
    """
    name, entry = _unreal_entry(cfg, registry)
    level = entry["level"]
    repeats = entry.get("repeats", 8)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 64:
        raise SystemExit(f"scene {name!r}: unreal.repeats must be a whole number in 1-64, not {repeats!r}")
    build = entry.get("build") or "its builder"
    umap = Path(paths.UE_PROJECT).parent / "Content" / "Levels" / f"{level}.umap"
    if not umap.is_file():
        raise SystemExit(f"{umap} is missing; build the Unreal level of scene {name!r}: {build}")
    source = Path(paths.UNREAL_ASSETS) / level / "source.json"
    if not source.is_file():
        raise SystemExit(f"{source} is missing; the level was not built by the builder: {build}")
    built = json.loads(source.read_text())
    if cfg["scene"].get("dir"):
        xml = ROOT / cfg["scene"]["dir"] / "scene_locomani.xml"
        if xml.is_file() and built.get("scene_xml_sha256") != hashlib.sha256(xml.read_bytes()).hexdigest():
            raise SystemExit(f"Unreal level {level!r} was built from another {xml}; rebuild it: {build}")
    if built.get("robot_look") != robot_look.NAME:
        raise SystemExit(
            f"Unreal level {level!r} carries the robot look {built.get('robot_look')}, not {robot_look.NAME} "
            f"(locomani.sim.robot_look, what Isaac binds); rebuild it: {build}"
        )
    return {"level": level, "level_path": f"/Game/Levels/{level}", "repeats": repeats}


def successful_traces(run_dir: Path, traces: Path) -> list[str]:
    """Labels of the state traces in `traces` whose round succeeded, as the manifest that judged it says.

    A closed-loop evaluation's traces (`<eval dir>/state_trace/crate_vla_NNN`) are judged by the manifest beside
    them (`closed_loop.manifest.jsonl`, key `video`); collection traces by the run's batch manifests
    (`manifest_NN.jsonl`, bimanual at the top, single-arm under `collect/`, key `label`). A trace no row names
    succeeded nowhere on record -- an interrupted batch's manifest is renamed `interrupted_manifest_*` -- so it
    is left out, as a failed round is: an Isaac re-render costs about two minutes a round (docs/rendering.md).
    """
    run_dir, traces = Path(run_dir), Path(traces)
    evaluated = traces.parent / "closed_loop.manifest.jsonl"
    if evaluated.is_file():
        judged, key = [evaluated], "video"
    else:
        judged = sorted(run_dir.glob("manifest_*.jsonl")) + sorted(run_dir.glob("collect/manifest_*.jsonl"))
        key = "label"
    ok = set()
    for manifest in judged:
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        ok |= {row[key] for row in rows if row.get("ok") and row.get(key)}
    labels = (p.name.removesuffix("_state_trace.npz") for p in traces.glob("*_state_trace.npz"))
    return sorted(label for label in labels if label in ok)


def checkpoint_of(run_dir: Path) -> Path:
    """The checkpoint the train stage recorded for this run.

    Read, never recomputed: `run_train.checkpoint_of` decides where the recipe leaves the weights
    and `summarise` writes that answer down, so everything downstream asks the run instead of
    re-deriving it. Copies of this lived in the two eval drivers and the policy server; the
    drivers are gone since 2026-09-22 and this is where the remaining callers look.
    """
    summary = Path(run_dir) / "summary.json"
    entry = json.loads(summary.read_text()).get("train") if summary.exists() else None
    if not entry or not entry.get("checkpoint"):
        raise SystemExit(f"{summary} has no checkpoint; run the train stage first")
    return Path(entry["checkpoint"])


def measure(gpu: str | None = None, *, seconds: float = 1.0) -> dict:
    """Cores in use (cgroup delta over `seconds`; 0 skips it), pids in the container, free MiB on the card."""
    live: dict = {}
    try:
        cpu = Path("/sys/fs/cgroup/cpu.stat")
        if seconds > 0:
            first = int(cpu.read_text().split()[1])
            time.sleep(seconds)
            live["cores"] = round((int(cpu.read_text().split()[1]) - first) / 1e6 / seconds, 2)
        live["pids"] = int(Path("/sys/fs/cgroup/pids.current").read_text())
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        live["quota"] = float(os.cpu_count() or 16) if quota == "max" else float(quota) / float(period)
        limit = Path("/sys/fs/cgroup/pids.max").read_text().strip()
        live["pids_max"] = 3840 if limit == "max" else int(limit)
    except (OSError, ValueError, IndexError):
        pass
    if gpu is not None:
        try:
            q = subprocess.run(
                ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            total, used = (int(x) for x in q.stdout.strip().splitlines()[0].split(","))
            live["gpu_free_mib"] = total - used
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    return live


def group_threads(pids) -> int:
    """Threads currently held by these processes and everything in their process groups.

    What a stack that is ALREADY RUNNING costs right now, so a process attaching to it is not
    charged for it a second time (see `already` in `shortfall`).
    """
    wanted = set()
    for pid in pids:
        try:
            wanted.add(int(Path(f"/proc/{int(pid)}/stat").read_text().split(") ")[1].split()[2]))
        except (OSError, ValueError, IndexError):
            continue
    total = 0
    for task in Path("/proc").iterdir():
        if not task.name.isdigit():
            continue
        try:
            fields = Path(f"/proc/{task.name}/stat").read_text().split(") ")[1].split()
            if int(fields[2]) in wanted:
                total += int(fields[17])  # num_threads
        except (OSError, ValueError, IndexError):
            continue
    return total


def shortfall(stage: str, live: dict, *, record: bool = True, already: int = 0) -> str | None:
    """Why this stage must not start now, or None. Pure, so the thresholds are testable.

    `already` is what a stack this process is about to ATTACH to already holds. Collection and
    evaluation are two processes since 2026-09-21 -- one starts the stack, the other drives it --
    and both were charged the whole stage budget, so the driver was refused for resources the
    stack it was going to use had already taken (pids 1864 + 1800 > 3640 with a live stack holding
    313 of those 1864). Subtracting keeps the reservation the same size above the pre-stack
    baseline instead of doubling it.
    """
    need = NEEDS[stage]
    pids_needed = max((need["pids"] if record else min(need["pids"], NO_EXPORTER_PIDS)) - already, 0)
    # the live stack's cores are in `cores`
    cores_needed = ATTACH_CORES if already > 0 else need["cores"] if record else min(need["cores"], NO_EXPORTER_CORES)
    if (cores := live.get("cores")) is not None and cores + cores_needed > live.get("quota", 16.0):
        return f"cpu: {cores} cores in use + {cores_needed} needed > quota {live.get('quota')}"
    if (pids := live.get("pids")) is not None and pids + pids_needed > live.get("pids_max", 3840) - PID_MARGIN:
        return f"pids: {pids} + {pids_needed} needed > {live.get('pids_max', 3840) - PID_MARGIN}"
    free = live.get("gpu_free_mib")
    if need["gpu_free_mib"] and free is not None and free < need["gpu_free_mib"]:
        return f"gpu: {free} MiB free < {need['gpu_free_mib']} needed"
    return None


def require_headroom(stage: str, *, gpu: str | None = None, record: bool = True, already: int = 0) -> dict:
    """Refuse rather than wait: a saturated quota makes the sim miss its wall clock and every
    timed expert phase fail as a control error, which reads like a robot problem for hours.

    `record=False` is what a run without an exporter passes -- every evaluation, and collection
    with `collect.record` off. It asks for NO_EXPORTER_PIDS instead of the recording budget.
    """
    live = measure(gpu)
    if (why := shortfall(stage, live, record=record, already=already)) is not None:
        raise SystemExit(f"{stage}: not enough headroom -- {why}")
    return live


def require_stack_of(stack, run_dir: Path, mode: str) -> None:
    """Refuse to drive a stack another run -- or the other mode -- started.

    Starting and driving are two processes since 2026-09-21, and the stack's children were told
    this run's scene, trace directory, frame-audit file and video directory when they started. A
    driver attached to someone else's stack (a `--keep-stack` left over from another run, or the
    two terminals given different `--name`s) records into THAT run's directories with nothing to
    show for it but a split run. The mode matters the same way: the collect stack runs the deploy
    with its encoder, the eval stack without, and neither works for the other's driver.
    """
    owner = getattr(stack, "run_dir", None)
    started = getattr(stack, "mode", None)
    here = Path(run_dir).resolve()
    restart = (
        f"stop it (bash scripts/stop_stack.sh --instance {stack.spec.instance}) and start this run's own "
        f"(bash scripts/start_stack.sh {mode} <same target> ...)"
    )
    if owner is None:
        raise SystemExit(f"the stack on instance {stack.spec.instance} does not say which run started it; {restart}")
    if Path(owner).resolve() != here:
        raise SystemExit(f"the stack on instance {stack.spec.instance} serves {owner}, not {here}; {restart}")
    if started != mode:
        raise SystemExit(
            f"the stack on instance {stack.spec.instance} was started for {started!r}, this is {mode!r}; {restart}"
        )


def adopt_stack_env(stack) -> None:
    """Take the attached stack's environment into this process: its port block and its scene.

    The drivers used to be subprocesses handed exactly this as their whole environment. In-process,
    the ports are read when used, but `locomani.paths` resolved the scene at import, so it is
    re-pointed here too -- before anything builds an IK model from it.
    """
    os.environ.update({k: str(v) for k, v in stack.env.items()})
    if os.environ.get("LOCOMANI_SIM_DIR"):
        paths.adopt_sim_dir(os.environ["LOCOMANI_SIM_DIR"])


def require_train_split(cfg: dict) -> None:
    """`train.holdout` kept episodes go to the validation set; with none left to train on, convert used to fail
    only at stats_train with pandas' "No objects to concatenate". Check before collecting, not after.

    A run keeps at most `collect.successes` episodes, and at most one per round of its `batch_rounds` x
    `max_batches` budget. Checking only the first let a copy cut to 10 rounds keep `holdout: 10`, which every
    line's convert then refused (docs reproduction 2026-09-25). A run that records nothing is never converted.
    """
    collect = cfg["collect"]
    if not collect["record"]:
        return
    holdout, successes = cfg["train"]["holdout"], collect["successes"]
    rounds = collect["batch_rounds"] * collect["max_batches"]
    if holdout >= successes:
        raise SystemExit(
            f"train.holdout {holdout} >= collect.successes {successes}: the training set would be empty; "
            "lower train.holdout (0 skips the validation set)"
        )
    if holdout >= rounds:
        raise SystemExit(
            f"train.holdout {holdout} >= collect.batch_rounds {collect['batch_rounds']} x collect.max_batches "
            f"{collect['max_batches']} = {rounds} rounds: the training set would be empty; "
            "lower train.holdout (0 skips the validation set)"
        )


def require_deploy_patches(cfg: dict) -> None:
    """The streamed-pose experts need a patched deploy; check before anything is started."""
    from locomani.pipeline.provenance import require_pose_ownership

    strategy = (cfg.get("task", {}).get("reach") or {}).get("strategy")
    require_pose_ownership(dict(os.environ), hand_handoff=strategy in ("simple", "whole_body"))


def show(name: str, argv: list, env: dict | None = None) -> None:
    """The one dry-run format: what would run, and the settings that are this command's own."""
    print(f"\n[{name}]\n  " + " ".join(str(a) for a in argv))
    for key, value in sorted((env or {}).items()):
        print(f"    {key}={value}")


def show_config(name: str, config: dict) -> None:
    """The dry-run format for a stage that runs in THIS process.

    There is no argv to print when nothing is spawned, so it prints the resolved configuration the
    stage will act on. For the gym that is `GymOptions.report()`, which pairs every option with
    where it came from -- config default, task profile, or this stage (decision 10).
    """
    print(f"\n[{name}]")
    print(json.dumps(config, indent=2, sort_keys=True, default=str))


def summarise(run_dir: Path, stage: str, **metrics) -> Path:
    """One line per stage in `<run>/summary.json`: what the next stage and the reader both read."""
    path = Path(run_dir) / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary[stage] = dict(metrics, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    path.write_text(json.dumps(summary, indent=1, sort_keys=True))
    return path
