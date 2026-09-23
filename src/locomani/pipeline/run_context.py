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

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from locomani import paths

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
    cfg = resolve_scene(yaml.safe_load(experiment.read_text()))
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
    return resolve_scene(yaml.safe_load(path.read_text()))


def resolve_scene(cfg: dict, registry: Path = SCENES) -> dict:
    """Turn `scene: {name: warehouse}` into the `scene.dir` every consumer reads; returns `cfg`.

    Resolved here, where both the experiment and a run's copy are read, so no consumer knows names
    exist. The copy keeps the name; `configs/scenes.yaml` promises a name never moves.
    """
    scene = cfg.get("scene") or {}
    if "name" not in scene:
        return cfg
    if scene.get("dir"):
        raise SystemExit(f"scene gives both name {scene['name']!r} and dir {scene['dir']!r}; give one")
    names = yaml.safe_load(Path(registry).read_text())
    if scene["name"] not in names:
        raise SystemExit(f"no scene called {scene['name']!r}; {registry} names {', '.join(sorted(names))}")
    scene["dir"] = names[scene["name"]]["dir"]
    return cfg


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


def measure(gpu: str | None = None) -> dict:
    """Cores in use (1 s cgroup delta), pids in the container, free MiB on the card."""
    live: dict = {}
    try:
        cpu = Path("/sys/fs/cgroup/cpu.stat")
        first = int(cpu.read_text().split()[1])
        time.sleep(1.0)
        live["cores"] = round((int(cpu.read_text().split()[1]) - first) / 1e6, 2)
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
    cores_needed = ATTACH_CORES if already > 0 else need["cores"]  # the live stack's cores are in `cores`
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
