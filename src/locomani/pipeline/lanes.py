"""How many collection lanes a run gets, and where: a lane is one stack (sim, deploy, exporter) on its own instance
and one driver process that runs batches on it.

    plan = plan_run(cfg, gpus="all", want=None)   # measures the machine and picks instances; starts nothing
    print("\\n".join(plan.describe()))
    write(run_dir, plan)                           # collect/lanes.json: what the stack starter started

Inside a lane collection is serial -- the sim and the deploy run on the wall clock, one round at a time -- so the only
way to collect faster is more lanes. What bounds them is the machine, not the task: the container's CPU quota (a
starved sim fails timed phases as control errors; memory container-cpu-quota-15-cores), its pid ceiling (3840
threads for every session together), each card's free memory, how many rendering sims one card may carry, and which
instances have their port block free. `fit` is the arithmetic, pure so it is testable; `plan_run` measures.

The stack starter (`run_stack --mode collect`) decides the plan and starts the stacks; the driver
(`tasks.crate.collect`) reads `collect/lanes.json` and runs one process per lane. Only the bimanual line collects in
lanes (`lines.Line.parallel`); the single-arm driver still drives one stack.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import time
from pathlib import Path

from locomani.pipeline import run_context

# One lane's cost -- its stack AND its driver -- by (collect.record, backend.render). Measured 2026-09-28 for the
# warehouse lane (recording, offline; outputs/warehouse_parallel_20260928, container deltas over a 30 s baseline,
# one 4-round batch, real-time factor 0.993): +2.6 cores on average and +3.7 at the 90th percentile, +1.5 GB on its
# card, and +180 threads once up, then about +130 per recorded episode until the exporter restarts at the next
# batch (the exporter's threads grow with every episode it saves), peaking at +715. So a recording lane's pids are
# RECORD_PIDS + RECORD_PIDS_PER_ROUND x collect.batch_rounds: about 1600 for batches of 10, the old gate's 1800.
# The live-render lanes keep the single-stack gate's CPU and card budgets (run_context.NEEDS plus the driver),
# which nobody has re-measured per lane.
LANE_COST = {
    (True, "offline"): dict(cores=4.0, gpu_mib=2000),
    (True, "live"): dict(cores=10.0, gpu_mib=6000),
    # five of these ran side by side on GPU0 at real-time factor 0.93-0.98 (ledger crate.md 2026-09-26)
    (False, "offline"): dict(cores=3.0, gpu_mib=2000),
    (False, "live"): dict(cores=7.0, gpu_mib=6000),
}
RECORD_PIDS, RECORD_PIDS_PER_ROUND = 200, 140
NO_RECORD_PIDS = run_context.NO_EXPORTER_PIDS
# Sims per card. A rendering sim beside another on one card made the deploy trip on LowState gaps and chopped
# recordings into stubs (2026-09-08, memory no-sim-probes-beside-recording-stack); an offline sim renders nothing.
PER_GPU = {"offline": 4, "live": 1}
MAX_LANES = 8
INSTANCES = range(10)  # port blocks 5555 + 10*i, all below the policy servers' 5700/5710
DRIVER_CORES = run_context.ATTACH_CORES  # what the lane's driver process adds once its stack is up
CPU_SAMPLE_S = 3.0  # the other sessions' load moves by a core or more second to second


@dataclasses.dataclass(frozen=True)
class Lane:
    index: int  # also the batch its stack's first exporter session was opened for
    instance: int
    gpu: str


@dataclasses.dataclass
class Plan:
    lanes: list[Lane]
    limits: dict  # what each resource allows, in lanes; the smallest is the answer
    live: dict  # the measurement the limits came from
    cost: dict  # one lane's cost that was assumed
    busy: dict = dataclasses.field(default_factory=dict)  # instance -> why it was skipped

    def describe(self) -> list[str]:
        """The lines a collection prints before it starts, and again when its driver attaches."""
        n = len(self.lanes)
        where = ", ".join(f"instance {lane.instance} on GPU {lane.gpu}" for lane in self.lanes)
        head = (
            f"collect: {n} lane{'s' if n != 1 else ''} in parallel -- one stack and one driver process each ({where})"
            if n
            else "collect: no lane fits on this machine now"
        )
        free_cores = self.live.get("quota", 0) - self.live.get("cores", 0)
        free_pids = self.live.get("pids_max", 3840) - run_context.PID_MARGIN - self.live.get("pids", 0)
        facts = {
            "cpu": f"{free_cores:.1f} cores free / {self.cost['cores']:g} per lane",
            "pids": f"{free_pids} free / {self.cost['pids']} per lane",
            "gpu": f"{self.cost['gpu_mib']} MiB per lane, "
            + ", ".join(f"GPU {g} {m} MiB free" for g, m in self.live.get("gpu_free_mib", {}).items()),
            "instances": "free port blocks"
            + (f" (skipped {', '.join(f'{i}: {why}' for i, why in self.busy.items())})" if self.busy else ""),
            "requested": "--lanes / collect.lanes",
            "ceiling": "lanes.MAX_LANES",
        }
        lines = [head, "  what fits: " + "; ".join(f"{k} {v} ({facts[k]})" for k, v in self.limits.items())]
        if n > 1:
            lines.append("  one lane only: --lanes 1 when starting the stack (or collect.lanes: 1 in the experiment)")
        return lines

    def to_dict(self) -> dict:
        return dict(
            lanes=[dataclasses.asdict(lane) for lane in self.lanes],
            limits=self.limits,
            live=self.live,
            cost=self.cost,
            busy={str(k): v for k, v in self.busy.items()},
            planned=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


def lane_cost(record: bool, render: str, batch_rounds: int) -> dict:
    """One lane's cores, pids and card MiB (LANE_COST; a recording lane's pids grow with its batch length)."""
    pids = RECORD_PIDS + RECORD_PIDS_PER_ROUND * int(batch_rounds) if record else NO_RECORD_PIDS
    return dict(LANE_COST[(bool(record), render)], pids=pids)


def fit(live: dict, cost: dict, gpu_free: dict, *, per_gpu: int, instances: int, want: int | None) -> tuple[dict, dict]:
    """How many lanes each resource allows, and how many go on each card: (limits, slots per GPU).

    A resource the measurement could not read does not limit. The answer is the smallest limit."""
    limits: dict[str, int] = {}
    if "cores" in live and "quota" in live:
        limits["cpu"] = max(0, math.floor((live["quota"] - live["cores"]) / cost["cores"]))
    if "pids" in live:
        room = live.get("pids_max", 3840) - run_context.PID_MARGIN - live["pids"]
        limits["pids"] = max(0, math.floor(room / cost["pids"]))
    slots = {
        gpu: per_gpu if free is None else max(0, min(per_gpu, math.floor(free / cost["gpu_mib"])))
        for gpu, free in gpu_free.items()
    }
    limits["gpu"] = sum(slots.values())
    limits["instances"] = instances
    if want is not None:
        limits["requested"] = want
    limits["ceiling"] = MAX_LANES
    return limits, slots


def spread(n: int, slots: dict, instances: list[int]) -> list[Lane]:
    """`n` lanes over the cards in turn (a card takes another only while it has a slot), one instance each."""
    remaining, order = dict(slots), []
    while len(order) < n and any(remaining.values()):
        for gpu in list(remaining):
            if remaining[gpu] > 0 and len(order) < n:
                order.append(gpu)
                remaining[gpu] -= 1
    return [Lane(index=k, instance=instances[k], gpu=gpu) for k, gpu in enumerate(order)]


def gpu_list(value: str | None) -> list[str]:
    """`--gpu` as cards: "0", "0,1" or "all" (every card nvidia-smi lists; also the default)."""
    if value and value != "all":
        return [g.strip() for g in str(value).split(",") if g.strip()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    return [line.strip() for line in out.splitlines() if line.strip()] or ["0"]


def wanted(cli: str | None, cfg: dict) -> int | None:
    """Lanes asked for: `--lanes`, else `collect.lanes`, else "auto" (None: as many as fit)."""
    value = cli if cli is not None else (cfg.get("collect") or {}).get("lanes", "auto")
    if str(value) == "auto":
        return None
    if not str(value).isdigit() or int(value) < 1:
        raise SystemExit(f"lanes must be a positive number or auto, not {value!r}")
    return int(value)


def plan_run(cfg: dict, *, gpus: str | None, want: int | None) -> Plan:
    """Measure the machine and plan the run's lanes. `want == 1` keeps the old contract exactly: the one stack
    goes on `collect.instance` or nowhere. More lanes start there and take the next free instances."""
    from locomani.stack import instance_busy

    record = bool(cfg["collect"]["record"])
    render = (cfg.get("backend") or {}).get("render", "live")
    cost = lane_cost(record, render, int(cfg["collect"]["batch_rounds"]))
    first = int(cfg["collect"].get("instance", 0))
    candidates = [first] if want == 1 else [first, *(i for i in INSTANCES if i != first)]
    free, busy = [], {}
    for instance in candidates:
        why = instance_busy(instance)
        if why is None:
            free.append(instance)
        else:
            busy[instance] = why
    live = run_context.measure(None, seconds=CPU_SAMPLE_S)
    cards = gpu_list(gpus)
    gpu_free = {gpu: run_context.measure(gpu, seconds=0).get("gpu_free_mib") for gpu in cards}
    live["gpu_free_mib"] = gpu_free
    limits, slots = fit(live, cost, gpu_free, per_gpu=PER_GPU[render], instances=len(free), want=want)
    return Plan(spread(min(limits.values()), slots, free), limits, live, cost, busy)


def path(run_dir: Path) -> Path:
    return Path(run_dir) / "collect" / "lanes.json"


def write(run_dir: Path, plan: Plan) -> Path:
    target = path(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.to_dict(), indent=1))
    return target


def read(run_dir: Path) -> Plan | None:
    """The plan the stack starter wrote, or None (a stack started before lanes existed: one lane)."""
    if not path(run_dir).exists():
        return None
    data = json.loads(path(run_dir).read_text())
    return Plan(
        [Lane(**lane) for lane in data["lanes"]],
        data["limits"],
        data["live"],
        data["cost"],
        {int(k): v for k, v in data.get("busy", {}).items()},
    )


def require_drivers(n: int, already: int, cost: dict) -> dict:
    """Before the drivers start: the lanes' stacks are up (their cores are in the measurement), their drivers are not.
    Refuse rather than start `n` drivers the quota cannot carry; the pids reserve stays `n` lanes' worth."""
    live = run_context.measure(None, seconds=CPU_SAMPLE_S)
    if "cores" in live and live["cores"] + n * DRIVER_CORES > live.get("quota", 16.0):
        raise SystemExit(
            f"collect: not enough headroom for {n} driver(s) -- cpu: {live['cores']} cores in use + "
            f"{n} x {DRIVER_CORES} > quota {live.get('quota')}; restart the stacks with fewer lanes (--lanes)"
        )
    need = max(n * cost["pids"] - already, 0)
    if "pids" in live and live["pids"] + need > live.get("pids_max", 3840) - run_context.PID_MARGIN:
        raise SystemExit(f"collect: not enough headroom -- pids: {live['pids']} + {need} needed")
    return live
