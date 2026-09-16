"""Launcher: a job list -> stage subprocesses on free GPUs, admitted by measured headroom, logged to a registry.

    python -m locomani.launch run configs/jobs/example.yaml [--gpus 0,1] [--dry-run]
    python -m locomani.launch ls [--registry outputs/registry.jsonl] [--name PATTERN]
    python -m locomani.launch compare RUN_A RUN_B [--stage eval]

Decisions 1, 4, 6 and 8 of docs/handoff/infra-plan-20260916.md. A job is `overrides` for
`locomani.cli` plus the stages to run; stages of one job run in order, each as its own process
(`python -m locomani.cli <stage> <overrides> gpu=<g>`), jobs run in parallel across GPUs. A stage
is skipped when its manifest exists, is ok, and its config hash and upstream hash still match.
Admission: static GPU ownership (one job per card at a time) plus a live check of the container's
CPU quota, pid count and the card's foreign memory before every stage. Every finished stage appends
one line to `outputs/registry.jsonl`.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from locomani import conf, paths
from locomani.stages import common

REGISTRY = Path(paths.OUTPUTS) / "registry.jsonl"
CPU_QUOTA_FALLBACK = 16.0
PID_LIMIT_FALLBACK = 3840
PID_MARGIN = 200
DEFAULT_RESOURCES = {  # declared per stage (decision 6); a job's `resources` overrides per stage
    "collect": dict(cpu_cores=8, pids=2700, gpu_mib_free=6000),
    "convert": dict(cpu_cores=4, pids=200, gpu_mib_free=0),
    "train": dict(cpu_cores=4, pids=300, gpu_mib_free=28000),
    "eval": dict(cpu_cores=8, pids=2700, gpu_mib_free=12000),
    "stats": dict(cpu_cores=1, pids=10, gpu_mib_free=0),
}
NEEDS_GPU = {"collect", "train", "eval"}


def cpu_quota() -> float:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return float(quota) / float(period) if quota != "max" else float(os.cpu_count() or CPU_QUOTA_FALLBACK)
    except (OSError, ValueError):
        return CPU_QUOTA_FALLBACK


def pid_limit() -> int:
    try:
        text = Path("/sys/fs/cgroup/pids.max").read_text().strip()
        return int(text) if text != "max" else PID_LIMIT_FALLBACK
    except (OSError, ValueError):
        return PID_LIMIT_FALLBACK


def gpu_total_mib(index: str) -> int:
    try:
        q = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(q.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0


def admissible(need: dict, live: dict, gpu: str | None, *, quota: float, pids_max: int, gpu_total: int) -> str | None:
    """None when the stage may start now, else the reason it must wait (pure function for tests)."""
    cores = live.get("cores_in_use")
    if cores is not None and cores + need["cpu_cores"] > quota:
        return f"cpu: {cores:.1f} in use + {need['cpu_cores']} > quota {quota:.1f}"
    pids = live.get("pids")
    if pids is not None and pids + need["pids"] > pids_max - PID_MARGIN:
        return f"pids: {pids} + {need['pids']} > {pids_max - PID_MARGIN}"
    if gpu is not None and need.get("gpu_mib_free", 0) and gpu_total:
        used = live.get("gpu_used_mib", {}).get(str(gpu))
        if used is not None and gpu_total - used < need["gpu_mib_free"]:
            return f"gpu {gpu}: {gpu_total - used} MiB free < {need['gpu_mib_free']}"
    return None


class Job:
    def __init__(self, spec: dict):
        self.name = spec["name"]
        self.overrides = list(spec.get("overrides", []))
        self.stages = list(spec.get("stages", ["collect", "convert", "train", "eval", "stats"]))
        self.gpu = spec.get("gpu", "any")
        self.resources = {s: dict(DEFAULT_RESOURCES[s], **(spec.get("resources", {}).get(s, {}))) for s in self.stages}
        for s in self.stages:
            if s not in conf.STAGES:
                raise ValueError(f"job {self.name}: unknown stage {s!r}")
        self.run_cfg, self.cfg = conf.load_run([*self.overrides, f"name={spec.get('run_name', self.name)}"])
        self.out = conf.resolve_repo_path(self.run_cfg.out)


def stage_is_current(job: Job, stage: str) -> bool:
    """Skip rule: manifest exists, ok, same config hash, same upstream manifest hash."""
    m = common.read_manifest(job.out, stage)
    if m is None or not m.get("ok"):
        return False
    if m.get("config_hash") != common.config_hash(job.cfg, stage):
        return False
    up = common.UPSTREAM[stage]
    if up and m.get("upstream", {}).get(up) != common.manifest_hash(job.out, up):
        return False
    return True


def append_registry(record: dict, registry: Path = REGISTRY) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


class Launcher:
    def __init__(
        self,
        jobs: list[Job],
        gpus: list[str],
        *,
        registry: Path = REGISTRY,
        dry_run: bool = False,
        poll_s: float = 30.0,
    ):
        self.jobs, self.gpus, self.registry, self.dry_run, self.poll_s = jobs, gpus, registry, dry_run, poll_s
        self.free = set(gpus)
        self.lock = threading.Lock()
        self.quota, self.pids_max = cpu_quota(), pid_limit()
        self.totals = {g: gpu_total_mib(g) for g in gpus}
        self.results: dict[str, list[dict]] = {}

    def _acquire_gpu(self, job: Job, stage: str) -> str | None:
        """Block until a card is free and the live headroom admits the stage; None for GPU-less stages."""
        need = job.resources[stage]
        wants = [job.gpu] if job.gpu != "any" else list(self.gpus)
        if self.dry_run:  # plan only: the first card the job may use, no live admission
            return wants[0] if stage in NEEDS_GPU else None
        waited = 0.0
        while True:
            live = common.resources_now()
            with self.lock:
                candidates = [g for g in wants if g in self.free] if stage in NEEDS_GPU else [None]
                for g in candidates:
                    why = admissible(
                        need,
                        live,
                        g,
                        quota=self.quota,
                        pids_max=self.pids_max,
                        gpu_total=self.totals.get(g, 0) if g else 0,
                    )
                    if why is None:
                        if g is not None:
                            self.free.discard(g)
                        return g
                why = why if candidates else "no free gpu"
            if waited == 0 or int(waited) % 300 == 0:
                print(f"[launch] {job.name}/{stage} waiting: {why}", flush=True)
            time.sleep(self.poll_s)
            waited += self.poll_s

    def _release_gpu(self, g: str | None) -> None:
        if g is not None:
            with self.lock:
                self.free.add(g)

    def _run_stage(self, job: Job, stage: str, gpu: str | None) -> dict:
        argv = [sys.executable, "-m", "locomani.cli", stage, *job.overrides, f"name={job.run_cfg.name}"]
        if gpu is not None:
            argv.append(f"gpu={gpu}")
        job.out.mkdir(parents=True, exist_ok=True)
        log = job.out / f"launch_{stage}.log"
        started = time.time()
        print(f"[launch] {job.name}/{stage} start gpu={gpu} log={log}", flush=True)
        if self.dry_run:
            return dict(stage=stage, ok=True, dry_run=True, argv=argv)
        with log.open("a") as f:
            f.write("COMMAND " + json.dumps(argv) + "\n")
            f.flush()
            rc = subprocess.call(
                argv,
                cwd=paths.REPO,
                env=dict(os.environ, PYTHONPATH=os.path.join(paths.REPO, "src")),
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        m = common.read_manifest(job.out, stage) or {}
        record = dict(
            job=job.name,
            name=job.run_cfg.name,
            stage=stage,
            ok=bool(rc == 0 and m.get("ok")),
            rc=rc,
            gpu=gpu,
            out=str(job.out),
            config_hash=m.get("config_hash"),
            upstream=m.get("upstream"),
            metrics=m.get("metrics"),
            resources_at_start=m.get("resources_at_start"),
            started=started,
            seconds=round(time.time() - started, 1),
            launcher_pid=os.getpid(),
            overrides=job.overrides,
        )
        append_registry(record, self.registry)
        print(f"[launch] {job.name}/{stage} {'ok' if record['ok'] else 'FAIL'} ({record['seconds']} s)", flush=True)
        return record

    def _run_job(self, job: Job) -> None:
        self.results[job.name] = []
        for stage in job.stages:
            if stage_is_current(job, stage):
                print(f"[launch] {job.name}/{stage} up to date, skipped", flush=True)
                self.results[job.name].append(dict(stage=stage, skipped=True, ok=True))
                continue
            gpu = self._acquire_gpu(job, stage)
            try:
                record = self._run_stage(job, stage, gpu)
            finally:
                self._release_gpu(gpu)
            self.results[job.name].append(record)
            if not record["ok"]:
                print(f"[launch] {job.name}: stopping at {stage}", flush=True)
                return

    def run(self) -> dict:
        threads = [threading.Thread(target=self._run_job, args=(job,), name=job.name) for job in self.jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return self.results


def load_jobs(path: Path) -> list[Job]:
    doc = yaml.safe_load(Path(path).read_text())
    return [Job(spec) for spec in doc["jobs"]]


def cmd_ls(registry: Path, pattern: str | None) -> None:
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()] if registry.exists() else []
    if pattern:
        rows = [r for r in rows if fnmatch.fnmatch(r.get("name", ""), pattern)]
    print(f"{'name':28s} {'stage':8s} {'ok':3s} {'gpu':4s} {'seconds':>8s}  metrics")
    for r in rows:
        print(
            f"{r.get('name', ''):28s} {r.get('stage', ''):8s} {'ok' if r.get('ok') else 'NO':3s} {str(r.get('gpu')):4s} {r.get('seconds', 0):8.1f}  {json.dumps(r.get('metrics'))[:80]}"
        )


def cmd_compare(a: Path, b: Path, stage: str) -> None:
    ma, mb = common.read_manifest(a, stage), common.read_manifest(b, stage)
    if ma is None or mb is None:
        sys.exit(f"both runs need a {stage} manifest")
    keys = sorted(set(ma["metrics"]) | set(mb["metrics"]))
    print(f"{'metric':24s} {a.name:>20s} {b.name:>20s}")
    for k in keys:
        print(f"{k:24s} {str(ma['metrics'].get(k)):>20s} {str(mb['metrics'].get(k)):>20s}")
    print(f"config_hash same: {ma['config_hash'] == mb['config_hash']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("jobs", type=Path)
    r.add_argument("--gpus", default="0", help="comma-separated GPU indices the launcher may use")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--registry", type=Path, default=REGISTRY)
    r.add_argument("--poll", type=float, default=30.0)
    ls = sub.add_parser("ls")
    ls.add_argument("--registry", type=Path, default=REGISTRY)
    ls.add_argument("--name", default=None)
    c = sub.add_parser("compare")
    c.add_argument("run_a", type=Path)
    c.add_argument("run_b", type=Path)
    c.add_argument("--stage", default="eval")
    args = ap.parse_args(argv)
    if args.verb == "run":
        jobs = load_jobs(args.jobs)
        results = Launcher(
            jobs, args.gpus.split(","), registry=args.registry, dry_run=args.dry_run, poll_s=args.poll
        ).run()
        failed = [j for j, recs in results.items() if any(not r.get("ok") for r in recs)]
        print(
            json.dumps(
                {
                    j: [dict(stage=r.get("stage"), ok=r.get("ok"), skipped=r.get("skipped", False)) for r in recs]
                    for j, recs in results.items()
                },
                indent=1,
            )
        )
        sys.exit(1 if failed else 0)
    if args.verb == "ls":
        cmd_ls(args.registry, args.name)
    if args.verb == "compare":
        cmd_compare(args.run_a, args.run_b, args.stage)


if __name__ == "__main__":
    main()
