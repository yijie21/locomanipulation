"""Launcher contract: manifest-based skipping, resource admission, registry lines (infra plan decisions 4, 6, 8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from locomani import launch
from locomani.stages import common


def test_admission_is_refused_on_cpu_pids_or_gpu_memory_and_granted_otherwise():
    need = dict(cpu_cores=8, pids=2700, gpu_mib_free=6000)
    live = dict(cores_in_use=6.0, pids=500, gpu_used_mib={"1": 2000})
    assert launch.admissible(need, live, "1", quota=15.36, pids_max=3840, gpu_total=32000) is None
    assert "cpu" in launch.admissible(
        need, dict(live, cores_in_use=9.0), "1", quota=15.36, pids_max=3840, gpu_total=32000
    )
    assert "pids" in launch.admissible(need, dict(live, pids=1500), "1", quota=15.36, pids_max=3840, gpu_total=32000)
    assert "gpu" in launch.admissible(
        need, dict(live, gpu_used_mib={"1": 30000}), "1", quota=15.36, pids_max=3840, gpu_total=32000
    )
    assert (
        launch.admissible(need, dict(cores_in_use=None, pids=None), None, quota=15.36, pids_max=3840, gpu_total=0)
        is None
    )


def test_stage_is_current_only_with_matching_config_and_upstream_hashes(tmp_path, monkeypatch):
    job = launch.Job(dict(name="j", overrides=["+experiment=bottle_simple_lowcom"], stages=["collect", "convert"]))
    monkeypatch.setattr(job, "out", tmp_path)
    assert not launch.stage_is_current(job, "collect")
    common.write_manifest(tmp_path, "collect", job.cfg, ok=True, outputs={}, metrics={}, started=0.0, resources={})
    assert launch.stage_is_current(job, "collect")
    common.write_manifest(tmp_path, "convert", job.cfg, ok=True, outputs={}, metrics={}, started=0.0, resources={})
    assert launch.stage_is_current(job, "convert")
    # the upstream manifest changes -> convert is stale even though its own manifest is fine
    common.write_manifest(
        tmp_path, "collect", job.cfg, ok=True, outputs={}, metrics={"kept": 1}, started=1.0, resources={}
    )
    assert not launch.stage_is_current(job, "convert")
    # a failed manifest never counts
    common.write_manifest(tmp_path, "collect", job.cfg, ok=False, outputs={}, metrics={}, started=0.0, resources={})
    assert not launch.stage_is_current(job, "collect")


def test_launcher_runs_stages_in_order_per_job_and_appends_the_registry(tmp_path, monkeypatch):
    jobs = launch.load_jobs(Path(launch.paths.REPO) / "configs/jobs/example.yaml")
    for j in jobs:
        monkeypatch.setattr(j, "out", tmp_path / j.name)
    registry = tmp_path / "registry.jsonl"
    calls = []

    def fake_call(argv, cwd=None, env=None, stdout=None, stderr=None):
        stage = argv[3]
        job = next(j for j in jobs if f"name={j.run_cfg.name}" in argv)
        calls.append((job.name, stage, [a for a in argv if a.startswith("gpu=")]))
        common.write_manifest(
            job.out, stage, job.cfg, ok=True, outputs={}, metrics=dict(n=len(calls)), started=0.0, resources={}
        )
        return 0

    monkeypatch.setattr(launch.subprocess, "call", fake_call)
    monkeypatch.setattr(
        launch.common, "resources_now", lambda: dict(cores_in_use=1.0, pids=100, gpu_used_mib={"0": 0, "1": 0})
    )
    monkeypatch.setattr(launch, "gpu_total_mib", lambda index: 32000)
    results = launch.Launcher(jobs, ["0", "1"], registry=registry, poll_s=0.01).run()
    assert all(r["ok"] for recs in results.values() for r in recs)
    order = {name: [s for n, s, _ in calls if n == name] for name in results}
    assert order["simple_fixed8"] == ["collect", "stats"]
    assert order["pick_place_seedA"] == ["collect", "convert", "train", "eval", "stats"]
    assert all(g == [] for n, s, g in calls if s in ("convert", "stats"))  # GPU-less stages get no card
    assert all(len(g) == 1 for n, s, g in calls if s in ("collect", "train", "eval"))
    lines = [json.loads(line) for line in registry.read_text().splitlines()]
    assert len(lines) == 7 and {line["stage"] for line in lines} == {"collect", "convert", "train", "eval", "stats"}
    # second run: everything is up to date, nothing is executed
    calls.clear()
    launch.Launcher(jobs, ["0", "1"], registry=registry, poll_s=0.01).run()
    assert calls == []


def test_launcher_stops_a_job_at_its_first_failed_stage(tmp_path, monkeypatch):
    jobs = launch.load_jobs(Path(launch.paths.REPO) / "configs/jobs/example.yaml")[1:]
    monkeypatch.setattr(jobs[0], "out", tmp_path / "j")
    monkeypatch.setattr(launch.subprocess, "call", lambda *a, **k: 1)
    monkeypatch.setattr(launch.common, "resources_now", lambda: dict(cores_in_use=1.0, pids=100, gpu_used_mib={"0": 0}))
    monkeypatch.setattr(launch, "gpu_total_mib", lambda index: 32000)
    results = launch.Launcher(jobs, ["0"], registry=tmp_path / "r.jsonl", poll_s=0.01).run()
    assert [r["stage"] for r in results["pick_place_seedA"]] == ["collect"] and not results["pick_place_seedA"][0]["ok"]


def test_dry_run_plans_without_executing(tmp_path, monkeypatch):
    jobs = launch.load_jobs(Path(launch.paths.REPO) / "configs/jobs/example.yaml")[:1]
    monkeypatch.setattr(jobs[0], "out", tmp_path / "j")
    monkeypatch.setattr(launch.common, "resources_now", lambda: dict(cores_in_use=1.0, pids=100, gpu_used_mib={"0": 0}))
    monkeypatch.setattr(launch, "gpu_total_mib", lambda index: 32000)
    results = launch.Launcher(jobs, ["0"], registry=tmp_path / "r.jsonl", dry_run=True, poll_s=0.01).run()
    assert [r["stage"] for r in results["simple_fixed8"]] == ["collect", "stats"]
    assert all(r["dry_run"] for r in results["simple_fixed8"])
    assert not (tmp_path / "r.jsonl").exists()
