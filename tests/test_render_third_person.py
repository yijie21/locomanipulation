"""scripts/render_third_person.sh -> locomani.sim.render_third_person: one command, run whenever wanted, that films a
run's third-person videos with the renderer of its images.source. Which rounds, and where the videos go."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from locomani.sim import render_third_person as T

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, source="mujoco"):
    run = tmp_path / "run"
    (run / "trace").mkdir(parents=True)
    (run / "eval" / "state_trace").mkdir(parents=True)
    cfg = yaml.safe_load((ROOT / "configs/experiments/warehouse_smoke.yaml").read_text())
    cfg["images"] = {"source": source}
    if source == "mujoco":
        cfg["eval"]["backend"]["render"] = "live"
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    for label in ("r0", "r1", "r2"):
        np.savez(run / "trace" / f"{label}_state_trace.npz", qpos=np.zeros((4, 7)), seq=np.arange(4))
    (run / "manifest_00.jsonl").write_text(
        "".join(json.dumps({"label": f"r{k}", "ok": k != 1}) + "\n" for k in range(3))
    )
    np.savez(run / "eval" / "state_trace" / "crate_vla_000_state_trace.npz", qpos=np.zeros((4, 7)), seq=np.arange(4))
    (run / "eval" / "closed_loop.manifest.jsonl").write_text(json.dumps({"video": "crate_vla_000", "ok": False}) + "\n")
    return run


def test_rounds_are_the_successful_ones_every_one_or_the_named_ones(tmp_path):
    run = _run(tmp_path)
    assert T.chosen_rounds(run, evaluation=False, every=False, labels=[]) == ["r0", "r2"]
    assert T.chosen_rounds(run, evaluation=False, every=True, labels=[]) == ["r0", "r1", "r2"]
    assert T.chosen_rounds(run, evaluation=False, every=False, labels=["r1"]) == ["r1"]
    assert T.chosen_rounds(run, evaluation=True, every=False, labels=[]) == []  # the closed loop failed it
    assert T.chosen_rounds(run, evaluation=True, every=True, labels=[]) == ["crate_vla_000"]
    with pytest.raises(SystemExit, match="no state trace"):
        T.chosen_rounds(run, evaluation=False, every=False, labels=["r9"])


@pytest.mark.parametrize(
    "source, flags, where",
    [
        ("mujoco", [], "run/videos"),  # MuJoCo's own directory: where its live sim records
        ("mujoco", ["--eval", "--all"], "run/eval/videos"),
        ("isaac", [], "run/videos_isaac"),
        ("unreal", ["--eval", "--all"], "run/eval/videos_unreal"),
    ],
)
def test_a_dry_run_names_the_renderer_directory_and_skips_what_is_filmed(
    tmp_path, capsys, monkeypatch, source, flags, where
):
    from locomani.pipeline import run_context

    monkeypatch.setattr(run_context, "isaac_scene", lambda cfg: {"dir": "stage"})
    monkeypatch.setattr(run_context, "unreal_scene", lambda cfg: {"level_path": "/Game/Levels/x"})
    run = _run(tmp_path, source)
    (run / where.removeprefix("run/")).mkdir(parents=True, exist_ok=True)
    first = "crate_vla_000" if "--eval" in flags else "r0"
    (run / where.removeprefix("run/") / f"{first}.mp4").write_bytes(b"x")
    assert T.main([str(run), "--dry-run", *flags]) == 0
    printed = capsys.readouterr().out
    assert f"into {where} " in printed and "(1 of " in printed
