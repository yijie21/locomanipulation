"""A stage's config must reach the process that consumes it.

`test_eval_config_reaches_the_client.py` covers the in-process consumers; this file covers the
seam the 2026-09-17 regression hid in: `stages.eval` builds a command line for
`eval_closed_loop.py`, and hand-copying four fields into `--set` left every other field at the
evaluator's own defaults (the 100-round run spawned at 0.7 m although the group said 1.1-1.3 m,
and the eval group's whole `backend` block -- including `sim_vis`, hence the missing videos --
was replaced by a freshly constructed one).
"""

from __future__ import annotations

import dataclasses
import json

from locomani import conf
from locomani.pipeline.eval_config import EvalConfig
from locomani.stages import eval_stage


def test_eval_stage_sends_the_whole_config_file(tmp_path):
    """The stage writes the resolved EvalConfig and points the client at it."""
    run, _ = conf.load_run(["+experiment=bottle_simple_learn200"])
    path = tmp_path / "eval_config.json"
    run.eval.dump(path)
    assert EvalConfig.load(path) == run.eval  # a round trip the client can read
    written = json.loads(path.read_text())
    assert written["spawn_edge"] == [1.1, 1.3]  # the group's value, not EvalConfig()'s 0.7 default
    assert written["initial_token"].endswith("whole_body_initial_token.json")


def test_eval_stack_follows_the_eval_group_backend(tmp_path):
    """Whatever the eval group's backend block says is what the eval stack gets."""
    run, _ = conf.load_run(["+experiment=bottle_simple_learn200"])
    spec, env = eval_stage.eval_stack(run, tmp_path, tmp_path / "eval")
    assert spec.encoder is not run.eval.backend.no_encoder  # token streaming needs the encoder off
    assert spec.sim_vis == run.eval.backend.sim_vis
    assert spec.video_dir == str(tmp_path / "eval" / "videos")
    assert spec.child_env(dict(env))["SIM_VIDEO_DIR"] == spec.video_dir  # reaches the sim process

    louder = dataclasses.replace(run, eval=run.eval.with_overrides(["backend.sim_vis=full"]))
    assert eval_stage.eval_stack(louder, tmp_path, tmp_path / "eval")[0].sim_vis == "full"
