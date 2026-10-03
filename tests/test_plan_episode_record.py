"""plan_episode --record: what it refuses, and the exporter keys and training_tail it leaves per round."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import plan_episode  # noqa: E402

EXP = "configs/experiments/bottle_simple_200.yaml"


@pytest.mark.parametrize(
    "extra, refused",
    [
        (["--agent"], "--agent"),
        (["--inject", "x.json"], "--inject"),
        (["--think-s", "8"], "--think-s"),
        (["--instance", "1"], "only instance 0"),
    ],
)
def test_record_refuses_what_a_training_episode_must_not_contain(extra, refused, capsys):
    base = [EXP, "--record"] + ([] if "--instance" in extra else ["--instance", "0"])
    with pytest.raises(SystemExit):
        plan_episode.parse(base + extra)
    assert refused in capsys.readouterr().err
    assert plan_episode.parse([EXP, "--record", "--instance", "0"]).record


class Cmd:
    def __init__(self):
        self.calls = []

    def key(self, k):
        self.calls.append(("key", k))

    def hold(self, seconds, **kw):
        self.calls.append(("hold", seconds))


class Stack:
    def __init__(self, seq=1234):
        self.cmd, self.seq = Cmd(), seq

    def obs(self):
        return {"seq": self.seq}


def session(metrics=None):
    return SimpleNamespace(record=SimpleNamespace(metrics=dict(metrics or {})))


def test_a_success_is_saved_with_a_training_tail_and_a_failure_is_discarded(tmp_path):
    rec = plan_episode.Recording(str(tmp_path / "session"), tmp_path / "manifest_00.jsonl")
    p, s = Stack(), session()
    rec.start(p)
    rec.stop(p, s, True, (1.0, 0.0, 0.0))
    assert p.cmd.calls[0] == ("key", "c") and ("key", "c") in p.cmd.calls[1:]
    assert s.record.metrics["training_tail"] == {"end_seq": 1234, "reason": "plan_end"}
    # the plan-end mark is held a moment so it falls inside the recorded frames, before the save
    assert p.cmd.calls[1] == ("hold", plan_episode.Recording.TAIL_HOLD_S) and p.cmd.calls[2] == ("key", "c")

    p, s = Stack(), session({"training_tail": {"end_seq": 99, "reason": "post_transport_hold_verification"}})
    rec.stop(p, s, True, (1.0, 0.0, 0.0))
    assert s.record.metrics["training_tail"]["end_seq"] == 99  # the primitive's own boundary wins
    assert p.cmd.calls[0] == ("key", "c")

    p, s = Stack(), session()
    rec.stop(p, s, False, (1.0, 0.0, 0.0))
    assert p.cmd.calls[0] == ("key", "x") and "training_tail" not in s.record.metrics


def test_rows_land_in_the_collect_manifest_the_converter_reads(tmp_path):
    rec = plan_episode.Recording(str(tmp_path / "session"), tmp_path / "manifest_00.jsonl")
    rec.write({"round": 0, "ok": True, "produced": True, "task": "pick up the water bottle"})
    rec.write({"round": 1, "ok": False, "produced": False, "task": "pick up the water bottle"})
    rows = [json.loads(line) for line in (tmp_path / "manifest_00.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows] == [0, 1] and rec.rows == rows
