"""`plan_episode.think` (TODO 18.3.5): the fixed plan's stand-in for an agent's decision latency.

The deploy drops the frozen upper body and the hand command after 1 s without a planner message
(zmq_manager.hpp PLANNER_TIMEOUT), so the wait must keep sending frames unless `--think-silent` asks for the
agent runner's gap; either way a single-arm load's palm-frame drift is traced and its stream saved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from locomani.skills import Executor, HeldObject, Session

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class Stack:
    """The load slides 1 mm along the palm's x per observation; the palm stays at the origin, unrotated."""

    def __init__(self):
        self.t, self.frames = 0.0, []
        self.cmd = SimpleNamespace(frame=lambda **kw: self.frames.append(kw))

    def obs(self):
        self.t += 0.02
        x = 0.1 + 0.05 * self.t
        return {
            "t": self.t,
            "fallen": False,
            "pelvis": {"pos": [0.0, 0.0, 0.79], "yaw": 0.5},
            "palms": {"right": {"pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}},
            "objects": {"soup_can": {"pos": [x, 0.0, 0.9], "touching": ["right"]}},
            "qpos": [self.t, x],
        }

    def pelvis_yaw(self, s=None):
        return 0.5


def executor(held: bool, tmp_path: Path) -> Executor:
    session = Session(stack=Stack(), scene=None)
    if held:
        session.held = HeldObject("soup_can", ("right",), {"right": np.zeros(3)})
        session.hands["right"] = 1.0
    return Executor(session, {}, trace_path=tmp_path / "trace.jsonl")


def test_keepalive_sends_frames_and_traces_the_drift(tmp_path):
    from plan_episode import think

    ex = executor(True, tmp_path)
    event = think(ex, 0.3, tmp_path / "hold.npz")
    frames = ex.session.stack.frames
    assert len(frames) >= 5 and all(f["facing"] == (np.cos(0.5), np.sin(0.5), 0.0) for f in frames)
    assert event["event"] == "think" and event["keepalive"] and event["held"] == "soup_can"
    assert event["drift_m"] > 0 and event["touching_end"] and 4.0 < event["drift_rate_cm_s"] < 6.0
    saved = np.load(event["npz"])
    assert saved["qpos"].shape[1] == 3 and len(saved["drift"]) == len(saved["qpos"])
    assert json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[-1])["event"] == "think"


def test_silent_sends_nothing_and_an_empty_hand_records_no_drift(tmp_path):
    from plan_episode import scripted_round, think

    ex = executor(True, tmp_path)
    assert think(ex, 0.1, keepalive=False)["keepalive"] is False and ex.session.stack.frames == []
    empty = think(executor(False, tmp_path), 0.05)
    assert empty["held"] is None and "drift_m" not in empty

    class Prim:
        name, requires, yields = "noop", frozenset(ex.session.arm_owner.__class__), None

        def preconditions(self, session):
            return []

        def run(self, session):
            from locomani.skills.contract import Outcome

            return Outcome(True, "")

    ex.catalogue["noop"] = Prim()
    entries = scripted_round(ex, [{"primitive": "noop"}], think_s=0.05, keepalive=False)
    assert [e.get("event", e.get("primitive")) for e in ex.session.trace][-2:] == ["noop", "think"]
    assert len(entries) == 1


# ---- Executor.idle: the wait every caller shares ------------------------------------------------------------------


def test_idle_until_a_callable_keeps_a_planner_arm_alive(tmp_path):
    ex = executor(True, tmp_path)
    frames = ex.session.stack.frames
    ex.idle(lambda: len(frames) >= 5)
    assert len(frames) == 5 and frames[0]["facing"] == frames[-1]["facing"]


def test_idle_repushes_a_streamed_arm_instead_of_planner_frames(tmp_path):
    from locomani.skills import ArmOwner

    ex = executor(True, tmp_path)
    pushes = []
    ex.session.arm_owner, ex.session.arm = ArmOwner.POSE_STREAM, SimpleNamespace(hold=pushes.append)
    ex.idle(lambda: len(pushes) >= 3)
    assert pushes == [0.02, 0.02, 0.02] and ex.session.stack.frames == []


def test_agent_round_commands_while_the_planner_answers(tmp_path, monkeypatch):
    import time

    from plan_episode import agent_round

    import locomani.skills.planner as planner_mod

    monkeypatch.setattr(planner_mod, "scene_view", lambda stack, scene, state: {})
    ex = executor(True, tmp_path)

    class SlowPlanner:
        def next_step(self, *, session, scene, last):
            time.sleep(0.3)  # claude -p takes 5-9 s
            return {"next": "done", "args": {}, "rationale": "holding"}

    entries = agent_round(ex, SlowPlanner(), ex.session, None, 4, lambda text: None)
    assert entries[-1]["primitive"] == "abort" and entries[-1]["reason"].startswith("done")
    assert len(ex.session.stack.frames) >= 5  # 50 Hz over the 0.3 s ask


def test_injection_settle_goes_through_idle(tmp_path):
    from locomani.skills.injection import Injection, Injector

    ex = executor(True, tmp_path)
    ex.session.reset = SimpleNamespace(pin_object=lambda *a, **k: None)
    ex.session.scene = SimpleNamespace(meta={"objects": {"soup_can": {"init_pos": [0.0, 0.0, 0.7]}}})
    inj = Injector(
        Injection(kind="drop_held", after="grasp_single", settle_s=0.1), log=lambda text: None, sleep=ex.idle
    )
    fired = inj.after({"primitive": "grasp_single", "args": {}, "outcome": {"ok": True}}, ex)
    assert fired is not None and len(ex.session.stack.frames) >= 10  # 0.1 s pin + 0.2 s rest at 50 Hz
