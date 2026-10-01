"""ClaudePlanner over a fake `claude` command, the catalogue manual, the scene view, and the agent round loop."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from locomani.skills import ArmOwner, Executor, SceneModel, Session
from locomani.skills.catalogue import build_catalogue
from locomani.skills.contract import Outcome
from locomani.skills.planner import ClaudePlanner, catalogue_manual, scene_view

REPO = Path(__file__).resolve().parents[1]

FAKE_CLAUDE = """#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
log = os.environ["FAKE_CLAUDE_LOG"]
with open(log, "a") as f:
    f.write(json.dumps(argv) + "\\n")
plan = json.load(open(os.environ["FAKE_CLAUDE_PLAN"]))
n = sum(1 for _ in open(log))
decision = plan[min(n - 1, len(plan) - 1)]
print(json.dumps({"structured_output": decision, "result": json.dumps(decision), "usage": {"output_tokens": 3}, "total_cost_usd": 0.0}))
"""


def fake_claude(tmp_path, plan):
    exe = tmp_path / "claude"
    exe.write_text(FAKE_CLAUDE)
    exe.chmod(0o755)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    os.environ["FAKE_CLAUDE_LOG"] = str(tmp_path / "argv.log")
    os.environ["FAKE_CLAUDE_PLAN"] = str(tmp_path / "plan.json")
    return exe


META = {
    "robot_start": {"xy": [0.0, -1.2], "yaw_deg": 90.0},
    "surfaces": {"A": {"top": 0.75, "xy": [0.0, 0.0], "half_extent": [0.6, 0.4]}},
    "objects": {
        "soup_can": {
            "box": {"half_extent": [0.03, 0.03, 0.07]},
            "bimanual": False,
            "container": False,
            "surface": "A",
            "height": 0.14,
            "init_pos": [0.2, -0.35, 0.82],
        },
        "basket": {
            "box": {"half_extent": [0.15, 0.10, 0.076], "wall": 0.01},
            "bimanual": False,
            "container": True,
            "surface": "A",
            "height": 0.152,
            "init_pos": [-0.2, -0.35, 0.826],
        },
    },
}


class Stack:
    cmd = SimpleNamespace(frame=lambda **kw: None)  # agent_round commands while the planner answers

    def obs(self):
        return {
            "fallen": False,
            "pelvis": {"pos": [0.0, -1.0, 0.79], "yaw": 1.5708},
            "objects": {
                "soup_can": {"pos": [0.2, -0.35, 0.82], "touching": [], "cam": [0.1, 0.1]},
                "basket": {"pos": [-0.2, -0.35, 0.826], "touching": []},
            },
        }

    def pelvis_xy(self, s):
        return np.array(s["pelvis"]["pos"][:2])

    def pelvis_yaw(self, s):
        return s["pelvis"]["yaw"]

    def world_to_root(self, pos, s):
        d = np.asarray(pos[:2]) - self.pelvis_xy(s)
        yaw = self.pelvis_yaw(s)
        return np.array(
            [np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1], pos[2] - 0.79]
        )


def test_planner_asks_resumes_and_logs(tmp_path):
    exe = fake_claude(
        tmp_path,
        [
            {"next": "approach", "args": {"target": "soup_can", "profile": "bottle_simple"}, "rationale": "start"},
            {"next": "done", "rationale": "held"},
        ],
    )
    planner = ClaudePlanner(command=(sys.executable, str(exe)), model="sonnet", log_path=tmp_path / "io.jsonl")
    planner.start_episode(
        task="put the soup can in the basket",
        menu=["put the soup can in the basket"],
        catalogue=build_catalogue(profiles={}),
        profiles=["bottle_simple"],
    )
    first = planner.next_step(session={"arm_owner": "planner_swing"}, scene={}, last=None)
    assert first == {
        "next": "approach",
        "args": {"target": "soup_can", "profile": "bottle_simple"},
        "rationale": "start",
    }
    second = planner.next_step(
        session={"arm_owner": "planner_frozen"}, scene={}, last={"primitive": "approach", "outcome": {"ok": True}}
    )
    assert second["next"] == "done"
    argvs = [json.loads(line) for line in (tmp_path / "argv.log").read_text().splitlines()]
    assert (
        "--session-id" in argvs[0]
        and "--resume" in argvs[1]
        and argvs[1][argvs[1].index("--resume") + 1] == planner.session_id
    )
    assert argvs[0][argvs[0].index("--tools") + 1] == "" and "--json-schema" in argvs[0]
    request0 = json.loads(argvs[0][argvs[0].index("-p") + 1])
    assert request0["task"].startswith("put the soup can") and any(
        row["name"] == "approach" for row in request0["catalogue"]
    )
    request1 = json.loads(argvs[1][argvs[1].index("-p") + 1])
    assert "catalogue" not in request1 and request1["last_call"]["primitive"] == "approach"
    lines = [json.loads(line) for line in (tmp_path / "io.jsonl").read_text().splitlines()]
    assert (
        [entry["call"] for entry in lines] == [1, 2]
        and lines[0]["decision"]["next"] == "approach"
        and lines[0]["error"] == ""
    )


def test_planner_failure_becomes_abort(tmp_path):
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\necho nope >&2\nexit 3\n")
    exe.chmod(0o755)
    planner = ClaudePlanner(command=(str(exe),), log_path=tmp_path / "io.jsonl")
    planner.start_episode(task="t", menu=[], catalogue={})
    decision = planner.next_step(session={}, scene={}, last=None)
    assert decision["next"] == "abort" and "exit 3" in decision["rationale"]


def test_catalogue_manual_and_scene_view():
    rows = {row["name"]: row for row in catalogue_manual(build_catalogue(profiles={}))}
    assert rows["walk_to"]["args"]["turn_mode"] == "slow_walk" and rows["approach"]["args"]["target"] == "required"
    assert (
        rows["grasp_single"]["starts_from"] == ["planner_swing"] and rows["grasp_single"]["yields"] == "planner_frozen"
    )
    assert sorted(rows["approach"]["yields"]) == ["planner_frozen", "planner_swing"]
    view = scene_view(Stack(), SceneModel(META), Stack().obs())
    assert view["robot"]["yaw_rad"] == 1.571 and view["objects"]["basket"]["container"] is True
    assert view["objects"]["soup_can"]["in_ego_view"] is True and view["objects"]["basket"]["in_ego_view"] is False
    assert abs(view["objects"]["soup_can"]["pelvis_frame_xyz"][0] - 0.65) < 0.01  # 0.65 m ahead along +y
    assert view["surfaces"]["A"]["top_z"] == 0.75


def test_agent_round_runs_until_done(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import plan_episode

    class Noop:
        name, requires, yields = "noop", frozenset(ArmOwner), None

        def preconditions(self, session, *, k: int = 0):
            return []

        def run(self, session, *, k: int = 0):
            return Outcome(True, "", metrics={"k": k})

    class Planner:
        calls = 0

        def next_step(self, *, session, scene, last):
            self.calls += 1
            return (
                {"next": "noop", "args": {"k": self.calls}, "rationale": "go"}
                if self.calls < 3
                else {"next": "done", "rationale": "enough"}
            )

    s = Session(stack=Stack(), scene=SceneModel(META), heading=0.0)
    ex = Executor(s, {"noop": Noop()})
    entries = plan_episode.agent_round(ex, Planner(), s, s.scene, 10, lambda text: None)
    assert [e["primitive"] for e in entries] == ["noop", "noop", "abort"]
    assert entries[1]["outcome"]["metrics"]["k"] == 2 and entries[-1]["reason"].startswith("done")
    steps = plan_episode.default_plan(SceneModel(META), {"target": "soup_can", "goal": "basket"})
    assert [(st["primitive"], st["args"].get("target") or st["args"].get("container")) for st in steps] == [
        ("approach", "soup_can"),
        ("grasp_single", "soup_can"),
        ("approach", "basket"),
        ("place", "basket"),
    ]
    assert plan_episode.default_plan(SceneModel(META), {"target": "soup_can"}, "other")[0]["args"]["profile"] == "other"
    coupled = {"c": SimpleNamespace(config=SimpleNamespace(reach=SimpleNamespace(strategy="coupled")))}
    direct = plan_episode.default_plan(SceneModel(META), {"target": "soup_can", "goal": "basket"}, "c", coupled)
    assert [st["primitive"] for st in direct] == ["approach", "grasp_single", "place"]
    assert isinstance(SimpleNamespace(), object)


def test_criterion_for_put_down_plans():
    sys.path.insert(0, str(REPO / "scripts"))
    import plan_episode

    meta = {
        "robot_start": {"xy": [-1.8, 0.0], "yaw_deg": 0.0},
        "surfaces": {"A": {"top": 0.30, "xy": [0.0, 0.0], "half_extent": [0.5, 0.5]}},
        "objects": {
            "blue_crate": {
                "box": {"half_extent": [0.2, 0.13, 0.085]},
                "bimanual": True,
                "surface": "A",
                "height": 0.17,
                "init_pos": [-0.3, 0, 0.385],
            }
        },
    }
    scene = SceneModel(meta)
    steps = [
        {"primitive": "grasp_bimanual", "args": {}},
        {"primitive": "put_down_bimanual", "args": {"profile": "warehouse_box"}},
    ]
    text, check = plan_episode.criterion_for(scene, {"target": "blue_crate"}, steps)
    assert text == "put blue crate down on A"
    resting = {"fallen": False, "objects": {"blue_crate": {"pos": [-0.3, 0.0, 0.385]}}}
    s = Session(stack=Stack(), scene=scene)
    assert check(resting, s) == (True, "")
    s.held = SimpleNamespace(key="blue_crate")
    assert check(resting, s)[1].startswith("[put_down] still held")
    s.held = None
    assert check({"fallen": False, "objects": {"blue_crate": {"pos": [-0.3, 0.0, 0.9]}}}, s)[0] is False
    text, check = plan_episode.criterion_for(scene, {"target": "blue_crate"}, [{"primitive": "walk_to", "args": {}}])
    assert text == "pick up blue crate" and check(resting, s)[1].startswith("[hold]")
    goal_task = {"target": "soup_can", "goal": "basket"}
    text, _ = plan_episode.criterion_for(SceneModel(META), goal_task, [{"primitive": "place", "args": {}}])
    assert text == "put soup can in basket"
    text, _ = plan_episode.criterion_for(SceneModel(META), goal_task, [{"primitive": "navigate_to", "args": {}}])
    assert text == "pick up soup can"
