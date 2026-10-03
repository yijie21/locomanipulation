"""The feasibility checker on a small scene built to land on each outcome: in the envelope, each gap, each limit."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from locomani.scene_discovery import feasibility

CAP = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/scene_discovery/capability.yaml").read_text())
DROP_OK = {"ok": True}


def obj(xy, top, half, *, grasp="single", mass=0.3, surface="T", container=False, free=True, room="kitchen"):
    z = top + half[2]
    return {
        "init_pos": [xy[0], xy[1], z],
        "init_quat": [1, 0, 0, 0],
        "box": {"half_extent": list(half), "centre": [0, 0, 0], "wall": 0.01},
        "surface": surface,
        "free": free,
        "container": container,
        "semantic": {
            "graspable": grasp,
            "movable": True,
            "mass_kg": mass,
            "friction": 0.6,
            "room": room,
            "drop_test": DROP_OK,
            "upright_axis": [0, 0, 1],
        },  # fmt: skip
    }


def surface(xy, top, half, edges=("-y",), room="kitchen"):
    return {
        "xy": list(xy),
        "top": top,
        "half_extent": list(half),
        "yaw": 0.0,
        "approach_edges": list(edges),
        "room": room,
    }


ROOM = {"polygon": [[-3, -3], [3, -3], [3, 3], [-3, 3]], "name_source": "claude"}
META = {
    "room": {"bounds": {"x": [-3, 3], "y": [-3, 3]}, "walls": [], "rooms": {"kitchen": ROOM}},
    "surfaces": {
        "T": surface([0.0, 0.0], 0.62, [0.5, 0.3]),  # the measured table: 0.62 m, stand at its -y edge
        "HIGH": surface([2.0, 0.0], 0.90, [0.4, 0.3]),
        "LOW": surface([-2.0, 0.0], 0.10, [0.4, 0.3]),
        "WALLED": surface([0.0, 2.2], 0.62, [0.4, 0.3], edges=()),
    },
    "objects": {
        "bottle": obj([0.0, -0.24], 0.62, [0.0225, 0.0225, 0.08]),  # 6 cm in from the -y edge: the bottle line
        "deep_bottle": obj([0.0, 0.10], 0.62, [0.0225, 0.0225, 0.08]),  # 0.40 m in from the open edge
        "book": obj([0.3, -0.24], 0.62, [0.03, 0.12, 0.10]),  # thin and tall: graspable, never measured
        "high_bottle": obj([2.0, -0.24], 0.90, [0.0225, 0.0225, 0.08], surface="HIGH"),
        "brick": obj([-0.3, -0.24], 0.62, [0.0225, 0.0225, 0.08], mass=4.0),
        "sofa_cushion": obj([0.0, 2.2], 0.62, [0.0225, 0.0225, 0.08], surface="WALLED"),
        "basket": obj([0.35, -0.15], 0.62, [0.151, 0.1025, 0.076], grasp="none", container=True),
        "cup": obj([-0.35, -0.2], 0.62, [0.04, 0.04, 0.05], grasp="none", container=True),
        "bin": obj([0.0, -2.0], 0.0, [0.12, 0.09, 0.27], grasp="none", container=True, surface=None, free=False),
    },
}


def checker(meta=META):
    return feasibility.Checker(copy.deepcopy(meta), CAP)


def gaps_of(goal, meta=META):
    r = checker(meta).check({"id": "t", "goal": goal})
    return r.bucket(), r.gaps, r.blocked, r.invalid


def test_a_bottle_like_pick_from_the_measured_table_is_bucket_1():
    assert gaps_of(["held(bottle)"]) == (1, [], [], "")


def test_each_untested_dimension_names_its_gap():
    assert gaps_of(["held(book)"])[:2] == (2, ["grasp_single.object_size"])
    assert gaps_of(["held(high_bottle)"])[:2] == (2, ["grasp_single.support_top"])
    bucket, gaps, *_ = gaps_of(["in(bottle, bin)"])  # a floor bin: rim 0.54 m, opening 0.22 x 0.16 < the basket's
    assert bucket == 2 and {"place_into.rim_height", "place_into.opening"} <= set(gaps)
    assert gaps_of(["on(bottle, HIGH)"])[1] == ["place_on_surface.single"]


def test_hard_limits_block():
    bucket, _gaps, blocked, _ = gaps_of(["held(deep_bottle)"])
    assert bucket == 3 and any("beyond the hand's reach" in b for b in blocked)
    assert any("heavier than the jaw" in b for b in gaps_of(["held(brick)"])[2])
    assert any("no edge" in b for b in gaps_of(["held(sofa_cushion)"])[2])
    assert any("does not fit the opening" in b for b in gaps_of(["in(book, cup)"])[2])
    assert any("neither the jaw" in b for b in gaps_of(["held(basket)"])[2])


def test_putting_into_a_basket_on_the_same_table_has_no_carry():
    r = checker().check({"id": "t", "goal": ["in(bottle, basket)"]})
    assert r.bucket() == 1, (r.gaps, r.blocked)
    assert not [c for c in r.checks if c.step.startswith("carry")]


def test_cross_room_and_unknown_or_satisfied_goals():
    meta = copy.deepcopy(META)
    meta["room"]["rooms"] = {"kitchen": ROOM, "hall": dict(ROOM, polygon=[[-3, 2], [3, 2], [3, 3], [-3, 3]])}
    meta["surfaces"]["HIGH"]["room"] = "hall"
    assert "navigate.cross_room" in gaps_of(["on(bottle, HIGH)"], meta)[1]
    assert gaps_of(["held(teapot)"])[3].startswith("held(teapot)")
    assert gaps_of(["on(bottle, T)"])[3] == "on(bottle, T): already on it"
    assert gaps_of(["upright(bottle)"])[3] == "upright(bottle): already upright"


def test_the_pilot_policy_attempts_untested_but_never_blocked_tasks():
    c = checker()
    assert c.check({"id": "a", "goal": ["held(book)"]}).bucket("pilot") == 1
    assert c.check({"id": "b", "goal": ["held(deep_bottle)"]}).bucket("pilot") == 3


def test_templates_and_the_gap_ranking():
    tasks = feasibility.templates(META)
    ids = {t["id"] for t in tasks}
    assert "put_bottle_in_basket" in ids and "put_bottle_on_WALLED" not in ids and "pick_basket" not in ids
    c = checker()
    ranking = feasibility.gap_ranking([c.check(t) for t in tasks])
    by = {d["gap"]: d for d in ranking}
    assert by["place_on_surface.single"]["unlocks_alone"] >= 1
    assert ranking == sorted(ranking, key=lambda d: (-d["unlocks_alone"], -d["in_tasks"], d["gap"]))


def test_unstable_objects_are_no_template_and_a_shelf_container_is_reached_at_its_shelf_edge():
    meta = copy.deepcopy(META)
    meta["objects"]["book"]["semantic"]["drop_test"] = {"ok": False}
    assert not [
        t for t in feasibility.templates(meta) if t["goal"][0].split("(")[1].split(",")[0].rstrip(")") == "book"
    ]
    meta["objects"]["mug_on_high"] = obj(
        [2.0, -0.2], 0.90, [0.05, 0.05, 0.05], grasp="none", container=True, surface="HIGH"
    )
    r = feasibility.Checker(meta, CAP).check({"id": "t", "goal": ["in(bottle, mug_on_high)"]})
    route = next(c for c in r.checks if c.step.startswith("carry") and c.name == "route")
    assert (
        route.outcome == "ok" and 1.5 < route.measured < 2.6
    )  # table edge to HIGH's -y edge, not a ring round the mug
