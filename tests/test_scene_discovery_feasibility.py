"""The feasibility checker on a small scene built to land on each outcome: in the envelope, each gap, each limit."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from locomani.scene_discovery import feasibility, scene_file

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
    # 0.90 m table: grip at 0.98 m, above what the skill plans for (0.66-0.89) -- a skill gap, not just a height
    assert gaps_of(["held(high_bottle)"])[:2] == (2, ["grasp_single.grip_height", "grasp_single.support_top"])
    bucket, gaps, *_ = gaps_of(["in(bottle, bin)"])  # a floor bin: rim 0.54 m, opening 0.22 x 0.16 < the basket's
    assert bucket == 2 and {"place_into.rim_height", "place_into.opening"} <= set(gaps)
    # since 2026-10-03 put_down_single is measured on one setting (0.52 m top, a 19.6 cm canister): a 16 cm bottle
    # onto a 0.90 m top is outside it on both counts, a range gap now, not a missing skill
    assert gaps_of(["on(bottle, HIGH)"])[1] == ["place_on_surface.object_height", "place_on_surface.support_top"]


def test_a_measured_surface_put_down_is_checked_by_its_range():
    cap = copy.deepcopy(CAP)
    cap["place"]["on_surface_single"] = {
        "status": "measured_one",
        "support_top_measured": [0.90, 0.90],
        "object_height_measured": [0.14, 0.20],
    }
    r = feasibility.Checker(copy.deepcopy(META), cap).check({"id": "t", "goal": ["on(bottle, HIGH)"]})
    assert not [g for g in r.gaps if g.startswith("place_on_surface")]
    cap["place"]["on_surface_single"] = {"status": "untested"}  # unmeasured: the missing skill, as before 2026-10-03
    r = feasibility.Checker(copy.deepcopy(META), cap).check({"id": "t", "goal": ["on(bottle, HIGH)"]})
    assert [g for g in r.gaps if g.startswith("place_on_surface")] == ["place_on_surface.single"]
    cap["place"]["on_surface_single"] = {
        "status": "measured_one",
        "support_top_measured": [0.90, 0.90],
        "object_height_measured": [0.14, 0.20],
    }
    cap["place"]["on_surface_single"]["support_top_measured"] = [0.52, 0.52]
    r = feasibility.Checker(copy.deepcopy(META), cap).check({"id": "t", "goal": ["on(bottle, HIGH)"]})
    assert [g for g in r.gaps if g.startswith("place_on_surface")] == ["place_on_surface.support_top"]


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
    covered = copy.deepcopy(META)  # a cabinet top 3 cm under the table, as the ADT kitchen's (plan section 12)
    covered["surfaces"]["UNDER"] = dict(covered["surfaces"]["T"], top=0.59)
    assert gaps_of(["on(bottle, UNDER)"], covered)[3] == "the goal holds at the scene pose"


def test_the_pilot_policy_attempts_untested_but_never_blocked_tasks():
    c = checker()
    assert c.check({"id": "a", "goal": ["held(book)"]}).bucket("pilot") == 1
    assert c.check({"id": "b", "goal": ["held(deep_bottle)"]}).bucket("pilot") == 3


def test_templates_and_the_gap_ranking():
    tasks = feasibility.templates(META)
    ids = {t["id"] for t in tasks}
    assert "put_bottle_in_basket" in ids and "put_bottle_on_WALLED" not in ids and "pick_basket" not in ids
    unmeasured = copy.deepcopy(CAP)  # the capability before put_down_single was measured (2026-10-03)
    unmeasured["place"]["on_surface_single"] = {"status": "untested"}
    c = feasibility.Checker(copy.deepcopy(META), unmeasured)
    ranking = feasibility.gap_ranking([c.check(t) for t in tasks])
    by = {d["gap"]: d for d in ranking}
    assert by["place_on_surface.single"]["unlocks_alone"] >= 1
    assert ranking == sorted(ranking, key=lambda d: (-d["unlocks_alone"], -d["in_tasks"], d["gap"]))
    now = {d["gap"] for d in feasibility.gap_ranking([checker().check(t) for t in tasks])}
    assert "place_on_surface.single" not in now and "place_on_surface.support_top" in now


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


def test_pilot_groups_split_envelope_gaps_from_skill_gaps():
    from locomani.scene_discovery import pilot_groups

    c = checker()
    results = [
        c.check(t).to_dict() for t in [{"id": "a", "goal": ["held(book)"]}, {"id": "b", "goal": ["held(high_bottle)"]}]
    ]
    groups = {tuple(g["gaps"]): g for g in pilot_groups.groups(results)}
    assert groups[("grasp_single.object_size",)]["pilotable"]
    high = groups[("grasp_single.grip_height", "grasp_single.support_top")]
    assert not high["pilotable"] and high["skill_gaps"] == ["grasp_single.grip_height"]


def _envelope_meta():
    meta = copy.deepcopy(META)
    meta["surfaces"]["CART"] = surface([2.0, 2.0], 0.70, [0.21, 0.15])  # the canister pilot's cart: 0.70 m, -y edge
    o = meta["objects"]
    o["canister"] = obj([2.0, 1.934], 0.70, [0.041, 0.0615, 0.098], mass=0.6, surface="CART")  # centre 8.4 cm in
    o["cart_bottle"] = obj([2.1, 1.934], 0.70, [0.0225, 0.0225, 0.08], surface="CART")
    o["deep_can"] = obj([0.0, -0.03], 0.62, [0.0225, 0.0225, 0.08])  # 0.27 m in: past the apple pilot
    o["short_deep"] = obj([0.2, -0.14], 0.62, [0.03, 0.03, 0.05])  # 0.16 m in, grip 5 cm over the top: the plant
    o["tall_deep"] = obj([-0.2, -0.14], 0.62, [0.0225, 0.0225, 0.10])  # 0.16 m in, grip 10 cm over the top
    return meta


def test_a_passing_pilot_is_a_setting_matched_whole():
    meta = _envelope_meta()
    r = feasibility.Checker(meta, CAP).check({"id": "t", "goal": ["held(canister)"]})
    assert r.bucket() == 1, (r.gaps, r.blocked)
    assert {c.setting for c in r.checks if c.setting} == {"adt_canister_cart"}
    # the bottle line's object on the canister's cart: each value was measured, the combination never was
    bucket, gaps, *_ = gaps_of(["held(cart_bottle)"], meta)
    assert bucket == 2 and gaps == ["grasp_single.support_top"]


def test_a_grasp_at_least_as_hard_as_a_failed_pilot_carries_its_skill_gap():
    from locomani.scene_discovery import pilot_groups

    meta = _envelope_meta()
    assert "grasp_single.deep_reach" in gaps_of(["held(deep_can)"], meta)[1]
    assert "grasp_single.deep_entry" in gaps_of(["held(short_deep)"], meta)[1]
    tall = gaps_of(["held(tall_deep)"], meta)[1]  # as deep, but gripped higher than either failed pilot
    assert "grasp_single.deep_entry" not in tall and "grasp_single.inboard" in tall
    assert pilot_groups.is_skill_gap("grasp_single.deep_reach") and pilot_groups.is_skill_gap("grasp_single.deep_entry")


def test_a_spot_under_another_surface_is_outside_every_measured_setting():
    meta = _envelope_meta()
    meta["surfaces"]["SHELF"] = surface([0.0, 0.0], 0.92, [0.5, 0.3])  # a board 0.28 m over the measured table
    assert "grasp_single.covered" in gaps_of(["held(bottle)"], meta)[1]
    assert "place_on_surface.covered" in gaps_of(["on(canister, T)"], meta)[1]
    assert "place_into.covered" in gaps_of(["in(canister, basket)"], meta)[1]
    meta["surfaces"]["SHELF"]["top"] = 1.50  # 0.86 m up: open enough
    assert gaps_of(["held(bottle)"], meta)[:2] == (1, [])


def test_upright_beside_a_move_is_a_constraint_on_the_move():
    meta = _envelope_meta()
    assert gaps_of(["on(canister, T)", "upright(canister)"], meta)[3] == ""  # set it down standing: not invalid
    assert "place_into.upright" in gaps_of(["in(bottle, basket)", "upright(bottle)"], meta)[1]
    assert gaps_of(["upright(bottle)"], meta)[3] == "upright(bottle): already upright"  # nothing moves it
    meta["objects"]["bottle"]["init_quat"] = [0.7071, 0.7071, 0, 0]  # lying on its side
    assert "reorient" in gaps_of(["on(bottle, CART)", "upright(bottle)"], meta)[1]


def _mars_cap(**reach):
    """The 10-03 table with the mars + ScaleBFM table's new keys (capability_mars_scalebfm.yaml, TODO 27.3)."""
    cap = copy.deepcopy(CAP)
    cap["reach_single_arm"].update(reach)
    return cap


def test_a_measured_hand_envelope_blocks_outside_it_and_a_lift_limit_blocks_above_it():
    meta = _envelope_meta()
    cap = _mars_cap(grip_height_accepted=[0.75, 1.25], grip_height_blocked_outside=True, mass_lifted_max=2.0)
    r = feasibility.Checker(copy.deepcopy(meta), cap).check({"id": "t", "goal": ["held(bottle)"]})
    assert r.bucket() == 3 and any("outside the hand's measured reach" in b for b in r.blocked)
    meta["objects"]["light_brick"] = obj([-0.3, -0.24], 0.70, [0.0225, 0.0225, 0.08], mass=2.2, surface="CART")
    cap = _mars_cap(grip_height_accepted=[0.15, 1.25], grip_height_blocked_outside=True, mass_lifted_max=2.0)
    r = feasibility.Checker(copy.deepcopy(meta), cap).check({"id": "t", "goal": ["held(light_brick)"]})
    assert any("heavier than any mass the hand lifted" in b for b in r.blocked)


def test_a_failed_task_names_its_object_and_untested_skills_are_gaps():
    meta = _envelope_meta()
    cap = _mars_cap(grasp_failures=[dict(name="pick_single.bottle", gap="grasp_single.bottle_pickup", objects=["bottle"],
                                         pilots=["bottle_pickup"], success="2/10")])  # fmt: skip
    gaps = feasibility.Checker(copy.deepcopy(meta), cap).check({"id": "t", "goal": ["held(bottle)"]}).gaps
    assert "grasp_single.bottle_pickup" in gaps
    cap["place"]["into_container_single_arm"] = dict(status="untested")
    r = feasibility.Checker(copy.deepcopy(meta), cap).check({"id": "t", "goal": ["in(canister, basket)"]})
    assert "place_into.single" in r.gaps


def test_the_body_radius_comes_from_the_table():
    c = feasibility.Checker(copy.deepcopy(META), dict(CAP, robot=dict(CAP["robot"], body_radius=0.35)))
    stand, _ = c.edge_stand("T", [0.0, -0.24])
    assert stand[1] == pytest.approx(-0.3 - 0.35 - feasibility.STAND_OUT)
    assert feasibility.Checker(copy.deepcopy(META), CAP).edge_stand("T", [0.0, -0.24])[0][1] == pytest.approx(-0.66)


def test_a_task_measured_on_its_own_object_is_bucket_1_whatever_the_proxies_say():
    meta = _envelope_meta()
    meta["objects"]["walled_bottle"] = obj([0.0, 2.2], 0.62, [0.0225, 0.0225, 0.08], surface="WALLED")  # no edge
    goal = {"id": "t", "goal": ["held(walled_bottle)"]}
    assert feasibility.Checker(copy.deepcopy(meta), CAP).check(goal).bucket() == 3
    cap = _mars_cap(grasp_settings=[dict(name="pick_single.walled", objects=["walled_bottle"], surface="WALLED",
                                         success="9/10", stance_xy=[0.0, 1.5], support_top=[0.62, 0.62],
                     object_width=[0.045, 0.045], object_height=[0.16, 0.16], mass_max=0.3, inboard=[0.0, 0.0])] + CAP["reach_single_arm"]["grasp_settings"])  # fmt: skip
    r = feasibility.Checker(copy.deepcopy(meta), cap).check(goal)
    assert r.bucket() == 1 and any(c.name == "measured_task" for c in r.checks)
    meta["objects"]["walled_bottle"]["surface"] = "T"  # moved to another surface: the measurement no longer applies
    assert not any(c.name == "measured_task" for c in feasibility.Checker(copy.deepcopy(meta), cap).check(goal).checks)


def test_a_passage_only_a_tighter_body_fits_is_a_gap_not_a_block():
    meta = _envelope_meta()
    meta["room"]["walls"] = [
        [[-3.0, 1.0], [-0.4, 1.0]],
        [[0.4, 1.0], [3.0, 1.0]],
    ]  # a 0.80 m doorway (wall cells eat into it)
    meta["surfaces"]["FAR"] = surface([1.8, 2.3], 0.62, [0.3, 0.2])
    goal = {"id": "t", "goal": ["on(bottle, FAR)"]}
    wide = dict(CAP, robot=dict(CAP["robot"], body_radius=0.35))
    r = feasibility.Checker(copy.deepcopy(meta), wide).check(goal)
    assert any("no route in the occupancy map" in b for b in r.blocked)
    tight = dict(CAP, robot=dict(CAP["robot"], body_radius=0.35, body_radius_tight=0.26))
    r = feasibility.Checker(copy.deepcopy(meta), tight).check(goal)
    assert not r.blocked and "navigate.narrow_passage" in r.gaps


def test_an_approach_edge_overlay_replaces_the_scene_files_edges():
    overlay = {"surfaces": {"WALLED": {"approach_edges": ["-y"]}}}  # TODO 27.3.1: the runner's stance search
    meta = scene_file.with_approach_edges(META, overlay)
    assert meta["surfaces"]["WALLED"]["approach_edges"] == ["-y"] and META["surfaces"]["WALLED"]["approach_edges"] == []
    assert meta["surfaces"]["T"]["approach_edges"] == ["-y"]  # not in the overlay: kept
    assert gaps_of(["held(sofa_cushion)"])[0] == 3
    assert gaps_of(["held(sofa_cushion)"], meta)[0] != 3
    assert any("put_bottle_on_WALLED" == t["id"] for t in feasibility.templates(meta))
    with pytest.raises(ValueError, match="not in the scene"):
        scene_file.with_approach_edges(META, {"surfaces": {"NOPE": {"approach_edges": []}}})
    with pytest.raises(ValueError, match="not in"):
        scene_file.with_approach_edges(META, {"surfaces": {"T": {"approach_edges": ["up"]}}})


def test_a_put_down_measured_on_its_own_object_and_surface_decides_the_put_and_its_carry():
    meta = _envelope_meta()
    meta["surfaces"]["FAR"] = surface([2.6, -2.4], 0.95, [0.3, 0.2])  # outside the 10-03 put-down's measured tops
    goal = {"id": "t", "goal": ["on(bottle, FAR)"]}
    cap = copy.deepcopy(CAP)
    before = feasibility.Checker(copy.deepcopy(meta), cap).check(goal)
    assert before.bucket() == 2 and not any(c.name == "measured_task" for c in before.checks)
    cap["place"]["on_surface_single"]["measured_tasks"] = [dict(name="r1_on_single.t", objects=["bottle"], surface="FAR",
                                                          success="7/8")]  # fmt: skip
    after = feasibility.Checker(copy.deepcopy(meta), cap).check(goal)
    assert [c.step for c in after.checks if c.name == "measured_task"] == [
        "carry(bottle -> FAR)",
        "put_on(bottle on FAR)",
    ]
    assert not any(g.startswith(("place_on_surface", "carry_single", "navigate")) for g in after.gaps)
    cap["place"]["on_surface_single"]["measured_tasks"] = []
    cap["place"]["on_surface_single"]["failed_tasks"] = [dict(name="r1_on_single.t", gap="place_on_surface.t",
                                                          objects=["bottle"], surface="FAR", success="0/8")]  # fmt: skip
    assert "place_on_surface.t" in feasibility.Checker(copy.deepcopy(meta), cap).check(goal).gaps


def test_a_put_in_measured_on_its_own_object_and_container_decides_the_put():
    goal = {"id": "t", "goal": ["in(bottle, bin)"]}  # the floor bin: a ring stand, no measured put-in
    cap = copy.deepcopy(CAP)
    cap["place"]["into_container_single_arm"]["measured_tasks"] = [dict(name="r1_in_single.t", objects=["bottle"],
                                                                       surface="bin", success="6/8")]  # fmt: skip
    r = feasibility.Checker(copy.deepcopy(META), cap).check(goal)
    assert [c.step for c in r.checks if c.name == "measured_task"] == ["carry(bottle -> bin)", "place(bottle in bin)"]
    assert not any(g.startswith(("place_into", "carry_single", "navigate")) for g in r.gaps)
    cap["place"]["into_container_single_arm"]["measured_tasks"] = []
    cap["place"]["into_container_single_arm"]["failed_tasks"] = [dict(name="r1_in_single.t", gap="place_into.t",
                                                                     objects=["bottle"], surface="bin", success="1/8")]  # fmt: skip
    assert "place_into.t" in feasibility.Checker(copy.deepcopy(META), cap).check(goal).gaps


def test_a_goal_measured_end_to_end_decides_its_near():
    goal = {"id": "t", "goal": ["held(bottle)", "near(bottle, HIGH, 0.8)"]}
    assert "goal.near" in feasibility.Checker(copy.deepcopy(META), CAP).check(goal).gaps
    cap = dict(copy.deepcopy(CAP), goals_measured=[dict(name="r1_held_near_single.t", success="7/10",
                                                        goal=["near(bottle,  HIGH, 0.8)", "held(bottle)"])])  # fmt: skip
    r = feasibility.Checker(copy.deepcopy(META), cap).check(goal)
    assert "goal.near" not in r.gaps and any(c.name == "measured_task" for c in r.checks)


def test_an_object_whose_centre_cannot_get_under_the_rim_is_bucket_3():
    meta = copy.deepcopy(META)  # a 2.4 cm dish: its floor 1.4 cm under the rim; the bottle lying down is 4.5 cm thick
    meta["objects"]["dish"] = obj([0.35, -0.15], 0.62, [0.08, 0.08, 0.012], grasp="none", container=True)
    bucket, _, blocked, _ = gaps_of(["in(bottle, dish)"], meta)
    assert bucket == 3 and blocked == ["place(bottle in dish): its centre cannot get under the rim"]
    assert gaps_of(["in(bottle, basket)"])[0] != 3  # the deep basket: the rule leaves it to the put-in checks
