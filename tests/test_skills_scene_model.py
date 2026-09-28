"""SceneModel over an inline scene_meta (soup_can / basket / blue_crate on surfaces A and B)."""

from __future__ import annotations

import copy

import pytest

from locomani.skills.scene_model import SceneModel

META = {
    "object_phrase": {"soup_can": "the soup can", "basket": "the basket"},
    "robot_start": {"xy": [0.0, -1.2], "yaw_deg": 90.0},
    "surfaces": {
        "A": {"top": 0.75, "xy": [0.0, 0.0], "half_extent": [0.6, 0.4], "flat_half_extent": [0.6, 0.4]},
        "B": {"top": 0.30, "xy": [2.0, 0.5], "half_extent": [0.3, 0.3]},
    },
    "objects": {
        "soup_can": {
            "box": {"half_extent": [0.03, 0.03, 0.07], "centre": [0.0, 0.0, 0.0]},
            "bimanual": False,
            "container": False,
            "surface": "A",
            "height": 0.14,
            "init_pos": [0.2, 0.0, 0.82],
        },
        "basket": {
            "box": {"half_extent": [0.15, 0.10, 0.076], "centre": [0.0, 0.0, 0.0], "wall": 0.01},
            "bimanual": False,
            "container": True,
            "surface": "A",
            "init_pos": [-0.2, 0.0, 0.826],
        },
        "blue_crate": {
            "box": {"half_extent": [0.20, 0.13, 0.085], "centre": [0.0, 0.0, 0.0]},
            "bimanual": True,
            "container": False,
            "surface": "A",
            "init_pos": [0.0, 0.3, 0.835],
        },
    },
}


def scene(**changes):
    meta = copy.deepcopy(META)
    for path, value in changes.items():
        node = meta
        *keys, last = path.split(".")
        for k in keys:
            node = node[k]
        node[last] = value
    return SceneModel(meta)


def state(**pos):
    objs = {k: {"pos": list(v), "quat": [1.0, 0.0, 0.0, 0.0]} for k, v in pos.items()}
    return {"fallen": False, "objects": objs}


def test_objects_surfaces_and_graspability():
    s = scene()
    assert s.objects["soup_can"].single_hand_graspable and not s.objects["basket"].single_hand_graspable
    assert not s.objects["blue_crate"].single_hand_graspable and s.objects["blue_crate"].bimanual
    assert s.surfaces["A"].top == 0.75 and s.objects["basket"].wall == 0.01 and s.objects["soup_can"].wall == 0.01
    assert s.heading() == pytest.approx(1.5708, abs=1e-3)


def test_front_line_prefers_the_box_footprint_then_the_surface():
    s = scene()
    box = s.front_line("blue_crate")
    assert box["kind"] == "box" and box["xy"] == [0.0, 0.3] and box["half_extent"] == [0.2, 0.13]
    assert box["top"] == pytest.approx(0.92)
    assert s.front_line("A")["kind"] == "surface"
    boxless = scene(**{"objects.soup_can.box": None})
    assert boxless.front_line("soup_can")["kind"] == "surface"
    with pytest.raises(KeyError):
        s.front_line("nothing")


def test_affordances_follow_the_scene_flags():
    kinds = {(a.kind, a.obj, a.target) for a in scene().affordances()}
    assert ("pick_single", "soup_can", None) in kinds
    assert ("pick_bimanual", "blue_crate", None) in kinds
    assert ("place_into", "soup_can", "basket") in kinds and ("place_into", "blue_crate", "basket") in kinds
    assert ("put_on", "soup_can", "B") in kinds and ("put_on", "soup_can", "A") not in kinds
    assert not any(a.obj == "basket" for a in scene().affordances())


def test_instances_are_the_template_menu_with_profiles():
    inst = {(i.obj, i.target): i for i in scene().instances()}
    can = inst[("soup_can", "basket")]
    assert can.kind == "in" and can.profile == "bottle_simple" and can.text == "put the soup can in the basket"
    assert can.template == "put <obj> in <container>"
    crate = inst[("blue_crate", "B")]
    assert crate.kind == "on" and crate.profile == "crate_low_table" and crate.text == "put blue crate on B"
    low = scene(**{"surfaces.A.top": 0.2})
    assert {i.profile for i in low.instances() if i.obj == "blue_crate"} == {"warehouse_box"}


def test_in_criterion_uses_the_container_pose_from_the_state():
    s = scene()
    inst = next(i for i in s.instances() if (i.obj, i.target) == ("soup_can", "basket"))
    check = s.criterion(inst)
    assert check(state(soup_can=[0.5, 0.0, 0.83], basket=[0.5, 0.0, 0.826])) == (True, "")
    ok, why = check(state(soup_can=[0.7, 0.0, 0.83], basket=[0.5, 0.0, 0.826]))
    assert not ok and why.startswith("[place]")
    ok, why = check(state(soup_can=[0.5, 0.0, 1.0], basket=[0.5, 0.0, 0.826]))
    assert not ok and "rim" in why
    st = state(soup_can=[0.5, 0.0, 0.83], basket=[0.5, 0.0, 0.826])
    st["fallen"] = True
    assert check(st) == (False, "[place] robot fell")


def test_on_criterion_checks_height_band_and_footprint():
    s = scene()
    inst = next(i for i in s.instances() if (i.obj, i.target) == ("soup_can", "B"))
    check = s.criterion(inst)
    assert check(state(soup_can=[2.0, 0.5, 0.37])) == (True, "")
    assert check(state(soup_can=[2.0, 0.5, 0.20]))[0] is False
    assert "footprint" in check(state(soup_can=[2.5, 0.5, 0.37]))[1]


def test_put_on_criterion_uses_the_box_underside_and_centre():
    meta = {
        "surfaces": {"P": {"top": 0.10, "xy": [0.0, -0.05], "half_extent": [0.6, 0.5]}},
        "objects": {
            "carton": {
                "box": {"half_extent": [0.225, 0.16, 0.3], "centre": [0.0, 0.0, 0.3]},
                "bimanual": True,
                "surface": "P",
                "height": 0.6,
                "init_pos": [0.0, -0.395, 0.1],
            }
        },
    }
    scene = SceneModel(meta)
    from locomani.skills.scene_model import TaskInstance

    inst = TaskInstance("put <obj> on <surface>", "carton", "P", "on", "warehouse_box", "put carton down on P")
    check = scene.criterion(inst)
    resting = {
        "fallen": False,
        "objects": {"carton": {"pos": [0.0, -0.395, 0.1]}},
    }  # origin on the top, underside at 0.10
    assert check(resting) == (True, "")
    assert (
        check({"fallen": False, "objects": {"carton": {"pos": [0.0, -0.395, 0.115]}}})[0] is True
    )  # 1.5 cm up: still on
    assert check({"fallen": False, "objects": {"carton": {"pos": [0.0, -0.395, 0.45]}}})[1].startswith(
        "[put_on] underside"
    )
    assert check({"fallen": False, "objects": {"carton": {"pos": [0.0, -0.60, 0.1]}}})[1].startswith(
        "[put_on] centre xy"
    )
    assert check({"fallen": True, "objects": {"carton": {"pos": [0.0, -0.395, 0.1]}}}) == (False, "[put_on] robot fell")
