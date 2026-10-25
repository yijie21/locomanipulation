"""TODO 33.24.3: floor furniture that moves -- the map follows the pieces, the floor goals, the prep layer's fourth kind,
and the draw filter that keeps only the starts needing it."""

import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "rsi")]

from locomani.primitives.navigator import build  # noqa: E402
from locomani.tasks.scene import floor as FL  # noqa: E402
from locomani.tasks.scene import graspability as G  # noqa: E402

TOP = 0.70
CHAIR = [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]  # its footprint, in its own frame
BENCH = [[-0.45, -0.2], [0.45, -0.2], [0.45, 0.2], [-0.45, 0.2]]


def scene(chair_xy=(0.0, -0.75), movable=True, wide=False) -> G.StaticScene:
    fp = BENCH if wide else CHAIR
    hx = 0.45 if wide else 0.2
    xml = ('<mujoco><worldbody><body name="cup" pos="0 -0.2 0.75"><freejoint/><geom type="box" size="0.03 0.03 0.05"/></body>'
           f'<body name="chair" pos="{chair_xy[0]} {chair_xy[1]} 0.45"><freejoint/><geom type="box" size="{hx} 0.2 0.45"/></body>'
           "</worldbody></mujoco>")  # fmt: skip
    meta = dict(
        room=dict(bounds=dict(x=[-3.0, 3.0], y=[-3.0, 3.0]), obstacles=[(np.asarray(fp) + chair_xy).tolist()]),
        surfaces={"table": dict(xy=[0.0, 0.0], top=TOP, half_extent=[0.8 if wide else 0.5, 0.3, 0.01], yaw=0.0,
                                approach_edges=["-y"])},
        objects={"cup": dict(body="cup", surface="table", init_pos=[0, -0.2, 0.75], init_quat=[1, 0, 0, 0], free=True,
                             box=dict(half_extent=[0.03, 0.03, 0.05])),
                 "chair": dict(body="chair", surface=None, init_pos=[*chair_xy, 0.45], init_quat=[1, 0, 0, 0], free=True,
                               box=dict(half_extent=[hx, 0.2, 0.45]))},
    )  # fmt: skip
    if movable:
        meta["movable"] = {"chair": dict(obstacle=0, footprint=fp)}
    d = Path(tempfile.mkdtemp())
    (d / "scene_meta.json").write_text(json.dumps(meta))
    return G.StaticScene(d, model=mujoco.MjModel.from_xml_string(xml))


def cells(occ, xy) -> bool:
    c = occ.cell_of(np.asarray(xy, float))
    return bool(occ.occupied[c[0], c[1]])


def test_the_map_is_the_scene_files_while_the_pieces_stand_where_they_began_and_follows_them():
    s = scene()
    occ0, _ = build(s.meta, body_radius=FL.BODY_RADIUS_M)
    occ1, _ = FL.navigation(s)
    assert (occ0.occupied == occ1.occupied).all() and np.allclose(occ0.clearance, occ1.clearance)
    moved, _ = FL.navigation(s, {"chair": (np.array([1.5, -0.75, 0.45]), np.array([1.0, 0, 0, 0]))})
    assert cells(occ1, (0.0, -0.75)) and not cells(moved, (0.0, -0.75)) and cells(moved, (1.5, -0.75))


def test_refresh_and_reset_swap_the_worlds_map():
    s = scene()
    s.occ, s.nav = build(s.meta, body_radius=FL.BODY_RADIUS_M)
    first = s.occ
    s.place({"chair": (np.array([1.5, -0.75, 0.45]), np.array([1.0, 0, 0, 0]))})
    FL.refresh(s)
    assert s.occ is not first and not cells(s.occ, (0.0, -0.75))
    FL.reset(s)
    assert s.occ is first


def test_a_scene_without_movable_pieces_is_untouched():
    s = scene(movable=False)
    assert FL.pieces(s) == {} and FL.goals(s, "cup") == []
    s.occ = "kept"
    FL.refresh(s)
    assert s.occ == "kept"


def test_goals_move_a_near_piece_onto_free_floor_away_or_aside():
    s = scene()
    gs = FL.goals(s, "cup")
    assert gs and {g["key"] for g in gs} == {"chair"}
    for g in gs:
        assert g["move_m"] in FL.MOVES_M
        d = np.asarray(g["pos"][:2]) - np.array([0.0, -0.75])
        assert math.isclose(float(np.linalg.norm(d)), g["move_m"], abs_tol=1e-6)
    aways = [g for g in gs if "away" in g["note"]]
    assert aways and all(g["pos"][1] < -0.75 for g in aways)  # away from the cup at y = -0.2
    far = scene(chair_xy=(2.5, 2.5))
    assert FL.goals(far, "cup") == []  # farther than NEAR_M


def test_the_prep_layer_proposes_moving_the_chair_when_it_blocks_the_only_side(tmp_path):
    from locomani.tasks.scene import body
    from locomani.tasks.scene import prep as P

    rows = [[x, y, z, att, pel, hand] for x in np.arange(0.2, 0.95, 0.1) for y in (-0.05, 0.05, 0.15)
            for z in np.arange(0.4, 1.2, 0.1) for att in (0.0, 1.0, 2.0) for pel in (0.78, 0.68, 0.62) for hand in (0.0, 1.0)]  # fmt: skip
    X = np.array(rows)
    miss = ((X[:, 0] > 0.65) | (X[:, 3] == 1.0)).astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    table = tmp_path / "t.npz"
    np.savez(table, X=((X - mu) / sd).astype(np.float32), y_miss=miss.astype(np.float32),
             y_err=np.where(miss > 0, 10.0, 1.0).astype(np.float32), mu=mu, sd=sd,
             features=np.array(["x", "y_out", "z", "att", "pelvis", "hand"]), k=np.array(15), miss_cm=np.array(3.0),
             stand_z=np.array(0.78))  # fmt: skip
    body.table.cache_clear()
    s = scene(chair_xy=(0.0, -0.62), wide=True)  # a bench along the only approachable edge, in front of the cup
    from locomani.tasks.scene import choose as C

    before = C.choose(s, "cup", table=str(table))
    assert before is None or before.needs_prep  # the side stances are beyond the synthetic grid's 0.65 m
    was = P.ACTIONS.get("floor")
    P.ACTIONS["floor"] = lambda ctx, plan: None
    try:
        plans = P.propose(s, "cup", table=str(table), kinds=("floor",))
    finally:
        P.ACTIONS.pop("floor", None) if was is None else P.ACTIONS.__setitem__("floor", was)
    assert plans and plans[0].goal.kind == "floor" and plans[0].goal.key == "chair"
    assert plans[0].goal.tol_m == P.TOL_FLOOR_M
    assert P.met(plans[0].goal, plans[0].goal.pos, plans[0].goal.quat)


def test_the_draw_filter_asks_the_selector(monkeypatch):
    import family as F

    calls = []
    fake = SimpleNamespace(place=lambda p: None)
    monkeypatch.setattr(F, "_STATIC", {"/s": fake})
    from locomani.tasks.scene import choose as C
    from locomani.tasks.scene import prep as P

    monkeypatch.setattr(C, "choose", lambda s, k, xy: SimpleNamespace(needs_prep=False))
    t = SimpleNamespace(objects=("cup",))
    assert not F.needs_prep(dict(scene_dir="/s"), t, dict(placements={}, robot_xy=[0, 0]), "floor")
    monkeypatch.setattr(C, "choose", lambda s, k, xy: None)
    monkeypatch.setattr(P, "propose", lambda s, k, xy, kinds: calls.append(kinds) or ["a plan"])
    assert F.needs_prep(dict(scene_dir="/s"), t, dict(placements={}, robot_xy=[0, 0]), "floor") and calls == [
        ("floor",)
    ]
    assert "floor" not in P.ACTIONS  # the stand-in action is taken out again
