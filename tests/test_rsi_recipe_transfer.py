"""TODO 33.13.1 / 33.13.2: the recipe transfer check -- pick_any's tasks, the similar objects by the package's own
features and match, the arm without the recipe, and the promotion verdict."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"
sys.path[:0] = [str(HERE), str(ROOT / "src")]

import recipe_transfer as RT  # noqa: E402


def test_pick_any_lists_free_one_hand_objects_on_supports_with_their_room(tmp_path):
    meta = dict(room=dict(rooms={"kitchen": dict(polygon=[[0, 0], [2, 0], [2, 2], [0, 2]]),
                                 "bed": dict(polygon=[[3, 0], [5, 0], [5, 2], [3, 2]])}),
                objects={"cup": dict(free=True, surface="t", body="cup_b", init_pos=[1, 1, 0.8]),
                         "book": dict(free=True, surface="s", body="book_b", init_pos=[4, 1, 0.5]),
                         "crate": dict(free=True, bimanual=True, surface="t", body="c_b", init_pos=[1, 1, 0.8]),
                         "chair": dict(free=False, init_pos=[1, 1, 0.4]),
                         "ball": dict(free=True, surface=None, body="ball_b", init_pos=[1, 1, 0.1])})  # fmt: skip
    (tmp_path / "scene_meta.json").write_text(json.dumps(meta))
    n = RT.write_tasks(tmp_path, tmp_path / "t.jsonl")
    rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert n == 2 and {r["id"]: r["room"] for r in rows} == {"pick_book": "bed", "pick_cup": "kitchen"}
    assert rows[0]["goal"] == ["held(book)"]


def test_the_shipped_family_lists_its_task_file():
    fam = json.loads((HERE / "families" / "pick_any.json").read_text())
    rows = [json.loads(x)["id"] for x in (ROOT / fam["tasks_file"]).read_text().splitlines() if x.strip()]
    assert fam["tasks"] == rows and len(rows) > 10 and "sandbox" not in fam and set(fam["pools"]) == {"transfer"}


class FakeStatic:
    def __init__(self, objects):
        self.meta = dict(objects=objects, surfaces={})
        ids = {o["body"]: i for i, o in enumerate(objects.values())}
        self.m = SimpleNamespace(body=lambda name: SimpleNamespace(id=ids[name]))

    def object_pose(self, key):
        return np.zeros(3), np.array([1.0, 0, 0, 0])


def test_similar_ranks_by_match_then_feature_distance(monkeypatch):
    objs = {k: dict(free=True, surface="t", body=k) for k in ("soap", "jam", "can", "vase", "spoon")}
    height = dict(soap=0.28, jam=0.12, can=0.20, vase=0.27, spoon=0.02)
    width = dict(soap=0.085, jam=0.07, can=0.088, vase=0.086, spoon=0.03)
    rec = types.ModuleType("skills.recipes.tall")
    rec.match = lambda f: 1.0 if f.height >= 0.18 and 0.08 <= f.width < 0.09 and f.key != "vase" else 0.0
    pkg = SimpleNamespace(recipes=SimpleNamespace(RECIPES=(rec,), features=lambda w, k, ux, z, flat, cover: SimpleNamespace(
        key=k, height=height[k], width=width[k], flat=height[k] < 0.04, container=False)))  # fmt: skip
    from locomani.tasks.scene import graspability as G

    monkeypatch.setattr(G, "resting_half_height", lambda w, k: height[k] / 2)
    monkeypatch.setattr(G, "grasp_height", lambda w, k: 0.9)
    out = RT.similar(pkg, FakeStatic(objs), family_objects=["soap", "spoon"], n=3)
    s = out["tall"]
    assert s["birth"] == ["soap"]
    assert [x["key"] for x in s["similar"]] == ["can", "vase", "jam"]  # can matches; then the nearest by features
    assert [x["score"] for x in s["similar"]] == [1.0, 0.0, 0.0]


def test_the_arm_without_the_recipe_drops_it_from_recipes(tmp_path):
    sk = tmp_path / "skills"
    (sk / "recipes").mkdir(parents=True)
    (sk / "__init__.py").write_text("from . import recipes\n")
    (sk / "recipes" / "__init__.py").write_text("from . import tall, flat\nRECIPES = (tall, flat)\n")
    (sk / "recipes" / "tall.py").write_text("X = 1\n")
    (sk / "recipes" / "flat.py").write_text("X = 2\n")
    off = RT.without(sk, "tall", tmp_path / "off")
    spec = importlib.util.spec_from_file_location("off_pkg", off / "__init__.py", submodule_search_locations=[str(off)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["off_pkg"] = mod
    spec.loader.exec_module(mod)
    assert [r.__name__.rsplit(".", 1)[-1] for r in mod.recipes.RECIPES] == ["flat"]
    assert "tall" in [r.__name__.rsplit(".", 1)[-1] for r in (mod.recipes.tall, mod.recipes.flat)]  # the module stays


@pytest.mark.parametrize(
    "rows,promotable,word",
    [
        ([dict(score=1, n=2, c=0), dict(score=1, n=2, c=0), dict(score=0, n=2, c=0)], True, "promotable"),
        ([dict(score=1, n=2, c=0), dict(score=1, n=2, c=1)], False, "lost"),
        ([dict(score=1, n=2, c=0), dict(score=0, n=2, c=0), dict(score=1, n=0, c=0)], False, "matches only 1"),
    ],
)
def test_the_promotion_verdict(rows, promotable, word):
    ok, why = RT.verdict(rows)
    assert ok is promotable and word in why
