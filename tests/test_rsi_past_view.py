"""TODO 39.7: the scene-graph page (scripts/rsi/report/past_view.py) on hand-made records, no simulation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("past_view", REPO / "scripts" / "rsi" / "report" / "past_view.py")
PV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PV)

META = {
    "objects": {
        "Mug": {"free": True, "body": "Mug", "box": {"half_extent": [0.04, 0.04, 0.05], "centre": [0, 0, 0]}},
        "Table": {"free": False, "body": None, "init_pos": [0, 0, 0], "box": {"half_extent": [0.6, 0.4, 0.35], "centre": [1.0, 0.0, 0.35]}},
        "Room": {"free": False, "body": None, "init_pos": [0, 0, 0], "box": {"half_extent": [6.5, 4.9, 4.2], "centre": [1.0, 0.0, 2.0]}},
        "Stool": {"free": False, "body": None, "init_pos": [0, 0, 0], "box": {"half_extent": [0.15, 0.15, 0.35], "centre": [0.45, 0.0, 0.35]}},
    },
    "surfaces": {"T": {"xy": [1.0, 0.0], "top": 0.70, "half_extent": [0.6, 0.4], "yaw": 0.0, "of_object": "Table"}},
}  # fmt: skip


def edge(s, r, o, v, src="derived", **num):
    return dict(s=s, r=r, o=o, v=v, src=src, **num)


def record(label, i, step, kind, ok, edges_before, edges_after, events=(), expected=(), disc=(), entities=None):
    ents = entities or {
        "Mug": {"type": "object", "half_extent_m": [0.04, 0.04, 0.05]},
        "robot": {"type": "robot"},
        "right_hand": {"type": "effector"},
        "T": {"type": "surface"},
        "Table": {"type": "furniture"},
        "Room": {"type": "furniture"},
        "Stool": {"type": "furniture"},
        "region:stance": {"type": "region", "shape": "disc", "centre": [0.4, 0.0], "radius_m": 0.35},
    }
    states = {"Mug": {"p": [1.0, 0.0, 0.75], "q": [1, 0, 0, 0]}, "robot": {"pelvis_p": [0.4, 0.0, 0.79], "yaw": 0.0},
              "right_hand": {"p": [0.7, -0.1, 0.9]}}  # fmt: skip
    return dict(schema="PAST-locomani-0.1", record_id=f"{label}/{i}.1", label=label, task="pick_Mug", step_index=i,
                attempt=1, step=step, kind=kind, action=dict(t0=float(i), t1=float(i) + 1, ok=ok, why="" if ok else "missed", calls=[], check={}),
                roles=dict(target="Mug", effector="right_hand", support="T"), scope=dict(initial=list(ents), added=[]), entities=ents,
                G_before=dict(t=float(i), states=states, edges=list(edges_before)), events=list(events),
                G_after=dict(t=float(i) + 1, states=states, edges=list(edges_after)), expected=list(expected),
                discrepancies=list(disc), outcome=dict(step_ok=ok, round_ok=ok, round_why="", failure_class=None),
                evidence=dict(npz=f"{label}.npz", k0=50 * i, k1=50 * i + 49))  # fmt: skip


def walk_and_grip(label, ok):
    blockers = [edge("Room", "blocking", "region:stance", True, depth_m=2.3), edge("Table", "blocking", "region:stance", True, depth_m=0.7),
                edge("Stool", "blocking", "region:stance", True, depth_m=0.1)]  # fmt: skip
    walk = record(label, 0, "walk", "walk", True, blockers, blockers + [edge("robot", "at", "region:stance", True, err_cm=2.0)],
                  expected=[dict(edge=["*", "blocking", "region:stance"], to=False)],
                  disc=[dict(edge=["*", "blocking", "region:stance"], to=False, observed=True)])  # fmt: skip
    ev = [{"t": 1.10, "edge": ["Mug", "touching", "right_hand"], "from": False, "to": True, "src": "observed"},
          {"t": 1.15, "edge": ["Mug", "touching", "right_hand"], "from": True, "to": False, "src": "observed"},
          {"t": 1.20, "edge": ["Mug", "touching", "right_hand"], "from": False, "to": True, "src": "observed"}]  # fmt: skip
    gb = [edge("Mug", "gripped", "right_hand", False, "observed"), edge("Mug", "supported_by", "T", True)]
    ga = [edge("Mug", "gripped", "right_hand", ok, "observed"), edge("Mug", "supported_by", "T", True),
          edge("Mug", "touching", "right_hand", True, "observed")]  # fmt: skip
    grip = record(label, 1, "grip", "grip", ok, gb, ga, events=ev, expected=[dict(edge=["Mug", "gripped", "right_hand"], to=True)],
                  disc=[] if ok else [dict(edge=["Mug", "gripped", "right_hand"], to=True, observed=False)])  # fmt: skip
    return [walk, grip]


def write(d, label, recs):
    p = Path(d) / f"{label}.past.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return Path(d) / label


class Page(unittest.TestCase):
    def test_single_page_has_every_step_its_edges_and_marks(self):
        with tempfile.TemporaryDirectory() as d:
            text = PV.single(write(d, "pick_Mug_s1", walk_and_grip("pick_Mug_s1", ok=False)), META)
        self.assertTrue(text.startswith("<!doctype html>"))
        for s in ("0.1 walk", "1.1 grip", "gripped", "supported_by", "期待 true ✗", '<svg class="map"', "放大到这一步"):
            self.assertIn(s, text, s)
        self.assertIn("整局 失败", text)

    def test_legacy_blockers_are_dropped_and_the_stools_kept(self):
        recs = walk_and_grip("pick_Mug_s1", ok=True)
        walk = PV.clean(recs[0], META)
        blockers = {e["s"] for e in walk["G_after"]["edges"] if e["r"] == "blocking"}
        self.assertEqual(blockers, {"Stool"})
        self.assertEqual(len(walk["discrepancies"]), 1)  # the stool still blocks the stance: the miss stays
        only_shell = dict(recs[0])
        for g in ("G_before", "G_after"):
            only_shell[g] = dict(recs[0][g], edges=[e for e in recs[0][g]["edges"] if e["s"] != "Stool"])
        self.assertEqual(PV.clean(only_shell, META)["discrepancies"], [])  # nothing real blocks it: the miss goes

    def test_chattering_flips_are_one_line(self):
        ev = walk_and_grip("pick_Mug_s1", ok=True)[1]["events"]
        merged = PV.merged_events(ev)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0]["n"], merged[0]["from"], merged[0]["to"]), (3, False, True))
        self.assertIn("抖动 3 次", PV.events_html(dict(events=ev)))

    def test_compare_aligns_by_kind_and_marks_the_difference(self):
        with tempfile.TemporaryDirectory() as d:
            a = write(d, "pick_Mug_s1", walk_and_grip("pick_Mug_s1", ok=False))
            b = write(d, "pick_Mug_s2", walk_and_grip("pick_Mug_s2", ok=True))
            text = PV.compare(a, b, META)
        self.assertIn("1. walk", text)
        self.assertIn("2. grip", text)
        self.assertIn("·差 1", text)  # gripped false vs true
        self.assertIn('class="diff', text)

    def test_a_round_that_never_moved_shows_its_note(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pick_Mug_s3.past.jsonl"
            p.write_text(
                json.dumps(
                    dict(schema="PAST-locomani-0.1", label="pick_Mug_s3", note="no records: no skill covers this goal")
                )
                + "\n"
            )
            text = PV.single(Path(d) / "pick_Mug_s3", META)
        self.assertIn("no records: no skill covers this goal", text)

    def test_main_writes_the_page(self):
        with tempfile.TemporaryDirectory() as d:
            src = write(d, "pick_Mug_s1", walk_and_grip("pick_Mug_s1", ok=True))
            scene = Path(d) / "scene"
            scene.mkdir()
            (scene / "scene_meta.json").write_text(json.dumps(META))
            out = Path(d) / "page.html"
            self.assertEqual(PV.main([str(src), "--scene", str(scene), "--out", str(out)]), 0)
            self.assertGreater(out.stat().st_size, 2000)


class English(unittest.TestCase):
    def tearDown(self):
        PV.set_lang("zh")

    def test_the_english_page_has_no_chinese_text(self):
        PV.set_lang("en")
        with tempfile.TemporaryDirectory() as d:
            a = write(d, "pick_Mug_s1", walk_and_grip("pick_Mug_s1", ok=False))
            b = write(d, "pick_Mug_s2", walk_and_grip("pick_Mug_s2", ok=True))
            pages = [PV.single(a, META), PV.compare(a, b, META)]
        for text in pages:
            self.assertIn("<html lang='en'>", text)
            cjk = [ch for ch in text if "一" <= ch <= "鿿"]
            self.assertEqual(cjk, [], "".join(cjk[:40]))
        self.assertIn("expected true ✗", pages[0])
        self.assertIn("Zoom to this step", pages[0])
        self.assertIn("round failed", pages[0])
        self.assertIn("(flickered 3 times)", PV.events_html(dict(events=walk_and_grip("x", True)[1]["events"])))
        self.assertIn(" · 1 differ", pages[1])

    def test_the_english_entry_point_writes_a_dot_en_page(self):
        spec = importlib.util.spec_from_file_location(
            "past_view_en", REPO / "scripts" / "rsi" / "report" / "past_view_en.py"
        )
        EN = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(EN)
        with tempfile.TemporaryDirectory() as d:
            src = write(d, "pick_Mug_s1", walk_and_grip("pick_Mug_s1", ok=True))
            scene = Path(d) / "scene"
            scene.mkdir()
            (scene / "scene_meta.json").write_text(json.dumps(META))
            out = Path(d) / "page.html"
            self.assertEqual(EN.main([str(src), "--scene", str(scene), "--out", str(out)]), 0)
            self.assertIn("<html lang='en'>", out.read_text())


class Batch(unittest.TestCase):
    def test_the_folder_script_writes_every_page_and_an_index(self):
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "diag"
            src.mkdir()
            for lab, ok in (("pick_Mug_s1", False), ("pick_Mug_s2", True), ("_retired_pick_Mug_s0", True)):
                write(src, lab, walk_and_grip(lab, ok=ok))
                (src / f"{lab}.json").write_text(
                    json.dumps(dict(label=lab, task="pick_Mug", ok=ok, failure_class=None if ok else "other"))
                )
            scene = Path(d) / "scene"
            scene.mkdir()
            (scene / "scene_meta.json").write_text(json.dumps(META))
            out = Path(d) / "pages"
            r = subprocess.run(["bash", str(REPO / "scripts/rsi/report/past_view_all.sh"), str(src), "--out", str(out),
                                "--no-frames", "--scene", str(scene)], capture_output=True, text=True)  # fmt: skip
            self.assertEqual(r.returncode, 0, r.stderr)
            names = sorted(p.name for p in out.iterdir())
            self.assertEqual(names, ["index.html", "pick_Mug_s1.en.html", "pick_Mug_s2.en.html"])  # retired left out
            index = (out / "index.html").read_text()
            self.assertIn("1 / 2 rounds succeeded", index)
            self.assertIn('href="pick_Mug_s1.en.html"', index)
            self.assertIn("<html lang='en'>", (out / "pick_Mug_s2.en.html").read_text())


class Camera(unittest.TestCase):
    class FakeScene:
        """Walls everywhere but towards azimuths near `open_az` (deg)."""

        def __init__(self, open_az=None, best=1.4):
            self.open_az, self.best = open_az, best

        def clear_distance(self, lookat, az, el):
            if self.open_az is not None and abs((az - self.open_az + 180) % 360 - 180) <= 20:
                return 99.0
            return self.best if abs((az - 135 + 180) % 360 - 180) <= 20 else 0.6

    def test_the_first_unobstructed_view_in_preference_order_wins(self):
        cam = PV.pick_view(self.FakeScene(open_az=-90), [1, 0, 0.8], bearing=0.0)  # +90 is walled, -90 is open
        self.assertEqual((cam["azimuth"], cam["elevation"], cam["distance"]), (-90.0, -28, PV.CAM_DIST_M))

    def test_when_every_view_is_walled_the_clearest_one_comes_closer(self):
        cam = PV.pick_view(self.FakeScene(open_az=None, best=1.4), [1, 0, 0.8], bearing=0.0)
        self.assertEqual(cam["azimuth"], 135.0)
        self.assertAlmostEqual(cam["distance"], 1.3)
        cam = PV.pick_view(self.FakeScene(open_az=None, best=0.6), [1, 0, 0.8], bearing=0.0)
        self.assertEqual(cam["distance"], PV.CAM_MIN_M)
