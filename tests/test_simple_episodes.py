"""TODO 49.2: the SIMPLE bed's episode pools: ids, the two practice pools, and the refusal of released (exam) episodes."""

import json
import tempfile
import unittest
from pathlib import Path

from locomani.bench.simple import episodes as EP
from locomani.bench.simple.config import SimpleConfig


def pools(t, n=3):
    q, p = Path(t) / "q", Path(t) / "p"
    for d, tag in ((q, "q"), (p, "p")):
        d.mkdir()
        (d / "XMovePickTeleop.jsonl").write_text(
            "".join(json.dumps(dict(instruction=f"{tag}{i}", cfg={})) + "\n" for i in range(n))
        )
    return SimpleConfig(practice_dir=str(q), dev_dir=str(p))


class Pools(unittest.TestCase):
    def test_q_and_p_lines_load(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = pools(t)
            e = EP.load("XMovePickTeleop/Q/2", cfg)
            self.assertEqual((e["task"], e["level"], e["index"], e["instruction"]), ("XMovePickTeleop", "Q", 2, "q2"))
            e = EP.load("XMovePickTeleop/P/1", cfg.with_overrides(["kinds=PQ"]))
            self.assertEqual((e["level"], e["instruction"]), ("P", "p1"))

    def test_released_ids_are_refused_whatever_the_config(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = pools(t).with_overrides(["kinds=PQL"])
            for bad in (
                "XMovePickTeleop/L0/e0",
                "XMovePickTeleop/L2/e9",
                "XMovePickTeleop/x/3",
                "XMovePickTeleop/Q/e1",
                "XMovePickTeleop",
            ):
                with self.assertRaises(EP.EpisodeRefused, msg=bad):
                    EP.load(bad, cfg)

    def test_a_pool_not_given_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = pools(t)
            with self.assertRaises(EP.EpisodeRefused):
                EP.check_id("XMovePickTeleop/P/0", cfg)  # kinds defaults to Q, as SIMPLE_LOOP=1 did
            EP.check_id("XMovePickTeleop/Q/0", cfg)

    def test_blank_line_and_unknown_task_refused(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = pools(t)
            f = Path(cfg.practice_dir) / "XMovePickTeleop.jsonl"
            f.write_text(f.read_text().replace('{"instruction": "q1", "cfg": {}}', ""))
            EP._lines.cache_clear()
            with self.assertRaises(EP.EpisodeRefused):
                EP.load("XMovePickTeleop/Q/1", cfg)
            with self.assertRaises(EP.EpisodeRefused):
                EP.check_id("NoSuchTask/Q/1", cfg)

    def test_ids_of(self):
        self.assertEqual(EP.ids_of("A/Q/0-2,B/Q/7"), ["A/Q/0", "A/Q/1", "A/Q/2", "B/Q/7"])
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "ids"
            f.write_text("A/Q/1\n\nA/Q/4\n")
            self.assertEqual(EP.ids_of(str(f)), ["A/Q/1", "A/Q/4"])

    def test_no_code_opens_released_files(self):
        src = Path(EP.__file__).read_text()
        self.assertNotIn("dr-level", src)
        self.assertNotIn("episodes.jsonl", src)

    def test_model_cache_choice_is_scoped(self):
        import locomani.paths as paths

        before = paths.MJB_CACHE
        with EP.no_model_cache():
            self.assertEqual(paths.MJB_CACHE, EP.NO_CACHE)
        self.assertEqual(paths.MJB_CACHE, before)  # (the old bed rebound it for good at import)

    def test_the_real_pools_if_present(self):
        cfg = SimpleConfig()
        f = Path(cfg.practice_dir) / "XMovePickTeleop.jsonl"
        if not f.exists():
            self.skipTest("esc outputs not on this machine")
        e = EP.load("XMovePickTeleop/Q/0", cfg)
        self.assertEqual(e["task"], "XMovePickTeleop")
        self.assertIn("cfg", e)


if __name__ == "__main__":
    unittest.main()
