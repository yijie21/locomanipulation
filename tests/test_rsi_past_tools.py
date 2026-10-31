"""TODO 39.3 / 39.4: the recorder's side files in the loop's tools (batch.retire, the digest section, past.py CLI)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

import batch as B  # noqa: E402


class Retire(unittest.TestCase):
    def test_retire_renames_the_past_side_file_too(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "pick_s1.json"
            f.write_text(json.dumps(dict(label="pick_s1", ok=True, rule=B.SUCCESS_RULE, motor="old")))
            (Path(d) / "pick_s1.npz").write_bytes(b"x")
            (Path(d) / "pick_s1.past.jsonl").write_text("{}\n")
            B.retire(f)
            names = sorted(p.name for p in Path(d).iterdir())
        self.assertEqual(names, ["_motorold_pick_s1.json", "_motorold_pick_s1.npz", "_motorold_pick_s1.past.jsonl"])
