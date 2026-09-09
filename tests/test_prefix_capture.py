import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from locomani.backends import sonic as sonic_pkg
from locomani.rtc.audit_prefix_execution import audit, candidates
from locomani.rtc.transition_capture import Capture


class CaptureTests(unittest.TestCase):
    def test_capture_preserves_actions_and_assigns_chunk_id(self):
        token = np.arange(128).reshape(2, 64)
        grip = np.array([0.02, 0.0])
        calls = []
        records = []

        class Link:
            def get_chunk(self, obs):
                return token, grip

        class Commander:
            def token_frame(self, tk, **kw):
                calls.append((tk, kw))
                return "sent"

        cap = Capture.__new__(Capture)
        cap.chunk_id = 0
        cap.chunk_lock = threading.Lock()
        cap.inference_context = threading.local()
        cap.put = lambda kind, **kw: records.append((kind, kw))
        with (
            patch.object(sonic_pkg, "agent_server", SimpleNamespace(AgentCommander=Commander)),
            patch.object(sonic_pkg, "vla_round", SimpleNamespace(PolicyLink=Link)),
        ):
            cap.instrument()
            link = Link()
            actual, _ = link.get_chunk({})
            self.assertIs(actual, token)
            self.assertEqual(link.capture_context.chunk_id, 0)
            source = dict(kind="chunk", chunk_id=0, action_index=1)
            self.assertEqual(Commander().token_frame(actual[1], trace_source=source), "sent")
        np.testing.assert_array_equal(calls[0][0], token[1])
        self.assertEqual(records[-1][1]["source"], source)
        self.assertEqual(records[-1][1]["receipt_status"], "unconfirmed")

    def test_file_audit_checks_provenance_and_keeps_response_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "chunks").mkdir()
            token = np.ones((2, 64), dtype=np.float32)
            token[0] = 0
            np.savez(root / "chunks/000007.npz", token=token)
            sent = dict(wall=1.0, token=token[1].tolist(), source=dict(chunk_id=7, action_index=1))
            (root / "sent.jsonl").write_text(json.dumps(sent) + "\n")
            debug = [
                dict(wall=0.9, state=dict(body_q=[0.0] * 29, token_state=[0.0] * 64)),
                dict(wall=1.1, state=dict(body_q=[0.3] * 29, token_state=token[1].tolist())),
            ]
            (root / "debug.jsonl").write_text("".join(json.dumps(r) + "\n" for r in debug))
            result = audit(root)
            self.assertEqual(result["provenance_mismatches"], [])
            self.assertEqual(result["chunks"][7]["indices"], [1])
            self.assertEqual(result["chunks"][7]["unique_payload_evidence"], 1)
            self.assertAlmostEqual(result["chunks"][7]["body_knee_change_next_1s_rad"], 0.3)
            sent["source"]["action_index"] = 0
            (root / "sent.jsonl").write_text(json.dumps(sent) + "\n")
            self.assertEqual(len(audit(root)["provenance_mismatches"]), 1)

    def test_repeated_payloads_are_not_unique_receipts(self):
        tokens = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        times = np.array([1.0, 1.1, 1.2])
        self.assertEqual(candidates([1.0, 2.0], tokens, times, 1.3), [0, 1])
        self.assertEqual(candidates([3.0, 4.0], tokens, times, 1.1), [])
        self.assertEqual(candidates([1.0, 2.0], tokens, times, 2.0), [])


if __name__ == "__main__":
    unittest.main()
