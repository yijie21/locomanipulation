import sys
import unittest
from pathlib import Path

from locomani.pipeline.audit_collection_timing import audit


class TimingAuditTests(unittest.TestCase):
    def test_failure_before_closing_still_reports_idle_execution_windows(self):
        row = dict(
            case_id="stalled_before_capture",
            ok=False,
            execution=dict(
                events=[
                    dict(stage="recovery_brake", wall_time=10.0),
                    dict(stage="grasp.local_align", wall_time=10.2),
                    dict(stage="failed", wall_time=12.0),
                ],
                metrics={},
            ),
        )
        trace = [
            dict(
                wall=10.0 + i * 0.02,
                u17=[0.0] * 17,
                hands={"right": [0.0245, 0.0245]},
                pelvis=[0.0, 0.0, 0.8],
                cmd=dict(mode=4, height=0.7),
            )
            for i in range(101)
        ]
        result = audit(row, trace)
        self.assertGreater(result["quiet_prefix_count"], 30)
        self.assertFalse(result["capture_reached"])

    def test_truncated_record_cannot_claim_a_complete_timing_audit(self):
        row = dict(
            case_id="interrupted",
            ok=False,
            execution=dict(events=[dict(stage="recovery_brake", wall_time=10.0)], metrics={}),
        )
        result = audit(row, [])
        self.assertFalse(result["timing_complete"])
        self.assertEqual(result["prefix_windows"], [])

    def test_matching_endpoints_do_not_hide_missing_middle_frames(self):
        row = dict(
            case_id="gap",
            ok=False,
            execution=dict(
                events=[dict(stage="recovery_brake", wall_time=10.0), dict(stage="failed", wall_time=11.0)], metrics={}
            ),
        )
        trace = [dict(wall=t, u17=[0.0] * 17, pelvis=[0.0, 0.0, 0.8]) for t in [10.0, 10.04, 10.4, 10.8, 11.0]]
        result = audit(row, trace)
        self.assertFalse(result["timing_complete"])
        self.assertEqual(result["prefix_windows"], [])
