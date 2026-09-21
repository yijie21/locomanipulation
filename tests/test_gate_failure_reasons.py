"""The failure-reason criterion, and why it is a share rather than a set difference.

Until 2026-09-21 any reason in the candidate that the baseline had not recorded failed the gate.
Measured over seven runs of ONE expert revision on one fixed bank and seed, that fired on 29 of 42
ordered pairs -- 69% -- against 75% on pairs the ledger says genuinely differ, so it was close to a
coin toss rather than a detector. Six failure families appeared across those runs and four of them
in a single run each: eight rounds cannot enumerate the tail, so absence from the baseline is not
evidence of impossibility.

These tests pin the shape of the replacement, not the seven measurements behind it: the bound comes
from the BASELINE's size, a rare new reason passes, a frequent one still fails, and the reasons are
reported either way so the signal is never silently dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gate  # noqa: E402


def _compare(base_reasons, base_rounds, cand_counts, cand_rounds):
    base = dict(rounds=base_rounds, yield_=1.0, failure_reasons=sorted(base_reasons))
    cand = dict(
        rounds=cand_rounds,
        yield_=1.0,
        failure_reasons=sorted(cand_counts),
        failure_counts=dict(cand_counts),
    )
    checks = gate.compare(base, cand)
    return next(c for c in checks if c[0] == "failure_reasons")


def test_a_reason_seen_once_in_eight_rounds_is_not_a_regression():
    """3/8 is what the rule of three allows a reason the baseline never saw; one round is 0.125."""
    name, ok, detail = _compare({"[grasp] known"}, 8, {"[grasp] known": 1, "[grasp] brand new": 1}, 8)
    assert ok, detail
    assert "brand new" in detail  # still reported: a passing check must not hide what it saw


def test_a_new_reason_that_takes_half_the_rounds_still_fails():
    name, ok, detail = _compare({"[grasp] known"}, 8, {"[grasp] brand new": 4}, 8)
    assert not ok, detail


def test_the_bound_tightens_as_the_baseline_grows():
    """A 20-round baseline can rule out more than an 8-round one: 3/20 against 3/8."""
    _, ok_small, _ = _compare({"a"}, 8, {"b": 2}, 20)  # 0.10 share, bound 0.375
    _, ok_large, _ = _compare({"a"}, 20, {"b": 4}, 20)  # 0.20 share, bound 0.15
    assert ok_small
    assert not ok_large


def test_no_new_reason_passes_whatever_the_counts():
    _, ok, _ = _compare({"a", "b"}, 8, {"a": 5, "b": 3}, 8)
    assert ok


def test_metrics_report_counts_so_the_share_can_be_computed():
    """Every arm emits failure_counts; without it the share would silently fall back to one each."""
    rows = [
        {"ok": False, "why": "[grasp] simple clearance: left_x:table"},
        {"ok": False, "why": "[grasp] simple clearance: right_x:table"},
        {"ok": True, "why": ""},
    ]
    for name in ("simple", "carry", "bottle", "crate"):
        counts = gate.METRICS[name](rows).get("failure_counts")
        assert counts, name
        assert sum(counts.values()) == 2, name
    # simple and carry collapse the side, so those two rows are ONE family, not two
    assert len(gate.METRICS["simple"](rows)["failure_counts"]) == 1


def test_the_simple_arm_depth_tolerance_covers_its_measured_spread():
    """4 cm, from seven runs of one revision: mean 0.1508, sd 0.0115, t(0.975,6)*sd*sqrt(2)=0.0398.

    2 cm came from three batches that happened to span 1 cm, and failed 5 of 21 same-revision
    pairs. carry keeps 3 cm because its three runs span 0.0066 m.
    """
    assert gate.ARM_TOL["simple"]["depth_m"] == 0.04
    assert gate.ARM_TOL["carry"]["depth_m"] == 0.03
    spread = 0.1695 - 0.1344  # the measured extremes of the same-revision set
    assert spread > gate.ARM_TOL["simple"]["depth_m"] - 0.01  # the tolerance is of that order
    base = dict(rounds=8, yield_=1.0, end_depth_m=0.1469, failure_reasons=[])
    cand = dict(rounds=8, yield_=1.0, end_depth_m=0.1695, failure_reasons=[], failure_counts={})
    depth = next(c for c in gate.compare(base, cand, arm="simple") if c[0] == "end_depth_m")
    assert depth[1], depth[2]  # the widest same-revision pair seen must not fail


def test_the_yield_tolerance_comes_from_the_baseline_size():
    """A fixed 0.15 claimed a precision eight rounds do not have.

    Seven runs of one revision yielded 0.50-1.00 with sd 0.157 against a binomial prediction of
    0.160, so the spread is sampling; the 95% bound on the difference of two 8-round runs is 0.44.
    Deriving it from n cut same-revision false alarms from 9/42 to 3/42 -- the design point of a
    95% interval -- while still catching four of the five regressions the ledger records.
    """
    wide = gate.yield_bound(dict(rounds=8, yield_=0.75))
    narrow = gate.yield_bound(dict(rounds=100, yield_=0.75))
    assert wide > 0.3, wide  # eight rounds cannot resolve less than that
    assert narrow < wide / 2, (narrow, wide)  # and a hundred rounds should not pretend otherwise
    assert gate.yield_bound(dict(rounds=8, yield_=1.0)) > 0  # finite at the edges, not zero
    assert gate.yield_bound({}) == gate.TOL["yield_abs"]  # a baseline without a count keeps the old number


def test_a_two_round_difference_in_eight_is_not_a_regression():
    """0.75 -> 0.5 is two rounds; the gate used to call that a failure."""
    base = dict(rounds=8, yield_=0.75, failure_reasons=[])
    cand = dict(rounds=8, yield_=0.5, failure_reasons=[], failure_counts={})
    check = next(c for c in gate.compare(base, cand, arm="simple") if c[0] == "yield")
    assert check[1], check[2]


def test_the_same_drop_at_a_hundred_rounds_is_a_regression():
    base = dict(rounds=100, yield_=0.75, failure_reasons=[])
    cand = dict(rounds=100, yield_=0.5, failure_reasons=[], failure_counts={})
    check = next(c for c in gate.compare(base, cand, arm="simple") if c[0] == "yield")
    assert not check[1], check[2]
