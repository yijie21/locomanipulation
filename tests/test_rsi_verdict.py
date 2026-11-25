"""TODO 54: scripts/rsi/verdict.py, the one verdict function of the check half, the gate (stage 1 and the full pool) and
the sealed set: the exact binomial edges, Holm, the per-task sign test, min_net ignored where p_max is set, and the scene
rule against recorded gate tallies."""

import importlib.util
import sys
from math import comb
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "rsi"
spec = importlib.util.spec_from_file_location("rsi_verdict_under_test", HERE / "verdict.py")
V = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = V
spec.loader.exec_module(V)


def tally(b, c, per_task=None, **kw):
    t = dict(n=b + c, b=b, c=c, candidate_ok=b + 10, incumbent_ok=10 + c, candidate_fell=0, incumbent_fell=0,
             candidate_dropped=0, incumbent_dropped=0, per_task=per_task or {})  # fmt: skip
    return {**t, **kw}


def test_sign_p_exact_edges():
    assert V.sign_p(0, 0) == 1.0
    assert V.sign_p(5, 0) == pytest.approx(1 / 32)
    assert V.sign_p(4, 0) == pytest.approx(1 / 16)
    assert V.sign_p(6, 1) == pytest.approx(sum(comb(7, i) for i in (6, 7)) / 2**7)
    assert V.sign_p(3, 1) == pytest.approx(5 / 16)  # the check half's trash: 1 fixed, 3 broken: p 0.31 of being worse
    assert V.sign_p(1, 3) == pytest.approx(15 / 16)


def test_the_smallest_accepted_tally_at_p_005_is_five_to_nothing():
    least = {c: next(b for b in range(60) if V.sign_p(b, c) <= 0.05) for c in range(5)}
    assert least == {0: 5, 1: 7, 2: 9, 3: 10, 4: 12}  # verify.md 8


def test_holm_step_down():
    out = V.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}, 0.05)
    assert [out[k]["reject"] for k in "abcd"] == [True, False, False, False]  # 0.03 > 0.05 / 3 ends the rejections
    assert out["a"]["p_holm"] == 0.04 and out["c"]["p_holm"] == 0.09 and out["b"]["p_holm"] == 0.09
    assert out["d"]["p_holm"] == 0.5
    out = V.holm({"a": 0.001, "b": 0.01, "c": 0.02}, 0.05)
    assert all(v["reject"] for v in out.values())  # 0.001 <= 0.05/3, 0.01 <= 0.05/2, 0.02 <= 0.05


def test_min_net_is_not_read_where_p_max_is_set():
    t = tally(5, 0, {"T": dict(n=5, b=5, c=0)})
    assert V.decide(t, "gate", dict(p_max=0.05, min_net=99))["decision"]  # it could never bind there
    assert not V.decide(t, "gate", dict(p_max=None, min_net=99))["decision"]  # the scene rule keeps it
    assert V.decide(t, "gate", dict(p_max=None, min_net=3))["decision"]
    assert not V.decide(tally(4, 0), "gate", dict(p_max=0.05))["decision"]  # p 0.0625


def test_stage_one_rejects_only_on_a_net_loss():
    assert V.decide(tally(0, 0), "stage1")["decision"]  # net 0 was rejected before
    assert V.decide(tally(2, 2), "stage1")["decision"]
    r = V.decide(tally(1, 2), "stage1")
    assert not r["decision"] and "net wins -1 < 0" in r["reasons"][0]


def test_the_per_task_test_rejects_a_significant_loss_only():
    small = {"TaskA": dict(n=15, b=14, c=0), "TaskB": dict(n=15, b=0, c=2)}  # the old rule (net <= -2) rejected this
    r = V.decide(tally(14, 2, small), "gate", dict(p_max=0.05, alpha_task=0.05))
    assert r["decision"] and not r["per_task"]["TaskB"]["regress"]
    big = {"TaskA": dict(n=15, b=14, c=0), "TaskB": dict(n=15, b=0, c=7)}  # 7 losses to 0: p 1/128
    r = V.decide(tally(14, 7, big), "gate", dict(p_max=0.05, alpha_task=0.05))
    assert not r["decision"] and r["per_task"]["TaskB"]["regress"]
    assert any("task TaskB regresses" in w for w in r["reasons"])
    # Holm across four tasks: 6 losses to 0 (p 1/64) passes 0.10 / 4 at the check half, not 0.05 / 4 at the gate
    four = {t: dict(n=15, b=2, c=0) for t in ("A", "B", "C")} | {"D": dict(n=15, b=0, c=6)}
    t = tally(6, 6, four)
    assert V.decide(t, "check", dict(check_min_net=0, alpha_task=0.10))["per_task"]["D"]["regress"]
    assert not V.decide(t, "check", dict(check_min_net=0, alpha_task=0.05))["per_task"]["D"]["regress"]


def test_the_check_half_sends_on_the_net_threshold_and_a_clean_tasks_test():
    t = tally(5, 1, {"A": dict(n=10, b=5, c=1)})
    assert V.decide(t, "check", dict(check_min_net=4, alpha_task=0.10))["decision"]
    assert not V.decide(t, "check", dict(check_min_net=5, alpha_task=0.10))["decision"]  # net 4 < 5


def test_the_sealed_kind_uses_the_family_p_and_the_task_test():
    final = {f"T1_s{i}": dict(ok=True, task="T1") for i in range(20)} | {
        f"T2_s{i}": dict(ok=False, task="T2") for i in range(10)
    }
    seed = {f"T1_s{i}": dict(ok=False, task="T1") for i in range(20)} | {
        f"T2_s{i}": dict(ok=True, task="T2") for i in range(10)
    }
    r = V.judge(final, seed, "sealed", dict(p_max=0.05, alpha_task=0.05))
    assert r["b"] == 20 and r["c"] == 10 and not r["decision"]  # 20:10 p 0.049 passes, but T2 lost 10 to 0
    assert r["per_task"]["T2"]["regress"]
    r = V.judge(final, seed, "sealed", dict(p_max=0.05, alpha_task=None))
    assert r["decision"]  # without the cap: p = 0.0494


# a sample of recorded gate tallies of the scene families (p_max None, gate rule 2): (accepted, b, c, candidate_fell,
# incumbent_fell, candidate_dropped, incumbent_dropped, candidate_ok, incumbent_ok, n), from /workspace/code/rsi-scene/*
RECORDED = [
    (False, 2, 4, 0, 0, 22, 20, 78, 80, 150), (False, 2, 0, 0, 0, 3, 3, 27, 25, 40),
    (False, 5, 2, 1, 1, 0, 1, 12, 9, 40),
    (True, 3, 0, 0, 0, 1, 1, 29, 26, 32), (True, 5, 1, 1, 1, 2, 1, 20, 16, 40),
    (True, 10, 0, 0, 0, 1, 7, 26, 16, 32), (True, 18, 1, 1, 1, 1, 1, 26, 9, 40),
    (True, 20, 3, 2, 1, 0, 1, 26, 9, 40), (True, 23, 1, 1, 1, 1, 1, 31, 9, 40),
    (True, 26, 0, 1, 1, 0, 1, 35, 9, 40), (True, 28, 1, 1, 1, 0, 1, 36, 9, 40),
    (True, 43, 0, 0, 0, 0, 0, 71, 28, 90), (True, 61, 0, 3, 3, 15, 0, 61, 0, 90),
]  # fmt: skip


@pytest.mark.parametrize("row", RECORDED)
def test_the_scene_rule_gives_the_recorded_verdict(row):
    acc, b, c, cf, inf, cd, ind, cok, iok, n = row
    t = tally(b, c, n=n, candidate_fell=cf, incumbent_fell=inf, candidate_dropped=cd, incumbent_dropped=ind,
              candidate_ok=cok, incumbent_ok=iok)  # fmt: skip
    t["p_one_sided"] = round(V.sign_p(b, c), 5)
    assert V.decide(t, "gate", dict(min_net=3, p_max=None, alpha_task=None))["decision"] is acc
