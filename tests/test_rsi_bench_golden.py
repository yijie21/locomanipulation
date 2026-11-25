"""One benchmark backend (TODO 54 phase 3): what the merge of backend_simple / backend_lml / backend_seeded into
backend_bench + bench_common / bench_pools / bench_rows / bench_<adapter> must not change. The golden file holds digests of what
the OLD modules produced: the diagnostic / acceptance / sealed pools (the anti-leaderboard wall: the hidden split seeds and
ids) for every family on real practice files, and rows -> rounds, blank rounds and cards on fixed synthetic records."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RSI = REPO / "scripts" / "rsi"
sys.path.insert(0, str(RSI))
import backend  # noqa: E402
import backend_bench as BB  # noqa: E402

GOLDEN = json.loads((Path(__file__).with_name("golden_bench_pools.json")).read_text())


def h(x) -> str:
    return hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:12]


def fam_of(name):
    try:
        return backend.of(name).load(name)
    except (OSError, FileNotFoundError) as e:  # a practice directory of the esc tree that this machine lacks
        pytest.skip(str(e))


@pytest.mark.parametrize("name", sorted(GOLDEN["draw"]))
def test_pools_draw_the_same_episodes_as_before_the_merge(name, tmp_path):
    fam = fam_of(name)
    if "practice_dir" in fam and not BB.seeded(fam) and not Path(fam["practice_dir"]).is_dir():
        pytest.skip("no practice files on this machine")
    be = backend.of(name)
    for pool in ("diag", "accept", "sealed"):
        d, d7 = tmp_path / pool, tmp_path / f"{pool}7"
        be.draw(fam, pool, d)
        be.draw(fam, pool, d7, 7, fam["tasks"][:2])
        files = {p.name: json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if p.name != "_pool.json"}
        names = sorted(p.name for p in d7.glob("*.json"))
        assert [h(files), h(names)] == GOLDEN["draw"][name][pool], (name, pool)


RECS = {
    "simple_pick": dict(episode_id="XMovePickTeleop/Q/3", success=False, loss="grasp", why="no lift", phases={"reach_grasp": 2.0, "close": 1.0, "lift": 1.5}, marks={"after_lift": {"rise_cm": -20.0, "held": False}, "before_close": {"moved_cm": 1.0}}, comply={"ok": True}, palm_err_cm=3.2, reach_ok=[1, 0], t_end=12.5, wall_s=40, grasp_desc=[0.1, 0.0, 0.02, 0, 0, 1], score=0.9, bench=dict(signature="simple/x")),
    "simple_ok": dict(episode_id="BendPickMP/Q/1", success=True, comply=dict(ok=True), phases={"a": 1.0}, t_end=3.0),
    "simple_comply": dict(episode_id="BendPickMP/Q/2", success=True, comply=dict(ok=False, why="used the body"), phases={"a": 1.0}, t_end=3.0),
    "simple_art": dict(episode_id="OpenOvenTeleop/Q/4", success=False, loss="short", phases={"walk": 3.0, "reach": 2.0, "close": 1.0, "move": 4.0}, art=dict(joint="door", start_deg=0, best_deg=20.0, need_deg=60, end_deg=10, sign=1, t_move=2.1), art_touch=dict(t_hand=3.0, hand_deg=5, body_links=["pelvis"], t_body=2.0), t_end=30),
    "seed_fail": dict(id="reach_target_single/P/5", success=False, why="fell", fell=True, steps=200, marks={"a": 10, "b": 50}, flow={"k": "v"}, body_parts=["l_wrist"], hand_steps=3, body_steps=1, bench_success=True, bench=dict(signature="bigym2/x")),
    "seed_ok": dict(id="pp_box/P/1", success=True, steps=100, marks={"a": 3}),
    "seed_to": dict(id="pp_box/P/2", success=False, why="not_success", termination="timeout", steps=100),
}  # fmt: skip


def lml_recs():
    p = Path(__file__).with_name("lml_real_rows.jsonl")
    recs = {f"lml_{i}": json.loads(x) for i, x in enumerate(p.read_text().splitlines()[:6])}
    recs["lml_err"] = dict(
        episode_id="IThorKitchenCupToSink_scene13/Q/2",
        success=False,
        error="boom [1.234, 2.345] x",
        loss="error",
        steps=300,
    )
    return recs  # fmt: skip


def rounds_and_cards():
    recs = {**RECS, **lml_recs()}
    plan = [("simple_art", n) for n in ("simple_pick", "simple_ok", "simple_comply", "simple_art")]
    plan += [("lml_q_bench", n) for n in recs if n.startswith("lml_")]
    plan += [
        ("bigym2_reach_drawer", "seed_fail"),
        ("humanoidarena_pp_box", "seed_ok"),
        ("humanoidarena_pp_box", "seed_to"),
    ]
    rounds = {}
    for fname, n in plan:
        rounds[f"{fname}:{n}"] = BB.to_round(recs[n], f"{n}_s1", "SH", {"a": 1}, "SIG", backend.of(fname).load(fname))
    return rounds


def test_rows_become_the_same_rounds_and_cards_as_before_the_merge():
    rounds = rounds_and_cards()
    assert {k: h(v) for k, v in rounds.items()} == GOLDEN["rounds"]
    for grp in ("simple", "lml", "seeded"):
        rs = [v for v in rounds.values() if grp in v]
        with tempfile.TemporaryDirectory() as d:
            text = BB.cards(rs, d).read_text().rstrip()
        assert h(text) == GOLDEN["cards"][grp], grp
        assert h([BB.card_text(r) for r in rs]) == GOLDEN["cards"][grp + "_text"], grp


def test_a_round_that_never_ran_keeps_its_shape():
    for name in ("simple_art", "lml_q_bench", "bigym2_reach_drawer"):
        fam = backend.of(name).load(name)
        assert h(BB._blank(fam, "X_s1", "why", "sh", {"a": 1}, "sig")) == GOLDEN["blank"][name]
