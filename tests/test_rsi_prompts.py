"""TODO 54 phase 2: one core rules file plus a generated appendix, one prompt set, and a size that stays small.

Every family resolves its rules and prompts; the assembled prompts of each role hold the rules the role acts on (by
stable id), none of the stale terms of the removed features (benchmark families), and a bounded size; the history a role
gets is capped. The sizes are printed (pytest -s) as the before / after record of the review: before, the roles read
the whole rules file (35.8 KB for simple_art, 6 reads an iteration) plus a 9-15 KB prompt each."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RSI = ROOT / "scripts" / "rsi"
sys.path.insert(0, str(RSI))

import backend as BK  # noqa: E402
import rules_build as RB  # noqa: E402

FAMILIES = sorted(p.stem for p in (RSI / "families").glob("*.json") if json.loads(p.read_text()).get("name"))
BENCH = [f for f in FAMILIES if "rules" in json.loads((RSI / "families" / f"{f}.json").read_text())]
SCENE = [
    f for f in FAMILIES if f not in BENCH and "sandbox" in json.loads((RSI / "families" / f"{f}.json").read_text())
]
ROLE_VARS = dict(
    FAMILY="fam",
    FAMILY_FILE="ff.json",
    CONTEXT="/c/context.json",
    CARDS="/c/cards.md",
    DIAG_INC="/d",
    POOL="/p",
    SKILLS="/s",
    SKILLS_NOTE="n",
    SIM_PY="py",
    LIBRARY="",
    N_DIAG=60,
    N_DIAG_TASK="15",
    N_ACCEPT=120,
    N_ACCEPT_TASK="30",
    WHAT="on X",
    REVIEW_NOTES="/r",
    ITER="/i",
    TMP="/t",
    PLAN="",
    CANDIDATES_NOTE="",
    SCENE_NOTE="in a room",
    SCENE_NOTE_="",
)
MUST_KEEP = (
    "frozen",
    "hidden",
    "win",
    "levels",
    "noise",
    "pool",
    "regress",
    "branch",
    "gate",
    "check-half",
    "sealed",
    "envelope",
)
STALE = ("manager", "prober", "second shot", "repair", "speculative")
OLD_FIXED_SIMPLE_ART = (
    293_000  # the review's measure: rules read 6 times plus the prompts (Reviewer, Manager, 3 Engineers, Composer)
)


def fam(name):
    spec = json.loads((RSI / "families" / f"{name}.json").read_text())
    return spec, BK.layout(spec)


def prompt(role, name):
    spec, lay = fam(name)
    t = RB.role_prompt(role, lay["prompts"], spec, lay["caps"], lay["rules"], lay.get("class_suggest"))
    for k, v in ROLE_VARS.items():
        t = t.replace("{" + k + "}", str(v))
    return t


def test_every_family_resolves_its_rules_and_prompts():
    assert BENCH and SCENE
    for name in BENCH + SCENE:
        spec, lay = fam(name)
        assert Path(lay["rules"]).is_file(), name
        assert (Path(lay["prompts"]) / "reviewer.md").is_file(), name
        assert Path(lay["prompts"]) == RSI / "prompts", name
        assert (RSI / "rules_core.md").is_file()
        roles = ("reviewer", "engineer", "composer") + (("manager",) if name in SCENE else ())
        for r in roles:
            t = prompt(r, name)
            assert "{RULES}" not in t and "{>" not in t, (name, r)


def test_the_old_forks_are_gone():
    left = [
        p.name
        for p in RSI.iterdir()
        if re.match(r"(rules_(simple|lml|bigym2|humanoidarena)|prompts_)", p.name) or p.name == "rules.md"
    ]
    assert left == []
    assert not list((RSI / "prompts").glob("NOTES.md")) and not list((RSI / "prompts").glob("RUNBOOK.md"))


@pytest.mark.parametrize("name", BENCH + SCENE)
def test_the_assembled_prompts_hold_every_must_keep_rule(name):
    allp = "".join(prompt(r, name) for r in ("reviewer", "engineer", "composer", "manager"))
    if name in BENCH:
        allp += RB.core_slice("manager")  # the gate and the sealed set are told to the roles that act on them
    if json.loads((RSI / "families" / f"{name}.json").read_text()).get("bed_fixed"):
        assert "[bed-fixed]" in allp, name
    for i in MUST_KEEP:
        assert f"[{i}]" in allp, (name, i)
    for role, ids in (
        ("reviewer", ("levels", "noise", "win")),
        ("engineer", ("guard", "levels", "scope", "tunable")),
        ("composer", ("gate", "scope", "segments")),
    ):
        t = prompt(role, name)
        for i in ids:
            assert f"### [{i}]" in t, (name, role, i)


@pytest.mark.parametrize("name", BENCH)
def test_benchmark_prompts_hold_no_stale_terms(name):
    for role in ("reviewer", "engineer", "composer"):
        t = prompt(role, name).lower()
        bad = [w for w in STALE if w in t]
        assert not bad, (name, role, bad)


def test_the_core_ids_are_unique_and_the_slices_name_real_ids():
    ids = RB.core_ids()
    assert len(ids) == len(set(ids)) and len(ids) >= 25
    for role, want in RB.ROLE_CORE.items():
        assert set(want.split()) <= set(ids), (role, set(want.split()) - set(ids))


def test_the_appendix_is_generated_from_the_family_file_and_the_guard():
    spec, lay = fam("simple_art")
    eng = RB.rules_for("engineer", spec, lay["caps"], lay["rules"], lay.get("class_suggest"))
    assert "OpenOvenTeleop (limit 20 s)" in eng and "skill.py, handover.py, pick.py, camera_view.py" in eng
    assert "locomani.bench.simple" in eng and "400 lines" in eng and "`feed_torque`" in eng
    rev = RB.rules_for("reviewer", spec, lay["caps"], lay["rules"], lay.get("class_suggest"))
    assert "`comply`" in rev and "Pools and caps" in rev and "check_alpha 0.1" in rev


def test_the_driver_cites_ids_not_numbers():
    bad = []
    for f in RSI.glob("*.py"):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"\brules?(\.md)? [0-9]+\b", line) and f.name != "rules_build.py":
                bad.append(f"{f.name}:{n}")
    assert bad == [] or all(x.startswith("gate.py") for x in bad), bad


def test_sizes_before_and_after(capsys):
    """Fixed text per iteration of simple_art (the Reviewer, 3 Engineers, the Composer; the Manager no longer runs): the
    rules slice plus the prompt, against the review's 293 KB (rules read 6 times plus 6 prompts)."""
    per = {r: len(prompt(r, "simple_art").encode()) for r in ("reviewer", "engineer", "composer")}
    total = per["reviewer"] + 3 * per["engineer"] + per["composer"]
    scene = {r: len(prompt(r, "grasp_atomic").encode()) for r in ("reviewer", "manager", "engineer", "composer")}
    with capsys.disabled():
        print(
            f"\nsimple_art fixed text per iteration: {per} total {total} B (before {OLD_FIXED_SIMPLE_ART} B, {OLD_FIXED_SIMPLE_ART / total:.1f}x)"
        )
        print(f"grasp_atomic: {scene}")
    assert total * 3 <= OLD_FIXED_SIMPLE_ART
    assert max(per.values()) < 30_000 and max(scene.values()) < 40_000


def test_the_history_is_capped_and_flat():
    import rsi_loop_ns as NS

    lp = NS.fresh_loop(scene=False)

    def rows(n_iter):
        es = []
        for n in range(n_iter):
            es += [dict(step="review", iter=n, usage={"x": 1}), dict(step="gate", iter=n, reason="r" * 5000)]
        return es

    sizes = [len(json.dumps(lp.compact_history(rows(k)))) for k in (3, 6, 12)]
    assert (
        sizes[1] == sizes[2] or abs(sizes[2] - sizes[1]) < 200
    )  # the same two iterations; only the count in the note moves
    assert all(len(json.dumps(r)) < 3500 for r in lp.compact_history(rows(4))[1:])
