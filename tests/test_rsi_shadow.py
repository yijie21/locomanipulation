"""TODO 45.7: the imagined score of an iteration's drafts stays out of the iteration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"
LOOP_NAMES = ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "records", "design", "loop")


@pytest.fixture
def lp(tmp_path, monkeypatch):
    saved, old_path = {k: sys.modules.pop(k, None) for k in LOOP_NAMES}, list(sys.path)
    sys.path[:0] = [str(HERE), str(ROOT / "src")]
    try:
        spec = importlib.util.spec_from_file_location("rsi_shadow_loop", HERE / "loop.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rsi_shadow_loop"] = mod
        spec.loader.exec_module(mod)
        monkeypatch.setattr(mod, "FAMILY", "fam")
        monkeypatch.setattr(mod, "LOOP", tmp_path / "loop")
        monkeypatch.setattr(mod, "HIDDEN", tmp_path / "hidden")
        monkeypatch.setattr(mod.B, "SIM_PY", Path(sys.executable))
        yield mod
    finally:
        sys.path[:] = old_path
        sys.modules.pop("rsi_shadow_loop", None)
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _drafts(tmp_path: Path) -> tuple[list[dict], dict]:
    d = tmp_path / "loop" / "iter_00" / "candidate_1"
    d.mkdir(parents=True)
    (d / "candidate.json").write_text(json.dumps({"set": {"a": 1}}))
    return [dict(number=1, hash="h1", dir=str(d))], dict(hash="inc", dir=str(tmp_path / "inc"), set={"b": 2})


def _script(tmp_path: Path, body: str) -> str:
    p = tmp_path / "fake_shadow.py"
    p.write_text(body)
    return str(p)


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def test_shadow_is_off_unless_asked_for(lp, tmp_path, monkeypatch):
    cands, inc = _drafts(tmp_path)
    monkeypatch.setattr(lp, "SHADOW", None)
    before = _tree(tmp_path)
    assert lp.shadow_drafts(0, cands, inc) is None
    assert _tree(tmp_path) == before


def test_the_score_lands_in_the_hidden_tree_and_nowhere_else(lp, tmp_path, monkeypatch):
    cands, inc = _drafts(tmp_path)
    monkeypatch.setattr(lp, "SHADOW", _script(tmp_path, "import sys, pathlib\n"
                        "a = sys.argv; pathlib.Path(a[a.index('--out') + 1], 'shadow.json').write_text('{}')\n"))  # fmt: skip
    loop_before = _tree(tmp_path / "loop")
    assert lp.shadow_drafts(3, cands, inc) is None
    out = tmp_path / "hidden" / "shadow" / "iter_03"
    assert (out / "shadow.json").exists()
    assert _tree(tmp_path / "loop") == loop_before  # the ledger and the contexts live under LOOP
    rows = json.loads((out / "candidates.json").read_text())
    assert [r["number"] for r in rows] == [1, "incumbent"]
    assert rows[0]["set"] == {"a": 1} and rows[1]["set"] == {"b": 2} and rows[1]["incumbent"] is True


@pytest.mark.parametrize("body", ["raise SystemExit(7)\n", "import time\ntime.sleep(30)\n"])
def test_a_failing_or_hanging_shadow_does_not_stop_the_iteration(lp, tmp_path, monkeypatch, capsys, body):
    cands, inc = _drafts(tmp_path)
    monkeypatch.setattr(lp, "SHADOW", _script(tmp_path, body))
    monkeypatch.setattr(lp, "SHADOW_TIMEOUT_S", 1)
    assert lp.shadow_drafts(0, cands, inc) is None
    assert "failed" in capsys.readouterr().out


def test_a_shadow_that_cannot_start_does_not_stop_the_iteration(lp, tmp_path, monkeypatch, capsys):
    cands, inc = _drafts(tmp_path)
    monkeypatch.setattr(lp, "SHADOW", str(tmp_path / "missing" / "shadow.py"))
    assert lp.shadow_drafts(0, cands, inc) is None
    assert "failed" in capsys.readouterr().out
