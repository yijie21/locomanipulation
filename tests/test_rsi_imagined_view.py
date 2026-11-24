"""TODO 45.7.2: the Composer gets the imagined result only when asked for, and a failing scorer cannot stop an iteration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts" / "rsi"
VIEW = ROOT / "scripts" / "experiments" / "imagine_20261108" / "composer_view.py"
LOOP_NAMES = ("gate", "batch", "family", "scene_round", "digest", "diag_pool", "records", "design", "loop")


@pytest.fixture
def lp(tmp_path, monkeypatch):
    saved, old_path = {k: sys.modules.pop(k, None) for k in LOOP_NAMES}, list(sys.path)
    sys.path[:0] = [str(HERE), str(ROOT / "src")]
    try:
        spec = importlib.util.spec_from_file_location("rsi_imagined_loop", HERE / "loop.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rsi_imagined_loop"] = mod
        spec.loader.exec_module(mod)
        monkeypatch.setattr(mod, "FAMILY", "fam")
        monkeypatch.setattr(mod, "LOOP", tmp_path / "loop")
        monkeypatch.setattr(mod, "HIDDEN", tmp_path / "hidden")
        monkeypatch.setattr(mod.B, "SIM_PY", Path(sys.executable))
        yield mod
    finally:
        sys.path[:] = old_path
        sys.modules.pop("rsi_imagined_loop", None)
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _drafts(tmp_path: Path) -> tuple[Path, list[dict], dict]:
    it = tmp_path / "loop" / "iter_00"
    d = it / "candidate_1"
    d.mkdir(parents=True)
    (d / "candidate.json").write_text(json.dumps({"set": {"a": 1}}))
    return it, [dict(number=1, hash="h1", dir=str(d), diag=str(tmp_path / "diag"))], dict(hash="inc", dir=str(tmp_path / "inc"), set={"b": 2})  # fmt: skip


def _script(tmp_path: Path, body: str) -> str:
    p = tmp_path / "fake_view.py"
    p.write_text(body)
    return str(p)


OK = (
    "import sys, pathlib, json\n"
    "a = sys.argv; out = pathlib.Path(a[a.index('--out') + 1])\n"
    "(out / 'imagined.json').write_text(json.dumps(dict(argv=a[1:])))\n"
)


def test_off_unless_asked_for(lp, tmp_path, monkeypatch):
    it, cands, inc = _drafts(tmp_path)
    monkeypatch.setattr(lp, "IMAGINE", None)
    assert lp.imagined_view(0, it, cands, inc, tmp_path / "base") is None
    assert not (it / "imagined").exists()


def test_the_view_comes_back_and_the_hidden_pools_are_forbidden(lp, tmp_path, monkeypatch):
    it, cands, inc = _drafts(tmp_path)
    (tmp_path / "hidden" / "pool_accept").mkdir(parents=True)
    monkeypatch.setattr(lp, "IMAGINE", _script(tmp_path, OK))
    view = lp.imagined_view(0, it, cands, inc, tmp_path / "base")
    assert view is not None
    argv = view["argv"]
    assert str((tmp_path / "hidden" / "pool_accept").resolve()) == argv[argv.index("--forbid") + 1]
    rows = json.loads((it / "imagined" / "candidates.json").read_text())
    assert [r["number"] for r in rows] == [1, "incumbent"]
    assert rows[0]["set"] == {"a": 1} and rows[1]["diag"] == str(tmp_path / "base")


@pytest.mark.parametrize("body", ["raise SystemExit(7)\n", "import time\ntime.sleep(30)\n", "pass\n"])
def test_a_failing_scorer_leaves_the_composer_with_the_real_rounds(lp, tmp_path, monkeypatch, capsys, body):
    it, cands, inc = _drafts(tmp_path)
    monkeypatch.setattr(lp, "IMAGINE", _script(tmp_path, body))
    monkeypatch.setattr(lp, "IMAGINE_TIMEOUT_S", 1)
    assert lp.imagined_view(0, it, cands, inc, tmp_path / "base") is None
    assert "failed" in capsys.readouterr().out


def test_the_prompt_addendum_has_only_placeholders_the_loop_fills():
    text = (HERE / "prompts" / "composer_imagined.md").read_text()
    assert "{CONTEXT}" in text or "imagined" in text
    left = {w for w in text.replace("{CONTEXT}", "").split() if w.startswith("{") and w.endswith("}")}
    assert not left


@pytest.mark.skipif(not VIEW.exists(), reason="the experiment's scripts are not in this checkout")
def test_the_view_lists_doubts_and_has_no_total_and_no_incumbent():
    spec = importlib.util.spec_from_file_location("composer_view_t", VIEW)
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    s = [dict(number=1, hash="h", incumbent=False), dict(number="incumbent", hash="i", incumbent=True)]
    row = lambda score: dict(task="T", score=score, why=[])  # noqa: E731
    rows = {1: dict(d1=row(0.0), d2=row(0.9), x1=row(0.0), x2=row(0.9)), "incumbent": dict(d1=row(0.0), d2=row(0.0), x1=row(0.9), x2=row(0.0))}  # fmt: skip
    v = cv.summarize(s, rows, ["d1", "d2"], ["x1", "x2"], {1: dict(d1=True, d2=True)})
    c = v["candidates"][0]
    assert [d["start"] for d in c["doubtful"]] == ["x1"]
    assert c["check_on_diagnostic_starts"] == dict(doubtful=1, doubtful_real_wins=1, others=1, others_real_wins=1)
    assert c["per_task"]["T"] == dict(starts=2, doubtful=1)
    assert [x["number"] for x in v["candidates"]] == [1]
    assert not {"score", "rank", "total", "mean", "new_doubtful"} & set(c)
