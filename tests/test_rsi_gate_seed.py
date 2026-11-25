"""The gate's first incumbent is the seed, copied from the backend's immutable seed package when it names one, never from
the working package the driver rewrites for every round (TODO 54 phase 4 follow-up, 2026-11-25: with the overlap, a copy
taken while a draft sat in the working package would have recorded the draft as the seed)."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

RSI = Path(__file__).resolve().parents[1] / "scripts" / "rsi"


def _gate():
    """scripts/rsi/gate.py by path under its own name: other tests put modules named `gate` on sys.path first."""
    if str(RSI) not in sys.path:
        sys.path.insert(0, str(RSI))
    spec = importlib.util.spec_from_file_location("rsi_gate_seedtest", RSI / "gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _package(d: Path, text: str) -> Path:
    d.mkdir(parents=True)
    (d / "skill.py").write_text(text)
    (d / "__init__.py").write_text("")
    return d


def test_the_first_incumbent_is_copied_from_the_seed_not_the_working_package(tmp_path, monkeypatch):
    G = _gate()
    seed = _package(tmp_path / "seed", "K = {'speed': 1.0}  # the seed\n")
    work = _package(tmp_path / "package", "K = {'speed': 9.9}  # a draft the driver has just installed\n")
    monkeypatch.setattr(G.BK, "layout", lambda family: dict(skills=work, seed=seed))
    rec = G.init_incumbent(tmp_path / "gate", "fam")
    copied = Path(rec["dir"]) / "skills" / "skill.py"
    assert copied.read_text() == (seed / "skill.py").read_text()
    assert json.loads((tmp_path / "gate" / "incumbent.json").read_text())["dir"] == rec["dir"]


def test_a_backend_without_a_seed_key_keeps_the_old_source(tmp_path, monkeypatch):
    G = _gate()
    work = _package(tmp_path / "package", "K = {'speed': 1.0}\n")
    monkeypatch.setattr(G.BK, "layout", lambda family: dict(skills=work))
    rec = G.init_incumbent(tmp_path / "gate", "fam")
    assert (Path(rec["dir"]) / "skills" / "skill.py").read_text() == (work / "skill.py").read_text()


def test_the_bench_layout_names_the_seed_package():
    if str(RSI) not in sys.path:
        sys.path.insert(0, str(RSI))
    BB = importlib.import_module("backend_bench")
    fam = BB.load("simple_art")
    assert BB.layout(fam)["seed"] == BB.base_dir(fam)
