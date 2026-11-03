"""TODO 39.7 (user 2026-11-02): the English page of a round's PAST scene-graph records -- past_view.py with `--lang en`
(the same map, relation list, events, `--compare` and `--frames`; only the words differ). Default output
outputs/scene_rsi_20261016/past_view/<label>.en.html, beside the Chinese page.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/report/past_view_en.py <diag dir>/<label> [--compare <round>] [--frames] [--out <page.html>]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import past_view  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return past_view.main([*argv, "--lang", "en"] if "--lang" not in argv else argv)


if __name__ == "__main__":
    raise SystemExit(main())
