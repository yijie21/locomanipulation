"""Read shadow mode out (TODO 45.7): each iteration's imagined draft ranking, written before its real rounds, beside them.

    python3 scripts/experiments/imagine_20261108/shadow_report.py '/workspace/code/rsi-scene/sh-*' grasp_floor

Fixed before any shadow round ran. The rule must be the frozen one, an iteration with another rule is listed and left out.
The first number is the real win rate of the top-scored draft against the mean draft of the same iteration (ties
share), over iterations with two drafts or more whose real rounds finished. Recorded drafts, for comparison: 0.904 against
0.787 on grasp_floor, 0.689 against 0.689 on grasp_atomic. Second, does the score tell a draft from the incumbent: the
sign of its score difference against the sign of its real difference.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

# the rule the recorded drafts were scored with; kin_motor was 0e87d7bbf446 until ruff reformatted it, same syntax tree
FROZEN = dict(kin_motor="418d79ab659c", body_table="0faf001db101")
BOOT = 4000


def real_wins(diag: Path, c: dict, labels: list[str]) -> float | None:
    d = diag / f"{'incumbent' if c['incumbent'] else 'candidate'}_{c['hash']}"
    files = [d / f"{l}.json" for l in labels]
    if not all(f.exists() for f in files):
        return None
    return float(np.mean([bool(json.loads(f.read_text())["ok"]) for f in files]))


def main(pattern: str, family: str) -> int:
    groups, other, partial, signs = [], [], 0, []
    for sb in sorted(glob.glob(pattern)):
        root = Path(sb) / "outputs"
        diag = root / "scene_rsi_20261016" / family / "diag"
        for f in sorted((root / "scene_rsi_20261016_hidden" / family / "shadow").glob("iter_*/shadow.json")):
            s = json.loads(f.read_text())
            name = f"{Path(sb).name} {f.parent.name}"
            if s["rule"] != FROZEN:
                other.append(name)
                continue
            rows = [dict(c, real=real_wins(diag, c, s["labels"])) for c in s["candidates"]]
            inc = next((c for c in rows if c["incumbent"]), None)
            drafts = [c for c in rows if not c["incumbent"] and c["real"] is not None]
            partial += sum(c["real"] is None and not c["incumbent"] for c in rows)
            if inc is not None and inc["real"] is not None:
                signs += [
                    (np.sign(round(c["score"] - inc["score"], 6)), np.sign(c["real"] - inc["real"])) for c in drafts
                ]
            if len(drafts) < 2:
                continue
            top = max(c["score"] for c in drafts)
            pick = [c for c in drafts if abs(c["score"] - top) < 1e-9]
            best = max(c["real"] for c in drafts)
            groups.append(dict(name=name, n=len(drafts), scores=[round(c["score"], 1) for c in drafts], real=[round(c["real"], 2) for c in drafts],
                               pick=float(np.mean([c["real"] for c in pick])), rand=float(np.mean([c["real"] for c in drafts])), best=best,
                               hit=float(np.mean([c["real"] == best for c in pick])), wall=s["wall_s"]))  # fmt: skip
    print(
        f"{family}: {len(groups)} iterations with two drafts or more; {partial} drafts without finished real rounds; other rule: {other or 'none'}"
    )
    for g in groups:
        print(
            f"  {g['name']:36s} score {g['scores']}  real {g['real']}  pick {g['pick']:.3f}  mean {g['rand']:.3f}  best {g['best']:.3f}  hit {g['hit']:.2f}  shadow {g['wall']:.0f} s"
        )
    if groups:
        d = np.array([g["pick"] - g["rand"] for g in groups])
        rng = np.random.default_rng(0)
        b = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(BOOT)]
        print(
            f"chosen draft's real win rate {np.mean([g['pick'] for g in groups]):.3f}; mean draft {np.mean([g['rand'] for g in groups]):.3f}; true best {np.mean([g['best'] for g in groups]):.3f}"
        )
        print(
            f"chosen minus mean draft {d.mean():+.3f}, 95% over iterations {np.percentile(b, 2.5):+.3f} .. {np.percentile(b, 97.5):+.3f}; real best chosen {np.mean([g['hit'] for g in groups]):.2f}"
        )
        wide = [g for g in groups if max(g["real"]) - min(g["real"]) >= 0.25]
        if wide:
            print(
                f"iterations whose drafts really differ by 25 points or more: {len(wide)}, real best chosen {np.mean([g['hit'] for g in wide]):.2f}"
            )
    judged = [(p, r) for p, r in signs if p != 0 and r != 0]
    if judged:
        print(
            f"draft against incumbent: the sign agrees in {sum(p == r for p, r in judged)} of {len(judged)} ({len(signs) - len(judged)} ties left out)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
