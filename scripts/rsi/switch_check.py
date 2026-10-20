"""TODO 27.2.1 (plan docs/handoff/scene-rsi-plan-20261016.md D5, §7): the motor layer starts every walk by switching
mode 4 -> 2, which drops the two ankle keypoints; D5 says switches only add keypoints (controller-compare S1: dropping
shoulder / elbow keypoints let the wrists fall 21-28 cm). Measure this one switch the way S1 did: stand in mode 4 with
the right jaw reached forward, switch to mode 2 with the same reference, and compare the 0.5 s after the switch with a
control round that stays in mode 4 (wrist, pelvis and foot displacement). Same for 2 -> 4 (adds the ankles).

    CUDA_VISIBLE_DEVICES=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/rsi/switch_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src")]

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.backends.scalebfm.motor import ANKLE, MODE_STAND, MODE_WALK, WRIST, Motor, MotorConfig  # noqa: E402

OUT = REPO / "outputs" / "scene_rsi_20261016" / "motor_check"
WALK = REPO / "outputs" / "scene" / "walk_mars"


def trial(policy, first: int, second: int | None, jitter: float, seed: int) -> dict:
    sim = R.Sim(policy.meta, backend=ScaleBfmBackendConfig(scene_dir=str(WALK), object_name=""))
    m = Motor(sim, policy, MotorConfig(), model_xml=str(WALK / "scene_locomani.xml"))
    m.reset((0.0, 0.0), 0.0)
    rng = np.random.default_rng(seed)
    sim.d.qpos[m.kp.qadr] += rng.uniform(-jitter, jitter, 29)  # repeats differ by the joints (flat floor, base frame)
    m.reach("right", np.array([0.40, -0.18, 0.95]), R.rot_z(0.0))
    m._set_mode(first)
    if first == MODE_WALK:
        m.pinned = None
    m.hold(1.0)
    before = dict(wr=sim.body(WRIST["right"])[0], wl=sim.body(WRIST["left"])[0], p=sim.d.qpos[:3].copy(),
                  fl=sim.body(ANKLE["left"])[0], fr=sim.body(ANKLE["right"])[0])  # fmt: skip
    if second is not None:
        if second == MODE_WALK:
            m.pinned = None
        else:
            m.pinned = {ANKLE[s]: sim.body(ANKLE[s]) for s in ("left", "right")}
        m._set_mode(second)
    m.hold(0.5)
    after = dict(wr=sim.body(WRIST["right"])[0], wl=sim.body(WRIST["left"])[0], p=sim.d.qpos[:3].copy(),
                 fl=sim.body(ANKLE["left"])[0], fr=sim.body(ANKLE["right"])[0])  # fmt: skip
    return {k: round(100 * float(np.linalg.norm(after[k] - before[k])), 2) for k in before} | dict(
        fell=m.fell_at is not None
    )


def main() -> int:
    policy = R.Policy("cuda", ScaleBfmBackendConfig())
    res = {}
    for name, first, second in (
        ("4_to_2", MODE_STAND, MODE_WALK),
        ("stay_4", MODE_STAND, None),
        ("2_to_4", MODE_WALK, MODE_STAND),
        ("stay_2", MODE_WALK, None),
    ):
        rows = [trial(policy, first, second, 0.03, s) for s in range(5)]
        res[name] = dict(
            rows=rows,
            max={k: max(r[k] for r in rows) for k in rows[0] if k != "fell"},
            falls=sum(r["fell"] for r in rows),
        )
        print(name, json.dumps(res[name]["max"]), "falls", res[name]["falls"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "switch.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
