"""TODO 25.9.1 phase 1 (docs/handoff/scalebfm-warehouse-rsi-20261014.md §6.3, §9): the first batch of interventions on
the diagnostic pool, each a fork of the incumbent from one phase on, run by the loop's own Prober (loop.probe) two at a
time (GPU 0 and GPU 1, 4 processes each). The predictions in HYPOTHESES were written before any of these ran, from
the incumbent's stall anatomy (§13): 40 of 42 failures stall in the descend with the left grip point ~2.4 cm and the
right ~1 cm above the grasp point, under the contact trigger's 0.5 cm, while the command is already 1-2.5 cm below it.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/scalebfm_warehouse_rsi_20261014/phase1.py [batch2]
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loop  # noqa: E402

OUT = loop.LOOP / "phase1"


def stalls(base: Path, max_up: float, max_across: float = 4.5) -> list[str]:
    """Incumbent stalls whose last capture error has both hands at most max_up cm above and max_across cm across."""
    out = []
    for f in sorted(base.glob("bfm_*.json")):
        j = json.loads(f.read_text())
        ce = (j.get("expert_diagnostics") or {}).get("capture_error")
        if j["expert_reason"] != "reach_stalled" or ce is None:
            continue
        if (
            max(100 * ce[0][2], 100 * ce[1][2]) <= max_up
            and max(abs(100 * ce[0][0]), abs(100 * ce[1][0])) <= max_across
        ):
            out.append(j["label"].replace("bfm", "sonic", 1))
    return out


def successes(base: Path) -> list[str]:
    return [
        f.stem.replace("bfm", "sonic", 1) for f in sorted(base.glob("bfm_*.json")) if json.loads(f.read_text())["ok"]
    ]


def hypotheses(base: Path) -> list[dict]:
    ok = successes(base)
    H = lambda id_, claim, phase, delta, fixed, broken, n_after: dict(  # noqa: E731
        id=id_, claim=claim, evidence="phase-1 stall anatomy (plan §13)", phase=phase, delta=delta,
        predicted_fixed=fixed, predicted_broken=broken, predicted_ok_after=n_after,
    )  # fmt: skip
    return [
        H("ff_up3", "the descend stalls on a 1-3 cm vertical delivery gap; a 3 cm downward feed-forward closes it",
          "reach", {"set": {"visual.reach_feedforward_up": -0.03}}, stalls(base, 3.5), [], 30),
        H("overdrive3", "same gap, closed by aiming the descend 3 cm below the grasp point (the expert's own knob for SONIC's undershoot)",
          "reach", {"set": {"visual.contact_overdrive": 0.03}}, stalls(base, 3.5), [], 30),
        H("pad", "the gap is ScaleBFM's wrist-attitude error (25.3); the jaw-centre + attitude servo removes it",
          "reach", {"servo": "pad"}, stalls(base, 5.0), [], 25),
        H("pad_g0", "with the pad servo the expert's own feedback double-corrects; servo alone",
          "reach", {"servo": "pad", "set": {"visual.tracking_gain": 0.0}}, stalls(base, 5.0), [], 25),
        H("retry2", "near misses close on a second descend", "reach", {"set": {"visual.descend_retries": 2}},
          stalls(base, 2.0), [], 15),
        H("gain0", "without feedback the gap is uncorrected", "reach", {"set": {"visual.tracking_gain": 0.0}}, [], ok, 1),
        H("gain1", "full feedback (known from the stub: 2 fixed, 8 broken; rerun for the new columns, not scored)",
          "reach", {"set": {"visual.tracking_gain": 1.0}}, [], [], 2),
        H("floor74", "§2's hypothesis: the crouch causes the stall; keep ScaleBFM's pelvis >= 0.74 m like SONIC (hands unchanged)",
          "ready", {"extensions": {"pelvis_floor": {"z": 0.74}}}, [], ok, 3),
        H("waist08", "more waist pitch in the solver lets the hands go lower with less crouch", "ready",
          {"set": {"carry.waist_pitch_max": 0.8}}, [], [], 8),
        H("near4", "4 cm closer: less forward reach, hands get lower", "walk", {"stance_shift": -0.04},
          stalls(base, 2.0), [], 15),
        H("near2", "2 cm closer", "walk", {"stance_shift": -0.02}, stalls(base, 1.0), [], 12),
        H("far2", "2 cm farther: harder", "walk", {"stance_shift": 0.02}, [], [], 6),
        H("walk015", "slower walk-in changes nothing in the descend", "walk", {"set": {"visual.walk_speed": 0.15}}, [], [], 8),
        H("walk035", "faster walk-in changes nothing in the descend", "walk", {"set": {"visual.walk_speed": 0.35}}, [], [], 8),
    ]  # fmt: skip


def batch2(base: Path) -> list[dict]:
    """Written after batch 1 (§14): floor74 8 -> 29 and retry2 8 -> 25 stand out of a churn of 9-19 flipped rounds per
    50. Their combination, the floor's dose, and two chaos controls (a 2% gain change at the reach fork; the slow walk
    batch 1 could not run because walk_speed_min 0.25 > 0.15)."""
    b1 = {r["id"]: r for r in json.loads((OUT / "probe.json").read_text())}
    ok = set(successes(base))
    union = sorted(set(b1["floor74"]["fixed"]) | set(b1["retry2"]["fixed"]))
    H = lambda id_, claim, phase, delta, fixed, broken, n_after: dict(  # noqa: E731
        id=id_, claim=claim, evidence="batch 1 (plan §14)", phase=phase, delta=delta,
        predicted_fixed=fixed, predicted_broken=broken, predicted_ok_after=n_after,
    )  # fmt: skip
    floor = lambda z: {"extensions": {"pelvis_floor": {"z": z}}}  # noqa: E731
    return [
        H("floor74_r2", "the crouch and the missing second try are separate causes: their fixes add up (union of the two "
          "fixed sets, floor74's three breaks stay)", "ready", {**floor(0.74), "set": {"visual.descend_retries": 2}},
          union, b1["floor74"]["broken"], 8 + len(union) - 3),
        H("floor70", "dose: a 0.70 m floor fixes about half of what 0.74 does", "ready", floor(0.70),
          b1["floor74"]["fixed"][::2], [], 20),
        H("floor78", "dose: 0.78 m (standing height) is past the useful range -- the hands cannot reach down", "ready",
          floor(0.78), [], sorted(ok), 4),
        H("floor74_r2_pad", "with the crouch and retries handled, the jaw servo cuts lost_attachment", "ready",
          {**floor(0.74), "servo": "pad", "set": {"visual.descend_retries": 2}}, union, [], 36),
        H("gain051", "chaos control: a 2% feedback gain change at the reach flips rounds both ways, net ~0", "reach",
          {"set": {"visual.tracking_gain": 0.51}}, [], [], 8),
        H("walk015", "chaos control: a slower walk-in flips rounds both ways, net ~0", "walk",
          {"set": {"visual.walk_speed": 0.15, "visual.walk_speed_min": 0.15}}, [], [], 8),
    ]  # fmt: skip


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inc = loop.incumbent()
    base = loop.LOOP / "diag" / f"incumbent_{inc['hash']}"
    second = len(sys.argv) > 1 and sys.argv[1] == "batch2"
    pred = OUT / ("predictions_b2.json" if second else "predictions.json")
    if not pred.exists():  # written once, before any probe of the batch runs
        pred.write_text(json.dumps(batch2(base) if second else hypotheses(base), indent=1))
    hyps = json.loads(pred.read_text())
    results, lock = {}, threading.Lock()
    probe_file = OUT / ("probe_b2.json" if second else "probe.json")
    queues = {"0": hyps[0::2], "1": hyps[1::2]}

    def worker(gpu):
        for h in queues[gpu]:
            r = loop.probe([h], inc, base, OUT, gpu=gpu)[0]
            r["predicted_ok_after"] = h["predicted_ok_after"]
            with lock:
                results[h["id"]] = r
                probe_file.write_text(json.dumps([results[x["id"]] for x in hyps if x["id"] in results], indent=1))
            print(
                h["id"],
                r.get("ok_before"),
                "->",
                r.get("ok_after"),
                "fixed",
                len(r.get("fixed", [])),
                "broken",
                len(r.get("broken", [])),
                flush=True,
            )

    ts = [threading.Thread(target=worker, args=(g,)) for g in queues]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("PHASE1 PROBES DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
