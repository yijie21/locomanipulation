#!/usr/bin/env python3
"""Label each round of a crate evaluation "jumpy" or not, from artifacts it already wrote.

Bimodal behaviour ("some rounds are clean, some twitch the whole way through") cannot be
diagnosed until the rounds are labelled, and the two kinds of jump need different evidence:

  transient  a step at the handover. It lands BEFORE the trace's first row and before the
             video's first frame, so only `placement.takeover` sees it (added 2026-09-18).
  sustained  twitching that lasts. The trace samples once per chunk (~0.73 s), which is far
             too slow for a 0.3 s step but perfectly adequate for 90 s of shaking.

Then the first question to ask of the labels is whether they are correlated WITHIN A BATCH.
The crate evaluation restarts the stack once per `--batch-rounds` and never restarts the
deploy per round (`restart_deploy` is not implemented on this line), so any state that
latches inside the deploy survives every round of a batch and nothing else. Runs of
`batch_rounds` mean deploy state; scattered rounds mean a per-round race at the switch.

    python scripts/experiments/crate_jump_label_20260918.py outputs/crate_eval100_<date>
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def rounds_of(run: Path) -> list[dict]:
    manifest = run / "closed_loop.manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"no {manifest}; give me a finished crate evaluation directory")
    return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]


def transient(row: dict) -> dict | None:
    """The handover step, if this run is new enough to have recorded it."""
    takeover = (row.get("placement") or {}).get("takeover")
    if not takeover or "held" not in takeover or "streamed" not in takeover:
        return None
    held, streamed = takeover["held"], takeover["streamed"]
    pitch = None
    if held.get("torso_pitch") is not None and streamed.get("torso_pitch") is not None:
        pitch = math.degrees(streamed["torso_pitch"] - held["torso_pitch"])
    return dict(
        yaw_deg=(streamed["yaw_deg"] - held["yaw_deg"] + 180) % 360 - 180,
        pitch_deg=pitch,
        pelvis_m=math.dist(streamed["pelvis_xy"], held["pelvis_xy"]),
    )


def sustained(run: Path, index: int) -> dict | None:
    """How much the body moved per second, across the whole round."""
    path = run / "traces" / f"round_{index:03d}.json"
    if not path.exists():
        return None
    trace = json.load(path.open()).get("trace") or []
    yaw_rate, speed = [], []
    for a, b in zip(trace, trace[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        yaw_rate.append(abs((b["pelvis_yaw"] - a["pelvis_yaw"] + math.pi) % (2 * math.pi) - math.pi) / dt)
        speed.append(math.dist(b["pelvis_xy"], a["pelvis_xy"]) / dt)
    if not yaw_rate:
        return None
    return dict(
        samples=len(yaw_rate),
        yaw_med=statistics.median(yaw_rate),
        yaw_p90=sorted(yaw_rate)[int(0.9 * (len(yaw_rate) - 1))],
        speed_med=statistics.median(speed),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--batch-rounds", type=int, default=10, help="the evaluation's stack-restart cadence")
    args = ap.parse_args(argv)

    rows = rounds_of(args.run)
    print(f"{len(rows)} rounds, batch size {args.batch_rounds}\n")
    print("局  批  ok     Δyaw    Δpitch   Δ骨盆 |  yaw中位  yaw_p90  速度中位  | why")
    labels = []
    for i, row in enumerate(rows):
        step, run_metrics = transient(row), sustained(args.run, i)
        s = (
            "    --      --      --"
            if not step
            else (
                f"{step['yaw_deg']:+7.2f} "
                f"{'  --  ' if step['pitch_deg'] is None else format(step['pitch_deg'], '+7.2f')} "
                f"{step['pelvis_m']:7.4f}"
            )
        )
        m = (
            "     --       --        --"
            if not run_metrics
            else (f"{run_metrics['yaw_med']:9.3f}{run_metrics['yaw_p90']:9.3f}{run_metrics['speed_med']:10.3f}")
        )
        print(f"{i:3d}{i // args.batch_rounds:3d}  {str(row.get('ok')):5s} {s} |{m}  | {str(row.get('why'))[:28]}")
        labels.append((i // args.batch_rounds, run_metrics["yaw_p90"] if run_metrics else None))

    known = [(b, v) for b, v in labels if v is not None]
    if len(known) > args.batch_rounds:
        cut = statistics.median(v for _, v in known)
        per_batch = {}
        for b, v in known:
            per_batch.setdefault(b, []).append(v > cut)
        print("\n批内一致性（每批被判为 jumpy 的比例；接近 0 或 1 = 状态锁在 deploy 里，")
        print("散布在 0.5 附近 = 每局独立的切换竞态）:")
        for b, flags in sorted(per_batch.items()):
            print(f"  批 {b:02d}: {sum(flags)}/{len(flags)}")
    else:
        print("\n只有一个批次的 trace 可读；要做批内一致性检验，需要每批的 traces 不互相覆盖。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
