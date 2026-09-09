"""Separate pre-close object topples from later no-close timeouts."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def upright(obj):
    w, x, y, z = obj["quat"]
    return 1 - 2 * (x * x + y * y)


def audit(root):
    read = lambda name: [json.loads(l) for l in (root / name).read_text().splitlines()]
    events = read("events.jsonl")
    priv = read("privileged.jsonl")
    sent = read("sent.jsonl")
    infer = read("inference.jsonl")
    starts = [e["wall"] for e in events if e["event"] == "round_start"]
    result = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else float("inf")
        boundaries = [
            e["wall"] for e in events if start < e["wall"] < end and e["event"] in ["round_end", "vla_suffix_exit"]
        ]
        if boundaries:
            end = min(boundaries)
        times = [r["started"] for r in infer if start <= r["started"] < end]
        if not times:
            continue
        t0 = min(times)
        close = next(
            (
                r["wall"]
                for r in sent
                if t0 <= r["wall"] < end and r.get("right_hand") is not None and r["right_hand"][0] < 0.005
            ),
            None,
        )
        states = [r for r in priv if t0 <= r["wall"] < end]
        initial = states[0]["state"]["objects"]["water_bottle"]
        tilted_since = None
        topple = None
        for r in states:
            if upright(r["state"]["objects"]["water_bottle"]) < 0.5:
                if tilted_since is None:
                    tilted_since = r["wall"]
                if r["wall"] - tilted_since >= 0.15:
                    topple = tilted_since
                    break
            else:
                tilted_since = None
        touches = Counter()
        furniture = Counter()
        if topple is not None:
            for r in states:
                if topple - 0.75 <= r["wall"] <= topple + 0.2:
                    o = r["state"]["objects"]["water_bottle"]
                    touches.update(o.get("touch_links", []))
                    furniture.update(r["state"].get("furniture_touch", []))
        at = next((r["state"] for r in states if topple is not None and r["wall"] >= topple), states[-1]["state"])
        pos = at["objects"]["water_bottle"]["pos"]
        row = dict(
            round=i,
            first_observation=t0,
            initial_up=upright(initial),
            first_close_s=None if close is None else close - t0,
            topple_s=None if topple is None else topple - t0,
            toppled_before_close=topple is not None and (close is None or topple < close),
            object_shift_xy=math.dist(pos[:2], initial["pos"][:2]),
            contact_links=dict(touches),
            furniture_contacts=dict(furniture),
            pelvis_z_at_event=at["pelvis"]["pos"][2],
        )
        result.append(row)
    (root / "topple_audit.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path, nargs="+")
    a = ap.parse_args()
    for root in a.capture:
        print(root, json.dumps(audit(root), ensure_ascii=False))
