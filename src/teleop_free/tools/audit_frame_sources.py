"""Summarize source freshness and verify audit-to-dataset row correspondence."""

import argparse
import json
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=float)
    return (
        dict(
            n=len(values),
            median=float(np.median(values)),
            p95=float(np.quantile(values, 0.95)),
            maximum=float(np.max(values)),
        )
        if len(values)
        else None
    )


def audit(path, session=None):
    rows = list(map(json.loads, Path(path).read_text().splitlines()))
    episodes = {}
    for r in rows:
        episodes.setdefault(r["episode_index"], []).append(r)
    result = {
        "scope": "Source publication/receipt times, not physical exposure/action timestamps",
        "rows": len(rows),
        "episodes": [],
        "correspondence_errors": [],
    }
    for ep, rs in episodes.items():
        times = np.array([r["row_build_wall"] for r in rs])
        record = dict(
            episode_index=ep,
            rows=len(rs),
            row_interval_s=distribution(np.diff(times)),
            state_receipt_age_s=distribution([r["row_build_wall"] - r["state_receipt_or_ros_wall"] for r in rs]),
            image_publish_age_s={},
        )
        for camera in sorted({k for r in rs for k in r["image_publish_wall"]}):
            record["image_publish_age_s"][camera] = distribution(
                [r["row_build_wall"] - r["image_publish_wall"][camera] for r in rs if camera in r["image_publish_wall"]]
            )
        ids = [r["state_index"] for r in rs]
        record["repeated_state_indices"] = sum(a is not None and a == b for a, b in zip(ids, ids[1:]))
        if [r["row_index"] for r in rs] != list(range(len(rs))):
            result["correspondence_errors"].append(f"episode {ep}: noncontiguous audit rows")
        if session is not None:
            source = Path(session) / f"priv_seq/episode_{ep:06d}.json"
            if not source.exists():
                result["correspondence_errors"].append(f"episode {ep}: missing raw sidecar")
            else:
                raw = json.loads(source.read_text())
                if raw["priv_seq"] != [r["priv_seq"] for r in rs]:
                    result["correspondence_errors"].append(f"episode {ep}: raw source sequences differ")
        result["episodes"].append(record)
    if not rows:
        result["correspondence_errors"].append("no audit rows")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    ap.add_argument("--session", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.receipt, args.session)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
    if result["correspondence_errors"]:
        raise SystemExit(1)
