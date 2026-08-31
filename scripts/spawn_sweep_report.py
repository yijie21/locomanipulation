#!/usr/bin/env python3
"""Read a spawn_sweep run and report success against SPAWN distance.

    python scripts/spawn_sweep_report.py outputs/eval/spawn_sweep

Spawn distance is recomputed per round from the manifest's own
`init.robot_xy` -- present on EVERY row including the ones the first-frame
view gate rejected, which is why the gate rejections can be reported
separately instead of silently depressing the far end of the curve.

Two denominators, and the difference matters:

  gross   every round, gate rejections counted as failures. This is what a
          deployment at that spawn distance would actually get.
  net     gate rejections removed. This is the TASK difficulty at that
          distance, with the camera-geometry limit factored out.

Quote the gross number for "where should the robot start", the net one for
"is the policy better or worse further out".
"""
import json
import sys
from pathlib import Path

import numpy as np

# table half-extent from outputs/scene/sim/scene_meta.json (flat_half_extent)
HALF_X, HALF_Y = 0.512854514191864, 0.5133445750697163


def edge_dist(xy):
    return max(abs(xy[0]) - HALF_X, abs(xy[1]) - HALF_Y)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval/spawn_sweep")
    rows = []
    for m in sorted(out.glob("*.manifest.jsonl")):
        for line in m.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            init = r.get("init") or {}
            if "robot_xy" not in init:
                continue
            rows.append(dict(
                edge=edge_dist(init["robot_xy"]),
                obj=float(np.linalg.norm(np.asarray(init["robot_xy"])
                                         - np.asarray(init["obj_xy"]))),
                ok=bool(r.get("ok")),
                gated="[view]" in (r.get("why") or ""),
                why=r.get("why") or "",
                half=m.stem))
    if not rows:
        sys.exit(f"no manifest rows under {out}")

    k = sum(r["ok"] for r in rows)
    print(f"{out}  n={len(rows)}  overall {k}/{len(rows)} = "
          f"{k/len(rows)*100:.1f}%   gate-rejected {sum(r['gated'] for r in rows)}")
    for half in sorted({r["half"] for r in rows}):
        h = [r for r in rows if r["half"] == half]
        print(f"   {half}: {sum(x['ok'] for x in h)}/{len(h)}")

    edges = np.arange(0.7, 1.45, 0.1)
    print(f"\n{'spawn-edge':>14} {'n':>4} {'gated':>6} "
          f"{'gross':>16} {'net (gate removed)':>22}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["edge"] < hi]
        if not sel:
            continue
        g = sum(r["gated"] for r in sel)
        ok = sum(r["ok"] for r in sel)
        n_net = len(sel) - g
        a, b = wilson(ok, len(sel))
        line = (f"  [{lo:.1f},{hi:.1f})  {len(sel):4d} {g:6d} "
                f"  {ok:3d}/{len(sel):-3d} {ok/len(sel)*100:5.1f}% "
                f"[{a*100:4.0f},{b*100:4.0f}]")
        if n_net:
            c, d = wilson(ok, n_net)
            line += (f"   {ok:3d}/{n_net:-3d} {ok/n_net*100:5.1f}% "
                     f"[{c*100:4.0f},{d*100:4.0f}]")
        print(line)

    # failure mix by distance -- a curve that falls for two different reasons
    # at the two ends should say so
    print("\n失败构成(net,已剔除 gate):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["edge"] < hi and not r["gated"]]
        if not sel:
            continue
        c = {}
        for r in sel:
            if r["ok"]:
                continue
            w = r["why"]
            stage = (w.split(":")[0].replace("[vla] ", "") if "[vla]" in w
                     else "pre-vla")
            c[stage] = c.get(stage, 0) + 1
        print(f"  [{lo:.1f},{hi:.1f})  n={len(sel):3d}  " +
              ", ".join(f"{s} {v}" for s, v in sorted(c.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
