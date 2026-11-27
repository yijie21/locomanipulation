# notebook: selfmodel_hadata_20261126
"""Walking in hr_real episodes, per directory (revision 2a addendum: gait_heft against hg_walk on the same dev seeds).
Per episode: end (fell / lost: info.fell is the step), rounds, the stand error (|stand.err|, pelvis heading frame); per walk: asked distance and the
share walked along it ((real . asked) / |asked|^2, odometry, the start pose's frame). Walks split at 0.3 m (gait_heft's
move_min; the smoke's short / long split).
    python hr_gait_cmp.py <run dir> [<run dir> ...] [--seeds 296-298]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--seeds", default=None)
    a = p.parse_args()
    keep = None
    if a.seeds:
        lo, hi = (int(v) for v in a.seeds.split("-"))
        keep = set(range(lo, hi + 1))
    for d in a.dirs:
        eps = []
        for f in sorted(glob.glob(os.path.join(d, "log", "*.json"))):
            e = json.load(open(f))
            if keep is not None and e["seed"] not in keep:
                continue
            eps.append(e)
        print(f"== {d}: {len(eps)} episodes")
        short, long_, errs = [], [], []
        for e in eps:
            st, info = e.get("stand") or {}, e["info"]
            err = float(np.linalg.norm(st["err"])) if st.get("err") is not None else float("nan")
            errs.append(err)
            ws = []
            for w in info.get("walks", []):
                ask, real = np.asarray(w["asked"][:2], float), np.asarray(w["real"][:2], float)
                n = float(np.linalg.norm(ask))
                r = float(real @ ask / max(n * n, 1e-9))
                (short if n < 0.3 else long_).append(r)
                ws.append(f"{n:.2f}->{r:.2f}" + ("" if w.get("plan", {}).get("gait") is None else "g"))
            print(f"  {e['task']}_s{e['seed']}: end={e['end']!r} at step {info.get('fell')} rounds={st.get('rounds')} "
                  f"stand_err={err:.3f} m segs={e.get('segments')} walks(asked m->share)=[{', '.join(ws)}]")
        f3 = lambda v: "nan" if not v else f"{np.median(v):.2f} (n {len(v)}, min {min(v):.2f}, max {max(v):.2f})"
        print(f"  stand_err median {f3(errs)}; walks < 0.3 m share walked median {f3(short)}; >= 0.3 m {f3(long_)}; "
              f"ends {dict(sorted(((k, [e['end'] for e in eps].count(k)) for k in {e['end'] for e in eps})))}")


if __name__ == "__main__":
    main()
