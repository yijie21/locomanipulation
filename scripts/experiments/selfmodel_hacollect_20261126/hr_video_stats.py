"""selfmodel_hadata_20261126 (diagnosis for viewing only): hr_video.py's per-step record summarised per walk and per
reach (no statistic of the entry; these episodes enter no criterion).

    python hr_video_stats.py <heft_video dir>

Per walk: asked distance, the reference's step count, how far the reference root moved and the actual root moved (true,
start heading frame), the reference's swings per foot (ankle > 1 cm over its lowest) against the actual lift-offs
(load < 15 N) and the highest actual ankle lift; per reach (phase window): the pelvis's and the reference root's
displacement, each foot's xy displacement (a planted foot that moves slips), the feet's load at start / end / min,
peak planar speed. The controller follows the reference about delay_f = 8 steps late; windows are on the actual."""

import glob
import json
import sys
from pathlib import Path

import numpy as np


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def main() -> int:
    V = Path(sys.argv[1])
    for fn in sorted(glob.glob(str(V / "steps" / "*.npz"))):
        z = dict(np.load(fn))
        ev = json.loads(Path(fn.replace(".npz", "_events.json")).read_text())
        name = Path(fn).stem
        f = z["f"].astype(int)
        print(f"== {name}: {len(f)} steps ({f[0]}..{f[-1]}), end: {[e['what'] for e in ev if e['what'].startswith('end')]}")
        i_of = {int(v): k for k, v in enumerate(f)}
        R = lambda yaw: np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])  # noqa: E731 world -> heading
        has_xy = "ankle_x_left" in z
        # walks
        wk = runs(z["walking"] > 0.5)
        wres = [e for e in ev if e["what"] == "walk result"]
        lab = [e for e in ev if e["what"].startswith("walk ") and e["what"] != "walk result" or e["what"].startswith("search")]
        print(" walks:")
        for n, (a, b) in enumerate(wk):
            Rh = R(z["yaw"][a])
            act = Rh @ np.array([z["x"][b - 1] - z["x"][a], z["y"][b - 1] - z["y"][a]])
            ref = Rh @ np.array([z["ref_x"][b - 1] - z["ref_x"][a], z["ref_y"][b - 1] - z["ref_y"][a]])
            sw, lo, lift = {}, {}, {}
            for s in ("left", "right"):
                rz = z[f"ref_ankle_z_{s}"][a:b]
                sw[s] = len(runs(rz > rz.min() + 0.01))
                lo[s] = len(runs(z[f"load_{s}"][a:b] < 15.0))
                az = z[f"ankle_z_{s}"][a:b]
                lift[s] = 100 * (az.max() - az.min())
            w = wres[n] if n < len(wres) else {}
            lb = lab[n]["what"] if n < len(lab) else ""
            print(f"  {n} f{f[a]}-{f[b - 1]} {lb}\n     asked {w.get('asked_m')} ({w.get('asked_dist')} m), ref steps {w.get('ref_steps')}; "
                  f"ref root moved {np.round(ref, 3).tolist()}, actual {np.round(act, 3).tolist()} (|act|/|ref| "
                  f"{np.linalg.norm(act) / max(np.linalg.norm(ref), 1e-6):.2f}); ref swings L {sw['left']} R {sw['right']}; "
                  f"actual lift-offs L {lo['left']} R {lo['right']}; max ankle rise L {lift['left']:.1f} R {lift['right']:.1f} cm; "
                  f"settle wait {w.get('wait_s')} s")  # fmt: skip
        # phases (reaches)
        ph = [e for e in ev if e["what"].startswith("reach") or e["what"].startswith("walk approach") or e["what"].startswith("search") or e["what"].startswith("end")]
        print(" reaches (window: the phase label to the next label):")
        for k, e in enumerate(ph):
            if not e["what"].startswith("reach"):
                continue
            f0 = e["f"]
            f1 = ph[k + 1]["f"] if k + 1 < len(ph) else int(f[-1])
            a, b = i_of.get(f0, 0), i_of.get(f1, len(f) - 1)
            if b <= a:
                continue
            Rh = R(z["yaw"][a])
            act = Rh @ np.array([z["x"][b] - z["x"][a], z["y"][b] - z["y"][a]])
            ref = Rh @ np.array([z["ref_x"][b] - z["ref_x"][a], z["ref_y"][b] - z["ref_y"][a]])
            ln = f"  {e['what']} f{f0}-{f1} ({(f1 - f0) / 50:.1f} s): pelvis moved {np.round(100 * act, 1).tolist()} cm (heading frame), ref root {np.round(100 * ref, 1).tolist()} cm; peak {100 * z['v_planar'][a:b].max():.0f} cm/s"
            if has_xy:
                for s in ("left", "right"):
                    d = Rh @ np.array([z[f"ankle_x_{s}"][b] - z[f"ankle_x_{s}"][a], z[f"ankle_y_{s}"][b] - z[f"ankle_y_{s}"][a]])
                    dr = Rh @ np.array([z[f"ref_ankle_x_{s}"][b] - z[f"ref_ankle_x_{s}"][a], z[f"ref_ankle_y_{s}"][b] - z[f"ref_ankle_y_{s}"][a]])
                    ln += f"\n     {s} foot moved {np.round(100 * d, 1).tolist()} cm (ref {np.round(100 * dr, 1).tolist()}), load {z[f'load_{s}'][a]:.0f} -> {z[f'load_{s}'][b]:.0f} N (min {z[f'load_{s}'][a:b].min():.0f}), lift-offs {len(runs(z[f'load_{s}'][a:b] < 15.0))}"
            print(ln)
        sl = [e for e in ev if e["what"].startswith("base sliding")]
        print(f" base-slide events: {len(sl)}")
        for e in sl[:12]:
            print(f"  t {e['t']} f{e['f']}: {e['what']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
