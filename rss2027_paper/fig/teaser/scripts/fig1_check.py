# notebook: skip render for paper Figure 1, not an experiment
"""fig1_check.py <task> <ep> <tag>: the replay against lmlroute's formal records, read-only.

(1) the episode records (A = alone, R = reachable): every JSON field, recursively, apart from wall-clock / machine fields
    (cmp_lmlroute's exclusions: cached, host, think_s, think_wall_s, wall_s, wall_wait_s); every array of segs/ and
    segs/arm/;
(2) the close (snapshot of the replay): the frame (waits['close'].fire), the grasp point plan.goal, close_dist_cm =
    |hand - goal| and close_obj_cm = |hand - object| (float64; the record keeps no hand / object vector at the close,
    whose frame need not be a 10 Hz row) against the formal record's; success / reason. For information only: how far
    the hand moved between the last 10 Hz W row before the close and the close.
"""

import json
import sys

import numpy as np

task, ep, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
O = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126/scalebfm"
F = "/wuji-vefps-D/wuji-il/yijie/loco_outputs/fig1_teaser_20261126"
N = f"{F}/out/scalebfm"
SKIP = {"cached", "host", "think_s", "think_wall_s", "wall_s", "wall_wait_s"}


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'formal' if k in a else 'replay'}")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (u, v) in enumerate(zip(a, b)):
                out += diff(u, v, f"{path}[{i}]")
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


res = {}
ok_all = True
for g, m in (("A", "alone"), ("R", "reachable")):
    rf = json.load(open(f"{O}/episodes/{task}/{m}/ep{ep:+05d}.json"))
    rr = json.load(open(f"{N}/episodes/{task}/{m}/ep{ep:+05d}.json"))
    d = diff(rf, rr)
    na, nd = 0, []
    for sub in ("", "arm/"):
        zf = np.load(f"{O}/segs/{task}/{m}/{sub}ep{ep:+05d}.npz")
        zr = np.load(f"{N}/segs/{task}/{m}/{sub}ep{ep:+05d}.npz")
        if set(zf.files) != set(zr.files):
            nd.append(f"{sub}keys differ")
        for k in zf.files:
            na += 1
            if k in zr.files and not (zf[k].shape == zr[k].shape and np.array_equal(zf[k], zr[k], equal_nan=zf[k].dtype.kind in 'fc')):
                nd.append(f"{sub}{k}")
    sn = json.load(open(f"{F}/state/{tag}/{m}.json"))
    fc = next(w["fire"] for w in rf["waits"] if w["key"] == "close")
    z = np.load(f"{O}/segs/{task}/{m}/ep{ep:+05d}.npz")
    off = np.r_[0, np.cumsum(z["n_steps"])]
    row, row_f = None, None  # the last 10 Hz row at or before the close (the close need not fall on the grid)
    for s_ in range(len(z["n_steps"])):
        frames = rf["d3"][s_]["f_cmd"] + 5 * (np.arange(int(z["n_steps"][s_])) - int(z["k0"][s_]))
        for i, fr_ in enumerate(frames):
            if fr_ <= fc and (row_f is None or fr_ >= row_f):
                row, row_f = int(off[s_] + i), int(fr_)
    hp = np.asarray(sn["hp"])
    w_hand = z["W"][row, 4:7].astype(np.float64)
    dh = float(np.linalg.norm(hp - w_hand))
    e = dict(group=g, method=m, formal_success=rf["success"], replay_success=rr["success"], formal_reason=rf["reason"],
             replay_reason=rr["reason"], formal_host=rf.get("host"), replay_host=rr.get("host"),
             json_differences=d, seg_arrays=na, seg_arrays_differing=nd,
             close_frame_formal=fc, close_frame_replay=sn["frame"],
             close_dist_cm_formal=rf["close_dist_cm"], close_dist_cm_replay=sn["close_dist_cm"],
             close_dist_abs_diff_m=abs(rf["close_dist_cm"] - sn["close_dist_cm"]) / 100,
             close_obj_cm_formal=rf["close_obj_cm"], close_obj_cm_replay=sn["close_obj_cm"],
             close_obj_abs_diff_m=abs(rf["close_obj_cm"] - sn["close_obj_cm"]) / 100,
             hand_replay=hp.tolist(), info_last_W_row_frame=row_f, info_hand_W_row_float32=w_hand.tolist(),
             info_hand_close_minus_last_row_m=dh, goal_formal=rf["plan"]["goal"], goal_replay=sn["goal"])
    ok = (not d and not nd and rf["success"] == rr["success"] and sn["frame"] == fc
          and e["close_dist_abs_diff_m"] <= 1e-6 and e["close_obj_abs_diff_m"] <= 1e-6
          and sn["goal"] == rf["plan"]["goal"])
    e["pass"] = bool(ok)
    ok_all &= ok
    res[g] = e
    print(f"{g} ({m}): formal success {rf['success']} replay {rr['success']} | reason {rf['reason']} / {rr['reason']} | "
          f"json differences {len(d)} {d[:5]} | seg arrays {na}, differing {len(nd)} {nd[:5]} | close frame {fc} / {sn['frame']} | "
          f"close_dist_cm {rf['close_dist_cm']!r} / {sn['close_dist_cm']!r} (|d| {e['close_dist_abs_diff_m']:.3g} m) | "
          f"close_obj_cm {rf['close_obj_cm']!r} / {sn['close_obj_cm']!r} (|d| {e['close_obj_abs_diff_m']:.3g} m) | "
          f"(info: hand moved {100 * dh:.3f} cm since the last 10 Hz row, frame {row_f}) | host {rf.get('host')} / {rr.get('host')} | PASS {ok}")
res["pass"] = bool(ok_all and not res["A"]["replay_success"] and res["R"]["replay_success"])
print(f"check {task} ep {ep}: A fail & R success in the replay {not res['A']['replay_success'] and res['R']['replay_success']}; "
      f"all checks {'PASS' if res['pass'] else 'FAIL'}")
json.dump(res, open(f"{F}/state/{tag}/check.json", "w"), indent=1)
