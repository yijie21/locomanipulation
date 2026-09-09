"""Compare the MOTION QUALITY of two LeRobot datasets, ours vs a reference.

    python -m teleop_free.tools.motion_smoothness DS_A DS_B [--n 60] [--json out]

WHY THIS EXISTS

"the dataset does not feel smooth enough" is not actionable until it is a
number, and the number has to be comparable against something. We have the
comparison on disk: `datasets/g1_sonic_orange_to_plate_locomani` is a human
teleop session recorded through the SAME exporter, same 43-dim `action.wbc`,
same 64-dim FSQ token, same 30 fps. So every statistic below is computed
identically on both and reported side by side.

WHAT IS MEASURED, AND WHY EACH ONE

  jerk RMS (rad/s^3), per joint group, on TWO channels
      `st_*` is `observation.state`, the MEASURED joints -- how the robot
      actually moved. `cmd_*` is `action.wbc`, the commanded joints.

      ONLY `st_*` IS CROSS-DATASET COMPARABLE. In the reference teleop session
      `action.wbc` is byte-identical to `observation.state` (checked: max abs
      difference exactly 0 over every episode), i.e. that session has no
      independent command channel, while ours does (max abs difference 4.4 rad).
      Comparing `action.wbc` across the two therefore compares their MEASURED
      joints against our COMMANDED ones and inflates every ratio by ~20x. The
      tool prints a WARNING when it detects the degenerate channel; treat
      `cmd_*` for such a dataset as a duplicate of `st_*`, not as a command.

  HF power fraction (>3 Hz share of the joint-velocity PSD)
      human arm motion is band-limited to ~2-3 Hz. A state machine that steps
      its setpoint puts power above that. Amplitude-invariant, so it does not
      punish us for simply moving further.

  SPARC (spectral arc length of the wrist speed profile)
      the standard movement-smoothness metric (Balasubramanian 2015):
      dimensionless, amplitude-invariant, more negative = less smooth. The
      wrist path comes from FK on `action.wbc`, because `observation.eef_state`
      is a constant placeholder in the reference session (checked: std 0).

  LDLJ (log dimensionless jerk of the same speed profile)
      the second standard metric; SPARC and LDLJ disagreeing is itself
      information (SPARC is sensitive to submovement count, LDLJ to amplitude).

  token index statistics
      the label the VLA actually fits. The FSQ grid is 1/16, so smoothing
      finer than one index is INVISIBLE downstream -- `tok_didx_mean` says how
      many grid steps the label moves per frame, i.e. how much headroom a
      smoothing change even has.

  10 Hz PSD ratio
      our converter resamples 50->30 by nearest neighbour
      (`convert_to_hf_format.py:632`), so consecutive rows are 1 or 2 source
      frames apart in a period-3 pattern = a 10 Hz timing artifact at 30 fps.
      This probe says whether that artifact is actually visible in the labels.
      A reference session published at 30 fps has no reason to show it.

  velocity lag-1 autocorrelation
      how much consecutive velocity samples agree. Smooth motion is temporally
      correlated; a setpoint that steps is closer to white. Scale-free, so it
      survives the two datasets moving different distances.

  still_frac / still_max_s / mode_switches / speed_levels
      the cadence and discrete-event axes -- the ones round 3 showed actually
      move closed-loop success (4.6 s -> 3.0 s pause, 13.7% -> 28.0%). Kept
      here so a smoothness change can be checked for not regressing them.

Everything is reported as median [IQR] over episodes; per-episode rows go to
--json for a paired test if one is ever wanted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

# the 29 actuated body joints inside the 43-wide vector, in MuJoCo order
BODY_IDX = np.array(list(range(0, 22)) + list(range(29, 36)))

# `observation.state` / `action.wbc` 43-wide layout. NOT the order
# docs/dataset.md lists (legs, waist, both arms, both hands): the two hand
# blocks are interleaved BETWEEN the arms, grouped per side. Verified three
# ways -- sonic_decoder.BODY_IDX (= 0:22 + 29:36) picks the 29 actuated body
# joints out of it; the two 7-wide blocks at 22:29 and 36:43 carry gripper-
# scale ranges (0.002 and 0.034) while 15:22 and 29:36 carry radian-scale arm
# motion; and FK on state[BODY_IDX] puts the RIGHT palm 0.14 m from the held
# bottle where the old slicing put it 0.24 m away.
GROUPS = {
    "legs": (0, 12),
    "waist": (12, 15),
    "l_arm": (15, 22),
    "l_hand": (22, 29),
    "r_arm": (29, 36),
    "r_hand": (36, 43),
}
TASK_ARM = (29, 36)  # the arm that does the grasping

STILL_RAD_S = 0.05  # arm joint speed below this = "not reaching"
HF_CUT_HZ = 3.0


def eps_paths(root: str, limit: int | None):
    d = os.path.join(root, "data", "chunk-000")
    fs = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    if limit:
        fs = fs[:limit]
    return [os.path.join(d, f) for f in fs]


def sparc(speed: np.ndarray, fs: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length. Reference implementation, Balasubramanian 2015."""
    if len(speed) < 8 or speed.max() <= 0:
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mf = np.abs(np.fft.fft(speed, nfft))
    mf = mf / mf.max()
    keep = np.flatnonzero(f <= fc)
    f_sel, m_sel = f[keep], mf[keep]
    inx = np.flatnonzero(m_sel >= amp_th)
    if len(inx) < 2:
        return float("nan")
    f_sel, m_sel = f_sel[inx[0] : inx[-1] + 1], m_sel[inx[0] : inx[-1] + 1]
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    return float(-np.sum(np.sqrt(df**2 + np.diff(m_sel) ** 2)))


def ldlj(speed: np.ndarray, fs: float) -> float:
    """Log dimensionless jerk of a speed profile. More negative = less smooth."""
    if len(speed) < 8 or speed.max() <= 0:
        return float("nan")
    dt = 1.0 / fs
    jerk = np.diff(speed, 2) / dt**2
    T = len(speed) * dt
    dlj = (T**3 / speed.max() ** 2) * np.sum(jerk**2) * dt
    return float(-np.log(dlj)) if dlj > 0 else float("nan")


def hf_fraction(vel: np.ndarray, fs: float, cut: float = HF_CUT_HZ) -> float:
    """Share of joint-velocity power above `cut`, averaged over joints."""
    n = len(vel)
    if n < 16:
        return float("nan")
    win = np.hanning(n)[:, None]
    P = np.abs(np.fft.rfft(vel * win, axis=0)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    P[0] = 0.0  # drop DC
    tot = P.sum(0)
    hi = P[f > cut].sum(0)
    ok = tot > 0
    return float(np.mean(hi[ok] / tot[ok])) if ok.any() else float("nan")


def band_ratio(vel: np.ndarray, fs: float, lo: float, hi: float, ref: tuple[float, float]) -> float:
    """Power in [lo,hi] over the mean density of a wider reference band."""
    n = len(vel)
    if n < 16:
        return float("nan")
    win = np.hanning(n)[:, None]
    P = (np.abs(np.fft.rfft(vel * win, axis=0)) ** 2).mean(1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    m, mr = (f >= lo) & (f <= hi), (f >= ref[0]) & (f <= ref[1])
    if not m.any() or not mr.any():
        return float("nan")
    return float(P[m].mean() / P[mr].mean())


def wrist_speed_fk(q29: np.ndarray, side: str = "right") -> np.ndarray | None:
    """|d/dt| of the wrist position from FK on the commanded joints.

    Uses the project's own ArmIK model so the frame and joint order match the
    rest of the pipeline. Returns None if mujoco/the scene are unavailable.
    """
    try:
        # arm_ik imports its siblings flat (`from g1_joints import ...`), the
        # way the stack runs them, so the package dir has to be on the path.
        d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if d not in sys.path:
            sys.path.insert(0, d)
        from arm_ik import ArmIK
    except Exception as e:
        print(f"  [fk] unavailable: {e}", file=sys.stderr)
        return None
    global _IK
    try:
        _IK  # noqa: B018  (probe: has the module-level cache been created yet?)
    except NameError:
        _IK = ArmIK()
    pos = np.stack([_IK.fk(q, side)[0] for q in q29])
    return pos


def _group_metrics(q: np.ndarray, fps: float, prefix: str) -> dict:
    dt = 1.0 / fps
    out = {}
    for g, (a, b) in GROUPS.items():
        x = q[:, a:b]
        v = np.diff(x, axis=0) / dt
        j = np.diff(x, n=3, axis=0) / dt**3
        out[f"{prefix}{g}_jerk_rms"] = float(np.sqrt((j**2).mean()))
        out[f"{prefix}{g}_vel_rms"] = float(np.sqrt((v**2).mean()))
        out[f"{prefix}{g}_vel_p99"] = float(np.percentile(np.abs(v), 99))
        # median of the per-frame fastest joint: says what the motion does
        # TYPICALLY, where the RMS above is dominated by the rare excursions
        out[f"{prefix}{g}_vel_med"] = float(np.median(np.abs(v).max(1)))
        if g in ("r_arm", "l_arm", "legs"):
            out[f"{prefix}{g}_hf_frac"] = hf_fraction(v, fps)
            out[f"{prefix}{g}_10hz_ratio"] = band_ratio(v, fps, 9.0, 11.0, (4.0, 14.0))
            if len(v) > 4:
                d = v - v.mean(0)
                num = (d[:-1] * d[1:]).sum(0)
                den = (d**2).sum(0)
                ok = den > 0
                out[f"{prefix}{g}_vel_ac1"] = float(np.mean(num[ok] / den[ok])) if ok.any() else float("nan")
    return out


def episode_stats(path: str, fps: float, want_fk: bool) -> dict:
    t = pq.read_table(path)
    col = lambda k: np.stack(t[k].to_numpy(zero_copy_only=False))
    state = col("observation.state").astype(float)
    wbc = col("action.wbc").astype(float)
    tok = col("action.motion_token").astype(float)
    n = len(state)
    dt = 1.0 / fps
    out = {
        "path": os.path.basename(path),
        "frames": n,
        "dur_s": n * dt,
        "cmd_equals_state": float(np.abs(wbc - state).max()),
    }

    out.update(_group_metrics(state, fps, "st_"))
    out.update(_group_metrics(wbc, fps, "cmd_"))

    # cadence, on the MEASURED arm joints
    varm = np.abs(np.diff(state[:, slice(*TASK_ARM)], axis=0) / dt).max(1)
    still = varm < STILL_RAD_S
    out["still_frac"] = float(still.mean())
    run = best = 0
    for sflag in still:
        run = run + 1 if sflag else 0
        best = max(best, run)
    out["still_max_s"] = best * dt

    # token: work in FSQ INDEX units, where 1 = one grid step = the finest
    # change the label can express
    idx = np.round(tok * 16.0)
    d = np.abs(np.diff(idx, axis=0))
    out["tok_didx_mean"] = float(d.mean())
    out["tok_change_frac"] = float((d > 0).mean())
    out["tok_jump_p99"] = float(np.percentile(d.max(1), 99))
    out["tok_levels"] = int(len(np.unique(idx)))
    out["tok_10hz_ratio"] = band_ratio(np.diff(tok, axis=0), fps, 9.0, 11.0, (4.0, 14.0))

    # discrete events from the planner channel
    for k, name in (("teleop.planner_mode", "mode_switches"), ("teleop.stream_mode", "stream_switches")):
        if k in t.column_names:
            a = np.asarray(t[k].to_numpy(zero_copy_only=False))
            out[name] = int((np.diff(a) != 0).sum())
    if "teleop.planner_speed" in t.column_names:
        sp = np.asarray(t["teleop.planner_speed"].to_numpy(zero_copy_only=False))
        out["speed_levels"] = int(len(np.unique(np.round(sp, 3))))
        out["speed_steps"] = int((np.abs(np.diff(sp)) > 1e-4).sum())

    if want_fk:
        pos = wrist_speed_fk(state[:, BODY_IDX], "right")
        if pos is not None:
            sp = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
            out["wrist_sparc"] = sparc(sp, fps)
            out["wrist_ldlj"] = ldlj(sp, fps)
            out["wrist_sparc_tail5s"] = sparc(sp[-int(5 * fps) :], fps)
            out["wrist_path_m"] = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    return out


def excursions(root: str, limit: int, fps: float) -> dict:
    """Where do the rare violent arm excursions come from?

    The group metrics show ours differs from the reference far more in the RMS
    and the jerk than in the median -- i.e. the gap is a TAIL, not general
    roughness. This attributes that tail: for each episode take the top 1% of
    frames by arm speed and ask what was happening at that moment.

    Measured on the TASK arm (`observation.state[29:36]`, the right arm -- see
    GROUPS for how that slicing was established).
    """
    from collections import Counter

    near = tot = 0
    by_mode, mode_frames, lead = Counter(), Counter(), Counter()
    steps = []
    for path in eps_paths(root, limit):
        t = pq.read_table(path)
        st = np.stack(t["observation.state"].to_numpy(zero_copy_only=False))
        cmd = np.stack(t["action.wbc"].to_numpy(zero_copy_only=False))
        pm = np.asarray(t["teleop.planner_mode"].to_numpy(zero_copy_only=False))
        sm = np.asarray(t["teleop.stream_mode"].to_numpy(zero_copy_only=False))
        v = np.abs(np.diff(st[:, slice(*TASK_ARM)], axis=0)).max(1) * fps
        hot = np.flatnonzero(v >= np.percentile(v, 99))
        sw = np.flatnonzero((np.diff(pm) != 0) | (np.diff(sm) != 0))
        for h in hot:
            tot += 1
            if len(sw) and np.abs(sw - h).min() <= int(0.5 * fps):
                near += 1
            by_mode[int(pm[h])] += 1
            lead[int(np.argmax(np.abs(st[h + 1, slice(*TASK_ARM)] - st[h, slice(*TASK_ARM)])))] += 1
        for m, c in Counter(pm.tolist()).items():
            mode_frames[int(m)] += c
        for k in sw:  # size of the commanded step at a switch
            a, b = max(0, k - 2), min(len(cmd) - 1, k + 8)
            steps.append(
                (
                    int(pm[a]),
                    int(pm[b]),
                    int(sm[a]),
                    int(sm[b]),
                    float(np.rad2deg(np.abs(cmd[b, slice(*TASK_ARM)] - cmd[a, slice(*TASK_ARM)]).max())),
                )
            )
    allf = sum(mode_frames.values()) or 1
    out = {
        "n_excursions": tot,
        "within_0.5s_of_a_switch": near / max(tot, 1),
        "by_planner_mode": {m: (by_mode[m] / max(tot, 1), mode_frames[m] / allf) for m in sorted(by_mode)},
        "leading_joint_in_block": {i: c / max(tot, 1) for i, c in lead.most_common(4)},
        "switch_steps_deg": {},
    }
    for key in sorted({s[:4] for s in steps}):
        sel = [s[4] for s in steps if s[:4] == key]
        out["switch_steps_deg"][str(key)] = (len(sel), float(np.median(sel)), float(np.percentile(sel, 90)))
    return out


def summarise(rows: list[dict], keys: list[str]) -> dict:
    s = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])], dtype=float)
        if len(v):
            s[k] = (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), len(v))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds_a", help="reference dataset root (e.g. the teleop session)")
    ap.add_argument("ds_b", help="our dataset root")
    ap.add_argument("--n", type=int, default=60, help="episodes per dataset")
    ap.add_argument("--fps", type=float, default=None, help="override meta fps")
    ap.add_argument("--no-fk", action="store_true", help="skip the FK wrist path")
    ap.add_argument("--json", default=None)
    ap.add_argument("--events", action="store_true", help="also attribute the top-1%% arm excursions in ds_b")
    args = ap.parse_args()

    res = {}
    for tag, root in (("A", args.ds_a), ("B", args.ds_b)):
        fps = args.fps or json.load(open(os.path.join(root, "meta", "info.json")))["fps"]
        paths = eps_paths(root, args.n)
        rows = []
        for i, p in enumerate(paths):
            rows.append(episode_stats(p, float(fps), not args.no_fk))
            if (i + 1) % 20 == 0:
                print(f"  [{tag}] {i + 1}/{len(paths)}", file=sys.stderr)
        res[tag] = {"root": root, "fps": float(fps), "rows": rows}
        degen = max(r["cmd_equals_state"] for r in rows)
        print(f"[{tag}] {root}  {len(rows)} episodes @ {fps} fps", file=sys.stderr)
        if degen == 0.0:
            print(
                f"[{tag}] WARNING: action.wbc == observation.state exactly -- "
                f"this dataset has NO independent command channel, so every "
                f"cmd_* row below is a duplicate of st_*. Compare st_* only.",
                file=sys.stderr,
            )

    keys = [k for k in res["A"]["rows"][0] if k not in ("path",) and isinstance(res["A"]["rows"][0][k], (int, float))]
    sa = summarise(res["A"]["rows"], keys)
    sb = summarise(res["B"]["rows"], keys)

    print(f"\n{'metric':24s} {'A (ref)':>22s} {'B (ours)':>22s} {'B/A':>8s}")
    print("-" * 80)
    for k in keys:
        if k not in sa or k not in sb:
            continue
        ma, la, ha, _ = sa[k]
        mb, lb, hb, _ = sb[k]
        ratio = (mb / ma) if ma not in (0.0,) else float("nan")
        print(f"{k:24s} {ma:9.4f} [{la:7.3f},{ha:7.3f}] {mb:9.4f} [{lb:7.3f},{hb:7.3f}] {ratio:8.2f}")

    if args.events:
        ev = excursions(args.ds_b, args.n, res["B"]["fps"])
        print("\n--- where the tail comes from (ds_b) ---")
        print(
            f"top-1% arm excursions: n={ev['n_excursions']}, "
            f"{ev['within_0.5s_of_a_switch']:.1%} within 0.5 s of a "
            f"planner/stream mode switch"
        )
        for m, (share, frames) in ev["by_planner_mode"].items():
            print(
                f"  planner mode {m}: {share:6.1%} of excursions vs "
                f"{frames:6.1%} of frames  (x{share / max(frames, 1e-9):.2f})"
            )
        print(
            "  leading joint within the task arm: "
            + ", ".join(f"#{i} {p:.0%}" for i, p in ev["leading_joint_in_block"].items())
        )
        print("  commanded step at a switch, task arm, deg (pm_before,pm_after,sm_before,sm_after) -> n, median, p90:")
        for k, (n, med, p90) in ev["switch_steps_deg"].items():
            print(f"    {k:20s} n={n:4d}  {med:6.1f}  {p90:6.1f}")
        res["events_b"] = ev

    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"\nper-episode rows -> {args.json}")


if __name__ == "__main__":
    main()
