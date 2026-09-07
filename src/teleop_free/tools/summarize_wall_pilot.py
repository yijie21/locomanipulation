"""Per-round summary of a wall pilot from its manifest + state traces.

    python summarize_pilot.py <pilot_dir>

Columns: ok, why, stand attempts, close trigger, crate first move (s) vs
episode length, static-frame fraction (50 Hz frames where every joint moved
< 1e-3 rad AND the pelvis < 0.5 mm AND the crate < 0.5 mm -- the frames a
single-frame policy cannot tell apart, i.e. the absorbing-state budget).
"""
import sys, json, glob, re, os, numpy as np

d = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl")) if l.strip()]
print(f"{'rd':>2} {'ok':>3} {'att':>3} {'trig':>8} {'move_s':>6} {'len_s':>5} {'static':>6} {'so':>5} {'yaw':>5}  why")
stat = []
for r in rows:
    rd = r["round"]
    f = os.path.join(d, "trace", f"wall_s0_{rd:02d}_state_trace.npz")
    move = length = static = float("nan")
    if os.path.exists(f):
        q = np.load(f)["qpos"]
        c = q[:, -7:]
        w, x, y, z = c[:, 3], c[:, 4], c[:, 5], c[:, 6]
        cy = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))); cy -= cy[0]
        dxy = np.linalg.norm(c[:, :2] - c[0, :2], axis=1)
        mv = np.where((dxy > 0.005) | (np.abs(cy) > 1.0))[0]
        move = mv[0] / 50 if len(mv) else float("nan")
        length = len(q) / 50
        dq = np.abs(np.diff(q[:, 7:-7], axis=0)).max(1)
        dp = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
        dc = np.linalg.norm(np.diff(c[:, :3], axis=0), axis=1)
        static = float(np.mean((dq < 1e-3) & (dp < 5e-4) & (dc < 5e-4)))
        stat.append(static)
    att = len(r.get("stand_attempts", [])) or 1
    so = r.get("stand_off_meas", float("nan")); yaw = r.get("yaw_err_deg", float("nan"))
    print(f"{rd:2d} {'OK' if r['ok'] else '--':>3} {att:3d} {str(r.get('close_trigger')):>8} {move:6.1f} {length:5.1f} {static:6.2f} {so:5.2f} {yaw:5.1f}  {r.get('why', '')[:60]}")
print(f"\nOK {sum(r['ok'] for r in rows)}/{len(rows)}; static-frame fraction median {np.median(stat):.2f} max {max(stat):.2f}" if stat else "")
