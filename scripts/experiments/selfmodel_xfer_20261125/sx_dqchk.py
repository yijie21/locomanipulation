import numpy as np, glob, sys
def load(paths):
    zs = [np.load(p) for p in paths]
    S = np.concatenate([z["S"] for z in zs]).astype(float); n = np.concatenate([z["n_steps"] for z in zs]).astype(int)
    k0 = np.concatenate([z["k0"] for z in zs]).astype(int); ch = np.concatenate([z["chain"] for z in zs]).astype(int)
    W = np.concatenate([z["W"] for z in zs]).astype(float)
    return S, n, k0, ch, W
for tag, paths in [("scalebfm_ha", ["/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125/scalebfm_ha/data/all.npz"]),
                   ("heft_ha (chunks so far)", sorted(glob.glob("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_xfer_20261125/heft_ha/data/w*_0*.npz")))]:
    paths = [p for p in paths if not p.endswith("_log.json")]
    if not paths: print(tag, "no chunks yet"); continue
    S, n, k0, ch, W = load(paths); so = np.r_[0, np.cumsum(n)[:-1]]; seg = np.repeat(np.arange(len(n)), n); same = seg[1:] == seg[:-1]
    q, dq = S[:, :29], S[:, 29:58]
    fd = ((q[1:] - q[:-1]) / 0.1)[same]; dqm = (0.5 * (dq[1:] + dq[:-1]))[same]; dq1 = dq[1:][same]
    print(f"== {tag}: {len(n)} segments, {len(S)} rows")
    # (a) dq[17], dq[18] vs the 0.1 s differences of all 29 joints, and vs every other dq / q column
    for j in (17, 18):
        c_fd = np.array([np.corrcoef(dqm[:, j], fd[:, k])[0, 1] for k in range(29)])
        c_dq = np.array([np.corrcoef(dq[:, j], dq[:, k])[0, 1] if k != j else np.nan for k in range(29)])
        c_q = np.array([np.corrcoef(dq[:, j], q[:, k])[0, 1] for k in range(29)])
        t = lambda c: f"max |r| {np.nanmax(abs(c)):.2f} at joint {int(np.nanargmax(abs(c)))}"
        print(f"  (a) dq[{j}] vs q differences of the 29 joints: {t(c_fd)} (own {c_fd[j]:+.2f}); vs other dq: {t(c_dq)}; vs q: {t(c_q)}")
    # (b) constant across resets / over time?
    first = np.where(ch == 0)[0]; r0 = so[first] + k0[first]
    for j in (17, 18):
        v = dq[r0, 29 - 29 + j] if False else S[r0, 29 + j]
        print(f"  (b) dq[{j}] at the first command over {len(first)} resets: unique values {len(np.unique(np.round(v, 4)))}, min / max {v.min():+.3f} / {v.max():+.3f}; q[{j}] there min / max {S[r0, j].min():+.5f} / {S[r0, j].max():+.5f}")
    sp = np.r_[np.linalg.norm(W[1:, :3] - W[:-1, :3], axis=1) / 0.1, 0][:len(S)]
    hs = np.r_[np.maximum(np.linalg.norm(W[1:, 4:7] - W[:-1, 4:7], axis=1), np.linalg.norm(W[1:, 7:10] - W[:-1, 7:10], axis=1)) / 0.1, 0]
    still = (sp < 0.02) & (hs < 0.02)
    for j in (17, 18):
        d = dq[still, j]; f = np.r_[(q[1:, j] - q[:-1, j]) / 0.1, 0][still]
        lag = np.corrcoef(dq[:-1, j][still[:-1] & still[1:] & same], dq[1:, j][still[:-1] & still[1:] & same])[0, 1]
        print(f"  (b) still rows ({still.sum()}): dq[{j}] median {np.median(d):+.3f}, |dq| p10/p50/p90 {np.percentile(abs(d),10):.3f}/{np.percentile(abs(d),50):.3f}/{np.percentile(abs(d),90):.3f}, positive {100*(d>0).mean():.0f} %, lag-1 (0.1 s) autocorr {lag:+.2f}; |q difference| p90 {np.percentile(abs(f),90):.4f} rad/s")
    for j in (15, 16, 19, 25):
        d = dq[still, j]; print(f"      reference joint {j}: still |dq| p50/p90 {np.percentile(abs(d),50):.3f}/{np.percentile(abs(d),90):.3f}")
    i = int(first[1]); a = so[i]; m = min(n[i], 30)
    print(f"  trace, reset of segment {i} rows 0-{m-1}: dq[17]", np.round(S[a:a+m, 46], 2).tolist())
    print(f"                                    dq[18]", np.round(S[a:a+m, 47], 2).tolist())
    print(f"                                    q[18]x1000", np.round(1000*S[a:a+m, 18], 2).tolist())
