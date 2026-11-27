"""Extra checks for coverage.py (read-only): data3 attitude error by commanded attitude, the real commands' attitude in
data3's yaw / pitch / roll form, the 0.0415 m point back-off, chain-1 self-NN, elbow ranges."""
import math, sys
import numpy as np
sys.path.insert(0, sys.argv[1] + "/scripts")
T = sys.argv[1]
sys.argv = [sys.argv[0], T]
import coverage as C

z = {c: np.load(f"{T}/d3/{c}/all.npz") for c in ("scalebfm", "heft", "sonic")}
for c, d in z.items():
    a = d["att_err_deg"]; m = d["side"] == 1
    yaw = np.degrees(d["dyaw"]); pit = np.degrees(d["pitch"]); rol = np.degrees(d["roll"])
    mild = (np.abs(yaw) < 45) & (pit > -10) & (pit < 60) & (np.abs(rol) < 30)
    print(f"data3 {c}: end attitude error med {np.median(a):.1f} [{np.percentile(a,10):.1f},{np.percentile(a,90):.1f}] deg; "
          f"mild commands (|yaw|<45, pitch -10..60, |roll|<30; n {mild.sum()}) med {np.median(a[mild]) if mild.any() else np.nan:.1f}; "
          f"jaw_err med {np.median(d['jaw_err_cm']):.1f} cm")
SM = {(c, col): C.simple(c, col) for c in ("scalebfm", "sonic") for col in ("A", "J")}
def ypr(R):  # R = Rz(yaw) Ry(pitch) Rx(roll)
    pitch = math.asin(-float(np.clip(R[2, 0], -1, 1)))
    yaw = math.atan2(R[1, 0], R[0, 0]); roll = math.atan2(R[2, 1], R[2, 2])
    return np.degrees([yaw, pitch, roll])
import json
for (c, col), L in SM.items():
    e = [d for d in L if d["stage"] == "entry"]
    d4 = np.array([d["dist4"] for d in e])
    print(f"SIMPLE {c} {col} entry: back-off at the 0.0415 m point med {np.median(d4):.1f} [{np.percentile(d4,10):.1f},{np.percentile(d4,90):.1f}] cm")
# command attitude of the SIMPLE grasps in the heading frame, data3's form
for c in ("scalebfm", "sonic"):
    Y = []
    for task in ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP"):
        for r in C.rows(f"{T}/simple/{c}/rows/JR_{task}.jsonl"):
            s = r["stages"][1]
            yaw = s["W"][3]
            Y.append(ypr(C.rz(yaw).T @ np.array(s["rot_want"])))
    Y = np.array(Y)
    ins = (np.abs(Y[:, 0]) <= 90) & (Y[:, 1] >= -30) & (Y[:, 1] <= 90) & (np.abs(Y[:, 2]) <= 90)
    print(f"SIMPLE {c} grasp attitude (heading frame) yaw med {np.median(Y[:,0]):.0f} [{np.percentile(Y[:,0],10):.0f},{np.percentile(Y[:,0],90):.0f}] "
          f"pitch {np.median(Y[:,1]):.0f} [{np.percentile(Y[:,1],10):.0f},{np.percentile(Y[:,1],90):.0f}] roll {np.median(Y[:,2]):.0f} "
          f"[{np.percentile(Y[:,2],10):.0f},{np.percentile(Y[:,2],90):.0f}]; inside data3 draw box {ins.mean()*100:.0f}%")
# LML commanded attitude
import glob
LY = []
for f in glob.glob(f"{T}/lml/episodes/*/alone/*.json"):
    r = json.load(open(f))
    for w in r.get("reaches") or []:
        if w["k"] == 1:
            LY.append((w["dyaw_deg"], w["pitch_deg"], (r.get("plan") or {}).get("roll", np.nan)))
LY = np.array(LY, float)
print(f"LML entry command dyaw med {np.median(LY[:,0]):.0f} [{np.percentile(LY[:,0],10):.0f},{np.percentile(LY[:,0],90):.0f}] pitch "
      f"{np.median(LY[:,1]):.0f} [{np.percentile(LY[:,1],10):.0f},{np.percentile(LY[:,1],90):.0f}] grasp roll (not sent) "
      f"{np.degrees(np.nanmedian(np.abs(LY[:,2]))):.0f} [|.| p90 {np.degrees(np.nanpercentile(np.abs(LY[:,2]),90)):.0f}]")
# chain-1 self NN (against all data3) and elbow range
full = ["palm", "pel_z", "dvec", "att", "ref_s", "q"]
for c in ("scalebfm", "sonic"):
    D, _ = C.data3(c)
    X = np.array([C.feat(d, full) for d in D]); ch = np.array([d["chain"] for d in D])
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    dxx = np.linalg.norm(Xs[:, None] - Xs[None], axis=2); np.fill_diagonal(dxx, np.inf)
    print(f"data3 {c}: chain-1 rows' NN (16-D) med {np.median(dxx.min(1)[ch==1]):.2f}, chain-0 rows' {np.median(dxx.min(1)[ch==0]):.2f}")
    el = np.degrees([d["q"][3] for d in D]); sp = np.degrees([d["q"][0] for d in D])
    print(f"   elbow p1 {np.percentile(el,1):.0f} p99 {np.percentile(el,99):.0f}; chain1 elbow <30 deg: {np.mean(el[ch==1]<30)*100:.0f}%;"
          f" shoulder pitch p1 {np.percentile(sp,1):.0f} p99 {np.percentile(sp,99):.0f}")
    for col in ("A", "J"):
        G = [d for d in SM[(c, col)] if d["stage"] == "entry"]
        Y = np.array([C.feat(d, full) for d in G]); Y = Y[np.isfinite(Y).all(1)]
        dy = np.linalg.norm(((Y - mu) / sd)[:, None] - Xs[None], axis=2).min(1)
        print(f"   SIMPLE {col} entry NN med {np.median(dy):.2f} / chain-1 self NN {np.median(dxx.min(1)[ch==1]):.2f} = "
              f"{np.median(dy)/np.median(dxx.min(1)[ch==1]):.2f}; beyond chain-1 self-NN p90 {np.mean(dy > np.percentile(dxx.min(1)[ch==1],90))*100:.0f}%")
