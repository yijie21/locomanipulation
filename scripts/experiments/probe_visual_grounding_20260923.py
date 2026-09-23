"""Is the bimanual expert's arm motion predictable from what the cameras see, or does it need hidden state?

A VLA can only learn a mapping observation -> action. If the expert's next arm motion (or its phase)
depends on information the cameras do not carry -- the simulator's box pose, how long a phase has
run -- the learner sees identical frames with different labels. Linear probes, rounds held out:

    P      proprioception: 29 measured joints + 2 hand openings
    P+I    + frozen DINOv2 ViT-B/14 CLS of ego / left_wrist / right_wrist
    P+I+H  + hidden: the expert's box pose (robot frame, from the simulator) + seconds in phase / in round
    P+H    the hidden set without images

Targets: the phase (walk/settle/observe/reach/handoff) and the arm reference 0.2 s ahead minus now
(14 arm joints; what the expert is about to do). The gap P+I+H - P+I is the share of the target the
cameras do not explain. Also reports how often the box is visible in each camera per phase.

    third_party/Isaac-GR00T/.venv/bin/python scripts/experiments/probe_visual_grounding_20260923.py \
        outputs/runs/box_room_20260922 --out outputs/diagnostics/visual_grounding_20260923/box_room
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAMS = ("ego_view", "left_wrist", "right_wrist")
PHASES = ("walk", "settle", "observe", "reach", "handoff")
ARM = slice(15, 29)  # G1 29-DoF order: legs 0-11, waist 12-14, arms 15-28
AHEAD = 5  # frames at 25 fps = 0.2 s


def read_round(d: Path):
    obs = [json.loads(line) for line in (d / "observations.jsonl").read_text().splitlines()]
    t0 = obs[0]["simulation_time"]
    rows, phase_start, last = [], t0, None
    for o in obs:
        if o["phase"] != last:
            phase_start, last = o["simulation_time"], o["phase"]
        v = o.get("vision") or {}
        pos = v.get("position") or [np.nan] * 3
        rot = np.asarray(v.get("rotation") or np.full((3, 3), np.nan))
        vis = (o.get("guidance") or {}).get("visible") or {}
        rows.append(
            dict(
                phase=o["phase"],
                prop=np.r_[o["joints"], o.get("hand_openings") or [0, 0]],
                ref=np.asarray(o["reference_joints"]),
                hidden=np.r_[
                    pos,
                    np.arctan2(rot[1, 0], rot[0, 0]) if np.isfinite(rot).all() else np.nan,
                    o["simulation_time"] - phase_start,
                    o["simulation_time"] - t0,
                ],
                visible=[bool(vis.get(c)) for c in CAMS],
            )
        )
    return rows


def embed(d: Path, n: int, model, device):
    import cv2
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for cam in CAMS:
        cap, frames = cv2.VideoCapture(str(d / f"{cam}.mp4")), []
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224)))
        frames += [frames[-1]] * (n - len(frames))  # a short video repeats its last frame
        feats = []
        with torch.no_grad():
            for i in range(0, n, 64):
                x = torch.from_numpy(np.stack(frames[i : i + 64])).to(device).permute(0, 3, 1, 2).float() / 255
                feats.append(model((x - mean) / std).float().cpu().numpy())
        out.append(np.concatenate(feats))
    return np.concatenate(out, axis=1)


def standardise(tr, te):
    mu, sd = np.nanmean(tr, 0), np.nanstd(tr, 0) + 1e-6
    return np.nan_to_num((tr - mu) / sd), np.nan_to_num((te - mu) / sd)


def ridge(Xtr, Ytr, Xte, alpha):
    Xtr, Xte = np.c_[Xtr, np.ones(len(Xtr))], np.c_[Xte, np.ones(len(Xte))]
    w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    return Xte @ w


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12))


def grouped_folds(groups, k=5):
    uniq = np.unique(groups)
    return [np.isin(groups, uniq[i::k]) for i in range(k)]


def evaluate(X, Y, groups, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0)):
    """Out-of-fold predictions; alpha chosen inside each training fold on a held-out round group."""
    pred = np.zeros_like(Y)
    for test in grouped_folds(groups):
        tr = ~test
        inner = grouped_folds(groups[tr], k=4)[0]
        fit, val = standardise(X[tr][~inner], X[tr][inner])
        best = min(alphas, key=lambda al: ((ridge(fit, Y[tr][~inner], val, al) - Y[tr][inner]) ** 2).mean())
        Xtr, Xte = standardise(X[tr], X[test])
        pred[test] = ridge(Xtr, Y[tr], Xte, best)
    return pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "features.npz"
    rounds = sorted(p for p in (a.run / "observations").iterdir() if p.is_dir())
    data = [read_round(d) for d in rounds]
    if not cache.exists():
        import torch

        model = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", source="github", trust_repo=True)
            .to(a.device)
            .eval()
        )
        np.savez(cache, *[embed(d, len(rows), model, a.device) for d, rows in zip(rounds, data)])
    images = list(np.load(cache).values())

    P, I, H, Y, phase, groups, visible = [], [], [], [], [], [], []
    for g, (rows, img) in enumerate(zip(data, images)):
        for t in range(len(rows) - AHEAD):
            r = rows[t]
            P.append(r["prop"]), I.append(img[t]), H.append(r["hidden"]), visible.append(r["visible"])
            Y.append(rows[t + AHEAD]["ref"][ARM] - r["ref"][ARM])
            phase.append(PHASES.index(r["phase"])), groups.append(g)
    P, I, H, Y = map(np.asarray, (P, I, H, Y))
    phase, groups, visible = np.asarray(phase), np.asarray(groups), np.asarray(visible)
    onehot = np.eye(len(PHASES))[phase]
    sets = {"P": P, "P+I": np.c_[P, I], "P+I+H": np.c_[P, I, H], "P+H": np.c_[P, H]}

    report = dict(run=str(a.run), rounds=len(rounds), frames=int(len(Y)), ahead_s=AHEAD / 25, sets={})
    for name, X in sets.items():
        arm = evaluate(X, Y, groups)
        ph = evaluate(X, onehot, groups).argmax(1)
        report["sets"][name] = dict(
            arm_r2=r2(Y, arm),
            arm_r2_by_phase={
                PHASES[k]: r2(Y[phase == k], arm[phase == k]) for k in np.unique(phase) if (phase == k).sum() > 30
            },
            phase_acc=float((ph == phase).mean()),
            phase_acc_by_phase={PHASES[k]: float((ph[phase == k] == k).mean()) for k in np.unique(phase)},
        )
    report["frames_by_phase"] = {PHASES[k]: int((phase == k).sum()) for k in np.unique(phase)}
    report["box_visible_by_phase"] = {
        PHASES[k]: dict(zip(CAMS, visible[phase == k].mean(0).round(3).tolist())) for k in np.unique(phase)
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
