"""TODO 47.8: a learned success predictor over (local geometry around the object, skill parameters).

Trained on the search's own round records (the incumbent skills package only). Input per round: a point cloud of what
is within 0.5 m of the object (target / other movable objects / fixed furniture) in a frame whose x axis is the nearest
approach edge's outward normal, a few scalars, and the parameter overrides of the round. Output: P(win) and the loss
class. Geometry is the scene file's truth; nothing is simulated to build an input.

    /venv/main/bin/python scripts/experiments/table_rsi_20261110/predict.py geom    # point clouds, one per start
    /venv/main/bin/python scripts/experiments/table_rsi_20261110/predict.py train   # 3 models x 5 seeds, held-out report
    /venv/main/bin/python scripts/experiments/table_rsi_20261110/predict.py pick    # a setting per held-out start
    $VENV_SIM_PY           scripts/experiments/table_rsi_20261110/predict.py test   # run the picks, count wins
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import feat as F  # noqa: E402
import table as T  # noqa: E402

D = T.OUT / "predict"
SR_SCENE = T.REPO / "outputs/scene_discovery/adt_apartment_mars_sr"  # sr.py build: the apartment + 128 SimReady objects
SR_D = T.OUT / "sr"
WITH_SR = False  # --with-sr: the SimReady rounds join the data, outputs go to predict_sr/


def scene_meta() -> dict:
    return json.loads((SR_SCENE / "scene_meta.json").read_text()) if WITH_SR else F.meta()


def proto(key: str, m: dict) -> str:
    return (
        m["objects"][key]["semantic"]["adt"]["prototype_name"] if key.startswith("sr_") else F.re.sub(r"_\d+$", "", key)
    )


def held_out_pools() -> list:
    """(name, pool, family, the incumbent's rounds, the feature table's rounds or None)"""
    out = [("apartment", F.D / "pool_feat_test", F.FAMILY, F.D / "test/base", F.D / "test/table")]
    if WITH_SR:
        out.append(("simready", SR_D / "pool_sr_test", "pick_sr", SR_D / "rounds/pool_sr_test/base", None))
    return out


INCUMBENT = "751e2bf9e0b0"  # the skills hash the round records carry for T.SKILLS
R_M, Z_LO, Z_HI = 0.5, -0.7, 0.5  # the crop around the object's centre
N_PTS = (192, 192, 640)  # target, other movable, fixed
DENSITY = 1600.0  # points per square metre on a box
CLASSES = ("won", "side_reach", "pinch_reach", "too_wide", "slip", "knocked", "miss_near", "walk_off", "other")
MARGIN = 0.05  # a start leaves the base setting only for this much predicted gain
SEEDS = 5


def rot2(a: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def quat_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def starts() -> dict:
    """label -> start file, from every pool under the experiment's output."""
    out = {}
    for f in sorted(T.OUT.glob("**/pool*/pick_*.json")):
        if WITH_SR or not f.stem.startswith("pick_sr_"):
            out.setdefault(f.stem, f)
    return out


def geom() -> None:
    import mujoco

    sys.path.insert(0, str(T.REPO / "src"))
    from locomani.tasks.scene.start import NORMALS, edge_depths

    meta = scene_meta()
    # with the SimReady objects: the scene's physics copy (no visual meshes; the colliders are the same)
    xml = SR_SCENE.parent / f"{SR_SCENE.name}_physics" / "scene_locomani.xml" if WITH_SR else meta["scene_xml"]
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(0)
    robot_root = m.body_rootid[m.body("pelvis").id]
    free = {o["body"]: k for k, o in meta["objects"].items() if o.get("free")}
    body_key = {m.body(b).id: k for b, k in free.items()}

    def local_points(g: int) -> np.ndarray:
        """Surface samples of a geom in its own frame."""
        t, s = int(m.geom_type[g]), m.geom_size[g]
        if t == 6:  # box
            area = 8 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
            n = int(np.clip(area * DENSITY, 6, 40000))
            p = rng.uniform(-1, 1, (n, 3)) * s
            w = np.array([s[1] * s[2], s[0] * s[2], s[0] * s[1]])
            ax = rng.choice(3, n, p=w / w.sum())
            p[np.arange(n), ax] = s[ax] * rng.choice((-1.0, 1.0), n)
            return p
        if t == 7:  # mesh
            i = m.geom_dataid[g]
            return m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]].copy()
        if t == 5:  # cylinder
            n = 200
            a = rng.uniform(0, 2 * np.pi, n)
            return np.c_[s[0] * np.cos(a), s[0] * np.sin(a), rng.uniform(-s[1], s[1], n)]
        return np.zeros((0, 3))

    fixed, moving = [], collections.defaultdict(list)  # world points; body-frame points per free object
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.body_rootid[m.geom_bodyid[g]] == robot_root:
            continue
        b, p = int(m.geom_bodyid[g]), local_points(g)
        if not len(p):
            continue
        if b in body_key:
            moving[body_key[b]].append(p @ quat_mat(m.geom_quat[g]).T + m.geom_pos[g])
        else:
            fixed.append(p @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    fixed = np.concatenate(fixed).astype(np.float32)
    moving = {k: np.concatenate(v).astype(np.float32) for k, v in moving.items()}
    home = {k: (d.xpos[m.body(b).id].copy(), d.xquat[m.body(b).id].copy()) for b, k in free.items() if k in moving}
    print("fixed points", len(fixed), "free objects", len(moving))

    def take(p: np.ndarray, n: int, r) -> tuple[np.ndarray, np.ndarray]:
        out, mask = np.zeros((n, 3), np.float32), np.zeros(n, np.float32)
        if len(p):
            i = r.choice(len(p), n, replace=len(p) < n)
            out, mask = p[i].astype(np.float32), np.ones(n, np.float32)
        return out, mask

    D.mkdir(exist_ok=True)
    pts, msk, sca, hand, labels = [], [], [], [], []
    for lab, f in starts().items():
        st = json.loads(f.read_text())["start"]
        key = T.task_of(lab)[5:]
        if key not in moving:
            continue
        pose = {k: home[k] for k in moving}
        pose.update(
            {k: (np.asarray(p, float), np.asarray(q, float)) for k, (p, q) in st["placements"].items() if k in moving}
        )
        world = {k: moving[k] @ quat_mat(q).T + p for k, (p, q) in pose.items() if np.linalg.norm(p[:2] - pose[key][0][:2]) < 1.2}  # fmt: skip
        tgt = world.pop(key)
        c = (tgt.min(0) + tgt.max(0)) / 2
        # the surface under the object and its nearest approach edge give the frame's x axis
        best = None
        for s in meta["surfaces"].values():
            he = np.asarray(s.get("flat_half_extent") or s["half_extent"], float)[:2]
            loc = rot2(float(s.get("yaw", 0.0))).T @ (c[:2] - np.asarray(s["xy"], float))
            if (
                np.all(np.abs(loc) <= he + 0.03)
                and s["top"] <= tgt.max(0)[2]
                and (best is None or s["top"] > best["top"])
            ):
                best = s
        rob = np.asarray(st["robot_xy"], float) - c[:2]
        if best is not None:
            dep = edge_depths(best, c[:2])
            name = min(dep, key=dep.get)
            n = rot2(float(best.get("yaw", 0.0))) @ np.asarray(NORMALS[name])
            depth, top = dep[name], best["top"]
        else:
            n, depth, top = rob / (np.linalg.norm(rob) + 1e-9), -1.0, tgt.min(0)[2]
        yaw = float(np.arctan2(n[1], n[0]))
        R = np.eye(3)
        R[:2, :2] = rot2(yaw).T

        def to_local(p):
            q = (p - c) @ R.T
            keep = (np.abs(q[:, 0]) <= R_M) & (np.abs(q[:, 1]) <= R_M) & (q[:, 2] >= Z_LO) & (q[:, 2] <= Z_HI)
            return q[keep]

        near = fixed[np.linalg.norm(fixed[:, :2] - c[:2], axis=1) < R_M * 1.5]
        t_l, o_l, f_l = (
            (tgt - c) @ R.T,
            to_local(np.concatenate(list(world.values()) or [np.zeros((0, 3))])),
            to_local(near),
        )
        r = np.random.default_rng(st["seed"])
        parts = [take(p, n_, r) for p, n_ in zip((t_l, o_l, f_l), N_PTS)]
        pts.append(np.concatenate([p for p, _ in parts]))
        msk.append(np.concatenate([k for _, k in parts]))
        ext = t_l.max(0) - t_l.min(0)
        rl = R[:2, :2] @ rob
        sca.append([c[2], top, depth, *ext, np.linalg.norm(rl), *(rl / (np.linalg.norm(rl) + 1e-9))])
        # hand-written placement features for the baseline: free gap beside the object, clearance above, behind, in front
        both = np.concatenate([o_l, f_l])
        band = both[(np.abs(both[:, 0]) <= ext[0] / 2) & (both[:, 2] > -ext[2] / 2 + 0.02) & (both[:, 2] < ext[2] / 2)]
        left, right = band[band[:, 1] > ext[1] / 2, 1], band[band[:, 1] < -ext[1] / 2, 1]
        col = both[
            (np.abs(both[:, 0]) <= ext[0] / 2) & (np.abs(both[:, 1]) <= ext[1] / 2) & (both[:, 2] > ext[2] / 2 + 0.005)
        ]
        lane = both[(np.abs(both[:, 1]) <= ext[1] / 2) & (both[:, 2] > -ext[2] / 2 + 0.02) & (both[:, 2] < ext[2] / 2)]
        back, front = lane[lane[:, 0] < -ext[0] / 2, 0], lane[lane[:, 0] > ext[0] / 2, 0]
        hand.append([min(left.min() - ext[1] / 2 if len(left) else R_M, R_M),
                     min(-ext[1] / 2 - right.max() if len(right) else R_M, R_M),
                     min(col[:, 2].min() - ext[2] / 2 if len(col) else Z_HI, Z_HI),
                     min(-ext[0] / 2 - back.max() if len(back) else R_M, R_M),
                     min(front.min() - ext[0] / 2 if len(front) else R_M, R_M)])  # fmt: skip
        labels.append(lab)
    cls = np.concatenate([np.full(n_, i) for i, n_ in enumerate(N_PTS)])
    np.savez_compressed(D / "geom.npz", labels=np.array(labels), pts=np.array(pts, np.float32), mask=np.array(msk, np.float32),
                        cls=cls, scalars=np.array(sca, np.float32), hand=np.array(hand, np.float32))  # fmt: skip
    print("starts", len(labels), "points per start", sum(N_PTS))


def records() -> list[dict]:
    """Every round of the incumbent package with a verdict and no runner error."""
    cache = D / "records.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = []
    for f in T.OUT.glob("**/pick_*.json"):
        if "pool" in f.parent.name or f.parts[len(T.OUT.parts)].startswith("predict"):
            continue
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not WITH_SR and r.get("label", "").startswith("pick_sr_"):
            continue
        if r.get("skills") != INCUMBENT or "ok" not in r or r.get("error"):
            continue
        out.append(dict(label=r["label"], ok=bool(r["ok"]), cls="won" if r["ok"] else str(r.get("failure_class")),
                        set=r.get("set") or {}, src=str(f.parent.relative_to(T.OUT))))  # fmt: skip
    cache.write_text(json.dumps(out))
    return out


def dataset():
    g = np.load(D / "geom.npz")
    row = {str(k): i for i, k in enumerate(g["labels"])}
    recs = [r for r in records() if r["label"] in row]
    seen = set()
    recs = [r for r in recs if not ((k := (r["label"], json.dumps(r["set"], sort_keys=True))) in seen or seen.add(k))]
    test_objs = set(json.loads((F.D / "split.json").read_text())["test"])
    if WITH_SR:
        test_objs |= {t[5:] for t in json.loads((SR_D / "split.json").read_text())["test"]}
    obj = [T.task_of(r["label"])[5:] for r in recs]
    dflt = T.defaults()
    vals = collections.defaultdict(set)
    for r, o in zip(recs, obj):
        if o not in test_objs:
            for k, v in r["set"].items():
                vals[k].add(float(v))
    keys = sorted(k for k, v in vals.items() if len(v | {dflt.get(k, next(iter(v)))}) >= 2)
    base = np.array([dflt.get(k, float(np.median(sorted(vals[k])))) for k in keys])
    scale = np.array([max(np.ptp([*vals[k], b]), 1e-6) for k, b in zip(keys, base)])
    P = np.array(
        [[(float(r["set"].get(k, b)) - b) / s for k, b, s in zip(keys, base, scale)] for r in recs], np.float32
    )
    y = np.array([r["ok"] for r in recs], np.float32)
    c = np.array([CLASSES.index(r["cls"]) if r["cls"] in CLASSES else len(CLASSES) - 1 for r in recs])
    gi = np.array([row[r["label"]] for r in recs])
    return g, recs, obj, test_objs, dict(keys=keys, base=base, scale=scale), P, y, c, gi


def cell_onehot(objs: list) -> np.ndarray:
    m = scene_meta()
    names = [("flat", "mid", "tall"), ("low", "table", "high"), ("fits", "tight", "wide")]
    out = []
    for o in objs:
        f = list(F.features(o, m).values())
        out.append([float(f[i] == n) for i in range(3) for n in names[i]])
    return np.array(out, np.float32)


def build(kind: str, n_scalar: int, n_param: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            if kind == "points":
                self.pt = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
            self.sc = nn.Sequential(nn.Linear(n_scalar, 64), nn.ReLU())
            self.pa = nn.Sequential(nn.Linear(n_param, 64), nn.ReLU())
            width = 128 + (256 if kind == "points" else 0)
            self.head = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
                                      nn.Linear(128, 1 + len(CLASSES)))  # fmt: skip

        def forward(self, pts, mask, cls1h, scalars, params):
            h = [self.sc(scalars), self.pa(params)]
            if self.kind == "points":
                z = self.pt(torch.cat([pts, cls1h.expand(len(pts), -1, -1)], -1))
                h.append(z.masked_fill(mask[..., None] == 0, -1e4).max(1).values)
            return self.head(torch.cat(h, -1))

    return Net()


def scalars_of(kind: str, g, objs_of_row: list) -> np.ndarray:
    if kind == "cells":
        return cell_onehot(objs_of_row)
    s = np.concatenate([g["scalars"], g["hand"]], 1) if kind == "hand" else g["scalars"]
    return s.astype(np.float32)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    pos, neg = p[y > 0.5], p[y < 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def fit_one(kind: str, seed: int, g, S, P, y, c, gi, tr, epochs: int = 25):
    import torch
    from torch.nn import functional as Fn

    torch.manual_seed(seed)
    dev = "cuda"
    net = build(kind, S.shape[1], P.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    pts, mask = torch.tensor(g["pts"], device=dev), torch.tensor(g["mask"], device=dev)
    cls1h = Fn.one_hot(torch.tensor(g["cls"], device=dev), 3).float()[None]
    mu, sd = S[np.unique(gi[tr])].mean(0), S[np.unique(gi[tr])].std(0) + 1e-6
    St = torch.tensor((S - mu) / sd, device=dev)
    Pt, yt, ct, git = (torch.tensor(a, device=dev) for a in (P, y, c, gi))
    idx = torch.tensor(tr, device=dev)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(idx) // 512 + 1))
    for _ in range(epochs):
        net.train()
        for b in idx[torch.randperm(len(idx), device=dev)].split(512):
            gb = git[b]
            x = pts[gb]
            if kind == "points":  # a small yaw turn and point noise
                a = (torch.rand(len(b), device=dev) - 0.5) * 0.3
                co, si = torch.cos(a)[:, None], torch.sin(a)[:, None]
                x = torch.stack([co * x[..., 0] - si * x[..., 1], si * x[..., 0] + co * x[..., 1], x[..., 2]], -1)
                x = x + 0.004 * torch.randn_like(x)
            out = net(x, mask[gb], cls1h, St[gb], Pt[b])
            loss = Fn.binary_cross_entropy_with_logits(out[:, 0], yt[b]) + 0.3 * Fn.cross_entropy(out[:, 1:], ct[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()

    def predict(rows: np.ndarray, params: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 4096):
                gb = torch.tensor(rows[i : i + 4096], device=dev)
                o = net(pts[gb], mask[gb], cls1h, St[gb], torch.tensor(params[i : i + 4096], device=dev))
                out.append(torch.sigmoid(o[:, 0]).cpu().numpy())
        return np.concatenate(out)

    return predict


def train() -> None:
    g, recs, obj, test_objs, pk, P, y, c, gi = dataset()
    row_obj = [T.task_of(str(x))[5:] for x in g["labels"]]
    is_test = np.array([o in test_objs for o in obj])
    src = np.array([r["src"] for r in recs])
    fresh = (
        np.array([s.startswith(("test", "jitter", "exact")) for s in src]) & ~is_test
    )  # seen objects, starts held back
    tr = np.where(~is_test & ~fresh)[0]
    sm = scene_meta()
    protos = sorted({proto(o, sm) for o, t in zip(obj, is_test) if not t})
    rng = np.random.default_rng(0)
    val_protos = set(rng.choice(protos, max(len(protos) // 5, 1), replace=False))
    is_val = np.array([proto(o, sm) in val_protos for o in obj])
    is_sr = np.array([o.startswith("sr_") for o in obj])
    base_rows = np.array([not (set(r["set"]) - set(T.BASE)) for r in recs])
    text = [f"records {len(recs)}: train {len(tr)} ({len(set(np.array(obj)[tr]))} objects, {len(set(gi[tr]))} starts), "
            f"seen objects on new starts {fresh.sum()}, unseen objects {is_test.sum()} ({len(set(np.array(obj)[is_test]))} objects)",
            f"parameters {len(pk['keys'])}; training win rate {y[tr].mean():.3f}; unseen-object win rate {y[is_test].mean():.3f}",
            f"validation prototypes (left out of training in the first pass): {sorted(val_protos)}", ""]  # fmt: skip
    preds = {}
    for kind in ("cells", "hand", "points"):
        S = scalars_of(kind, g, row_obj)
        # pass 1: prototypes held out of training, to see generalisation inside the training side
        f = fit_one(kind, 0, g, S, P, y, c, gi, tr[~is_val[tr]])
        v = np.where(is_val & ~is_test)[0]
        a_val = auc(f(gi[v], P[v]), y[v])
        fs = [fit_one(kind, s, g, S, P, y, c, gi, tr) for s in range(SEEDS)]
        preds[kind] = fs
        te, fr = np.where(is_test)[0], np.where(fresh)[0]
        p_te = np.mean([f(gi[te], P[te]) for f in fs], 0)
        p_fr = np.mean([f(gi[fr], P[fr]) for f in fs], 0)
        tb = np.where(is_test & base_rows)[0]
        p_tb = np.mean([f(gi[tb], P[tb]) for f in fs], 0)
        brier = float(np.mean((p_te - y[te]) ** 2))
        text.append(f"{kind:7s} AUC: left-out training prototypes {a_val:.3f} | seen objects, new starts {auc(p_fr, y[fr]):.3f} | "
                    f"unseen objects, all rounds {auc(p_te, y[te]):.3f} (Brier {brier:.3f}) | unseen objects, base setting {auc(p_tb, y[tb]):.3f}")  # fmt: skip
        for dom, mask in (("apartment", ~is_sr), ("simready", is_sr)):
            tb = np.where(is_test & base_rows & mask)[0]
            if len(tb):
                text.append(
                    f"          unseen {dom} objects, base setting: AUC {auc(np.mean([f(gi[tb], P[tb]) for f in fs], 0), y[tb]):.3f} over {len(tb)} rounds"
                )
        np.savez(D / f"pred_{kind}.npz", labels=g["labels"][gi[te]], p=p_te, y=y[te], P=P[te])
    (D / "train.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    # the candidates of every held-out start, scored by each model
    cand = F.variants()
    te_labels = sorted(p.stem for _, pool, *_ in held_out_pools() for p in pool.glob("pick_*.json"))
    row = {str(k): i for i, k in enumerate(g["labels"])}
    rows = np.array([row[x] for x in te_labels])
    Pc = np.array([[(float({**T.BASE, **st}.get(k, b)) - b) / s for k, b, s in zip(pk["keys"], pk["base"], pk["scale"])]
                   for _, st in cand], np.float32)  # fmt: skip
    score = {}
    for kind, fs in preds.items():
        sc = np.stack(
            [np.mean([f(rows, np.repeat(Pc[j : j + 1], len(rows), 0)) for f in fs], 0) for j in range(len(cand))], 1
        )
        score[kind] = sc.tolist()
    (D / "scores.json").write_text(json.dumps(dict(labels=te_labels, cand=[[n, st] for n, st in cand], score=score)))


def pick() -> None:
    s = json.loads((D / "scores.json").read_text())
    out = {}
    for kind, sc in s["score"].items():
        sc = np.array(sc)
        j = sc.argmax(1)
        j[sc.max(1) - sc[:, 0] < MARGIN] = 0  # candidate 0 is the base setting
        out[kind] = {lab: int(k) for lab, k in zip(s["labels"], j)}
        print(
            kind,
            "starts leaving the base setting",
            int((j > 0).sum()),
            "of",
            len(j),
            "| distinct settings",
            len(set(j)),
        )
    (D / "picks.json").write_text(json.dumps(out))


def test(procs: int, gpus: str) -> None:
    import fast

    s, picks = json.loads((D / "scores.json").read_text()), json.loads((D / "picks.json").read_text())
    text = []
    for dom, te, family, base_dir, table_dir in held_out_pools():
        base = T.rounds(base_dir)
        table = T.rounds(table_dir) if table_dir else {}
        text.append(f"held-out {dom} objects, {len(base)} starts: incumbent {sum(r['ok'] for r in base.values())}"
                    + (f", feature cells {sum(table.get(k, r)['ok'] for k, r in base.items())}" if table_dir else ""))  # fmt: skip
        for kind, pk in picks.items():
            by = collections.defaultdict(list)
            for lab, j in pk.items():
                if j and lab in base:
                    by[j].append(lab)
            out = D / "test" / dom / kind
            vs = [dict(name=f"{kind}{j}", skills=str(T.SKILLS), set={**T.BASE, **s["cand"][j][1]}, starts=ls, out=str(out))
                  for j, ls in sorted(by.items())]  # fmt: skip
            job = D / f"test_{dom}_{kind}.json"
            job.write_text(json.dumps(dict(family=family, pool=str(te), memo=False, starts=sorted(base), variants=vs)))
            if vs:
                fast.run(job, procs, gpus)
            new = T.rounds(out)
            won = sum(new[x]["ok"] and not base[x]["ok"] for x in new)
            lost = sum(base[x]["ok"] and not new[x]["ok"] for x in new)
            text.append(f"  {kind:7s}: {sum(new.get(k, r)['ok'] for k, r in base.items())} "
                        f"({len(new)} starts ran a changed setting, gained {won}, lost {lost})")  # fmt: skip
    (D / "test.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("geom", "train", "pick", "test"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--with-sr", action="store_true", help="add the SimReady rounds (sr.py); outputs in predict_sr/")
    a = ap.parse_args()
    if a.with_sr:
        WITH_SR, D = True, T.OUT / "predict_sr"
    if a.cmd == "test":
        test(a.procs, a.gpus)
    else:
        globals()[a.cmd]()
