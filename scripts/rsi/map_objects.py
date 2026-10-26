"""Vision RSI D2 step 2c (docs/handoff/vision-rsi-design-20261023.md section 17.4, item 4): the map's object layer.
SAM 3 (scripts/rsi/sam3_server.py, port 18822) is prompted on the tour's head-camera frames with the category phrases
of a family's targets (design D6, services.prompt_of); each instance at or above `--min-score` is back-projected with
its depth (the same stereo error model as map_build.py) and placed with the estimated poses; instances whose 3-D extent
is more than `--size-factor` times the category's object (the known object models L2 and L3 fit already) are dropped
(the size check of section 16); the rest merge across views into object nodes (DBSCAN on their centres, 8 cm).

Scored against the truth (the tour ran with the scene's default placements): per target, the node of its phrase nearest
its true position, the error; per phrase, the nodes that are no object of that phrase (more than 10 cm from all).

    PYTHONPATH=src /workspace/envs/vision_map/bin/python scripts/rsi/map_objects.py grasp_atomic \\
        outputs/vision_rsi_20261023/tour/sensors outputs/vision_rsi_20261023/tour/map_kiss_loop \\
        --poses outputs/vision_rsi_20261023/tour/poses/kiss_loop.npy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

from map_build import load  # noqa: E402
from map_common import MAX_DEPTH_M, kmat, se3, stereo_noise  # noqa: E402

from locomani.tasks.scene.services import SAM3_ADDRESS, Client, prompt_of, unpack_mask  # noqa: E402


def frames_to_segment(frames: list[dict], walk_every_s: float) -> list[int]:
    """Every head-scan frame, and walking frames every `walk_every_s`."""
    out, last = [], -1e9
    for f in frames:
        if f["kind"] == "scan":
            out.append(f["i"])
        elif f["t"] - last >= walk_every_s:
            out.append(f["i"])
            last = f["t"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("sensors", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--poses", default="truth")
    ap.add_argument("--min-score", type=float, default=0.4)
    ap.add_argument("--size-factor", type=float, default=1.6)
    ap.add_argument("--walk-every-s", type=float, default=1.0)
    a = ap.parse_args(argv)
    fam = json.loads((HERE / "families" / f"{a.family}.json").read_text())  # family.load pulls in the sim stack
    meta = json.loads((REPO / fam["scene"] / "scene_meta.json").read_text())
    rows = [json.loads(x) for x in (REPO / fam["tasks_file"]).read_text().splitlines() if x.strip()]
    goal = {r["id"]: r["goal"][0] for r in rows}
    targets = [goal[t].split("(")[1].rstrip(")") for t in fam["tasks"]]
    prompts = {k: prompt_of(meta["objects"][k], k) for k in targets}
    size = {}  # per phrase: the largest diagonal of a true object of that phrase (its collision box)
    by_phrase: dict[str, list[str]] = {}
    for k, o in meta["objects"].items():
        if not o.get("free"):
            continue
        ph = prompt_of(o, k)
        by_phrase.setdefault(ph, []).append(k)
        if ph in prompts.values():
            size[ph] = max(size.get(ph, 0.0), 2 * float(np.linalg.norm(o["box"]["half_extent"])))
    lz, lt, le, cz = load(a.sensors, a.poses)
    frames = json.loads((a.sensors / "frames.json").read_text())
    pick = frames_to_segment(frames, a.walk_every_s)
    K = kmat(cz["K"])
    sam = Client(SAM3_ADDRESS)
    rng = np.random.default_rng(0)
    dets, t0, phrases = [], time.time(), sorted(set(prompts.values()))
    for i in pick:
        rgb = cv2.cvtColor(cv2.imread(str(a.sensors / "rgb" / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
        z = cv2.imread(str(a.sensors / "depth" / f"{i:05d}.png"), cv2.IMREAD_UNCHANGED) / 1000.0
        z = stereo_noise(z, K[1, 1], rng)
        n = int(np.argmin(np.abs(lz["t"] - cz["t"][i])))
        cam = le[n] @ np.linalg.inv(lt[n]) @ se3(cz["R"][i], cz["pos"][i])
        res = sam.call(dict(image=rgb, prompts=phrases))
        if "error" in res:
            raise SystemExit(res["error"])
        for ph, inst in res["results"].items():
            for score, packed in inst:
                if score < a.min_score:
                    continue
                mask = unpack_mask(packed) & (z > 0.1) & (z < MAX_DEPTH_M)
                if mask.sum() < 15:
                    continue
                v, u = np.nonzero(mask)
                d = z[v, u]
                p = np.stack([(u - K[0, 2]) / K[0, 0] * d, -(v - K[1, 2]) / K[1, 1] * d, -d], -1)
                w = p @ cam[:3, :3].T + cam[:3, 3]
                med = np.median(w, axis=0)
                w = w[np.linalg.norm(w - med, axis=1) < 0.5]  # a mask edge on the background
                if len(w) < 15:
                    continue
                ext = float(np.linalg.norm(np.percentile(w, 95, axis=0) - np.percentile(w, 5, axis=0)))
                dets.append(dict(frame=i, phrase=ph, score=round(float(score), 3), centre=np.median(w, axis=0).tolist(),
                                 extent_m=round(ext, 3), points=int(len(w)),
                                 too_big=bool(ext > a.size_factor * size[ph])))  # fmt: skip
    seconds = time.time() - t0
    nodes = []
    for ph in phrases:
        ok = [d for d in dets if d["phrase"] == ph and not d["too_big"]]
        if not ok:
            continue
        c = np.array([d["centre"] for d in ok])
        lab = DBSCAN(eps=0.08, min_samples=1).fit_predict(c)
        for g in set(lab):
            mem = [ok[j] for j in np.where(lab == g)[0]]
            nodes.append(dict(phrase=ph, centre=np.median([m["centre"] for m in mem], axis=0).round(4).tolist(),
                              views=len(mem), score=round(max(m["score"] for m in mem), 3),
                              extent_m=round(float(np.median([m["extent_m"] for m in mem])), 3)))  # fmt: skip
    truth = {k: np.asarray(o["init_pos"], float) for k, o in meta["objects"].items() if o.get("free")}
    rows = {}
    for k in targets:
        mine = [n for n in nodes if n["phrase"] == prompts[k]]
        best = min(mine, key=lambda n: np.linalg.norm(np.subtract(n["centre"], truth[k])), default=None)
        err = float(np.linalg.norm(np.subtract(best["centre"], truth[k]))) if best else None
        rows[k] = dict(phrase=prompts[k], nodes_of_phrase=len(mine), err_cm=round(100 * err, 1) if best else None,
                       views=best["views"] if best else 0)  # fmt: skip
    false = {}
    for ph in phrases:
        same = [truth[k] for k in by_phrase.get(ph, [])]
        false[ph] = sum(
            1
            for n in nodes
            if n["phrase"] == ph and min(np.linalg.norm(np.subtract(n["centre"], s)) for s in same) > 0.10
        )
    a.out.mkdir(parents=True, exist_ok=True)
    res = dict(poses=a.poses, frames=len(pick), detections=len(dets), dropped_too_big=sum(d["too_big"] for d in dets),
               nodes=len(nodes), seconds=round(seconds, 1), targets=rows, false_nodes=false,
               found_within_5cm=sum(1 for r in rows.values() if r["err_cm"] is not None and r["err_cm"] <= 5))  # fmt: skip
    (a.out / "objects.json").write_text(json.dumps(dict(res, node_list=nodes, detection_list=dets), indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("targets",)}))
    for k, r in rows.items():
        print(k, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
