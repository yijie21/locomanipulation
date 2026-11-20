"""Step 2 (grasp env): the target's points -> the right hand's grasp and pregrasp.

    python plan_grasp.py OUT/<look>.target.npz [...] [--alternatives 4]

-> OUT/<look>.grasp.json     the chosen grasp (poses in the base frame, 4 x 4, metres) and the next ones
   OUT/<look>.grasps.npz     every grasp GraspGen-X gave: palm (n x 4 x 4), score, takeable
   OUT/<look>.check.png      the picture to check it by eye
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1grasp import grasp as G  # noqa: E402
from g1grasp.config import Config  # noqa: E402
from g1grasp.frames import wrist_of  # noqa: E402
from g1grasp.viz import check_image  # noqa: E402

FRAME = (
    "base: origin at the pelvis, x the robot's heading made level, z up; metres; poses are 4 x 4 (rotation | position)"
)


def pose_entry(T_palm: np.ndarray, score: float, rank: int, cfg: Config) -> dict:
    P = G.pregrasp(T_palm, cfg)
    r = lambda M: np.round(M, 5).tolist()  # noqa: E731
    return dict(rank=rank, score=round(float(score), 4), palm=r(T_palm), pregrasp_palm=r(P),
                wrist=r(wrist_of(T_palm)), pregrasp_wrist=r(wrist_of(P)))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--alternatives", type=int, default=4)
    ap.add_argument("--assets-dir", default="", help="GraspGen-X's gripper description assets (default: its own)")
    ap.add_argument("--num-grasps", type=int, default=Config.num_grasps)
    a = ap.parse_args()
    cfg = Config(num_grasps=a.num_grasps)
    t0 = time.time()
    gen = G.GraspGenX(cfg, a.assets_dir)
    print(json.dumps(dict(loaded_s=round(time.time() - t0, 1))), file=sys.stderr, flush=True)
    for f in a.targets:
        t = time.time()
        z = np.load(f, allow_pickle=False)
        stem = str(f).removesuffix(".target.npz")
        name = Path(stem).name
        out = dict(look=name, frame=FRAME, phrase=str(z["phrase"]), mask_score=round(float(z["score"]), 3))
        if "cloud" not in z.files:
            out["why"] = "the target was not seen"
            json.dump(out, open(f"{stem}.grasp.json", "w"), indent=1)
            check_image(
                f"{stem}.check.png", z["rgb"], z["K"], z["T_base_cam"], None, None, 0.0, title=f"{name}: not seen"
            )
            print(json.dumps(dict(look=name, why=out["why"])), flush=True)
            continue
        cloud = z["cloud"]
        palms, scores = gen(cloud)
        ok = G.takeable(palms, cloud, cfg)
        np.savez_compressed(f"{stem}.grasps.npz", palm=palms, score=scores, takeable=ok)
        idx = np.flatnonzero(ok)
        out.update(points=len(cloud), centre=np.median(cloud, axis=0).round(4).tolist(),
                   table_top=round(float(z["table_top"]), 4), grasps=len(scores), takeable=len(idx),
                   fingers_open=G.fingers(gen.fingers["open"]), fingers_close=G.fingers(gen.fingers["close"]))  # fmt: skip
        chosen = pose_entry(palms[idx[0]], scores[idx[0]], int(idx[0]), cfg) if len(idx) else None
        if chosen is None:
            out["why"] = "no grasp the right arm can take"
        out["grasp"] = chosen
        out["alternatives"] = [pose_entry(palms[i], scores[i], int(i), cfg) for i in idx[1 : 1 + a.alternatives]]
        json.dump(out, open(f"{stem}.grasp.json", "w"), indent=1)
        check_image(f"{stem}.check.png", z["rgb"], z["K"], z["T_base_cam"], z["mask"], cloud, float(z["table_top"]), chosen,
                    [e["palm"] for e in out["alternatives"]], title=f"{name}: {out['phrase']} (mask {out['mask_score']}), "
                    f"{len(idx)} of {len(scores)} grasps takeable" + (f", chosen score {chosen['score']}" if chosen else ""))  # fmt: skip
        row = dict(look=name, grasps=len(scores), takeable=len(idx), s=round(time.time() - t, 2))
        if chosen:
            row.update(score=chosen["score"], palm_xyz=np.round(palms[idx[0]][:3, 3], 3).tolist())
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
