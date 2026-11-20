"""Step 1 (perception env): looks -> the target's points.

    python perceive.py LOOK.npz [LOOK.npz ...] --out OUT [--object "red cup"] [--depth auto|raw|lingbot|stereo]

--depth  auto     the look's depth completed by LingBot-Depth; without a depth, its stereo pair matched first (default)
         raw      the look's depth as it is
         lingbot  the look's depth completed by LingBot-Depth
         stereo   the stereo pair matched, then completed by LingBot-Depth (needs right and baseline)

-> OUT/<look>.target.npz, which plan_grasp.py reads; a line of JSON per look on stdout.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1grasp import depth as D  # noqa: E402
from g1grasp import look as L  # noqa: E402
from g1grasp import target as T  # noqa: E402
from g1grasp.config import Config  # noqa: E402
from g1grasp.frames import apply, backproject  # noqa: E402


def depth_of(look: dict, mode: str, lingbot) -> np.ndarray:
    if mode == "auto":
        mode = "lingbot" if "depth" in look else "stereo"
    if mode == "raw":
        return look["depth"]
    if mode == "stereo":
        start = D.stereo(look["rgb"], look["right"], float(look["K"][0, 0]), float(look["baseline"]))
    else:
        start = look["depth"]
    return lingbot()(look["rgb"], start, look["K"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("looks", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--object", default="", help="the object's name for the segmenter; wins over the look's")
    ap.add_argument("--depth", default="auto", choices=("auto", "raw", "lingbot", "stereo"))
    a = ap.parse_args()
    cfg, out = Config(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sam, lb = T.Sam3(cfg.sam_threshold), None

    def lingbot():
        nonlocal lb
        lb = lb or D.LingBot()
        return lb

    print(json.dumps(dict(loaded_s=round(time.time() - t0, 1))), file=sys.stderr, flush=True)
    for f in a.looks:
        t = time.time()
        look = L.load(f)
        depth = depth_of(look, a.depth, lingbot)
        ok = (depth > cfg.near) & (depth < cfg.far)
        pts = apply(look["T_base_cam"], backproject(depth, look["K"]).reshape(-1, 3)).reshape(*depth.shape, 3)
        obj = a.object or look.get("object", "")
        tg = T.find(sam, look["rgb"], pts, ok, look.get("instruction", ""), obj, cfg)
        name = Path(f).name.removesuffix(".npz")
        row = dict(look=name, phrase=tg["phrase"], score=round(tg["score"], 3), seen=tg["mask"] is not None,
                   depth_mode=a.depth, depth_known=round(float(ok.mean()), 3))  # fmt: skip
        keep = dict(rgb=look["rgb"], K=look["K"], T_base_cam=look["T_base_cam"], depth=depth, phrase=tg["phrase"],
                    score=tg["score"], instruction=look.get("instruction", ""))  # fmt: skip
        if tg["mask"] is not None:
            c = tg["cloud"]
            row.update(points=len(c), centre=np.median(c, axis=0).round(3).tolist(), table_top=round(tg["table_top"], 3),
                       size=(np.percentile(c, 98, axis=0) - np.percentile(c, 2, axis=0)).round(3).tolist())  # fmt: skip
            keep.update(mask=tg["mask"], cloud=c, table_top=tg["table_top"])
        np.savez_compressed(out / f"{name}.target.npz", **keep)
        print(json.dumps(dict(row, s=round(time.time() - t, 2))), flush=True)


if __name__ == "__main__":
    main()
