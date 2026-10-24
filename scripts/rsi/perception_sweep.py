"""Vision RSI step 1 (docs/handoff/vision-rsi-design-20261023.md section 9): one skills package on one pool under
several scene.view perturbations (L1: object position offset, per-read noise, heading error, delay), each condition a
batch job in one queue, then every condition against the unperturbed rounds (L0) paired by start -- wins and the 95%
Wilson interval, falls and drops, the starts each side wins alone (b: L0 failed, the condition succeeded; c: the other
way) and the one-sided McNemar p of losing. Rounds land in <out>/<condition>/ (a rerun skips rounds already written);
the summary goes to <out>/summary.json. The package must set WORLD_VIEW.

Run it from the checkout, detached (nohup setsid), like sealed.py:
    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/perception_sweep.py grasp_atomic <pool> scripts/rsi/seeds/grasp_c11_view <out> \\
        --base <L0 rounds> --set '{"pick.VIA_BACK_M": 0.2}' [--grid 1a | 1bc | 2 | 3 | 3d | --conditions <json>] [--procs 6 --gpus 0,1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing.connection import Client as Connect
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import batch as B  # noqa: E402
from gate import mcnemar_one_sided, wilson  # noqa: E402

from locomani.tasks.scene.perceive import RenderConfig  # noqa: E402
from locomani.tasks.scene.services import AUTHKEY  # noqa: E402
from locomani.tasks.scene.view import Perturbation  # noqa: E402

# the design's step 1 grid, one error at a time: position offsets 0.5-4 cm (one direction per object and round, the
# same for every size), delays of 100 and 200 ms, 1 cm per-axis noise at every read, a 10 degree heading error
CONDITIONS = {
    "bias_005": {"obj_bias_m": 0.005},
    "bias_010": {"obj_bias_m": 0.01},
    "bias_020": {"obj_bias_m": 0.02},
    "bias_040": {"obj_bias_m": 0.04},
    "latency_100": {"latency_s": 0.1},
    "latency_200": {"latency_s": 0.2},
    "jitter_010": {"obj_jitter_m": 0.01},
    "yaw_10": {"obj_yaw_deg": 10.0},
}
# step 1's second half: the robot's own position off by 1-4 cm against the map, and the object unseen while a hand is
# within 10 or 20 cm of it and not holding it
GRIDS = {
    "1a": CONDITIONS,
    "1bc": {
        "self_010": {"self_bias_m": 0.01},
        "self_020": {"self_bias_m": 0.02},
        "self_040": {"self_bias_m": 0.04},
        "blind_010": {"blind_within_m": 0.10},
        "blind_020": {"blind_within_m": 0.20},
    },
    # step 2 (L2): poses from rendered depth and masks -- three cameras with the map's memory 5 cm off, the head camera
    # alone, and the memory exact or 10 cm off (needs MUJOCO_GL=egl)
    "2": {
        "l2": {"rendered": {}},
        "l2_head": {"rendered": {"cameras": ["head_camera"]}},
        "l2_prior0": {"rendered": {"prior_bias_m": 0.0}},
        "l2_prior10": {"rendered": {"prior_bias_m": 0.10}},
    },
    # step 3 (L3): SAM 3 masks on Unreal's RGB, depth still MuJoCo's, at Unreal's 640 x 480; the exact masks at the same
    # size as the control (needs the Unreal editor, locomani.sim.unreal.rgb_server and scripts/rsi/sam3_server.py up)
    "3": {
        "l3": {"rendered": {"masks": "sam3", "width": 640, "height": 480}},
        "l2_640": {"rendered": {"width": 640, "height": 480}},
    },
    # step 3 with the mask and its depth from one surface: Unreal's depth (the plugin's "fresh_depth",
    # patches/urlab_fresh_depth.md, so the editor must run the patched copy of the host project) and the objects'
    # visual meshes as the fitted shape, from the full scene model the level was built from (path from the checkout)
    "3d": {
        "l3d": {
            "rendered": {
                "masks": "sam3",
                "width": 640,
                "height": 480,
                "depth": "unreal",
                "shape": "visual",
                "visual_scene": "outputs/scene_discovery/adt_apartment_mars/scene_locomani.xml",
            }
        },
    },
}


def paired(base: dict, cond: dict) -> dict:
    keys = sorted(set(base) & set(cond))
    k = sum(cond[x]["ok"] for x in keys)
    b = [x for x in keys if cond[x]["ok"] and not base[x]["ok"]]
    c = [x for x in keys if base[x]["ok"] and not cond[x]["ok"]]
    return dict(n=len(keys), ok=k, ci=wilson(k, len(keys)), base_ok=sum(base[x]["ok"] for x in keys),
                fell=sum(cond[x]["fell"] for x in keys), dropped=sum(cond[x]["dropped"] for x in keys),
                b=len(b), c=len(c), p_worse=mcnemar_one_sided(len(c), len(b)), only_base=c, only_cond=b)  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("pool", type=Path)
    ap.add_argument("skills", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--base", type=Path, required=True, help="the L0 rounds of the same package, pool and --set")
    ap.add_argument("--set", default="{}")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="1a", help="a named set of conditions")
    ap.add_argument("--conditions", type=Path, help="a JSON file {name: Perturbation spec}, in place of --grid")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("RSI_PROCS", "6")))
    ap.add_argument("--gpus", default=os.environ.get("RSI_GPUS", "0,1"))
    ap.add_argument("--npz", action="store_true")
    a = ap.parse_args(argv)
    conditions = json.loads(a.conditions.read_text()) if a.conditions else GRIDS[a.grid]
    for spec in conditions.values():
        if "rendered" in spec:
            cfg = RenderConfig.of(spec["rendered"])
            if os.environ.get("MUJOCO_GL") != "egl":
                raise SystemExit("a rendered condition needs MUJOCO_GL=egl in the environment (the workers render)")
            if cfg.masks == "sam3":
                for name, port in (("Unreal RGB (locomani.sim.unreal.rgb_server)", cfg.unreal_port),
                                   ("SAM 3 (scripts/rsi/sam3_server.py)", cfg.sam3_port)):  # fmt: skip
                    try:
                        Connect(("127.0.0.1", port), authkey=AUTHKEY).close()
                    except OSError as e:
                        raise SystemExit(f"a sam3 condition needs the {name} service on port {port}: {e}") from e
        else:
            Perturbation.of(spec)  # an unknown key fails here, not in every worker
    overrides = json.loads(a.set)
    jobs = [B.Job(a.skills.resolve(), (a.out / name).resolve(), overrides, None, a.npz, perception=spec)
            for name, spec in conditions.items()]  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    missing = B.run_jobs(jobs, a.family, a.pool.resolve(), a.procs, a.out / "run.log", a.gpus)
    base = B.outcomes(a.base)
    summary = dict(pool=str(a.pool), skills=str(a.skills), set=overrides, base=str(a.base),
                   base_ok=sum(r["ok"] for r in base.values()), base_n=len(base),
                   conditions={name: dict(perception=spec, **paired(base, B.outcomes(a.out / name)))
                               for name, spec in conditions.items()},
                   missing={name: missing[k] for k, name in enumerate(conditions) if missing[k]})  # fmt: skip
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, r in summary["conditions"].items():
        print(
            f"{name:14s} {r['ok']:3d}/{r['n']}  base {r['base_ok']}/{r['n']}  b {r['b']} c {r['c']}  p {r['p_worse']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
