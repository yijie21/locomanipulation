"""Grasps of the target's points (grasp env): GraspGen-X with its description of the G1's right three-finger hand,
and the first of them, best score first, that the right arm can take.

GraspGen-X gives the pose of the hand's grasp frame and no finger angles; its hand description carries one open and
one closed finger posture. They are given here by the Dex3's joint names; the description's angles go past the
URDF's limits in places, so the clipped ones are given too. Close the hand by feel (until the fingers stop), not by
driving it to the closed posture.
"""

import json
import os

import numpy as np

from .config import Config
from .frames import DEX3_RIGHT

HAND = "unitree_g1"
DESCRIPTION_JOINTS = ("zero", "one", "two", "three", "four", "five", "six")  # thumb 0-2, index 0-1, middle 0-1
# The description's palm link in its grasp frame (gripper.urdf, world_joint: xyz 0.07 0 0.02, rpy 90 deg 0 180 deg), then
# to the URDF's right_hand_palm_link: the two hands differ a little, the finger bases line up 7.3 mm further along x.
T_GRASP_PALM = np.array([[-1, 0, 0, 0.07 - 0.0073], [0, 0, 1, 0.0], [0, 1, 0, 0.02], [0, 0, 0, 1.0]])


class GraspGenX:
    def __init__(self, cfg: Config | None = None, assets_dir: str = ""):
        from graspgenx import get_checkpoints_version_dir, get_gripper_descriptions_assets
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.utils.checkpoint_io import load_model_cfg

        self.cfg = cfg = cfg or Config()
        assets = (assets_dir or str(get_gripper_descriptions_assets())).rstrip("/")
        if os.path.basename(assets) == "x_grippers":  # the descriptions' getter names the folder of the grippers
            assets = os.path.dirname(assets)
        ckpt = str(get_checkpoints_version_dir())
        gen, dis = os.path.join(ckpt, "gen"), os.path.join(ckpt, "dis")
        self.sampler = GraspGenXSampler(load_model_cfg(gen, dis), HAND, assets_dir=assets)
        desc = json.load(open(os.path.join(assets, "x_grippers", HAND, "config.json")))
        self.fingers = {
            k: np.array([desc[k][f"right_{j}_joint"] for j in DESCRIPTION_JOINTS], float) for k in ("open", "close")
        }

    def __call__(self, cloud: np.ndarray):
        """Every grasp of the points (N x 3, base frame): right_hand_palm_link poses (n x 4 x 4, base frame) and their
        scores, best first."""
        import torch
        from graspgenx.samplers import run_planner_on_object

        cfg = self.cfg
        pts = cloud[np.linalg.norm(cloud - np.median(cloud, axis=0), axis=1) < cfg.radius].astype(np.float32)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        G, conf, _, _ = run_planner_on_object(pts, self.sampler, planner=cfg.planner, grasp_threshold=-1.0,
                                              num_grasps=cfg.num_grasps, topk_num_grasps=-1)  # fmt: skip
        conf = np.asarray(conf, np.float32)
        o = np.argsort(-conf)
        return np.asarray(G, float)[o] @ T_GRASP_PALM, conf[o]


def takeable(T_palm: np.ndarray, cloud: np.ndarray, cfg: Config | None = None, yaw: float = 0.0) -> np.ndarray:
    """Which palm poses the right arm can take: the fingers ahead, the palm facing between left and down, the palm link
    `cfg.clear` above the target's foot. `yaw`: the robot's heading in the points' frame (0 in the base frame)."""
    cfg = cfg or Config()
    ahead, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    foot = np.percentile(cloud[:, 2], 2)
    R = T_palm[:, :3, :3]
    roll = np.degrees(np.arctan2(-R[:, 2, 1], R[:, :, 1] @ left))
    return (
        (R[:, :, 0] @ ahead >= cfg.ahead)
        & (cfg.roll[0] <= roll)
        & (roll <= cfg.roll[1])
        & (T_palm[:, 2, 3] >= foot + cfg.clear)
    )


def pregrasp(T_palm: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    cfg = cfg or Config()
    P = T_palm.copy()
    P[..., :3, 3] -= cfg.pre * T_palm[..., :3, 1]
    return P


def fingers(posture: np.ndarray) -> dict:
    return {
        "raw": {n: round(float(q), 4) for (n, _, _), q in zip(DEX3_RIGHT, posture)},
        "clipped": {n: round(float(np.clip(q, lo, hi)), 4) for (n, lo, hi), q in zip(DEX3_RIGHT, posture)},
    }
