"""Serve a finetuned GR00T checkpoint over ZMQ for the closed-loop driver.

    .venv (Isaac-GR00T) $ python serve_policy.py --model-path <ckpt> [--port 5590]

Why a server at all: the closed-loop driver has to live in the SONIC .venv_sim
(it needs gear_sonic's camera client, the pinocchio robot model for the wrist
FK, and the protocol-v4 commander), and that venv has no gr00t. The model side
has the opposite problem. A msgpack ZMQ boundary between them is smaller than
making either venv carry the other's stack.

The one endpoint that matters, `get_action_raw`, takes the FLAT observation
dict ({"video.ego_view": (1,H,W,3) u8, "state.left_leg": (1,6), ...,
"annotation...": str}) and does the gr00t-side parsing here, so the driver
needs zero gr00t imports -- plain zmq + msgpack_numpy, the same wire format
PolicyServer already speaks.

Port 5590: 5555-5582 are taken by the stack (camera, commander, debug,
keyboard, privileged, episode control).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--port", type=int, default=5590)
    ap.add_argument("--modality-config", default=None)
    ap.add_argument("--embodiment-tag", default="UNITREE_G1_SONIC")
    ap.add_argument("--denoising-steps", type=int, default=4)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    # src/gr00t is the vendored, COMMITTED copy (src/gr00t/UPSTREAM.md);
    # the third_party clone is reference only and must not shadow it
    sys.path.insert(0, str(repo / "src"))
    cfg_path = Path(args.modality_config or repo / "configs/gr00t_sonic_2cam.py")
    sys.path.insert(0, str(cfg_path.parent))
    __import__(cfg_path.stem)

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval.open_loop_eval import parse_action_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    tag = EmbodimentTag[args.embodiment_tag.upper()]
    cfg = MODALITY_CONFIGS[tag.value]
    policy = Gr00tPolicy(model_path=args.model_path, embodiment_tag=tag, device="cuda")
    policy.model.action_head.num_inference_timesteps = args.denoising_steps

    def get_action_raw(obs: dict) -> dict:
        chunk, _ = policy.get_action(parse_observation_gr00t(obs, cfg))
        return {k: np.asarray(v) for k, v in parse_action_gr00t(chunk).items()}

    server = PolicyServer(policy, port=args.port)
    server.register_endpoint("get_action_raw", get_action_raw)
    print(f"[serve_policy] {args.model_path} on :{args.port} "
          f"(denoising {args.denoising_steps})")
    server.run()


if __name__ == "__main__":
    main()
