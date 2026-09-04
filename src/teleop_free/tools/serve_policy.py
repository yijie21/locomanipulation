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
    ap.add_argument("--rtc-ramp-rate", type=float, default=3.0,
                    help="RTC exponential ramp rate for the overlap window "
                         "(only used when a client sends rtc.* keys)")
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

    import torch
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    class RTCPolicy(Gr00tPolicy):
        """Gr00tPolicy + test-time RTC (frozen-prefix inpainting).

        The model-layer primitive (gr00t_n1d7.get_action_with_features) wants
        the PREVIOUS chunk in NORMALIZED action space as action_input["action"]
        plus an options dict. The normalized chunk only exists inside
        _get_action, so this subclass replicates that method's five steps and
        (a) caches `action_pred` after every RTC-mode call, (b) injects the
        cache + options into the next call. Alignment convention: the cached
        chunk is passed WHOLE (H rows) and rtc_overlap_steps = H - elapsed, so
        the primitive's tail slice action[H-overlap:H] == prev[elapsed:H] --
        exactly the rows that overlap the new chunk's [0, H-elapsed).

        A client opts in per request via rtc.* keys; without them behaviour is
        bit-identical to Gr00tPolicy (the cache stays empty). Single-client
        server, so one cache slot, no locking.
        """
        rtc_ramp_rate = 3.0

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._rtc_prev = None    # torch [1, H, D] normalized action_pred
            self._rtc_meta = None    # {"elapsed": int, "frozen": int} | None
            self._rtc_applied = None

        def _get_action(self, observation, options=None):
            unbatched_observations = self._unbatch_observation(observation)
            processed_inputs = []
            states = []
            for obs in unbatched_observations:
                vla_step_data = self._to_vla_step_data(obs)
                states.append(vla_step_data.states)
                messages = [{"type": MessageType.EPISODE_STEP.value,
                             "content": vla_step_data}]
                processed_inputs.append(self.processor(messages))
            collated_inputs = self.collate_fn(processed_inputs)
            collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

            meta, self._rtc_meta = self._rtc_meta, None
            self._rtc_applied = None
            opts = None
            if meta is not None and self._rtc_prev is not None:
                H = int(self._rtc_prev.shape[1])
                e = int(meta["elapsed"])
                if 0 < e < H - 1:
                    overlap = H - e
                    frozen = max(0, min(int(meta["frozen"]), overlap - 1))
                    inner = (collated_inputs["inputs"]
                             if "inputs" in collated_inputs else collated_inputs)
                    inner["action"] = self._rtc_prev
                    opts = {"action_horizon": H,
                            "rtc_overlap_steps": overlap,
                            "rtc_frozen_steps": frozen,
                            "rtc_ramp_rate": self.rtc_ramp_rate}
                    self._rtc_applied = opts

            with torch.inference_mode():
                model_pred = self.model.get_action(**collated_inputs, options=opts)
            normalized_action = model_pred["action_pred"].float()
            # Cache only for RTC-mode clients: a plain client keeps the slot
            # empty and this class stays behaviour-identical to Gr00tPolicy.
            self._rtc_prev = (normalized_action.detach().clone()
                              if meta is not None else None)

            batched_states = {}
            for k in self.modality_configs["state"].modality_keys:
                batched_states[k] = np.stack([s[k] for s in states], axis=0)
            unnormalized_action = self.processor.decode_action(
                normalized_action.cpu().numpy(), self.embodiment_tag,
                batched_states)
            casted_action = {key: value.astype(np.float32)
                             for key, value in unnormalized_action.items()}
            return casted_action, {}

    tag = EmbodimentTag[args.embodiment_tag.upper()]
    cfg = MODALITY_CONFIGS[tag.value]
    policy = RTCPolicy(model_path=args.model_path, embodiment_tag=tag, device="cuda")
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    policy.rtc_ramp_rate = args.rtc_ramp_rate

    def get_action_raw(obs: dict) -> dict:
        # rtc.* keys ride inside the flat obs dict (the wire format already
        # exists; three scalars beat a second endpoint). Pop them BEFORE the
        # gr00t parse, which would choke on unknown keys.
        # policy.seed: deterministic denoising noise for the lockstep client
        # (same seed + same obs -> same chunk). Absent = torch's own stream.
        seed = obs.pop("policy.seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))
        if obs.pop("rtc.reset", 0):
            policy._rtc_prev = None
        if "rtc.elapsed" in obs:
            policy._rtc_meta = {"elapsed": int(obs.pop("rtc.elapsed")),
                                "frozen": int(obs.pop("rtc.frozen", 0))}
        chunk, _ = policy.get_action(parse_observation_gr00t(obs, cfg))
        out = {k: np.asarray(v) for k, v in parse_action_gr00t(chunk).items()}
        if policy._rtc_applied is not None:
            out["rtc.applied"] = np.array(
                [policy._rtc_applied["rtc_overlap_steps"],
                 policy._rtc_applied["rtc_frozen_steps"]], dtype=np.int64)
        return out

    server = PolicyServer(policy, port=args.port)
    server.register_endpoint("get_action_raw", get_action_raw)
    print(f"[serve_policy] {args.model_path} on :{args.port} "
          f"(denoising {args.denoising_steps})")
    server.run()


if __name__ == "__main__":
    main()
