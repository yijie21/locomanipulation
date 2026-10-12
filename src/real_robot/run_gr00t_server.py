"""GR00T inference server for the real-robot client (`real_robot.inference`), on the GPU machine.

    third_party/Isaac-GR00T/.venv $ PYTHONPATH=src python -m real_robot.run_gr00t_server \\
        --model-path outputs/runs/<run>/checkpoint-XXXX --embodiment-tag unitree_g1_sonic

`scripts/run_gr00t_server.sh` is one line over this. The structure is NVIDIA's Isaac-GR00T
`gr00t/eval/run_gr00t_server.py` (vendored unchanged at `src/gr00t/eval/run_gr00t_server.py`): a tyro
`ServerConfig`, a `Gr00tPolicy` (or a `ReplayPolicy` over a dataset), and `PolicyServer` run as a context
manager. What differs, and why:

- One more endpoint, `identity`: the checkpoint's file hashes, the field order of each modality and the gripper
  keys' training ranges (`real_robot.statistics`), which the client checks against its schema and its gripper
  units before it starts. The client calls the standard `get_action` with the nested observation, as NVIDIA's
  `run_vla_inference.py` does.
- `--backbone-path` and `--denoising-steps`: a checkpoint trained on the cluster records a backbone directory
  that does not exist here, and the action head is run with the step count the checkpoint was evaluated with.
- Default port 5590 (5555-5580 are the robot's camera, action, state and keyboard ports), default embodiment
  tag `unitree_g1_sonic`. No sim policy wrapper: the real robot is not a gym environment.

The GR00T code is this repository's vendored `src/gr00t`, the copy the checkpoint was trained with.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
from real_robot.statistics import gripper_ranges

DEFAULT_MODEL_SERVER_PORT = 5590


def _load_json_modality_configs(config_path: Path) -> dict[str, ModalityConfig]:
    """Load a JSON file whose values are ModalityConfig field dicts (not a dataset's meta/modality.json)."""
    with open(config_path, "r") as f:
        raw = json.load(f)
    try:
        return {k: ModalityConfig(**v) for k, v in raw.items()}
    except TypeError as exc:
        raise ValueError(
            f"{config_path} is not a ModalityConfig JSON: each value must hold ModalityConfig fields "
            f"(delta_indices, modality_keys, ...). Pass a .py modality config (e.g. configs/gr00t_sonic_3cam_bimanual.py)."
        ) from exc


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag (name or value, case-insensitive)"""

    device: str = "cuda"
    """Device to run the model on"""

    backbone_path: str | None = None
    """Local Cosmos-Reason2-2B directory, for a checkpoint trained on another machine (None: the checkpoint's record)"""

    denoising_steps: int = 4
    """Flow-matching steps of the action head"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file (.py registers itself on import, or .json)"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""


def _replay_modality_configs(config: ServerConfig) -> dict[str, ModalityConfig]:
    modality_configs: dict[str, ModalityConfig] | None = None
    if config.modality_config_path is not None:
        config_path = Path(config.modality_config_path)
        if config_path.suffix == ".py":
            # The .py file calls register_modality_config() as an import side effect.
            sys.path.append(str(config_path.parent))
            importlib.import_module(config_path.stem)
            print(f"Loaded modality config: {config_path}")
        elif config_path.suffix == ".json":
            modality_configs = _load_json_modality_configs(config_path)
        else:
            raise ValueError(f"Unsupported modality config format: {config_path.suffix}. Use .py or .json")
    if modality_configs is None:
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

        modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
        if modality_configs is None:
            raise ValueError(
                f"No modality config for embodiment tag '{config.embodiment_tag.name}'; "
                f"pass --modality-config-path (JSON or .py) when using ReplayPolicy."
            )
    return modality_configs


def identity_of(config: ServerConfig, policy) -> dict:
    """What the client checks before it starts: which files are served and each modality's field order."""
    if config.model_path is not None:
        from locomani.pipeline.provenance import checkpoint_identity

        identity = checkpoint_identity(config.model_path)
    else:
        identity = {"path": str(Path(config.dataset_path).resolve()), "replay": True}
    identity["denoising_steps"] = config.denoising_steps if config.model_path is not None else None
    identity["backbone_path"] = config.backbone_path
    identity["modalities"] = {
        kind: list(value.modality_keys) for kind, value in policy.get_modality_config().items() if kind != "rl_info"
    }
    # The gripper's training range names its units (simulator metres or Dex1 motor units), which the field names
    # do not; the client refuses to drive the hands in the wrong unit (real_robot.statistics, policy_client).
    identity["gripper_ranges"] = (
        gripper_ranges(config.model_path, config.embodiment_tag.value) if config.model_path is not None else None
    )
    return identity


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    if config.model_path is not None:
        if not os.path.isdir(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
            backbone_path=config.backbone_path,
        )
        policy.model.action_head.num_inference_timesteps = config.denoising_steps
    elif config.dataset_path is not None:
        if config.execution_horizon is None or config.execution_horizon <= 0:
            raise ValueError(
                "--execution-horizon must be a positive integer when --dataset-path is set (ReplayPolicy)."
            )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=_replay_modality_configs(config),
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    identity = identity_of(config, policy)
    print(f"  Modalities: {identity['modalities']}")
    print(f"  Gripper ranges: {(identity['gripper_ranges'] or {}).get('action')}")

    with PolicyServer(policy=policy, host=config.host, port=config.port) as server:
        server.register_endpoint("identity", lambda: identity, requires_input=False)
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
