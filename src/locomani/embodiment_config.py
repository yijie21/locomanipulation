"""Embodiment: the robot's action/state layout, cameras and GR00T modality — shared by every task that runs on it.

Decision 5 of docs/handoff/infra-plan-20260916.md: training and evaluation depend on the
embodiment, never on the task, so "single-arm" and "bimanual" are one file each under
`configs/embodiment/` and a task declares which of them it can run on (`conf.TASK_EMBODIMENTS`).
"""

from __future__ import annotations

from dataclasses import dataclass

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EmbodimentConfig:
    name: str = "g1_single_arm"  # the file name under configs/embodiment/; tasks list the names they support
    arms: str = "single"  # single | bimanual: which arms the expert commands and the dataset records as actions
    state_dim: int = 46  # observation.state width in the exported LeRobot dataset (GR00T's unitree_g1_sonic layout)
    action_dim: int = 46  # action width in the exported dataset
    cameras: tuple[str, ...] = ("ego_view", "left_wrist", "right_wrist")  # video streams the exporter must record
    gr00t_tag: str = "unitree_g1_sonic"  # GR00T EmbodimentTag the modality config rewrites
    gr00t_modality_config: str = (
        "configs/gr00t_sonic_3cam.py"  # --modality_config_path for training/serving (repo-relative)
    )
    dataset_kind: str = "sonic_single_arm"  # what check_dataset expects: sonic_single_arm | sonic_bimanual

    def __post_init__(self):
        if self.arms not in ("single", "bimanual"):
            raise ValueError("embodiment.arms must be single or bimanual")
        if not (self.state_dim > 0 and self.action_dim > 0 and self.cameras):
            raise ValueError("embodiment needs positive state/action widths and at least one camera")

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> EmbodimentConfig:
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> EmbodimentConfig:
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, pairs) -> EmbodimentConfig:
        return apply_overrides(self, pairs)
