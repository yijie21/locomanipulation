"""ScaleBFM backend knobs (decision 8): what the in-process runtime loads and how it steps.

One frozen dataclass per Backend, mirroring `locomani/backends/sonic/config.py`. Every default
is the ScaleBFM line's ruled value (docs/ledger/scalebfm.md), so `ScaleBfmBackendConfig()` is
the yplus200 collection setting; the 2026-09-08 crate spike's values are noted per field.
`Sim` and `Policy` (runtime.py) take the config as a plain argument -- nothing in this
backend reads `os.environ`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict
from locomani.paths import REPO

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScaleBfmBackendConfig:
    checkpoint: str = "/workspace/models/scalebfm/humanoid_transformer_m_22200.pt"  # HF WeishuaiZeng/ScaleBFM, step 22200 (the only checkpoint ever used)
    scene_dir: str = ""  # MJCF + scene_meta.json directory; "" = outputs/scene/sim (the bottle scene; the spike used outputs/scene/crate_bimanual)
    object_name: str = "water_bottle"  # scene_meta `objects` entry the sim tracks as the target (spike: blue_crate)
    grip_kp: float = 1.0  # dex1 gripper PD gain, ctrl units x gear 250 N; 1.0/0.025 is what every visual-expert driver set (yplus200); spike default 40/1.0 limit-cycled the 87 g finger (runtime.py)
    grip_kd: float = 0.025  # dex1 gripper PD damping, see grip_kp

    def scene_path(self) -> str:
        return self.scene_dir or os.path.join(REPO, "outputs", "scene", "sim")

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "ScaleBfmBackendConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> "ScaleBfmBackendConfig":
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> "ScaleBfmBackendConfig":
        return apply_overrides(self, overrides)
