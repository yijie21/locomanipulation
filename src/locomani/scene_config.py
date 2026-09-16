"""Scene: which frozen scene directory a run simulates in, and the fixed case bank it draws from.

Decision 5 of docs/handoff/infra-plan-20260916.md: the scene is its own config group so a task can
be collected on any table/object set by changing one word on the command line (`scene=lowcom`).
The randomisation ranges stay in the task's spawn config for now; moving them here is a later step.
"""

from __future__ import annotations

from dataclasses import dataclass

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SceneConfig:
    name: str = "sim"  # the file name under configs/scene/
    dir: str = (
        "outputs/scene/sim"  # frozen scene directory (scene_locomani.xml, scene_meta.json), repo-relative or absolute
    )
    cases: str | None = (
        None  # fixed sampling bank (grasp_gym --sample-only --write-cases) matched to this scene, or null
    )
    preset: str | None = None  # scene_random preset name recorded for provenance; the frozen dir is the truth

    def __post_init__(self):
        if not self.dir:
            raise ValueError("scene.dir is required")

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> SceneConfig:
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> SceneConfig:
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, pairs) -> SceneConfig:
        return apply_overrides(self, pairs)
