"""Named configuration profiles: `configs/<group>/<name>.yaml` read as its own dataclass.

An experiment (`configs/experiments/*.yaml`) carries every value a run needs inline. These files are
the other half: the named profiles that the experts, the evaluators and the tests still refer to by
name -- "the bottle_visible task", "the sonic backend" -- without wanting a whole run around them.

Each file names the dataclass it is written against in its `defaults` list, which is the only thing
left of the composition that used to assemble these. Reading is otherwise plain YAML into the same
strict `from_dict` everything else uses, so an unknown field is an error here too.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from locomani import paths
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.bench.options import BenchOptions
from locomani.pipeline.eval_config import EvalConfig
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle_visual.config import BottleVisualTaskConfig
from locomani.tasks.crate.config import CrateTaskConfig

CONFIGS = Path(paths.REPO) / "configs"
SCHEMAS = {  # the name a profile gives in its `defaults` -> the dataclass that validates it
    "_bottle": BottleTaskConfig,
    "_bottle_visual": BottleVisualTaskConfig,
    "_crate": CrateTaskConfig,
    "_sonic": SonicBackendConfig,
    "_scalebfm": ScaleBfmBackendConfig,
}
GROUP_SCHEMAS = {"eval": EvalConfig, "bench": BenchOptions}  # groups with one dataclass, whose files just say `_schema`


def load(group: str, name: str):
    """`load("task", "bottle_simple")` -> a BottleTaskConfig."""
    path = CONFIGS / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no {group} profile called {name!r} ({path})")
    data = yaml.safe_load(path.read_text())
    named = [d for d in data.pop("defaults", []) if isinstance(d, str) and d in SCHEMAS]
    schema = SCHEMAS[named[-1]] if named else GROUP_SCHEMAS.get(group)
    if schema is None:
        raise ValueError(f"{path} does not say which dataclass it is written against")
    return schema.from_dict(data)


def names(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIGS / group).glob("*.yaml") if not p.stem.startswith("_"))
