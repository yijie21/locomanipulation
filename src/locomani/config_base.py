"""Shared machinery for the config dataclasses (decision 8, refactor step 3).

Every knob lives in one of a small number of frozen dataclasses -- one per
Task, one per Backend, one for eval, one for the learn pilot -- instead of in
`os.environ`. This module gives all of them the same four operations
(`to_dict` / `from_dict` / `load` / `dump`) plus `--set key=value` parsing, so
each config module only has to declare its fields and defaults.

Nested dataclasses (e.g. `SonicBackendConfig.lockstep`, `BottleTaskConfig.reach`)
round-trip automatically as long as the nested type is itself a dataclass (or,
for `ReachConfig`, exposes `to_dict`/`from_dict`).
"""

from __future__ import annotations

import dataclasses
import json
import typing
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def _resolved_field_types(cls) -> dict:
    """Field name -> real type object.

    Every config module uses `from __future__ import annotations`, so
    `dataclasses.fields(cls)[i].type` is a STRING ("ReachConfig",
    "tuple[float, float]", ...), not the type itself; `get_type_hints`
    resolves those against the class's defining module.
    """
    return typing.get_type_hints(cls)


def _to_plain(value: Any) -> Any:
    """Recursively unwrap dataclasses/tuples into JSON-safe plain data."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {f.name: _to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [_to_plain(v) for v in value]
    return value


def _from_plain(tp, value: Any) -> Any:
    """Reconstruct a field value of static type `tp` from plain JSON data."""
    if dataclasses.is_dataclass(tp) and isinstance(value, dict):
        if hasattr(tp, "from_dict"):
            return tp.from_dict(value)
        hints = _resolved_field_types(tp)
        return tp(**{k: _from_plain(hints.get(k), v) for k, v in value.items()})
    if typing.get_origin(tp) is tuple and isinstance(value, list):
        return tuple(value)
    return value


def config_to_dict(obj, schema_version: int) -> dict:
    d = {"schema_version": schema_version}
    for f in dataclasses.fields(obj):
        d[f.name] = _to_plain(getattr(obj, f.name))
    return d


def config_from_dict(cls: type[T], data: dict) -> T:
    data = dict(data)
    data.pop("schema_version", None)
    hints = _resolved_field_types(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _from_plain(hints.get(f.name), data[f.name])
    return cls(**kwargs)


def config_load(cls: type[T], path) -> T:
    return config_from_dict(cls, json.loads(Path(path).read_text()))


def config_dump(obj, path) -> None:
    Path(path).write_text(json.dumps(obj.to_dict(), indent=1) + "\n")


def _cast(current: Any, raw: str) -> Any:
    """Type a `--set key=value` RHS from the current (default) value's type."""
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a bool: {raw!r}")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, (tuple, list)):
        inner = type(current[0]) if current else float
        return type(current)(inner(v) for v in raw.split(","))
    if current is None:
        return raw
    return raw


def apply_overrides(obj: T, overrides: list[str]) -> T:
    """Apply `["a.b=1", "c=2"]`-style `--set` overrides, dotted for nested fields.

    Returns a new instance (the dataclasses are frozen); `obj` is unchanged.
    """
    values = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"--set expects key=value, got {kv!r}")
        key, raw = kv.split("=", 1)
        parts = key.split(".")
        if len(parts) == 1:
            if parts[0] not in values:
                raise ValueError(f"unknown field {parts[0]!r} on {type(obj).__name__}")
            values[parts[0]] = _cast(values[parts[0]], raw)
        elif len(parts) == 2:
            top, sub = parts
            if top not in values:
                raise ValueError(f"unknown field {top!r} on {type(obj).__name__}")
            nested = values[top]
            nested_values = {f.name: getattr(nested, f.name) for f in dataclasses.fields(nested)}
            if sub not in nested_values:
                raise ValueError(f"unknown field {top}.{sub!r}")
            nested_values[sub] = _cast(nested_values[sub], raw)
            values[top] = type(nested)(**nested_values)
        else:
            raise ValueError(f"--set keys nest at most one level deep, got {key!r}")
    return type(obj)(**values)
