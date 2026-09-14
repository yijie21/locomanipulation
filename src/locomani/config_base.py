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


_SCALARS = (bool, int, float, str)


def _scalar_ok(tp, value) -> bool:
    """Does `value` fit the static scalar type `tp` (bool/int/float/str, optionally `| None`)?

    Only scalars are checked; nested dataclasses, tuples and untyped fields pass through.
    bool is not an int here (rtc=1 for a bool field is a typo, not a value), int is a float.
    """
    origin = typing.get_origin(tp)
    if origin is typing.Union or (origin is not None and origin.__name__ == "UnionType"):
        return any(_scalar_ok(a, value) for a in typing.get_args(tp))
    if tp is type(None):
        return value is None
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp in _SCALARS:
        return isinstance(value, tp)
    return True


def validate(obj: T) -> T:
    """Reject values a config can hold but no entry point can act on (2026-09-10 review, refactor.md
    Spec 1): wrong scalar types, and strings outside a field's declared `metadata["choices"]`.
    Called by config_from_dict and apply_overrides; returns `obj` unchanged."""
    hints = _resolved_field_types(type(obj))
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if not _scalar_ok(hints.get(f.name), value):
            raise ValueError(f"{type(obj).__name__}.{f.name}: {value!r} is not a {hints.get(f.name)}")
        choices = f.metadata.get("choices")
        if choices is not None and value not in choices:
            raise ValueError(f"{type(obj).__name__}.{f.name}: {value!r} not in {choices}")
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            validate(value)
    return obj


def config_from_dict(cls: type[T], data: dict, schema_version: int | None = None) -> T:
    """Strict: unknown keys, a mismatched schema_version and wrong scalar types all raise --
    a config file that does not describe this code must not run (2026-09-10 review)."""
    data = dict(data)
    version = data.pop("schema_version", None)
    if schema_version is not None and version is not None and version != schema_version:
        raise ValueError(f"{cls.__name__}: schema_version {version} != {schema_version}")
    hints = _resolved_field_types(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown field(s) {unknown}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _from_plain(hints.get(f.name), data[f.name])
    return validate(cls(**kwargs))


def config_load(cls: type[T], path) -> T:
    return config_from_dict(cls, json.loads(Path(path).read_text()))


def config_dump(obj, path) -> None:
    Path(path).write_text(json.dumps(obj.to_dict(), indent=1) + "\n")


def _cast(current: Any, raw: str, tp: Any = None) -> Any:
    """Type a `--set key=value` RHS from the current (default) value's type.

    `tp` is the field's resolved static type (e.g. `tuple[str, ...]`), used to
    pick the element type for a tuple/list field whose *current* value is
    empty -- `type(current[0])` has nothing to look at then.
    """
    members = typing.get_args(tp)
    if type(None) in members:
        if raw.strip().lower() == "null":
            return None
        non_null = [member for member in members if member is not type(None)]
        if current is None and len(non_null) == 1 and non_null[0] in (bool, int, float, str):
            return _cast(non_null[0](), raw, non_null[0])
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
        if current:
            inner = type(current[0])
        else:
            args = typing.get_args(tp) if tp is not None else ()
            inner = args[0] if args else str
        return type(current)(inner(v) for v in raw.split(","))
    if current is None:
        return raw
    return raw


def apply_overrides(obj: T, overrides: list[str]) -> T:
    """Apply `["a.b=1", "c=2"]`-style `--set` overrides, dotted for nested fields.

    Returns a new instance (the dataclasses are frozen); `obj` is unchanged.
    """
    hints = _resolved_field_types(type(obj))
    values = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"--set expects key=value, got {kv!r}")
        key, raw = kv.split("=", 1)
        parts = key.split(".")
        if len(parts) == 1:
            if parts[0] not in values:
                raise ValueError(f"unknown field {parts[0]!r} on {type(obj).__name__}")
            values[parts[0]] = _cast(values[parts[0]], raw, hints.get(parts[0]))
        else:
            top = parts[0]
            if top not in values:
                raise ValueError(f"unknown field {top!r} on {type(obj).__name__}")
            nested = values[top]
            if not dataclasses.is_dataclass(nested):
                raise ValueError(f"field {top!r} is not a nested configuration")
            values[top] = apply_overrides(nested, [f"{'.'.join(parts[1:])}={raw}"])

    return validate(type(obj)(**values))
