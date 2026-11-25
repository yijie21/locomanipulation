"""Capture and restore the mutable state of live objects (TODO 52, mechanism 2: replay from a saved state).

Standalone (numpy, and mujoco / torch only when the objects hold them): the environment server loads this file BY PATH under
SIMPLE's venv, which has no `locomani`, the bed imports it as a module. No import of anything of ours.

    snap = capture(obj, skip=("m", "link"), owned=(LegOdometry,))   # -> a detached copy of obj's attributes
    restore(obj, snap)                                              # obj's attributes are what they were; identities kept

What is copied (so what a snapshot is complete for):
  * scalars, strings, None, numpy arrays and scalars, torch tensors (cloned), lists / tuples / dicts / sets of these (recursively);
  * mujoco.MjData (every field of fixed size: qpos, qvel, act, ctrl, qacc_warmstart, time, mocap, sensordata, xpos ... ; see copy_mjdata),
    restored IN PLACE (other objects hold the same MjData);
  * instances of the classes in `owned`: their attributes, recursively, restored in place when the live
    attribute is still an instance of that class, else replaced by a copy;
  * a *list* attribute is restored by slice assignment and a dict by clear/update, so references to the container stay valid.
Functions, methods, lambdas and partials are kept by reference and written back (the same object, not a copy: a closure over
immutable data); what is shared, never copied nor written back (the live one stays): classes, mujoco.MjModel, sockets / connections, torch
modules, and every other object that is not in `owned`: their own state is not in the snapshot. A skipped attribute (`skip`) is
neither captured nor touched. `fingerprint(obj)` hashes the captured state, for tests that two states are the same.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import types

import numpy as np

try:  # the objects may hold these; the module itself needs neither
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

_SCALARS = (bool, int, float, complex, str, bytes, type(None), np.generic)


def _is_mjdata(x) -> bool:
    return mujoco is not None and isinstance(x, mujoco.MjData)


def _models_of(obj) -> list:
    out = []
    for v in vars(obj).values():
        if mujoco is not None and isinstance(v, mujoco.MjModel):
            out.append(v)
    return out


_FIELDS: dict = {}


def copy_mjdata(dst, src) -> None:
    """Every field of `src` into `dst` (same model), in place: the numeric arrays (qpos, qvel, act, ctrl, qacc_warmstart, mocap,
    sensordata, xpos ... and the derived ones, which MuJoCo's step leaves one substep behind the state: sensors and body poses
    are read between steps) and the scalars (time, ...). SIMPLE's venv has MuJoCo 3.3.6, whose python has no mj_copyData. Not
    copied: what has another length at each state (contacts, constraint rows: ncon / nefc), recomputed by the next step."""
    names = _FIELDS.get(type(dst))
    if names is None:
        names = _FIELDS[type(dst)] = [
            n for n in dir(type(dst)) if not n.startswith("_") and isinstance(getattr(type(dst), n), property)
        ]
    for n in names:
        try:
            v = getattr(src, n)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            try:
                a = getattr(dst, n)
                if isinstance(a, np.ndarray) and a.shape == v.shape and a.flags.writeable and v.dtype.kind in "fiub":
                    a[...] = v
            except Exception:
                pass
        elif isinstance(v, (int, float, np.generic)) and not isinstance(v, bool):
            try:
                setattr(dst, n, v)
            except (AttributeError, TypeError, ValueError):
                pass


class Packed:
    """A detached copy of one value: the kind of node it is and its content."""

    __slots__ = ("kind", "cls", "val")

    def __init__(self, kind, val, cls=None):
        self.kind, self.val, self.cls = kind, val, cls


def pack(x, owned: tuple, ctx: list):
    if isinstance(x, _SCALARS):
        return Packed("atom", x)
    if isinstance(x, np.ndarray):
        return Packed("nd", x.copy())
    if torch is not None and isinstance(x, torch.Tensor):
        return Packed("tensor", x.detach().clone())
    if _is_mjdata(x):
        return Packed("mjdata", copy.copy(x))
    if isinstance(x, list):
        return Packed("list", [pack(v, owned, ctx) for v in x])
    if isinstance(x, tuple) and not hasattr(x, "_fields"):
        return Packed("tuple", [pack(v, owned, ctx) for v in x])
    if isinstance(x, dict):
        return Packed("dict", [(pack(k, owned, ctx), pack(v, owned, ctx)) for k, v in x.items()], type(x))
    if isinstance(x, (set, frozenset)):
        return Packed("set", [pack(v, owned, ctx) for v in x], type(x))
    if isinstance(x, (types.FunctionType, types.MethodType, functools.partial)):
        return Packed(
            "fn", x
        )  # a callable attribute is state too (Motor.ref: the reference trajectory of the last primitive)
    if isinstance(x, (types.BuiltinFunctionType, type, types.ModuleType)):
        return Packed("ref", x)
    if isinstance(x, owned):  # (a dataclass is owned only when listed: an episode facade holds the whole run)
        local = ctx + _models_of(x)
        return Packed("obj", {k: pack(v, owned, local) for k, v in vars(x).items()}, type(x))
    return Packed("ref", x)  # shared: connections, models, modules, anything not owned


def unpack(p: Packed, live=None):
    """A fresh value from `p`; when `live` is a compatible live value it is updated in place and returned."""
    k = p.kind
    if k == "atom" or k == "ref" or k == "fn":
        return p.val
    if k == "nd":
        if isinstance(live, np.ndarray) and live.shape == p.val.shape and live.dtype == p.val.dtype:
            live[...] = p.val
            return live
        return p.val.copy()
    if k == "tensor":
        if (
            torch is not None
            and isinstance(live, torch.Tensor)
            and live.shape == p.val.shape
            and live.dtype == p.val.dtype
        ):
            live.copy_(p.val)
            return live
        return p.val.clone()
    if k == "mjdata":
        if _is_mjdata(live):
            raise AssertionError("mjdata is restored by restore_obj, which knows the model")
        return copy.copy(p.val)
    if k == "list":
        items = [unpack(v, live[i] if isinstance(live, list) and i < len(live) else None) for i, v in enumerate(p.val)]
        if isinstance(live, list):
            live[:] = items
            return live
        return items
    if k == "tuple":
        return tuple(unpack(v) for v in p.val)
    if k == "dict":
        items = [(unpack(a), unpack(b, live.get(unpack(a)) if isinstance(live, dict) else None)) for a, b in p.val]
        if isinstance(live, dict):
            live.clear()
            live.update(items)
            return live
        return p.cls(items) if p.cls is not dict else dict(items)
    if k == "set":
        return p.cls(unpack(v) for v in p.val)
    if k == "obj":
        tgt = live if isinstance(live, p.cls) else p.cls.__new__(p.cls)
        _restore_attrs(tgt, p.val)
        return tgt
    raise ValueError(k)


def _restore_attrs(obj, attrs: dict) -> None:
    for name, p in attrs.items():
        if (
            p.kind == "ref"
        ):  # shared handles are never written back: the live one stays (a rebuilt scene's, not the captured one's)
            continue
        cur = obj.__dict__.get(name)
        if p.kind == "mjdata":
            if _is_mjdata(cur):
                copy_mjdata(cur, p.val)
            else:
                obj.__dict__[name] = copy.copy(p.val)
        else:
            obj.__dict__[name] = unpack(p, cur)


class _Ctx:
    """The shared models a restore may need for MjData objects whose model is not an attribute of the same object."""

    def __init__(self):
        self.v: list = []

    def get(self):
        return self.v


_ctx_models = _Ctx()


def capture(obj, skip: tuple = (), owned: tuple = (), models: tuple = ()) -> dict:
    """A detached copy of obj's attributes. `models`: MjModels MjData attributes may belong to besides obj's own."""
    ctx = list(models) + _models_of(obj)
    return {k: pack(v, owned, ctx) for k, v in vars(obj).items() if k not in skip}


def restore(obj, snap: dict, models: tuple = ()) -> None:
    """obj's captured attributes put back; objects, lists, dicts and arrays that exist are updated in place."""
    _ctx_models.v = list(models)
    try:
        # an attribute added after the capture is left alone (the skip list is the caller's contract for such)
        _restore_attrs(obj, snap)
    finally:
        _ctx_models.v = []


def fingerprint(snap: dict) -> str:
    """sha1 over the content of a snapshot (MjData: its numeric fields), for equality tests."""
    h = hashlib.sha1()

    def walk(p):
        h.update(p.kind.encode())
        v = p.val
        if p.kind == "atom":
            h.update(repr(v).encode())
        elif p.kind == "nd":
            h.update(str(v.shape).encode() + str(v.dtype).encode() + v.tobytes())
        elif p.kind == "tensor":
            h.update(v.cpu().numpy().tobytes())
        elif p.kind == "mjdata":
            for f in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "sensordata", "qfrc_applied",
                      "xfrc_applied", "xpos", "xquat", "qacc"):  # fmt: skip
                h.update(np.asarray(getattr(v, f)).tobytes())
            h.update(repr(float(v.time)).encode())
        elif p.kind in ("list", "tuple", "set"):
            for q in v:
                walk(q)
        elif p.kind == "dict":
            for a, b in v:
                walk(a)
                walk(b)
        elif p.kind == "obj":
            for k in sorted(v):
                h.update(k.encode())
                walk(v[k])
        elif p.kind == "fn":
            h.update(str(id(v)).encode())  # the same function object, not an equal one
        elif p.kind == "ref":
            h.update(type(v).__name__.encode())

    for k in sorted(snap):
        h.update(k.encode())
        walk(snap[k])
    return h.hexdigest()
