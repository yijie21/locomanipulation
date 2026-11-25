"""TODO 52 mechanism 2: replay from a saved state. Rounds that start the same way share the simulation of their beginning.

Ports TODO 46.1's memo (scripts/experiments/table_rsi_20261110/memo.py) to the SIMPLE bed, where the physics is in another process.
A round is a sequence of top-level Motor primitives (walk_to, turn, sidestep, posture, reach, adopt_hand, grip, release, hold).
The simulation is deterministic, so two rounds that start alike and whose first n primitives (name and arguments) are identical are
in the same state after them. `install` wraps the primitives of the episode's Motor: each primitive's key chains the start's label,
every earlier primitive and its own arguments. After a miss the primitive has run for real and the state it left is stored; on a
hit the stored state is put back and the stepping is skipped, while the skill's Python runs unchanged (so a variant's Python between
primitives, its perception included, still runs on the restored state).

The state is complete on both sides of the socket:
  * the server (server.py, `snap` / `restore`): SIMPLE's MuJoCo data (every field, statecopy), the environment's and the task's
    counters and flags, the robot's own, the episode's judge (compliance checker / articulation watcher) and counters; the Isaac
    scene is synced to the restored state before any render;
  * this side: the local mirror Sim (its MjData and fields), ScaleBFM's policy history, the Motor's fields, the Belief with its
    leg odometry and feed, the odometry root, the clock (counted steps, phases, the judge's answers), the record's lists.
Only what the primitive CHANGED is stored and put back (attribute by attribute), so what the skill's Python set between two
primitives is never overwritten by a stored state of another variant.

Perception is memoised with the state: the images (and the depth) `eyes` returns at a state are stored under the chain key and the
number of the call since the last primitive, and a later round at the same state gets those back. Rendering is not repeatable run to
run (two runs of one start differ in the last bits of the head images, so a skill's goals differ in the last bits and no key would
ever match), so without this a round could never reach the states of another; with it, variants of one start also share the
perception of their common beginning (their differences are then the parameter's, not the renderer's noise).

The chain breaks (no more hits or stores for the round) when the simulation was stepped outside a primitive, as a hand program does
through `tick`; and a primitive that ends the episode (Done) is never stored. `max_states` bounds the stored states (the oldest go
first, on the server too).

TODO 53, the disk (`disk`, memostore.DiskStore): every state stored after a miss is also written to a file (client changes, the
primitive's return value, trace, calls and row growth, and the server's state as bytes from its `snap_dump`), and every perception
output too; a state that is not in memory is looked for on disk (the server takes it with `snap_load`). So a fresh process, with a
fresh server, starts from what an earlier process stored. What cannot be written (a callable that is not a constant reference, a
shared handle inside a container) keeps that state in memory only and is counted (`unpersistable`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import types
import zlib
from collections import OrderedDict

import numpy as np

from locomani.bench.simple import statecopy as SC

PRIMITIVES = ("walk_to", "turn", "sidestep", "posture", "reach", "adopt_hand", "grip", "release", "squeeze", "hold")


def plain(x):
    """A JSON-able image of a primitive argument, exact for floats and arrays (hex), for the chain key."""
    if isinstance(x, np.ndarray):
        return ["nd", list(x.shape), str(x.dtype), x.astype(float).tobytes().hex()]
    if isinstance(x, (np.floating, float)):
        return float(x).hex()
    if isinstance(x, (np.integer, int, bool, str, type(None))):
        return x
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    return repr(x)


def start_key(label: str) -> str:
    return hashlib.sha256(f"start:{label}".encode()).hexdigest()


def next_key(key: str, name: str, args, kw) -> str:
    """The chain: this primitive's key from the previous key, its name and its arguments."""
    return hashlib.sha256(json.dumps([key, name, plain(args), plain(kw)]).encode()).hexdigest()


# what each captured object keeps out of the snapshot: heavy shared handles (identity kept anyway) and the lists handled as deltas
POL_SKIP = (
    "w",
    "meta",
    "toff",
    "isel",
    "gin",
    "gmode",
    "pinned",
    "gout",
    "g",
    "host",
    "parts",
    "sel",
    "fidx",
    "root_source",
)
DELTA_LISTS = ("trace", "calls")


class Entry:
    __slots__ = ("changes", "out", "trace", "calls", "row", "sid", "after")

    def __init__(self, changes, out, trace, calls, row, sid, after=None):
        self.changes, self.out, self.trace, self.calls, self.row, self.sid, self.after = (
            changes,
            out,
            trace,
            calls,
            row,
            sid,
            after,
        )


def _owned():
    from locomani.backends.scalebfm import motor as M
    from locomani.perception import odometry as LO

    own = [M.MotorConfig, M.Pose, M.Outcome, LO.LegOdometry]
    for n in ("OdoRoot", "SeatedOdometry"):
        if hasattr(LO, n):
            own.append(getattr(LO, n))
    from locomani.bench.simple import feed as FL

    own.append(FL.SeatedOdometry)
    return tuple(own)


class Memo:
    def __init__(
        self,
        env,
        label: str,
        max_states: int = 400,
        owned: tuple | None = None,
        store: OrderedDict | None = None,
        check: bool = False,
        eyes_store: OrderedDict | None = None,
        eyes_cache: bool = True,
        max_eyes: int = 120,
        disk=None,
    ):
        self.disk = disk
        self.persisted = self.unpersistable = self.disk_hits = self.eyes_disk_hits = self.eyes_persisted = 0
        self.why_not: dict = {}
        self.max_eyes = max_eyes
        self.check, self.diffs = check, []
        self.env, self.link = env, env.sim.link
        self.mo = env.mo
        self.owned = _owned() if owned is None else owned
        self.max_states = max_states
        self.store: OrderedDict[str, Entry] = (
            OrderedDict() if store is None else store
        )  # the Bed's: shared by its episodes
        self.key, self.live, self.k_seen, self.depth = start_key(label), True, self.mo.k, 0
        self.hits = self.misses = self.saved_ticks = 0
        self.eyes_hits = self.eyes_misses = 0
        self.eyes_store: OrderedDict = eyes_store if eyes_store is not None else OrderedDict()
        self.n_eyes = 0
        if eyes_cache:
            self._wrap_eyes()
        for name in PRIMITIVES:
            if hasattr(self.mo, name):
                setattr(self.mo, name, self._wrap(name, getattr(self.mo, name)))

    # ---- what is captured: the live objects of the bed's process, by name

    def objects(self) -> dict:
        e = self.env
        bel = e.bel
        out = dict(sim=e.sim, mo=self.mo, pol=self.mo.pol, bel=bel, clock=e.clock)
        for n in ("body", "feed"):
            if hasattr(bel, n):
                out[n] = getattr(bel, n)
        body = getattr(bel, "body", None)
        if body is not None and getattr(body, "odo_root", None) is not None:
            out["odo_root"] = body.odo_root
        return out

    def _skip(self, name: str) -> tuple:
        if name == "pol":
            return POL_SKIP
        if name == "mo":
            return DELTA_LISTS + PRIMITIVES + ("tick", "reset", "row")
        if name in ("bel", "sim"):
            return ("row",)  # the record is not state: the lists a primitive grows are carried as deltas
        return ()

    def pack_all(self) -> dict:
        models = (self.env.sim.m,)
        return {
            n: SC.capture(o, skip=self._skip(n), owned=self.owned, models=models) for n, o in self.objects().items()
        }

    def apply(self, changes: dict) -> None:
        models = (self.env.sim.m,)
        objs = self.objects()
        for n, part in changes.items():
            SC.restore(objs[n], part, models=models)

    @staticmethod
    def diff(before: dict, after: dict) -> dict:
        """The attributes whose captured content differs: {object: {attribute: packed}}."""
        out: dict = {}
        for n, snap in after.items():
            old = before.get(n, {})
            ch = {
                k: p for k, p in snap.items() if k not in old or SC.fingerprint({k: p}) != SC.fingerprint({k: old[k]})
            }
            if ch:
                out[n] = ch
        return out

    # ---- perception at a state

    def _wrap_eyes(self) -> None:
        sim = self.env.sim
        eyes = sim.eyes

        def cached():
            if not self.live:
                return eyes()
            k = (self.key, self.n_eyes)
            self.n_eyes += 1
            hit = self.eyes_store.get(k)
            if hit is not None:
                self.eyes_hits += 1
                self.eyes_store.move_to_end(k)
                sim.last_eyes = dict(hit[2])
                return hit[0], hit[1]
            got = self._eyes_from_disk(k)
            if got is not None:
                self.eyes_disk_hits += 1
                self.eyes_store[k] = got
                sim.last_eyes = dict(got[2])
                return got[0], got[1]
            self.eyes_misses += 1
            out = eyes()
            self.eyes_store[k] = (out[0], out[1], dict(sim.last_eyes))
            self._eyes_to_disk(k, self.eyes_store[k])
            while len(self.eyes_store) > self.max_eyes:
                self.eyes_store.popitem(last=False)
            return out

        sim.eyes = cached

    def _eyes_from_disk(self, k):
        if self.disk is None:
            return None
        blob = self.disk.get(f"{k[0]}.e{k[1]}")
        if blob is None:
            return None
        try:
            rgb, depth, last = pickle.loads(zlib.decompress(blob))
            return rgb, depth, dict(last)
        except Exception:
            self.disk.corrupt += 1
            return None

    def _eyes_to_disk(self, k, val) -> None:
        if self.disk is None:
            return
        try:
            blob = zlib.compress(pickle.dumps(val, protocol=5), 3)
        except Exception:
            return
        if self.disk.put(f"{k[0]}.e{k[1]}", blob):
            self.eyes_persisted += 1

    # ---- the disk: a state as bytes, and back

    def _ref_hook(self, f, path):
        """A plain function without a closure (a class's method, set once) stays as it is on the other side. Motor.ref is a closure over the primitive's trajectory (and the motor); on disk it is the pose it holds, which is
        all a later primitive reads of it when it is constant over the policy's look-ahead (a finished primitive's is)."""
        from locomani.backends.scalebfm import motor as M

        m = self.mo
        if path != "mo.ref":
            if isinstance(f, types.FunctionType) and not f.__closure__:
                return SC.KEEP
            raise SC.Unpersistable(f"callable at {path}: {f!r}")
        try:
            p0 = f(m.t)
            if not isinstance(p0, M.Pose):
                raise SC.Unpersistable("callable that is not a reference")
            for j in [*[int(x) for x in m.pol.fidx], 1, 50]:
                q = f(m.t + j * 0.02)
                if any(
                    not np.array_equal(getattr(p0, a), getattr(q, a)) for a in ("pos", "quat", "joints", "velocity")
                ):
                    raise SC.Unpersistable("reference still moving")
        except SC.Unpersistable:
            raise
        except Exception as e:
            raise SC.Unpersistable(f"callable {e!r}") from e
        return copy.deepcopy(p0)

    @staticmethod
    def _ref_load(p0):
        return lambda _t, p=p0: p

    def _persist(self, key: str, e: Entry) -> None:
        if self.disk is None or self.disk.has(key):
            return
        try:
            self.link.send(("snap_dump", e.sid))
            server = self.link.recv()
            if not isinstance(server, bytes):
                raise SC.Unpersistable(f"server: {server}")
            client = {
                n: {a: SC.to_plain(p, self._ref_hook, True, f"{n}.{a}") for a, p in part.items()}
                for n, part in e.changes.items()
            }
            blob = zlib.compress(
                pickle.dumps(
                    dict(v=1, client=client, out=e.out, trace=e.trace, calls=e.calls, row=e.row, server=server),
                    protocol=5,
                ),
                3,
            )
        except SC.Unpersistable as ex:
            self.unpersistable += 1
            self.why_not[str(ex)[:200]] = self.why_not.get(str(ex)[:200], 0) + 1
            return
        except Exception as ex:  # a return value or a row that does not pickle
            self.unpersistable += 1
            self.why_not[repr(ex)[:200]] = self.why_not.get(repr(ex)[:200], 0) + 1
            return
        if self.disk.put(key, blob):
            self.persisted += 1

    def _from_disk(self, key: str) -> Entry | None:
        if self.disk is None:
            return None
        blob = self.disk.get(key)
        if blob is None:
            return None
        try:
            d = pickle.loads(zlib.decompress(blob))
            if d.get("v") != 1:
                raise SC.Unpersistable("version")
            classes = SC.class_map(self.owned)
            changes = {
                n: {a: SC.from_plain(x, classes, self._ref_load) for a, x in part.items()}
                for n, part in d["client"].items()
            }
            self.link.send(("snap_load", d["server"]))
            sid = self.link.recv()
            if not isinstance(sid, int):
                raise SC.Unpersistable(f"server: {sid}")
        except Exception:
            self.disk.corrupt += 1
            return None
        self.disk_hits += 1
        return Entry(changes, d["out"], d["trace"], d["calls"], d["row"], sid)

    # ---- the record's lists grown inside a primitive (reach_err_cm, ...)

    def row_now(self) -> dict:
        return {k: len(v) for k, v in self.env.row.items() if isinstance(v, list)}

    def row_delta(self, before: dict) -> dict:
        row = self.env.row
        return {
            k: copy.deepcopy(v[before.get(k, 0) :])
            for k, v in row.items()
            if isinstance(v, list) and len(v) > before.get(k, 0)
        }

    # ---- the primitives

    def _wrap(self, name: str, fn):
        def call(*args, **kw):
            m = self.mo
            if self.depth or not self.live:
                return fn(*args, **kw)
            if m.k != self.k_seen:  # stepped outside a primitive: the state is no longer a function of the commands
                self.live = False
                return fn(*args, **kw)
            self.key = next_key(self.key, name, args, kw)
            self.n_eyes = 0
            hit = self.store.get(self.key)
            if hit is None:
                hit = self._from_disk(self.key)
                if hit is not None:
                    self.store[self.key] = hit
                    self._bound()
            if hit is not None:
                self.store.move_to_end(self.key)
                try:
                    out = self._restore(hit)
                    self.hits += 1
                    return out
                except LookupError:  # the server no longer has that state (a new server, a new task): run it for real
                    del self.store[self.key]
            self.misses += 1
            before, n_row = self.pack_all(), self.row_now()
            n_trace, n_calls = len(m.trace), len(m.calls)
            k0 = m.k
            self.depth += 1
            try:
                out = fn(*args, **kw)
            finally:
                self.depth -= 1
            self.link.send(("snap",))
            sid = self.link.recv()
            after = self.pack_all()
            self.store[self.key] = Entry(
                self.diff(before, after), copy.deepcopy(out), m.trace[n_trace:],
                copy.deepcopy(m.calls[n_calls:]), self.row_delta(n_row), sid, after if self.check else None,
            )  # fmt: skip
            self.store[self.key].trace = list(self.store[self.key].trace)
            self.k_seen = m.k
            self._persist(self.key, self.store[self.key])
            self._bound()
            return out

        return call

    def _restore(self, e: Entry):
        m = self.mo
        self.saved_ticks += 0  # (the ticks saved are counted below from the restored k)
        k_before = m.k
        self.link.send(("restore", e.sid))
        r = self.link.recv()
        if isinstance(r, str) and r.startswith("restore failed"):
            raise RuntimeError(f"state {e.sid} left the server half restored: {r}")  # the episode cannot go on
        if r is not True:
            raise LookupError(f"the server could not restore state {e.sid}: {r}")
        self.apply(e.changes)
        m.trace.extend(e.trace)
        m.calls.extend(copy.deepcopy(e.calls))
        for k, items in e.row.items():
            self.env.row.setdefault(k, []).extend(copy.deepcopy(items))
        if self.check:  # debug: what differs from the state just stored, on both sides
            self.link.send(("snap_diff", e.sid))
            srv = self.link.recv()
            cli = (
                [f"{n}.{k}" for n, part in self.diff(self.pack_all(), e.after).items() for k in part] if e.after else []
            )
            self.diffs.append(dict(key=self.key[:8], server=srv, client=cli))
        self.saved_ticks += m.k - k_before
        self.k_seen = m.k
        return copy.deepcopy(e.out)

    def _bound(self) -> None:
        old = []
        while len(self.store) > self.max_states:
            _, e = self.store.popitem(last=False)
            old.append(e.sid)
        if old:
            self.link.send(("snap_free", old))
            self.link.recv()

    def stats(self) -> dict:
        return dict(
            hits=self.hits,
            misses=self.misses,
            eyes_hits=self.eyes_hits,
            eyes_misses=self.eyes_misses,
            saved_ticks=self.saved_ticks,
            stored=len(self.store),
            live=self.live,
            **self.disk_stats(),
        )

    def disk_stats(self) -> dict:
        if self.disk is None:
            return {}
        return dict(
            disk=dict(
                self.disk.stats(),
                state_hits=self.disk_hits,
                eyes_hits=self.eyes_disk_hits,
                persisted=self.persisted,
                eyes_persisted=self.eyes_persisted,
                unpersistable=self.unpersistable,
                why_not=self.why_not,
            )
        )


def install(env, label: str, **kw) -> Memo:
    env.memo = Memo(env, label, **kw)
    return env.memo
