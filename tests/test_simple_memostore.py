"""TODO 53: the memo's disk store (memostore.py), the snapshot <-> bytes format (statecopy.dumps / loads), and the memo reading
states another process wrote. No simulator."""

import multiprocessing as mp
import os
import pickle
import struct
import time
import types
import zlib

import numpy as np
import pytest
from test_simple_memo import FakeServer, make_env

from locomani.bench.simple import memo as M
from locomani.bench.simple import memostore as MS
from locomani.bench.simple import statecopy as SC


# ---------------------------------------------------------------------------------------------- the store
def test_round_trip_and_stats(tmp_path):
    s = MS.DiskStore(tmp_path, "sig")
    assert s.get("abcd") is None
    assert s.put("abcd", b"payload") and s.has("abcd")
    assert s.get("abcd") == b"payload"
    st = s.stats()
    assert (st["hits"], st["misses"], st["writes"], st["corrupt"]) == (1, 1, 1, 0)
    assert st["bytes_written"] == st["bytes_read"] == MS.HEAD.size + len(b"payload")
    assert MS.DiskStore(tmp_path, "sig").get("abcd") == b"payload"  # another process, same signature


def test_a_changed_signature_reaches_no_old_state(tmp_path):
    MS.DiskStore(tmp_path, "sig-a").put("abcd", b"old")
    assert MS.DiskStore(tmp_path, "sig-b").get("abcd") is None
    assert MS.DiskStore(tmp_path, "sig-a").get("abcd") == b"old"


@pytest.mark.parametrize("damage", ["truncate", "flip", "magic", "empty", "extra"])
def test_a_corrupt_or_partial_file_is_a_miss(tmp_path, damage):
    s = MS.DiskStore(tmp_path, "sig")
    s.put("abcd", b"x" * 1000)
    f = s.path("abcd")
    raw = f.read_bytes()
    f.write_bytes(
        {
            "truncate": raw[:500],
            "flip": raw[:600] + bytes([raw[600] ^ 1]) + raw[601:],
            "magic": b"XXXX" + raw[4:],
            "empty": b"",
            "extra": raw + b"junk",
        }[damage]
    )
    assert s.get("abcd") is None
    assert s.corrupt == 1 and not f.exists()  # counted, and the bad file is gone
    assert s.put("abcd", b"again") and s.get("abcd") == b"again"


def test_no_temporary_file_is_left_and_a_failed_write_is_a_false(tmp_path, monkeypatch):
    s = MS.DiskStore(tmp_path, "sig")
    s.put("abcd", b"1")
    assert [p.name for p in s.dir.rglob("*") if p.is_file()] == ["abcd"]
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    assert s.put("ffff", b"2") is False
    assert not [p for p in s.dir.rglob("*") if p.is_file() and p.name.startswith(MS.TMP)]


def _writer(root, name, tag, n):
    s = MS.DiskStore(root, "sig", prune_every=10**9)
    for i in range(n):
        s.put(name, (tag * 200_000 + str(i)).encode())


def test_concurrent_writers_and_readers_never_see_a_partial_file(tmp_path):
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=_writer, args=(tmp_path, "abcd", chr(97 + k), 40)) for k in range(3)]
    for p in ps:
        p.start()
    reader = MS.DiskStore(tmp_path, "sig")
    seen = 0
    t0 = time.time()
    while any(p.is_alive() for p in ps) and time.time() - t0 < 60:
        got = reader.get("abcd")
        if got is not None:
            seen += 1
            assert got[:200_000] == got[:1] * 200_000  # one writer's whole payload
    for p in ps:
        p.join()
        assert p.exitcode == 0
    assert reader.corrupt == 0 and seen > 0
    assert not [p for p in reader.dir.rglob("*") if p.is_file() and p.name.startswith(MS.TMP)]


def test_pruning_drops_the_oldest_first_and_keeps_what_was_read(tmp_path):
    s = MS.DiskStore(tmp_path, "sig", max_bytes=10_000, prune_every=10**9)
    for i in range(10):
        s.put(f"{i:02d}aa", b"x" * 1500)  # 10 x 1.5 KB = 15 KB over a 10 KB cap
        os.utime(s.path(f"{i:02d}aa"), (1000 + i, 1000 + i))
    assert s.get("00aa") is not None  # a hit refreshes the file: it is the newest now
    freed = s.prune()
    left = sorted(p.name for p in s.dir.rglob("*aa"))
    assert freed > 0 and s.size() <= 9_000 and s.pruned >= 1
    assert "00aa" in left and "01aa" not in left  # the oldest unread one went first
    assert s.prune() == 0  # under the cap: nothing


def test_the_cap_covers_every_signature_and_prune_runs_while_writing(tmp_path):
    MS.DiskStore(tmp_path, "old").put("aaaa", b"y" * 6000)
    os.utime(MS.DiskStore(tmp_path, "old").path("aaaa"), (1, 1))
    s = MS.DiskStore(tmp_path, "new", max_bytes=10_000, prune_every=2)
    for i in range(4):
        s.put(f"{i}bbb", b"z" * 2000)
    assert not MS.DiskStore(tmp_path, "old").has("aaaa") and s.size() <= 10_000


def test_old_temporary_files_are_removed_by_prune(tmp_path):
    s = MS.DiskStore(tmp_path, "sig")
    t = s.dir / "ab" / f"{MS.TMP}1.2.abcd"
    t.parent.mkdir()
    t.write_bytes(b"half")
    os.utime(t, (1, 1))
    s.prune()
    assert not t.exists()


# ---------------------------------------------------------------------------------------------- snapshot <-> bytes
class Owned:
    def __init__(self):
        self.a, self.b = np.arange(3.0), [1, 2]


def test_snapshot_bytes_round_trip_in_place():
    live_handle = object()
    o = types.SimpleNamespace(
        x=np.arange(4.0),
        y=[1, (2.0, "s")],
        z={"k": np.ones(2)},
        h=live_handle,
        hs=[live_handle, live_handle],
        own=Owned(),
    )
    snap = {"o": SC.capture(o, owned=(Owned,))}
    back = SC.loads(SC.dumps(snap), owned=(Owned,))
    o.x[:] = 0
    o.y[0] = 9
    o.z["k"][:] = 5
    o.own.a[:] = 7
    o.own.b.append(3)
    live_ids = (id(o.x), id(o.hs[0]), id(o.own))
    SC.restore(o, back["o"])
    assert list(o.x) == [0, 1, 2, 3] and o.y == [1, (2.0, "s")] and list(o.z["k"]) == [1, 1]
    assert list(o.own.a) == [0, 1, 2] and o.own.b == [1, 2]
    assert o.h is live_handle and o.hs[0] is live_handle and (id(o.x), id(o.hs[0]), id(o.own)) == live_ids


def test_a_callable_needs_the_hook_and_the_hook_may_keep_the_live_one():
    o = types.SimpleNamespace(f=lambda: 1, n=2)
    snap = {"o": SC.capture(o)}
    with pytest.raises(SC.Unpersistable, match="callable"):
        SC.dumps(snap)
    with pytest.raises(SC.Unpersistable):
        SC.dumps(snap, fn=lambda f, path: (_ for _ in ()).throw(SC.Unpersistable(path)))
    live = o.f
    back = SC.loads(SC.dumps(snap, fn=lambda f, path: SC.KEEP))
    o.n = 5
    SC.restore(o, back["o"])
    assert o.n == 2 and o.f is live
    back = SC.loads(SC.dumps(snap, fn=lambda f, path: ("const", f())), fn=lambda x: lambda: x[1] + 10)
    SC.restore(o, back["o"])
    assert o.f() == 11


def test_a_handle_as_a_dict_key_or_a_unknown_class_is_refused():
    h = object()
    snap = {"o": SC.capture(types.SimpleNamespace(d={h: 1}))}
    with pytest.raises(SC.Unpersistable):
        SC.dumps(snap)
    blob = SC.dumps({"o": SC.capture(types.SimpleNamespace(w=Owned()), owned=(Owned,))})
    with pytest.raises(SC.Unpersistable):  # the reader does not know the class
        SC.loads(blob, owned=())


def test_garbage_bytes_are_unpersistable_not_a_crash():
    for blob in (b"", b"abc", zlib.compress(b"not a pickle"), zlib.compress(pickle.dumps((99, {})))):
        with pytest.raises(SC.Unpersistable):
            SC.loads(blob)


# ---------------------------------------------------------------------------------------------- the memo on the disk
class DiskServer(FakeServer):
    """The fake server with snap_dump / snap_load: its saved states as bytes."""

    def send(self, msg):
        if msg[0] == "snap_dump":
            self.reply = pickle.dumps(self.saved[msg[1]]) if msg[1] in self.saved else "no state"
        elif msg[0] == "snap_load":
            self.n += 1
            self.saved[self.n] = pickle.loads(msg[1])
            self.reply = self.n
        else:
            super().send(msg)


def script(env, speed=0.5):
    env.bel.flag = speed
    return [env.mo.walk_to((1.0, 2.0), speed=speed), env.mo.reach("right", (0.1, 0.0)), env.mo.hold(0.1)]


def test_a_fresh_process_restores_what_another_stored(tmp_path):
    env, srv = make_env(DiskServer())
    a = M.install(env, "start", disk=MS.DiskStore(tmp_path, "sig"))
    out_a = script(env)
    end_a = (
        srv.state["x"],
        env.mo.pose.copy(),
        env.mo.k,
        list(env.mo.trace),
        dict(env.row),
        env.mo.pol.buf["h"].copy(),
    )
    assert a.persisted == 3 and a.unpersistable == 0 and a.hits == 0

    env2, srv2 = make_env(DiskServer())  # another process: its own server, its own (empty) memory
    b = M.install(env2, "start", disk=MS.DiskStore(tmp_path, "sig"))
    out_b = script(env2)
    assert b.hits == 3 and b.misses == 0 and b.disk_hits == 3 and b.saved_ticks == env.mo.k
    assert out_b == out_a
    assert (srv2.state["x"], env2.mo.k, list(env2.mo.trace), dict(env2.row)) == (end_a[0], end_a[2], end_a[3], end_a[4])
    assert np.array_equal(env2.mo.pose, end_a[1]) and np.array_equal(env2.mo.pol.buf["h"], end_a[5])
    assert b.stats()["disk"]["state_hits"] == 3 and b.stats()["disk"]["bytes_read"] > 0


def test_a_variant_in_a_fresh_process_hits_the_shared_beginning(tmp_path):
    env, _ = make_env(DiskServer())
    M.install(env, "start", disk=MS.DiskStore(tmp_path, "sig"))
    script(env, 0.5)
    env2, srv2 = make_env(DiskServer())
    b = M.install(env2, "start", disk=MS.DiskStore(tmp_path, "sig"))
    env2.bel.flag = 0.5
    env2.mo.walk_to((1.0, 2.0), speed=0.5)  # shared
    env2.mo.reach("right", (0.2, 0.0))  # a variant
    assert b.hits == 1 and b.misses == 1
    assert b.persisted == 1  # the variant's state is stored too


def test_a_signature_change_or_a_corrupt_state_runs_for_real(tmp_path):
    env, _ = make_env(DiskServer())
    M.install(env, "start", disk=MS.DiskStore(tmp_path, "sig-a"))
    script(env)
    env2, _ = make_env(DiskServer())
    b = M.install(env2, "start", disk=MS.DiskStore(tmp_path, "sig-b"))
    script(env2)
    assert b.hits == 0 and b.misses == 3
    store = MS.DiskStore(tmp_path, "sig-a")
    for f in store.dir.rglob("*"):
        if f.is_file():
            f.write_bytes(f.read_bytes()[:-5])  # every file truncated
    env3, _ = make_env(DiskServer())
    c = M.install(env3, "start", disk=MS.DiskStore(tmp_path, "sig-a"))
    out = script(env3)
    assert c.hits == 0 and c.misses == 3 and out[0]["ok"]


def test_a_state_the_server_cannot_dump_stays_in_memory(tmp_path):
    class NoDump(DiskServer):
        def send(self, msg):
            if msg[0] == "snap_dump":
                self.reply = "unpersistable: a callable"
            else:
                super().send(msg)

    env, _ = make_env(NoDump())
    a = M.install(env, "start", disk=MS.DiskStore(tmp_path, "sig"))
    script(env)
    assert a.persisted == 0 and a.unpersistable == 3 and a.why_not
    assert not [f for f in MS.DiskStore(tmp_path, "sig").dir.rglob("*") if f.is_file()]


def test_eyes_are_read_from_disk_by_the_next_process(tmp_path):
    calls = []

    def eyes():
        calls.append(1)
        return np.full(2, float(len(calls))), np.zeros(2)

    env, _ = make_env(DiskServer())
    env.sim.eyes = eyes
    M.install(env, "start", disk=MS.DiskStore(tmp_path, "sig"))
    first = env.sim.eyes()
    env2, _ = make_env(DiskServer())
    env2.sim.eyes = eyes
    b = M.install(env2, "start", disk=MS.DiskStore(tmp_path, "sig"))
    again = env2.sim.eyes()
    assert len(calls) == 1 and np.array_equal(first[0], again[0]) and b.eyes_disk_hits == 1
