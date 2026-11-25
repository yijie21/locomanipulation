"""TODO 52 mechanism 2: replay from a saved state -- key chaining, the stored-state round trip, the bounds and fallbacks of memo.py,
with a fake Motor and a fake environment server (no simulator, no GPU)."""

import types
import unittest
from collections import OrderedDict

import numpy as np

from locomani.bench.simple import memo as M


class FakeServer:
    """The server's side of the socket: `snap` stores a copy of its state, `restore` puts it back."""

    def __init__(self):
        self.state, self.saved, self.n = {"x": 0.0}, {}, 0
        self.sent = []

    def send(self, msg):
        self.sent.append(msg[0])
        k = msg[0]
        if k == "snap":
            self.n += 1
            self.saved[self.n] = dict(self.state)
            self.reply = self.n
        elif k == "restore":
            self.reply = True if msg[1] in self.saved else f"no state {msg[1]}"
            if msg[1] in self.saved:
                self.state = dict(self.saved[msg[1]])
        elif k == "snap_free":
            for i in msg[1]:
                self.saved.pop(i, None)
            self.reply = True
        elif k == "snap_diff":
            self.reply = []

    def recv(self):
        return self.reply


class FakeMotor:
    def __init__(self, server):
        self.server, self.k, self.trace, self.calls = server, 0, [], []
        self.pose = np.zeros(2)
        self.pol = types.SimpleNamespace(buf={"h": np.zeros(3)}, eager_calls=3)
        self.row = {}

    def _steps(self, n, dx):
        for _ in range(n):
            self.k += 1
            self.pose = self.pose + dx
            self.server.state["x"] += float(dx[0])
            self.trace.append(self.k)
            self.pol.buf["h"] = self.pol.buf["h"] + 1
        self.calls.append(dict(k=self.k))

    def walk_to(self, xy, speed=0.5):
        self._steps(5, np.asarray(xy, float) * speed)
        self.row.setdefault("walks", []).append(float(xy[0]))
        return dict(ok=True, k=self.k)

    def reach(self, side, p):
        self._steps(3, np.asarray(p, float))
        return dict(ok=True, k=self.k, side=side)

    def hold(self, seconds):
        self._steps(int(seconds * 50), np.zeros(2))
        return dict(ok=True, k=self.k)

    def adopt_hand(self, side):
        self.k += 0
        return None

    turn = sidestep = posture = grip = release = squeeze = walk_to


def make_env(server=None):
    server = server or FakeServer()
    mo = FakeMotor(server)
    sim = types.SimpleNamespace(link=server, m=None, last=None, eyes=lambda: (np.zeros(2), np.zeros(2)), steps=0)
    sim.last_eyes = {}
    bel = types.SimpleNamespace(flag=0, pb=np.zeros(2), row=mo.row)
    clock = types.SimpleNamespace(steps=0)
    return types.SimpleNamespace(sim=sim, mo=mo, bel=bel, clock=clock, row=mo.row), server


class Keys(unittest.TestCase):
    def test_chain_is_order_and_argument_exact(self):
        a = M.next_key(M.start_key("s"), "walk_to", ((1.0, 2.0),), {"speed": 0.5})
        self.assertEqual(a, M.next_key(M.start_key("s"), "walk_to", ((1.0, 2.0),), {"speed": 0.5}))
        for other in (
            M.next_key(M.start_key("s2"), "walk_to", ((1.0, 2.0),), {"speed": 0.5}),  # another start
            M.next_key(M.start_key("s"), "turn", ((1.0, 2.0),), {"speed": 0.5}),  # another primitive
            M.next_key(M.start_key("s"), "walk_to", ((1.0, np.nextafter(2.0, 3.0)),), {"speed": 0.5}),  # one ulp
            M.next_key(M.start_key("s"), "walk_to", ((1.0, 2.0),), {"speed": 0.6}),
        ):
            self.assertNotEqual(a, other)
        ab = M.next_key(M.next_key(M.start_key("s"), "a", (), {}), "b", (), {})
        ba = M.next_key(M.next_key(M.start_key("s"), "b", (), {}), "a", (), {})
        self.assertNotEqual(ab, ba)

    def test_arrays_and_numpy_scalars_key_by_value(self):
        k1 = M.next_key("k", "reach", ("right", np.array([0.1, 0.2])), {})
        k2 = M.next_key("k", "reach", ("right", np.array([0.1, 0.2])), {})
        k3 = M.next_key("k", "reach", ("right", np.array([0.1, 0.2000001])), {})
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(M.plain(np.float64(0.5)), M.plain(0.5))


class RoundTrip(unittest.TestCase):
    def script(self, env, speed=0.5, reach=(0.1, 0.0)):
        env.bel.flag = speed  # what the skill's Python sets between primitives
        out = [env.mo.walk_to((1.0, 2.0), speed=speed), env.mo.reach("right", reach), env.mo.hold(0.1)]
        return out

    def test_second_round_is_all_hits_and_ends_in_the_same_state(self):
        env, srv = make_env()
        memo = M.install(env, "start")
        a = self.script(env)
        end_a = (
            srv.state["x"],
            env.mo.pose.copy(),
            env.mo.k,
            list(env.mo.trace),
            dict(env.row),
            env.mo.pol.buf["h"].copy(),
        )
        # a fresh round: the world back at its start, the same chain label
        env.mo.k, env.mo.trace, env.mo.pose, env.mo.calls = 0, [], np.zeros(2), []
        env.mo.pol.buf["h"] = np.zeros(3)
        env.row.clear()
        srv.state["x"] = 0.0
        memo.key, memo.k_seen = M.start_key("start"), 0
        b = self.script(env)
        self.assertEqual((memo.hits, memo.misses), (3, 3))
        self.assertEqual(a, b)
        self.assertEqual(srv.state["x"], end_a[0])
        np.testing.assert_array_equal(env.mo.pose, end_a[1])
        self.assertEqual((env.mo.k, env.mo.trace, dict(env.row)), (end_a[2], end_a[3], end_a[4]))
        np.testing.assert_array_equal(env.mo.pol.buf["h"], end_a[5])
        self.assertEqual(len(env.mo.calls), 3)
        self.assertEqual(memo.saved_ticks, env.mo.k)

    def test_variant_hits_the_shared_beginning_and_runs_the_rest(self):
        env, srv = make_env()
        memo = M.install(env, "start")
        self.script(env)
        env.mo.k, env.mo.trace, env.mo.pose = 0, [], np.zeros(2)
        srv.state["x"] = 0.0
        memo.key, memo.k_seen = M.start_key("start"), 0
        self.script(env, reach=(0.2, 0.0))  # the second primitive differs
        self.assertEqual((memo.hits, memo.misses), (1 + 0, 3 + 2))
        self.assertEqual(env.mo.k, 5 + 3 + 5)

    def test_python_set_between_primitives_is_not_overwritten_by_a_stored_state(self):
        env, srv = make_env()
        memo = M.install(env, "start")
        self.script(env, speed=0.5)
        env.mo.k, env.mo.trace, env.mo.pose = 0, [], np.zeros(2)
        memo.key, memo.k_seen = M.start_key("start"), 0
        env.bel.flag = "set by the variant's python"
        env.mo.walk_to((1.0, 2.0), speed=0.5)  # a hit; the stored run had flag 0.5 at that time
        self.assertEqual(env.bel.flag, "set by the variant's python")

    def test_stepping_outside_a_primitive_breaks_the_chain(self):
        env, srv = make_env()
        memo = M.install(env, "start")
        env.mo.walk_to((1.0, 2.0))
        env.mo.k += 1  # a hand program's tick
        env.mo.walk_to((1.0, 2.0))
        self.assertFalse(memo.live)
        self.assertEqual(memo.misses, 1)
        self.assertEqual(len(memo.store), 1)

    def test_a_primitive_that_raises_is_not_stored(self):
        env, srv = make_env()
        memo = M.install(env, "start")

        def boom(xy):
            raise RuntimeError("Done")

        env.mo.turn = memo._wrap("turn", boom)
        with self.assertRaises(RuntimeError):
            env.mo.turn((1.0, 0.0))
        self.assertEqual(len(memo.store), 0)
        self.assertEqual(memo.depth, 0)

    def test_states_are_bounded_on_both_sides_and_a_lost_state_runs_for_real(self):
        env, srv = make_env()
        memo = M.install(env, "start", max_states=2)
        for i in range(4):
            env.mo.hold(0.02 * (i + 1))
        self.assertEqual(len(memo.store), 2)
        self.assertEqual(len(srv.saved), 2)
        self.assertIn("snap_free", srv.sent)
        # the server forgets everything (a new server): the next hit falls back to running the primitive
        srv.saved.clear()
        env.mo.k, env.mo.trace = 0, []
        memo.key, memo.k_seen = M.start_key("start"), 0
        memo.store = OrderedDict((k, v) for k, v in memo.store.items())
        k_before = env.mo.k
        env.mo.hold(0.02 * 3)  # the 3rd hold's state is in the store, but not on the server
        self.assertGreater(env.mo.k, k_before)  # it stepped

    def test_eyes_are_stored_with_the_state(self):
        env, srv = make_env()
        calls = []

        def eyes():
            calls.append(1)
            env.sim.last_eyes = {"n": len(calls)}
            return np.full(2, float(len(calls))), np.zeros(2)

        env.sim.eyes = eyes
        store = OrderedDict()
        memo = M.install(env, "start", eyes_store=store)
        first = env.sim.eyes()
        env.mo.hold(0.02)
        second = env.sim.eyes()
        # a new round of the same start
        memo.key, memo.n_eyes, memo.k_seen = M.start_key("start"), 0, env.mo.k
        again = env.sim.eyes()
        np.testing.assert_array_equal(again[0], first[0])
        self.assertEqual(env.sim.last_eyes, {"n": 1})
        self.assertEqual((memo.eyes_hits, memo.eyes_misses), (1, 2))
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(float(second[0][0]), float(first[0][0]))


if __name__ == "__main__":
    unittest.main()
