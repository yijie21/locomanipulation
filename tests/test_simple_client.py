"""TODO 49.2: the SIMPLE client's reset contract (bench/simple/client.py OffSim.reset)."""

import unittest
from unittest import mock

from locomani.bench.simple import client


class Reset(unittest.TestCase):
    def test_reset_lays_the_environment_reset_state_over_the_local_default_pose(self):
        # the port first dropped this override, so Motor.reset's start hold ran on the local model's default pose
        calls = []
        with mock.patch.object(client.R.Sim, "reset", lambda self, *a: calls.append(("default", a))):
            sim = client.OffSim.__new__(client.OffSim)
            sim.pull = lambda st=None: calls.append(("pull", st))
            client.OffSim.reset(sim, (0.1, 0.2), 0.3, 0.0)
        self.assertEqual(calls, [("default", ((0.1, 0.2), 0.3, 0.0)), ("pull", None)])

    def test_every_client_sim_class_keeps_the_override(self):
        for cls in (c for c in vars(client).values() if isinstance(c, type) and issubclass(c, client.OffSim)):
            self.assertIs(cls.reset, client.OffSim.reset, cls.__name__)


if __name__ == "__main__":
    unittest.main()
