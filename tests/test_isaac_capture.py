import numpy as np

from locomani.sim.isaac import stage


def test_capture_does_not_reuse_an_earlier_step_for_a_missing_camera():
    class World:
        tick = 0

        def step(self, render=True):
            self.tick += 1

    class Sensor:
        def __init__(self, world, missing=False):
            self.world, self.missing = world, missing

        def get_rgba(self):
            if self.missing and self.world.tick == 2:
                return None
            return np.full((2, 2, 4), self.world.tick, dtype=np.uint8)

        def get_current_frame(self):
            return {"rendering_frame": self.world.tick, "rgba": self.get_rgba()}

    world = World()
    got = stage.step_and_grab_all(world, {"head": Sensor(world), "wrist": Sensor(world, True)})
    assert got["head"][0, 0, 0] == 2
    assert got["wrist"] is None


def test_nonempty_but_stale_or_differently_stamped_images_are_rejected():
    class World:
        tick = 10

        def step(self, render=True):
            self.tick += 1

    class Sensor:
        def __init__(self, stamp):
            self.stamp = stamp

        def get_current_frame(self):
            return {"rendering_frame": self.stamp(), "rgba": np.ones((2, 2, 4))}

    world = World()
    got = stage.step_and_grab_all(
        world,
        {
            "head": Sensor(lambda: world.tick),
            "stale": Sensor(lambda: 9),
            "lagging": Sensor(lambda: world.tick - 1),
        },
    )
    assert got["head"] is not None
    assert got["stale"] is None
    assert got["lagging"] is None
