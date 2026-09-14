"""Exercise the complete two-pass consumer without requiring an Isaac runtime."""

import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from locomani.sim.isaac import isaac_render as render
from locomani.sim.isaac import replay


@pytest.mark.parametrize("missing", ["warmup", "frame", None])
def test_all_requested_insets_are_required_on_every_written_frame(tmp_path, monkeypatch, missing):
    state = SimpleNamespace(t=0, calls=0)

    class Replay:
        _obj = {}

        def __init__(self, *args):
            pass

        def set(self, qpos, qvel):
            state.t = int(qpos[0])

        def dof_positions(self, *args):
            return []

        def body_pose(self, *args):
            return [0, 0, 0, 1, 0, 0, 0]

        def object_poses(self):
            return {}

        def camera_pose(self, name):
            return [0, 0, 0], [1, 0, 0, 0]

    class Writer:
        frames = []
        released = False

        def __init__(self, *args):
            pass

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    layout = render.dataset_inset_layout(720, 540)
    sensors = {name: SimpleNamespace(set_world_pose=lambda **kwargs: None) for name in layout}
    colours = dict(zip(layout, [60, 120, 180]))

    def grab_all(world, sensors):
        state.calls += 1
        # Head is quiet from the start; wrist images settle later. Warm-up must
        # consider all cameras before writing anything.
        warm_change = max(0, 4 - state.calls) * 4
        return {
            name: None
            if name == "right_wrist" and (missing == "warmup" or (missing == "frame" and state.t == 1))
            else np.full(
                (layout[name][2], layout[name][1], 4),
                colours[name] + state.t + (warm_change if name != "head_camera" else 0),
                dtype=np.uint8,
            )
            for name in sensors
        }

    patches = {
        "assemble_stage": lambda *a: object(),
        "make_world": lambda: object(),
        "bind_robot": lambda *a: (SimpleNamespace(dof_names=[]), [], []),
        "mount_camera": lambda *a: (object(), "/World/ego_cam"),
        "mount_inset_cameras": lambda *a: {n: (s, n) for n, s in sensors.items()},
        "root_body": lambda *a: "torso_link",
        "sync_replay": lambda *a, **kw: None,
        "step_and_grab": lambda *a: np.full((540, 720, 4), 20 + state.t, dtype=np.uint8),
        "step_and_grab_all": grab_all,
    }
    for name, value in patches.items():
        monkeypatch.setattr(render.S, name, value)
    monkeypatch.setattr(replay, "MujocoReplay", Replay)
    monkeypatch.setattr(cv2, "VideoWriter", Writer)
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(UsdGeom=object()))
    args = SimpleNamespace(
        out=str(tmp_path),
        scene_xml="",
        ghost_trace="",
        debug_cam=False,
        end_frame=0,
        cam_traj="",
        width=720,
        height=540,
        fps=50,
        stride=1,
    )
    messages = []

    def run():
        render._render(
            args,
            render.S.StageCfg(),
            [],
            None,
            None,
            None,
            None,
            messages.append,
            qpos=np.array([[0], [1]]),
            seq=np.arange(2),
            inset_specs={n: () for n in layout},
            inset_layout=layout,
        )

    if missing:
        with pytest.raises(RuntimeError, match="warm-up" if missing == "warmup" else "frame 1.*right_wrist"):
            run()
        assert "done" not in messages
        assert len(Writer.frames) == (0 if missing == "warmup" else 1)
    else:
        run()
        assert messages[-1] == "done"
        assert state.calls >= 9  # 7 warm-up/settle samples plus two source frames
        assert len(Writer.frames) == 2
        for t, frame in enumerate(Writer.frames):
            assert (frame[300, 360] == 20 + t).all()
            for name, (_, w, h, x, y) in layout.items():
                assert (frame[y + h // 2, x + w // 2] == colours[name] + t).all()
    assert not list(tmp_path.glob("main_*"))
