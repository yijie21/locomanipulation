"""The one Unreal renderer both drivers call: a MuJoCo state in, the dataset's camera images out.

The Unreal counterpart of `locomani.sim.isaac.view.IsaacView`, with the same two methods: the eval mirror
(`unreal.mirror`) renders the sim's newest state with it and the dataset re-render
(`unreal.render_dataset_cameras`) a trace's rows, so training and evaluation see one function of the state.

URLab runs in puppet mode: a qpos (the run's scene, mapped onto URLab's import joint by joint name) is pushed with
zero steps, then the three dataset cameras are captured in the patched plugin's "fresh" mode: a camera cut, then
`repeats` captures of the frozen pose (urlab.FreshCaptureRepeats) for Lumen and TSR to converge, then a synchronous
readback (patches/urlab_fresh_capture.md). Every request is a jump for the renderer's history, as the mirror's
requests are 0.4 s apart and the dataset's rows would otherwise depend on the rows before them.

The editor must hold the scene's level (`build_level`) and be in its render defaults (`editor.require_render_defaults`);
the view loads the level, sets the capture count and starts PIE itself. `close()` stops PIE before it lets go of
the bridge (editor.stop_pie says why).
"""

from __future__ import annotations

import mujoco
import numpy as np

from locomani.sim.dataset_cameras import CAMERA_CONFIGS
from locomani.sim.unreal import editor

WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}  # mjtJoint: free, ball, slide, hinge


def sim_joint(name: str, sim_names) -> str | None:
    """The sim joint a URLab joint `name` is: the same name, else the LONGEST sim name it ends with after "_" or "/"
    (URLab prefixes its actor). The longest, not the first: with both `hip_joint` and `left_hip_joint` in the sim,
    `g1_left_hip_joint` is the second."""
    if name in sim_names:
        return name
    hits = [k for k in sim_names if name.endswith("_" + k) or name.endswith("/" + k)]
    return max(hits, key=len) if hits else None


def qpos_map(sim_model, urlab_model) -> list[tuple[int, int, int]]:
    """(sim adr, URLab adr, width) per URLab joint. URLab prefixes names with its actor, so match by suffix;
    a URLab joint the sim does not have means the level was built from another scene."""
    sim = {
        sim_model.joint(j).name: (int(sim_model.jnt_qposadr[j]), WIDTH[int(sim_model.jnt_type[j])])
        for j in range(sim_model.njnt)
    }
    out = []
    for j in range(urlab_model.njnt):
        name = urlab_model.joint(j).name
        key = sim_joint(name, sim)
        if key is None:
            raise SystemExit(f"URLab joint {name!r} is not in the sim scene: the level was built from another scene")
        adr, width = sim[key]
        out.append((adr, int(urlab_model.jnt_qposadr[j]), width))
    return out


class UnrealView:
    """A running editor's level mirroring one MuJoCo scene, read through the dataset's cameras."""

    def __init__(self, scene_xml: str, level_path: str, repeats: int, streams=tuple(CAMERA_CONFIGS), log=print):
        self.streams = tuple(streams)
        self.log = log
        editor.load_level(level_path)
        editor.checked(
            editor.uecmd(f"unreal.SystemLibrary.execute_console_command(None, 'urlab.FreshCaptureRepeats {repeats}')")
        )
        editor.require_render_defaults()
        client = editor.connect(local_model=False)
        if not client.manager_present:
            res = client.sim.start(raise_on_failure=False, timeout_s=300)
            if "READY" not in str(res.state):
                raise SystemExit(f"PIE did not start in {level_path}: {res.state} {res.compile_error[:300]!r}")
        client.local_model = True
        client.connect()
        if client.model is None:
            raise SystemExit("the URLab handshake carried no model (the bridge venv's mujoco must be URLab's, 3.10.0)")
        client.runtime.set_mode("puppet")
        self.client = client
        names = client.camera_names()
        self.all_cameras = names
        self.cameras = {}
        for k in self.streams:
            mjcf = CAMERA_CONFIGS[k]["mjcf_name"]
            found = [n for n in names if n.split("/")[-1].endswith(mjcf)]
            if len(found) != 1:
                raise SystemExit(f"stream {k}: {len(found)} URLab cameras end in {mjcf!r} among {names}")
            self.cameras[k] = found[0]
        sim = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.nq = sim.nq
        self.map = qpos_map(sim, client.model)
        self.data = client.data
        log(f"unreal view: level {level_path}, {repeats} captures a request, cameras {self.cameras}")

    def _push(self, include: dict | None = None) -> dict:
        request = {
            "mode": "puppet",
            "n_steps": 0,
            "observations": "minimal",
            "time": float(self.data.time),
            "qpos": np.asarray(self.data.qpos, dtype=np.float64).tolist(),
            "qvel": [0.0] * self.client.model.nv,
            "per_articulation": {},
        }
        if include:
            request["include_cameras"] = include
        return self.client._rpc("step", request, expected_op="step_ok")

    def push(self, qpos) -> None:
        """One MuJoCo state into the level: the URLab import's joints set by name, forward kinematics, one puppet step."""
        qpos = np.asarray(qpos, dtype=np.float64)
        if len(qpos) != self.nq:
            raise RuntimeError(f"a qpos of {len(qpos)} values for a scene of {self.nq}")
        for s, d, w in self.map:
            self.data.qpos[d : d + w] = qpos[s : s + w]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.client.model, self.data)
        self._push()

    def grab(self, cameras: dict | None = None) -> dict:
        """`{stream: HxWx3 uint8}` captured fresh from the pose last pushed. Raises rather than return a partial set."""
        cameras = cameras or self.cameras
        reply = self._push({n: "fresh" for n in cameras.values()})
        out = {}
        for k, name in cameras.items():
            blk = (reply.get("cameras") or {}).get(name)
            if blk is None:
                raise RuntimeError(f"URLab returned no frame for {name}")
            px = np.frombuffer(blk["data"], np.uint8).reshape(int(blk["height"]), int(blk["width"]), 4)
            out[k] = np.ascontiguousarray(px[..., [2, 1, 0]])  # BGRA -> RGB
        return out

    def render(self, qpos) -> dict:
        """The images of this state."""
        self.push(qpos)
        return self.grab()

    def warm_up(self, qpos) -> int:
        """Capture every camera of the import once (each then renders only on request; before its first capture
        a camera renders every frame), render this state once, and check the dataset resolution; returns 2."""
        self.push(qpos)
        self.grab({n: n for n in self.all_cameras})
        images = self.render(qpos)
        for k, img in images.items():
            want = (CAMERA_CONFIGS[k]["height"], CAMERA_CONFIGS[k]["width"], 3)
            if img.shape != want:
                raise SystemExit(f"stream {k}: URLab renders {img.shape}, the dataset wants {want}; rebuild the level")
        return 2

    def close(self) -> None:
        """Stop PIE, then let go of the bridge (close() alone would switch a running PIE back to live)."""
        import time

        if self.client.manager_present:
            self.client.sim.stop()
            time.sleep(2.0)
        self.client.close()
