"""Is the warehouse expert's 1/3-real-time sim a property of the scene or of the expert's camera reading?

Warehouse scene, one live stack (see warehouse_speed_probe_20260923.sh). Each trial is diag_brake.trial's guard
walk (SLOW_WALK 0.25 from the scene's robot_start, line --marker ahead = the box front at 0.65 m), timed in wall
and simulated seconds. The first --repeats trials read no cameras (the walk-scene probes' condition); the next
--repeats run a thread that reads the three MJPEG streams the way crate.expert does (ComposedCameraClientSensor).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from locomani import paths
from locomani.backends.sonic.diag_brake import trial
from locomani.backends.sonic.primitives import Primitives
from locomani.sim import ports
from locomani.sim.privileged_pub import EpisodeControlClient

GUARD = dict(extra=0.20, margin=0.03, stride0=0.14)


class CameraReader:
    def __init__(self):
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

        self.camera = ComposedCameraClientSensor(server_ip="localhost", port=ports.port("TF_CAMERA_PORT"))
        self.frames, self.stop = 0, threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            if self.camera.read() is not None:
                self.frames += 1
            time.sleep(0.002)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.camera.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--marker", type=float, default=0.65)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    rows = []
    for cameras in (False, True):
        reader = CameraReader() if cameras else None
        try:
            for i in range(a.repeats):
                label = f"speed_{'cam' if cameras else 'nocam'}_{i:02d}"
                s0, w0 = p.obs()["seq"], time.time()
                r = trial(p, reset, 0.25, a.marker, 3.0, label, guard=GUARD)
                s1, w1 = p.obs()["seq"], time.time()
                r.update(label=label, cameras=cameras, sim_s=(s1 - s0) / 50, wall_s=w1 - w0)
                r["ratio"] = r["sim_s"] / r["wall_s"]
                rows.append(r)
                print(
                    f"{label}: ratio {r['ratio']:.2f}  "
                    + (
                        f"brake v {r['brake_speed']:.2f} carry {100 * r['carry']:+.1f} cm "
                        f"stop {100 * (r['stop_fwd'] - a.marker):+.1f} lat {100 * r['stop_lat']:+.1f} cm "
                        f"yaw {r['yaw_deg']:+.1f} stride {r['brake_stride']:.2f}"
                        if "carry" in r
                        else str({k: r[k] for k in ("fell", "timeout") if k in r})
                    ),
                    flush=True,
                )
                (a.out / "results.json").write_text(json.dumps(dict(policy=paths.SONIC_POLICY, rows=rows), indent=1))
        finally:
            if reader is not None:
                print(f"camera reads: {reader.frames}", flush=True)
                reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
