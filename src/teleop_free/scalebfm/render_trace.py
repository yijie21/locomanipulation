"""Render a state trace (qpos at 50 Hz) with the spike's third-person camera.

    MUJOCO_GL=egl python render_trace.py <trace.npz> <out.mp4> [start_s] [end_s]
Same Sim.frame() as run_spike, so SONIC and ScaleBFM clips are comparable.
"""
import json, os, subprocess, sys
import mujoco, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_spike as R

tr = np.load(sys.argv[1])["qpos"]
out = sys.argv[2]
k0 = int(float(sys.argv[3]) * 50) if len(sys.argv) > 3 else 0
k1 = int(float(sys.argv[4]) * 50) if len(sys.argv) > 4 else len(tr)
sim = R.Sim(json.load(open(R.META)), video=True)
ffm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
                        "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], stdin=subprocess.PIPE)
for k in range(k0, min(k1, len(tr)), 2):
    sim.d.qpos[:] = tr[k]; mujoco.mj_forward(sim.m, sim.d)
    ffm.stdin.write(sim.frame().tobytes())
ffm.stdin.close(); ffm.wait()
print(out, (min(k1, len(tr)) - k0) / 50, "s")
