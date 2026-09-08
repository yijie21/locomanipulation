"""Side-by-side of the two envelope sweeps, ScaleBFM left, SONIC right.

SONIC's loop runs on wall-clock and lags 50 Hz, so its trace is retimed per
target from envelope.json's t_s: each target's span is resampled to 2 s (50
frames at 25 fps), which is exactly the ScaleBFM clip's schedule.
    MUJOCO_GL=egl python render_envelope_sbs.py <sonic_dir> <scalebfm_mp4> <out.mp4>
"""
import json, os, subprocess, sys, glob
import mujoco, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_spike as R

sonic_dir, bfm_mp4, out = sys.argv[1:4]
rows = json.load(open(os.path.join(sonic_dir, "envelope.json")))
tr = np.load(sorted(glob.glob(os.path.join(sonic_dir, "trace", "envelope*_state_trace.npz")))[0])["qpos"]
print("trace", tr.shape, "rows", len(rows))
idx = []
t_prev = 0.0
scale = (len(tr) - 1) / rows[-1]["t_s"]      # sim clock vs wall clock: map the schedule onto the trace length
for r in rows:
    k0, k1 = int(t_prev * scale), min(int(r["t_s"] * scale), len(tr) - 1)
    idx.extend(np.linspace(k0, k1, 50).astype(int).tolist())
    t_prev = r["t_s"]
sim = R.Sim(json.load(open(R.META)), video=True)
tmp = out + ".sonic.mp4"
ffm = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
                        "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", tmp], stdin=subprocess.PIPE)
for k in idx:
    sim.d.qpos[:] = tr[k]; mujoco.mj_forward(sim.m, sim.d)
    ffm.stdin.write(sim.frame().tobytes())
ffm.stdin.close(); ffm.wait()
# labels + target index/coords burnt in from the row schedule (2 s per row)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", bfm_mp4, "-i", tmp, "-filter_complex",
                "[0]drawtext=text='ScaleBFM mode 2 (pelvis+wrists)':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[a];"
                "[1]drawtext=text='SONIC (IK joints, planner channel)':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[b];"
                "[a][b]hstack", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out], check=True)
os.remove(tmp)
print(out)
