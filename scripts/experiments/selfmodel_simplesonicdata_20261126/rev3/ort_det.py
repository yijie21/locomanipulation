# notebook: selfmodel_simplesonicdata_20261126 -- revision 3's pre-registration check, no episode: is each path
# deterministic run to run (same input, 3 fresh sessions x 20 calls): CPU 1 thread, CPU 2 threads, GPU
import os
import sys

import numpy as np

sys.path.insert(0, os.environ["R"] + "/src")
from locomani import paths  # noqa: E402

import onnxruntime as ort  # noqa: E402

P = {"encoder": f"{paths.SONIC_POLICY_DIR}/model_encoder.onnx", "decoder": f"{paths.SONIC_POLICY_DIR}/model_decoder.onnx",
     "planner": f"{paths.SONIC_DEPLOY}/planner/target_vel/V2/planner_sonic.onnx"}


def sess(path, threads, gpu):
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = threads
    so.log_severity_level = 3
    if not gpu:
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    import torch  # noqa: F401

    ort.preload_dlls()
    return ort.InferenceSession(path, so, providers=[("CUDAExecutionProvider", {"device_id": 0})])


rng = np.random.default_rng(1)
for name, path in P.items():
    s0 = sess(path, 1, False)
    feed = {}
    for i in s0.get_inputs():
        shp = [d if isinstance(d, int) else 1 for d in i.shape]
        feed[i.name] = (rng.standard_normal(shp) * 0.5).astype(np.float32) if "float" in i.type else np.zeros(shp, np.int64)
    for tag, thr, gpu in (("cpu 1 thread", 1, False), ("cpu 2 threads", 2, False), ("gpu", 1, True)):
        outs = []
        for _ in range(3):
            s = sess(path, thr, gpu)
            outs += [s.run(None, feed) for _ in range(20)]
        same = all(all(np.array_equal(a, b) for a, b in zip(outs[0], o)) for o in outs[1:])
        print(f"{name:8s} {tag:14s}: 60 calls (3 sessions x 20) bitwise identical: {same}")
