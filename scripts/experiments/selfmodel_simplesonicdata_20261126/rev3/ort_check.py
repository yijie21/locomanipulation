# notebook: selfmodel_simplesonicdata_20261126 -- revision 3's pre-registration check, no episode: do SONIC's three ONNX
# nets give bitwise different outputs with 1 / 2 / 4 CPU threads, and on the GPU (CUDA provider)?
import os
import sys

import numpy as np

sys.path.insert(0, os.environ["R"] + "/src")
from locomani import paths  # noqa: E402

import onnxruntime as ort  # noqa: E402

P = {"encoder": f"{paths.SONIC_POLICY_DIR}/model_encoder.onnx", "decoder": f"{paths.SONIC_POLICY_DIR}/model_decoder.onnx",
     "planner": f"{paths.SONIC_DEPLOY}/planner/target_vel/V2/planner_sonic.onnx"}
print("onnxruntime", ort.__version__, "providers", ort.get_available_providers())


def sess(path, threads, gpu):
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = threads
    so.log_severity_level = 3
    if not gpu:
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    import torch  # noqa: F401

    ort.preload_dlls()
    s = ort.InferenceSession(path, so, providers=[("CUDAExecutionProvider", {"device_id": 0})])
    assert "CUDAExecutionProvider" in s.get_providers()
    return s


rng = np.random.default_rng(0)
for name, path in P.items():
    s1 = sess(path, 1, False)
    feed = {}
    for i in s1.get_inputs():
        shp = [d if isinstance(d, int) else 1 for d in i.shape]
        feed[i.name] = (rng.standard_normal(shp) * 0.5).astype(np.float32) if "float" in i.type else np.zeros(shp, np.int64)
    ref = s1.run(None, feed)
    print(f"{name}: inputs {[(i.name, i.shape, i.type) for i in s1.get_inputs()]}")
    for tag, s in (("cpu 2 threads", sess(path, 2, False)), ("cpu 4 threads", sess(path, 4, False)), ("gpu", sess(path, 1, True))):
        out = s.run(None, feed)
        d = max(float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()) for a, b in zip(ref, out))
        same = all(np.array_equal(a, b) for a, b in zip(ref, out))
        print(f"  {tag:14s} vs cpu 1 thread: bitwise equal {same}, max |diff| {d:.3e}")
