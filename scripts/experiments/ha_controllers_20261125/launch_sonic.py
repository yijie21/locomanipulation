#!/usr/bin/env python3
"""CTRL=sonic for run_ctrl.sh: HA's probe_sonic.py, unchanged (native SONIC path), launched under CUDA_VISIBLE_DEVICES=<one
card>. The only addition is ha_g1ctrl.kit_single_gpu after sim_eval_vla._configure_renderer_gpu (the renderer on the
physical card; see there). # notebook: skip ha controller integration smoke"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ha_g1ctrl as HC  # noqa: E402

R = Path(os.environ.get("HA_ROOT", "/wuji-vefps-D/wuji-il/yijie/tools/humanoidarena"))
spec = importlib.util.spec_from_file_location("probe_sonic", R / "scripts" / "probe_sonic.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
_orig = ps.sev._configure_renderer_gpu


def _configure_renderer_gpu(args):
    _orig(args)
    HC.kit_single_gpu(args)


ps.sev._configure_renderer_gpu = _configure_renderer_gpu
if __name__ == "__main__":
    try:
        code = ps.main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
