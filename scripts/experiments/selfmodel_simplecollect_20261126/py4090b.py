"""Run a script with the 4090's SIMPLE venv (third_party/GR00T-WholeBodyControl/.venv_sim = ws/venvs/sonic_ort over
ws/venvs/sim) on a machine without the /workspace -> /wuji-vefps-D/wuji-il/yijie/ws symlink (4090b), nothing installed:
    /usr/bin/python3.10 -S py4090b.py SCRIPT ARGS...
sys.path as the venv's own (stdlib, sonic_ort's site-packages with its .pth files, GR00T-WholeBodyControl, sim's
site-packages), every /workspace path spelled on vePFS. With LOCOMANI_WORKSPACE, SONIC_POLICY and SX_REPO=<repo_b>
(scene copies with the vePFS paths) a reset driven this way is bit-identical to the 4090's (development, 11-26); the
4090b environment itself is not changed."""
import os, runpy, site, sys
WS = "/wuji-vefps-D/wuji-il/yijie/ws"
SP = "lib/python3.10/site-packages"
sys.prefix = sys.exec_prefix = f"{WS}/code/loco/third_party/GR00T-WholeBodyControl/.venv_sim"
site.addsitedir(f"{WS}/venvs/sonic_ort/{SP}")  # its .pth files' /workspace paths do not exist here: skipped
# GR00T-WholeBodyControl: loco's copy holds gear_sonic_deploy and /workspace symlinks to locomanipulation's gear_sonic /
# decoupled_wbc (broken here), so locomanipulation's copy follows it for those two packages
for p in (f"{WS}/code/loco/third_party/GR00T-WholeBodyControl", f"{WS}/code/locomanipulation/third_party/GR00T-WholeBodyControl",
          f"{WS}/venvs/sim/{SP}"):
    sys.path.append(p)
script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
