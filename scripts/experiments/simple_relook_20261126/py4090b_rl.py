"""The 4090's SIMPLE venv (.venv_sim) as a python interpreter on 4090b, where /workspace does not exist: the sys.path of
selfmodel_simplecollect_20261126/py4090b.py (stdlib, sonic_ort's site-packages, GR00T-WholeBodyControl, sim's
site-packages, all spelled on vePFS), plus the command-line forms the SIMPLE bench scripts call python with:
    python [-I] SCRIPT ARGS...   python [-I] -m MODULE ARGS...   python [-I] - < code   python [-I] -c CODE ARGS...
-I as in python: PYTHONPATH's entries and the script's directory are left out. mkrepo_b.sh puts a two-line wrapper
`exec /usr/bin/python3.10 -S <this file> "$@"` at <repo_b>/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python, so
bench_via.sh and sfull_run.sh run unchanged from <repo_b>. simple_relook_20261126, SONIC on 4090b's CPU."""
import os, runpy, site, sys

WS = "/wuji-vefps-D/wuji-il/yijie/ws"
SP = "lib/python3.10/site-packages"
if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    del sys.path[0]  # this launcher's own directory, which python put first
sys.prefix = sys.exec_prefix = f"{WS}/code/loco/third_party/GR00T-WholeBodyControl/.venv_sim"
argv = sys.argv[1:]
isolated = False
while argv and argv[0] in ("-I", "-u", "-B"):
    isolated |= argv[0] == "-I"
    argv = argv[1:]
if isolated:
    pp = {os.path.abspath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p}
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) not in pp]
site.addsitedir(f"{WS}/venvs/sonic_ort/{SP}")  # its .pth files' /workspace paths do not exist here: skipped
for p in (f"{WS}/code/loco/third_party/GR00T-WholeBodyControl", f"{WS}/code/locomanipulation/third_party/GR00T-WholeBodyControl",
          f"{WS}/venvs/sim/{SP}"):
    sys.path.append(p)
if not argv or argv[0] == "-":
    sys.argv = [""] + argv[1:]
    if not isolated:
        sys.path.insert(0, "")
    code = sys.stdin.read()
    exec(compile(code, "<stdin>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
elif argv[0] == "-c":
    sys.argv = ["-c"] + argv[2:]
    if not isolated:
        sys.path.insert(0, "")
    exec(compile(argv[1], "<string>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
elif argv[0] == "-m":
    sys.argv = [argv[1]] + argv[2:]
    if not isolated:
        sys.path.insert(0, os.getcwd())
    runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
else:
    script = os.path.abspath(argv[0])
    sys.argv = argv
    if not isolated:
        sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")
