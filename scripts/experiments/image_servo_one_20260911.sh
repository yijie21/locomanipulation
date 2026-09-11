#!/bin/bash
# Run a COPY from its own output directory. Stops after one recorded success.
set -euo pipefail
cd /workspace/code/locomanipulation
OUT=${1:?absolute output directory}
SEED=${2:-31000}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=/workspace/code/locomanipulation/src
export INSTANCE=0 LOCOMANI_SIM_DIR=/workspace/code/locomanipulation/outputs/scene/sim
export TF_EXPORTER_SCRIPT=/workspace/code/locomanipulation/src/locomani/pipeline/run_instance_exporter.py
export SIM_VIDEO_DIR="$OUT/videos" SIM_STATE_TRACE_DIR="$OUT/state_trace" TF_FRAME_AUDIT_PATH="$OUT/frame_audit.jsonl"
SIM=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
headroom=$(bash scripts/cpu_headroom.sh 3)
echo "$headroom"
[[ "$headroom" == *"OK for a stack"* ]] || exit 3
mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$mem" -lt 500 ]] || exit 3
"$SIM" - <<'PY'
import socket
for port in (5555,5556,5557,5580,5581,5582):
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',port)) == 0:
            raise SystemExit(f'instance 0 port {port} is occupied')
PY
"$SIM" - "$OUT" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
from locomani.tasks.bottle.config import BottleTaskConfig
out=Path(sys.argv[1])
BottleTaskConfig.load('configs/tasks/bottle_image_servo.json').dump(out/'task_config.json')
hashes={}
for source in Path('src/locomani').rglob('*.py'):
    target=out/'source'/source
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
(out/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
PY
git rev-parse HEAD > "$OUT/git_sha.txt"
git diff HEAD > "$OUT/source.patch"
cleanup() {
  # Never tear down a stack whose admission lock belongs to another run.
  if grep -Fq "SIM_VIDEO_DIR=$OUT/videos " /workspace/tmp/locomani_locks/instance_0.lock; then
    bash scripts/stack/stop_instance.sh 0 > "$OUT/stop.log" 2>&1
  fi
}
trap cleanup EXIT
bash scripts/stack/start_stack.sh 'pick up the water bottle' > "$OUT/start.log" 2>&1
"$SIM" -u src/locomani/tasks/bottle/grasp_gym.py --task-config "$OUT/task_config.json" \
  --rounds 4 --until-ok 1 --seed "$SEED" --stand-retries 0 --record --video-include-init \
  --manifest "$OUT/manifest.jsonl" > "$OUT/collect.log" 2>&1
"$SIM" - "$OUT/manifest.jsonl" <<'PY'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
ok=[r for r in rows if r.get('ok') and r.get('produced')]
print(f'{len(ok)} successful recorded episodes / {len(rows)} attempts')
if not ok: raise SystemExit(1)
PY
