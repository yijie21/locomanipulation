"""Build a private SONIC encoder-tensor capture binary using existing build flags.

Production source, binary, models, and archives are read-only. One compilation
process; no GPU use. Requires the existing CMake build artifacts.
"""

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
OUT = ROOT / "outputs/diagnostics/sonic_lean_phase2_20260910"
PRIVATE = OUT / "deploy"
source = DEPLOY / "src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp"
marker = "      const size_t token_dim = encoder_engine_->GetTokenDimension();"
insert = r"""
      // [DEBUG-sonic-encoder-phase2] PRIVATE BUILD ONLY: the exact float buffer
      // passed to Encode(), plus its resulting token; buffered binary I/O.
      static std::ofstream encoder_audit;
      static bool audit_initialized = false;
      if (!audit_initialized) {
        const char* path = std::getenv("SONIC_ENCODER_AUDIT_FILE");
        if (path) encoder_audit.open(path, std::ios::binary | std::ios::out);
        audit_initialized = true;
      }
      if (encoder_audit.is_open()) {
        const auto& actual_input = encoder_engine_->GetInputBuffer();
        const uint32_t magic = 0x53454e43;
        const double wall = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        const uint32_t ni = actual_input.size(), nt = token_dim;
        const int32_t mode = current_motion_->GetEncodeMode();
        encoder_audit.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        encoder_audit.write(reinterpret_cast<const char*>(&wall), sizeof(wall));
        encoder_audit.write(reinterpret_cast<const char*>(&ni), sizeof(ni));
        encoder_audit.write(reinterpret_cast<const char*>(&nt), sizeof(nt));
        encoder_audit.write(reinterpret_cast<const char*>(&mode), sizeof(mode));
        encoder_audit.write(reinterpret_cast<const char*>(actual_input.data()), ni * sizeof(float));
        encoder_audit.write(reinterpret_cast<const char*>(token_buffer.data()), nt * sizeof(float));
      }
"""
PRIVATE.mkdir(parents=True, exist_ok=True)
(PRIVATE / "target/release").mkdir(parents=True, exist_ok=True)
text = source.read_text()
assert text.count(marker) == 1
patched = PRIVATE / "encoder_capture.cpp"
patched.write_text("#include <cstdlib>\n#include <cstdint>\n" + text.replace(marker, marker + insert))
for name in ["policy", "reference", "planner"]:
    link = PRIVATE / name
    if not link.exists():
        link.symlink_to(DEPLOY / name, target_is_directory=True)
commands = json.loads((DEPLOY / "build/compile_commands.json").read_text())
entry = next(x for x in commands if x["file"].endswith("/src/g1_deploy_onnx_ref.cpp"))
old = "/workspace/code/locomanipulation/GR00T-WholeBodyControl/gear_sonic_deploy"


def remap(arg):
    return arg.replace(old, str(DEPLOY))


compile_args = [remap(x) for x in shlex.split(entry["command"])]
compile_args[compile_args.index("-o") + 1] = str(PRIVATE / "encoder_capture.o")
compile_args[-1] = str(patched)
link_dir = DEPLOY / "build/src/g1/g1_deploy_onnx_ref"
link_args = [remap(x) for x in shlex.split((link_dir / "CMakeFiles/g1_deploy_onnx_ref.dir/link.txt").read_text())]
link_args = [str(PRIVATE / "encoder_capture.o") if x.endswith("g1_deploy_onnx_ref.cpp.o") else x for x in link_args]
link_args[link_args.index("-o") + 1] = str(PRIVATE / "target/release/g1_deploy_onnx_ref")
(OUT / "build_commands.json").write_text(json.dumps(dict(compile=compile_args, link=link_args), indent=2))
if "--link-only" not in sys.argv:
    subprocess.run(compile_args, cwd=DEPLOY / "build", check=True)
logger_entry = next(x for x in commands if x["file"].endswith("/state_logger.cpp"))
logger_args = [remap(x) for x in shlex.split(logger_entry["command"])]
logger_args[logger_args.index("-o") + 1] = str(PRIVATE / "state_logger.o")
subprocess.run(logger_args, cwd=DEPLOY / "build", check=True)
link_args.insert(link_args.index("libg1_deploy_shared.a"), str(PRIVATE / "state_logger.o"))
subprocess.run(link_args, cwd=link_dir, check=True)
(OUT / "build_manifest.json").write_text(
    json.dumps(
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                source,
                patched,
                PRIVATE / "target/release/g1_deploy_onnx_ref",
                DEPLOY / "target/release/g1_deploy_onnx_ref",
            ]
        },
        indent=2,
    )
)
print("PRIVATE BUILD COMPLETE", flush=True)
