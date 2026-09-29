#!/usr/bin/env python3
"""Knobs live in config dataclasses, not in the environment (standards.md, decision 8).

Only process-level variables may be read from os.environ in main code: devices, ports,
paths, and the few external services. Everything else is a knob and must move into a
config object (refactor step 3).

    python scripts/lint_env.py            # report (exit 0)
    python scripts/lint_env.py --strict   # exit 1 on any non-whitelisted read
"""

import glob
import re
import sys
from collections import defaultdict

WHITELIST = {
    "CUDA_VISIBLE_DEVICES",
    "MUJOCO_EGL_DEVICE_ID",
    "MUJOCO_GL",  # render backend (egl/osmesa), the same device category; bottle_visual/round.py records it as provenance
    "INSTANCE",
    "HF_HOME",
    "PYTHONUNBUFFERED",
    "WANDB",
    "LOCOMANI_ROOT",
    "LOCOMANI_WORKSPACE",
    "LOCOMANI_TMP",  # process-level session/log root, shared with scripts/paths.sh
    "LOCOMANI_SIM_DIR",
    "LOCOMANI_BACKEND_JSON",
    "LOCOMANI_DEPLOY_RECEIPTS",  # launcher/restart content-identity journal path  # one serialized config at the Python child-process entry
    "SONIC_ROOT",
    "SONIC_DEPLOY_ROOT",
    "SONIC_DEPLOY_BIN",
    # Which SONIC checkpoint the deploy runs (scripts/paths.sh is the default,
    # locomani.paths the Python half). A path, in the same category as the two
    # above -- and it MUST be readable here, because the offline encoder/decoder
    # have to resolve the same checkpoint as the running deploy.
    "SONIC_POLICY",
    "SIM_VIDEO_DIR",
    "SIM_STATE_TRACE_DIR",
    "TF_CAMERA_PORT",
    "TF_SONIC_PORT",
    "TF_OUT_PORT",
    "TF_KEYBOARD_PORT",
    "TF_PRIVILEGED_PORT",
    "TF_EPISODE_PORT",
    "TF_ISAAC_MIRROR_PORT",
    "TF_LOCKSTEP_PORT",
    "TF_LOCKSTEP_XSUB_PORT",
    "TF_SONIC_PUB_CONNECT",
    "PORT",
    "HOME",
    "PATH",
    "TMPDIR",
    # The finetune recipe's two hardware escape hatches, from scripts/train_bottle.sh when it was
    # ported into locomani.pipeline.run_train (2026-09-21). They describe the CARD, not the
    # experiment -- pin the optimizer, or refuse to pick one from VRAM at all -- which is why they
    # stay in the environment while everything about the run comes from its YAML.
    "OPTIM",
    "OPTIM_AUTO",
    # -- refactor step 3, step 6 additions -----------------------------------
    # Genuinely process-level: torchrun's own rank, not a knob of ours.
    "RANK",
    # Shell/process boundary (decision 8's stack.py doc): these five run
    # inside the MuJoCo sim, a separate OS process launched by
    # start_stack.sh with no channel to receive a SonicBackendConfig other
    # than inherited env (upstream's SimLoopConfig/tyro CLI is not ours to
    # extend). SonicBackendConfig.{sim_vis,scene_random,sonic_dds_domain,
    # lockstep.enabled} (locomani/backends/sonic/config.py) is the config
    # side; locomani/pipeline/stack.py is the sole writer.
    "SIM_VIS",
    "SCENE_RANDOM",
    "SONIC_DDS_DOMAIN",
    "LOCKSTEP",
    # Paths in files outside this refactor's scope (locomani/paths.py /
    # isaac_scene.py's own Isaac-render path resolution), the same category
    # as the already-whitelisted LOCOMANI_ROOT/LOCOMANI_SIM_DIR above.
    "SIMPLE_ROOT",
    "SIMPLE_PIN",
    "RENDER_PY",
    "ISAAC_ASSETS",
    # The Unreal install (images.source unreal), resolved in paths.py like the Isaac render paths above.
    "UNREAL_ROOT",
    "UE_ENGINE",
    "UE_PROJECT",
    "BRIDGE_PY",
    "UNREAL_ASSETS",
    "SCENE_META",
    "SCENE_XML",
    "TF_FRAME_AUDIT_PATH",
    # Out of this refactor's file scope, both already narrow: latency_dist.py
    # has its own --fps flag (this is just its default's fallback);
    # diag_transition_cross.py's GYM_CREEP_OBS is written into a diagnostic
    # event log as metadata, not read back to gate anything.
    "TF_DATASET_FPS",
    "GYM_CREEP_OBS",
    # Dead-variable guards (2026-09-11): the two crate preflights read these
    # ONLY to refuse to start when a caller still sets them (they were the
    # knobs before the refactor-step-3 argparse conversion; now --pose-left-deg,
    # --pw/--out-dist/--up/--over/--warm/--save/--sag/--rot/--pitch0). Never
    # used as a value -- same pattern as the wall expert's WALL_WALK check.
    "POSE_LEFT_DEG",
    "PW",
    "OUT",
    "UP",
    "OVER",
    "WARM",
    "SAVE",
    "SAG",
    "ROT",
    "PITCH0",
}
EXCLUDE = ("third_party/", "src/gr00t/", "outputs/", ".claude/", "legacy/")
PAT = re.compile(r"os\.environ(?:\.get)?\s*[\(\[]\s*['\"]([A-Za-z_0-9]+)['\"]")


def main():
    strict = "--strict" in sys.argv
    hits = defaultdict(list)
    for f in sorted(set(glob.glob("src/**/*.py", recursive=True) + glob.glob("scripts/*.py"))):
        if any(x in f for x in EXCLUDE):
            continue
        for i, line in enumerate(open(f), 1):
            for var in PAT.findall(line):
                if var not in WHITELIST:
                    hits[var].append(f"{f}:{i}")
    for var, where in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        more = f" (+{len(where) - 1})" if len(where) > 1 else ""
        print(f"{var:32s} {len(where):3d}  {where[0]}{more}")
    total = sum(len(v) for v in hits.values())
    fail = strict and bool(hits)
    print(f"[lint_env] {len(hits)} non-whitelisted variables, {total} reads" + (" -- FAIL" if fail else ""))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
