"""HEFT (LocoManiLab's whole-body motion tracker, g1_backend `checkpoints/heft/pmg`) run in this process from its ONNX file,
on our MuJoCo sim, under our Motor layer.

`obs` is g1_backend's HEFT observation (vendored, provenance in its docstring); `runtime` holds the policy (`HeftPolicy`:
the four ONNX inputs, `default + action * scale`) and the sim (`HeftSim`: `scalebfm.runtime.Sim` with HEFT's PD); `motor`
holds `HeftMotor`, the `scalebfm.motor.Motor` interface on top of them (the reference as qpos frames, walks on `gait`, a
footstep reference vendored from hardgrasp_20261125's hg_walk). The policy files are not in the repo: HeftConfig.policy_dir
(default `paths.HEFT_POLICY_DIR`: `$LOCOMANI_WORKSPACE/models/heft/pmg`).
"""
