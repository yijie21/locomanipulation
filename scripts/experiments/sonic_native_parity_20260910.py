"""Read-only phase3 provenance and compiled model comparison; no GPU simulation."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/diagnostics/sonic_native_phase3_20260910"
WBC = ROOT / "third_party/GR00T-WholeBodyControl"
models = {
    "local": ROOT / "outputs/scene/sim/scene_locomani.xml",
    "official_local": WBC / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml",
}
report = {
    "models": {},
    "hashes": {},
    "limits": [
        "Official means the locally checked-out upstream asset, not collaborator hardware configuration.",
        "XML timestep is overridden to 1/sim_frequency; default 200 Hz gives 0.005 s.",
        "No training PD specification available: consistency with current C++ is not training parity.",
        "Initial native trial collapsed before playback and is excluded from tracking conclusions.",
    ],
}
for key, path in models.items():
    m = mujoco.MjModel.from_xml_path(str(path))
    pelvis = m.body("pelvis").id
    report["models"][key] = dict(
        path=str(path),
        robot_mass=float(m.body_mass[m.body_rootid == pelvis].sum()),
        xml_timestep=m.opt.timestep,
        gravity=m.opt.gravity.tolist(),
        solver=int(m.opt.solver),
        integrator=int(m.opt.integrator),
        iterations=int(m.opt.iterations),
        tolerance=float(m.opt.tolerance),
        robot_bodies=int((m.body_rootid == pelvis).sum()),
    )
for folder in ("release", "sonic_v1_1"):
    base = WBC / "gear_sonic_deploy/policy" / folder
    for p in sorted(base.glob("*")):
        if p.is_file() and p.suffix in (".onnx", ".yaml"):
            report["hashes"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads((OUT / "native_release_1.json").read_text())
s = r["trace"][0]["state"]
pd = np.loadtxt(OUT / "deploy_pd.csv", delimiter=",")
report["active_pd"] = {
    "source": "First initialization-failure sample LowCmd; not evidence of successful tracking",
    "kp": s["cmd_kp"],
    "kd": s["cmd_kd"],
    "kp_max_error_vs_cpp_6digit_csv": float(np.max(np.abs(np.array(s["cmd_kp"]) - pd[:, 1]))),
    "kd_max_error_vs_cpp_6digit_csv": float(np.max(np.abs(np.array(s["cmd_kd"]) - pd[:, 2]))),
}
(OUT / "parity_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "hashes"}, indent=2))
