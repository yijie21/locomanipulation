"""After-episode truth evaluation, with no action or policy callback."""

import numpy as np
from scipy.spatial.transform import Rotation


def evaluate(initial, final, reason, *, lift_height, max_tilt_deg, key="blue_crate"):
    obj = final["objects"][key]
    rise = float(obj["pos"][2] - initial["objects"][key]["pos"][2])
    quat = obj["quat"]
    rot = Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
    tilt = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))
    touching = list(obj.get("touching", []))
    if reason == "completed":
        if final["fallen"]:
            reason = "evaluation_fallen"
        elif rise < lift_height * 0.6:
            reason = "evaluation_insufficient_rise"
        elif tilt >= max_tilt_deg:
            reason = "evaluation_tilt"
        elif not {"left", "right"} <= set(touching):
            reason = "evaluation_missing_contact"
    return {"ok": reason == "completed", "why": reason, "rise": rise, "tilt_deg": tilt, "touching": touching}
