"""Decode end-effector contact observations; task acceptance remains with its owner."""


def bilateral_dex1_contact(obj, side):
    prefix = "R" if side == "right" else "L"
    links = set(obj.get("touch_links", []))
    return {f"{prefix}.dex1_finger_1", f"{prefix}.dex1_finger_2"} <= links
