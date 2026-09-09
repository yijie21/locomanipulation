"""Constants shared by the scene builder and the runtime.

`scene_lib` cannot import `primitives` — that pulls in zmq, the commander and
the navigator, none of which a scene build should need — but both have to agree
on where the hand grips, or the ghost marker lands somewhere the servo is not
aiming.
"""

import os as _os

HAND = _os.environ.get("HAND", "dex1")

# Palm(wrist)-frame vector to the point the jaw closes on.
# dex3: measured on the three-finger hand. dex1: centre of the finger pads
# (pads span x 0.077..0.143 in the gripper-base frame, base at wrist
# +0.0415, fingers symmetric about y=0).
if HAND == "dex1":
    GRASP_CENTRE_OFFSET_CONST = (0.1515, 0.0, 0.0)
else:
    GRASP_CENTRE_OFFSET_CONST = (0.112, 0.049, 0.001)
