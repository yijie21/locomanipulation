"""Constants shared by the scene builder and the runtime.

`scene_lib` cannot import `primitives` — that pulls in zmq, the commander and
the navigator, none of which a scene build should need — but both have to agree
on where the hand grips, or the ghost marker lands somewhere the servo is not
aiming.
"""

GRASP_CENTRE_OFFSET_CONST = (0.112, 0.049, 0.001)
