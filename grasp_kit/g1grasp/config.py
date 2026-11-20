"""The pipeline's settings, at the values the simulated benchmark ran with."""

from dataclasses import dataclass


@dataclass
class Config:
    # depth
    near: float = 0.2  # m: depths outside [near, far] count as unknown
    far: float = 5.0
    # the target
    sam_threshold: float = 0.1  # SAM 3's own confidence floor for the instances it returns
    sam_floor: float = 0.3  # the lowest score taken for the named object
    generic: str = "object"  # the prompt for an instruction that names no object
    own: float = 0.2  # m ahead of the pelvis: nearer points are the robot's own body (unnamed target only)
    reach: float = 1.0  # m ahead of the pelvis: the farthest an unnamed target is looked for
    min_px: int = 150  # an unnamed target's size in pixels: at least this many
    max_frac: float = 0.08  # and at most this fraction of the image
    min_seen: int = 100  # fewer masked pixels with a depth: the target counts as not seen
    erode: int = 1  # px taken off the mask's rim, where the depth mixes the object and what lies behind it
    radius: float = 0.3  # m from the target's median point: farther points are strays of the mask
    # the grasp
    num_grasps: int = 500
    planner: str = "graspmoe"
    seed: int = 0
    ahead: float = 0.7  # the cosine between the fingers (the palm link's x) and the robot's forward, at least
    roll: tuple = (-30.0, 110.0)  # deg, the palm's facing: 0 faces left (back of the hand up), 90 faces down
    clear: float = 0.06  # m, the palm link above the target's foot, at least (the middle finger hangs 4 cm below it)
    pre: float = 0.08  # m, the pregrasp: back from the grasp against the palm's normal (the palm link's y)
