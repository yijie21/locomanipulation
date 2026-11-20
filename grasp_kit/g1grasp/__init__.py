"""From one look of the G1's head camera to a grasp of the named object for the right Dex3 hand.

perceive (perception env): depth -> SAM 3's mask of the target -> the target's points and the table under it
plan     (grasp env):      GraspGen-X's hand poses on those points -> the first one the right arm can take, and its pregrasp
"""

__version__ = "2026.11.20"
