"""View: what a skill asks about the scene (TODO 49.1).

The interface is the one the SIMPLE and LocoManiLab beds already share (escalation_20261116: TruthView and CameraView
in seed_pkg/skill_simple.py and view_camera.py, LmlCameraView in view_lml.py), with `basket` named `container`.
Positions are in the robot's frame: the world as it was at the episode's start, carried along by the SelfPose.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import numpy as np


class Target(TypedDict):
    T_wo: np.ndarray  # (4, 4) the object's frame in the robot's frame
    lo: np.ndarray  # (3,) lower corner of its points (camera) or vertices (truth), robot frame
    hi: np.ndarray  # (3,) upper corner
    top: float  # height of its lowest point: the surface it stands on


class Container(TypedDict):
    centre: np.ndarray  # (2,) xy of the container's middle
    rim: float  # height of its rim
    near: float  # its near side along the heading it was asked for (coordinate p . ahead)


@runtime_checkable
class View(Protocol):
    search: bool  # True: the container's place is not known until the robot turns and looks for it

    def heading(self) -> float:
        """The robot's heading at the start, in its own frame (radians)."""
        ...

    def look(self) -> dict:
        """Take one look now (camera: images, masks, points; truth: nothing to do). Returns what to record."""
        ...

    def target(self) -> Target | None:
        """The object the instruction names; None when it cannot be found."""
        ...

    def grasp(self, tgt: Target, yaw: float) -> dict | None:
        """The grasp to use on `tgt` when the robot faces `yaw`: palm pose and approach, or None when none fits."""
        ...

    def edge(self, ahead: np.ndarray, lateral: float, table: str = "table") -> float | None:
        """Where the named table begins along `ahead` (the coordinate p . ahead of its nearest point)."""
        ...

    def container(self, yaw: float) -> Container | None:
        """The container to place into, seen along heading `yaw`."""
        ...


@runtime_checkable
class ArticulatedView(View, Protocol):
    """A view for tasks on hinged objects (SIMPLE's oven, door, faucet, trash can)."""

    def art(self) -> dict | None:
        """The object and its moving part: masks and points in the robot's frame (CameraView.art on the esc bed)."""
        ...
