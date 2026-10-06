"""The Dex1 gripper in one scalar, and how that scalar is written on each side of the pipeline.

The real G1's Dex1 is a parallel jaw with one motor. The robot's own exporter records it in MOTOR UNITS: the
teleop command is 0 (closed) .. 5.5 (open) in component 0 of the 7-wide `hand_joints` field, and the state is the
motor position broadcast to all 7 hand slots (toy-bear dataset, 2026-10-06). The simulator's Dex1 model records
METRES on two finger joints, with the opposite sign: +0.0245 open, -0.02 closed (`sonic_token.GRIPPER_MIN/MAX`),
the command going to -0.06 to load the jaw. One scalar, two unit systems, two wire layouts -- so every conversion
lives here and nowhere else (decisions D6, D7 of the plan).

    model output ──to_dex1(units)──▶ Dex1 scalar ──hand_field(wire)──▶ 7-wide protocol-v4 hand field
    robot hand_q ──dex1_state────▶ Dex1 scalar ──hand_state(width, units)──▶ the model's `state.*_hand`
"""

from __future__ import annotations

import numpy as np

from locomani.backends.sonic.sonic_token import GRIPPER_MAX, GRIPPER_MIN

DEX1_CLOSED, DEX1_OPEN = 0.0, 5.5  # the robot's teleop command range; its state rests at 5.39-5.43 when open
SIM_CLOSED_M, SIM_OPEN_M = float(GRIPPER_MIN), float(GRIPPER_MAX)  # -0.02 .. +0.0245 m, the training data's range
HAND_WIDTH = 7  # protocol v4 and the dataset's hand slots (Dex3's seven motors)
SIM_JAW_SLOTS = (0, 1)  # where the sim's wire and teleop columns carry the jaw (sonic_token.hand_from_gripper)
SIM_STATE_SLOTS = (4, 5)  # where the sim's `observation.state` hand group carries the two finger joints

MODEL_UNITS = ("dex1", "sim_m")  # what the checkpoint was trained to predict and observe
HAND_WIRES = ("dex1", "sim")  # what the deploy on the other end of the pose socket reads


def sim_m_to_dex1(metres) -> np.ndarray:
    """Simulator metres to Dex1 motor units, linear on the training range and clipped to the command range: a
    command past -0.02 (the sim's -0.06 jaw loading) is 'closed', not a negative motor position."""
    m = np.asarray(metres, dtype=np.float64)
    t = (m - SIM_CLOSED_M) / (SIM_OPEN_M - SIM_CLOSED_M)
    return np.clip(DEX1_CLOSED + t * (DEX1_OPEN - DEX1_CLOSED), DEX1_CLOSED, DEX1_OPEN)


def dex1_to_sim_m(dex1) -> np.ndarray:
    """The inverse: Dex1 motor units to the simulator's metres, clipped to the training range."""
    d = np.asarray(dex1, dtype=np.float64)
    t = (d - DEX1_CLOSED) / (DEX1_OPEN - DEX1_CLOSED)
    return np.clip(SIM_CLOSED_M + t * (SIM_OPEN_M - SIM_CLOSED_M), SIM_CLOSED_M, SIM_OPEN_M)


def to_dex1(value, units: str) -> np.ndarray:
    """A model's gripper output in `units` -> Dex1 motor units (clipped to the command range)."""
    if units == "dex1":
        return np.clip(np.asarray(value, dtype=np.float64), DEX1_CLOSED, DEX1_OPEN)
    if units == "sim_m":
        return sim_m_to_dex1(value)
    raise ValueError(f"gripper units {units!r} is not one of {MODEL_UNITS}")


def dex1_state(hand_q) -> float:
    """The gripper's measured position from the deploy's 7-wide `hand_q`: the robot broadcasts one motor position
    to every slot, so slot 0 is the number (a 1-wide field reads the same)."""
    return float(np.ravel(np.asarray(hand_q, dtype=np.float64))[0])


def hand_field(dex1_value: float, wire: str) -> np.ndarray:
    """The 7-wide protocol-v4 hand field for one Dex1 scalar.

    `dex1`: the value in every slot -- whichever slot the robot's Dex1 driver reads gets the right number (D7).
    `sim`: this repository's simulation bridge, which reads metres from the two jaw slots (`sonic_token`)."""
    field = np.zeros(HAND_WIDTH, dtype=np.float32)
    if wire == "dex1":
        field[:] = np.clip(dex1_value, DEX1_CLOSED, DEX1_OPEN)
    elif wire == "sim":
        field[list(SIM_JAW_SLOTS)] = dex1_to_sim_m(dex1_value)
    else:
        raise ValueError(f"hand wire {wire!r} is not one of {HAND_WIRES}")
    return field


def hand_state(dex1_value: float, width: int, units: str) -> np.ndarray:
    """The model's `state.left_hand` / `state.right_hand` group for one measured Dex1 scalar.

    `dex1`: the value in every one of the `width` slots, as the robot's exporter records it (a 1-wide group is the
    scalar itself). `sim_m`: the simulator's layout, metres on the two finger slots, for a checkpoint trained on
    unconverted simulation data (the interim before the converter writes Dex1 units)."""
    state = np.zeros(width, dtype=np.float32)
    if units == "dex1":
        state[:] = np.clip(dex1_value, DEX1_CLOSED, DEX1_OPEN)
    elif units == "sim_m":
        if width != HAND_WIDTH:
            raise ValueError(f"sim_m hand state needs the {HAND_WIDTH}-wide group, got {width}")
        state[list(SIM_STATE_SLOTS)] = dex1_to_sim_m(dex1_value)
    else:
        raise ValueError(f"gripper units {units!r} is not one of {MODEL_UNITS}")
    return state
