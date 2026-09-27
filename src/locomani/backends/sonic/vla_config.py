"""Immutable settings consumed by SONIC policy links and round clients.

EvalConfig inherits these fields; one resolved object is passed to each consumer.
The remaining evaluation fields belong to pipeline scheduling and reporting.
"""

from dataclasses import dataclass, field

from locomani.backends.sonic.config import SonicBackendConfig

LIFT_M = 0.06
NEAR_M = 0.60
STALL_ABORT_M = 0.03


@dataclass(frozen=True)
class VlaConfig:
    backend: SonicBackendConfig = field(default_factory=SonicBackendConfig)
    budget_s: float = 90.0  # --budget: per-round VLA time budget (the ladder's setting)
    ensemble_n: int = 1  # --ensemble N: chunks to blend (1 = off, the ladder's setting)
    ensemble_m: float = 0.3  # --ensemble-m: exponential weight decay per chunk of age
    rtc: bool = False  # --rtc: async/lockstep real-time chunking (test-time inpainting)
    transition_commit: bool = False  # --transition-commit: experimental async RTC crouch-prefix commitment
    transition_decoder: str | None = None  # --transition-decoder: ONNX decoder matching the live deploy
    grip_gain: float = 1.0  # --grip-gain K: amplify the predicted gripper's deviation from fully-open
    snap: bool = True  # not --no-snap: send the predicted token snapped to the FSQ grid
    stall_abort_s: float = 0.0  # --stall-abort: end a round early after this many stalled seconds (0 = off)
    # Where the policy's camera images come from: the sim's own MuJoCo stream (:5555), or the Isaac mirror
    # rendering the sim's newest state on request (sim/isaac/mirror.py, scripts/run_isaac_mirror.sh). A run
    # decides it once, as `images.source`; run_context.evaluation_of delivers it here, and an experiment's eval
    # block does not set it.
    image_source: str = field(default="mujoco", metadata={"choices": ("mujoco", "isaac")})


DEFAULT_VLA = VlaConfig()
