"""The seed skill package for LocoManiLab's camera track: `parameters()` is the seed's default dict, `apply_set` overrides
parameters by name (an unknown name raises, as the bed does). Loaded by path (run.load_package), frozen for the loop."""

# Keys the camera flows read that skill.py's defaults do not hold. `lift` is the height the lift check asks for;
# `nav_gap` is the clearance (m) the walking planner keeps round what it has seen.
EXTRA = dict(lift=0.34, nav_gap=0.30)


def parameters(task: str = "") -> dict:
    """The seed's parameters for a task: skill.defaults on the K_CUT base, with the keys of EXTRA."""
    from . import skill as S

    return S.defaults(task, dict(S.K_CUT, **EXTRA))


def apply_set(overrides: dict, task: str = "") -> dict:
    """The seed's parameters with `overrides` set. Unknown names raise KeyError (a typo must not run silently)."""
    K = parameters(task)
    unknown = sorted(set(overrides or {}) - set(K))
    if unknown:
        raise KeyError(f"unknown parameters {unknown}")
    K.update(overrides or {})
    return K
