"""The seed skill package for SIMPLE (escalation plan; TODO 49.2: one package for all nine tasks): `parameters()` is the seed's default dict, `apply_set` overrides
parameters by name (an unknown name raises, as bed_simple does)."""


def parameters(task: str = "XMovePickTeleop") -> dict:
    """The seed's parameters for a task: skill.defaults on the K_CUT base, which is bed_simple's default."""
    from . import skill as S

    return S.defaults(task, S.K_CUT)


def apply_set(overrides: dict, task: str = "XMovePickTeleop") -> dict:
    """The seed's parameters with `overrides` set. Unknown names raise KeyError (a typo must not run silently)."""
    K = parameters(task)
    unknown = sorted(set(overrides or {}) - set(K))
    if unknown:
        raise KeyError(f"unknown parameters {unknown}")
    K.update(overrides or {})
    return K
