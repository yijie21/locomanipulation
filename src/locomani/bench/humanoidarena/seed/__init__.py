"""The seed skill package for HumanoidArena (target=truth, controller=native): `parameters(task)` is the default dict, `apply_set` overrides
parameters by name (an unknown name raises), `run(env)` makes one episode. Loaded by path (session.load_package), frozen for the loop."""


def parameters(task: str = "") -> dict:
    from . import skill as S

    return dict(S.K)


def apply_set(overrides: dict, task: str = "") -> dict:
    """The seed's parameters with `overrides` set. Unknown names raise KeyError (a typo must not run silently)."""
    K = parameters(task)
    unknown = sorted(set(overrides or {}) - set(K))
    if unknown:
        raise KeyError(f"unknown parameters {unknown}")
    K.update(overrides or {})
    return K


def run(env) -> None:
    """One episode on the facade `env` (client.PackageEnv). Returns when the skill is done; the episode may end first (client.Done)."""
    from . import skill as S

    return S.run(env)
