"""Image evidence gates for admitting an episode, never a student input."""


def head_visible(evidence):
    """Head camera must resolve the bottle; wrist-only sight is insufficient."""
    pixels = evidence.get('pixels', [])
    fractions = evidence.get('fractions', [])
    return bool(pixels and fractions and pixels[0] >= 32 and fractions[0] >= .2)


def initial_visibility(observations, first_record=None, first_record_time=None):
    checks = {}
    for name, t in [('reset', 0.), ('task_start', 1.6)]:
        evidence = next((r for r in observations if abs(r['t'] - t) < 1e-7), None)
        checks[name] = bool(evidence is not None and head_visible(evidence))
    if first_record_time is not None:
        candidates = observations + ([first_record] if first_record is not None else [])
        first_record = next((r for r in candidates if abs(r['t'] - first_record_time) < 1e-7), None)
        checks['first_record'] = bool(first_record is not None and head_visible(first_record))
    elif first_record is not None:
        checks['first_record'] = head_visible(first_record)
    return checks
