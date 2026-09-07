"""Opt-in small arrival advance and measured squat handover."""
from grasp_capture import HEIGHT_TOL

class ArrivalAdvance:
    def __init__(self, advance):
        self.advance = advance
        self.first_reachable = None

    def stop(self, depth, reachable):
        if reachable and self.first_reachable is None:
            self.first_reachable = float(depth)
        return (self.first_reachable is not None and
                depth <= self.first_reachable-self.advance)


def squat_handover(geometry, delivered, requested):
    if geometry['ready']:
        return 'capture_region'
    if abs(geometry['world_height_error_m']) < HEIGHT_TOL:
        return 'grip_height_aligned'
    if delivered >= requested-.005:
        return 'requested_drop_delivered'
    return None
