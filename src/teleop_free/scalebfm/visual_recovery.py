"""Assessment/reset interventions, outside the expert's information boundary.

The task planner never receives the intervention kind or its timing. Object
shifts and blackouts are visible environmental changes. Wrist offsets construct
a displaced robot state; recovery recordings begin after that reset, so the
dataset does not teach an unexplained teleport/action discontinuity.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

KINDS = ('none', 'stop_short', 'wrist_offset', 'blackout', 'object_shift')


class RecoveryIntervention:
    def __init__(self, kind='none'):
        if kind not in KINDS:
            raise ValueError('Unknown recovery intervention')
        self.kind = kind
        self.triggered = kind in ('none', 'stop_short')
        self.reach_started = None
        self.blackout_until = -1.
        self.events = []
        self.error = None

    def recording_ready(self):
        return self.kind not in ('wrist_offset', 'object_shift') or self.triggered

    def update(self, sim, t, phase):
        if phase == 'reach' and self.reach_started is None:
            self.reach_started = t
        if self.triggered or self.reach_started is None or t - self.reach_started < 1.8 or phase not in ('reach', 'entry'):
            return None
        self.triggered = True
        rotation = Rotation.from_quat(sim.d.qpos[3:7], scalar_first=True)
        yaw = rotation.as_euler('xyz')[2]
        lateral = Rotation.from_euler('z', yaw).apply([0., 1., 0.])
        event = dict(kind=self.kind, t=t)
        if self.kind == 'blackout':
            self.blackout_until = t + .4
            event['until'] = self.blackout_until
        elif self.kind == 'object_shift':
            delta = .04 * lateral
            sim.d.qpos[sim.crate_q:sim.crate_q + 3] += delta
            mujoco.mj_forward(sim.m, sim.d)
            event['delta'] = delta.tolist()
        elif self.kind == 'wrist_offset':
            names = ('right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                     'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint')
            joints = np.array([sim.m.joint(n).id for n in names])
            qa, va = sim.m.jnt_qposadr[joints], sim.m.jnt_dofadr[joints]
            data = mujoco.MjData(sim.m); data.qpos[:] = sim.d.qpos
            before, quat = sim.body('right_wrist_yaw_link')
            target = before + .03 * lateral
            desired = Rotation.from_quat(quat, scalar_first=True)
            initial = data.qpos[qa].copy()
            bounds = sim.m.jnt_range[joints]
            bid = sim.bid['right_wrist_yaw_link']
            def residual(q):
                data.qpos[qa] = q; mujoco.mj_forward(sim.m, data)
                rot = Rotation.from_quat(data.xquat[bid], scalar_first=True)
                return np.r_[data.xpos[bid] - target, .1 * (rot * desired.inv()).as_rotvec(), .001 * (q - initial)]
            solved = least_squares(residual, np.clip(initial, bounds[:, 0], bounds[:, 1]),
                                   bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=100)
            error = float(np.linalg.norm(residual(solved.x)[:3]))
            event['position_error_m'] = error
            if error > .003:
                self.error = 'Cannot construct the requested 3cm wrist offset'
            else:
                sim.d.qpos[qa] = solved.x
                sim.d.qvel[va] = 0
                sim.d.qacc_warmstart[:] = 0
                mujoco.mj_forward(sim.m, sim.d)
                event['achieved_offset_m'] = (sim.body('right_wrist_yaw_link')[0] - before).tolist()
        if self.kind in ('wrist_offset', 'object_shift'):
            event['qpos_after'] = sim.d.qpos.tolist()
            event['qvel_after'] = sim.d.qvel.tolist()
        self.events.append(event)
        return self.kind
