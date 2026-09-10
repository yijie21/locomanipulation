"""Diagnostic interventions on decoded references; no task/object inputs."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from visual_executor import ActionExecutor
from visual_action import encode_action

@dataclass(frozen=True)
class InterventionConfig:
    start_s: float = 1.68
    height_m: float = .76
    trigger_height_m: float = .72
    trigger_dwell_s: float = .10
    speed_m_s: float = .12
    accel_m_s2: float = .4
    hold_s: float = 1.
    release_s: float = .5

class DiagnosticExecutor(ActionExecutor):
    def __init__(self, hand_rel, arm='A', config=None):
        super().__init__(hand_rel)
        if arm not in 'ABCD' or len(arm)!=1:raise ValueError(arm)
        self.arm=arm;self.config=config or InterventionConfig()
        self.raw_hold=None;self.logs=[];self.trigger=None;self.low_s=0.
        self.z=None;self.v=0.;self.held=None

    def step(self, root_position, root_quaternion, action=None):
        t=self.tick*self.dt;cfg=self.config
        previous=self.previous_wrist
        # The VLA sees the actually applied previous row; its next raw row is
        # held separately so interventions never overwrite the prediction stream.
        self.row=self.raw_hold
        targets,grip=super().step(root_position,root_quaternion,action)
        self.raw_hold=self.row.copy()
        root=np.asarray(root_position)
        before=targets['pelvis'][0].copy()
        modified=False
        if t>=cfg.start_s-1e-9:
            self.low_s=self.low_s+self.dt if root[2]<cfg.trigger_height_m else 0.
            if self.trigger is None and self.low_s>=cfg.trigger_dwell_s-1e-9:
                self.trigger=t
                self.held=tuple(x.copy() for x in (previous or (targets['right_wrist_yaw_link'][0][0],targets['right_wrist_yaw_link'][1][0])))
            height_active=self.arm=='B' or (self.arm=='C' and self.trigger is not None)
            if height_active:
                if self.z is None:self.z=float(before[0,2])
                desired=np.clip(2*(cfg.height_m-self.z),-cfg.speed_m_s,cfg.speed_m_s)
                self.v+=float(np.clip(desired-self.v,-cfg.accel_m_s2*self.dt,cfg.accel_m_s2*self.dt))
                self.z+=self.v*self.dt
                positions,quaternions=targets['pelvis']
                positions=positions.copy();positions[:,2]=self.z+np.arange(len(positions))*self.dt*self.v
                targets['pelvis']=positions,quaternions
                # Left rest hand follows pelvis z, retaining its relative pose.
                lp,lq=targets['left_wrist_yaw_link'];lp=lp.copy();lp[:,2]+=positions[:,2]-before[:,2]
                targets['left_wrist_yaw_link']=lp,lq
                modified=True
            if self.arm=='D' and self.trigger is not None:
                elapsed=t-self.trigger
                if elapsed<cfg.hold_s+cfg.release_s:
                    p,q=targets['right_wrist_yaw_link'];hp,hq=self.held
                    u=float(np.clip((elapsed-cfg.hold_s)/cfg.release_s,0,1));alpha=u*u*(3-2*u)
                    p=hp[None]+alpha*(p-hp)
                    rotations=Rotation.from_quat(q,scalar_first=True)
                    base=Rotation.from_quat(hq,scalar_first=True)
                    rotations=Rotation.from_rotvec(alpha*(rotations*base.inv()).as_rotvec())*base
                    targets['right_wrist_yaw_link']=p,rotations.as_quat(scalar_first=True)
                    modified=True
        if modified:
            self.row=encode_action(targets,root_position,root_quaternion,grip,dt=self.dt).astype(np.float32)
            p,q=targets['right_wrist_yaw_link'];self.previous_wrist=(p[0].copy(),q[0].copy())
        self.logs.append(dict(t=t,root_z=float(root[2]),raw_target_z=float(before[0,2]),
            applied_target_z=float(targets['pelvis'][0][0,2]),height_velocity=self.v,
            trigger_t=self.trigger,modified=modified,raw_row=self.raw_hold.tolist(),applied_row=self.row.tolist()))
        return targets,grip
