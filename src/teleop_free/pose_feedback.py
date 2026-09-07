"""Bounded palm-pose feedback proposal, independent of publishing and safety.

The kinematic Jacobian is a local approximation to reference response, not an
identified WBC Jacobian. Callers must screen proposals and verify actual motion.
"""
from copy import copy
import mujoco
import numpy as np
from g1_joints import ARM_MJ


def rotation_vector(target, actual):
    quat=np.empty(4)
    mujoco.mju_mat2Quat(quat,np.ascontiguousarray(target@actual.T).ravel())
    if quat[0]<0:quat=-quat
    length=np.linalg.norm(quat[1:])
    return quat[1:]*(2*np.arctan2(length,quat[0])/length) if length>1e-10 else np.zeros(3)


class PoseFeedback:
    def __init__(self, ik, initial, side, *, excursion=.35, speed=.5):
        self.ik=copy(ik);self.ik.data=mujoco.MjData(ik.model)
        self.side=side;self.indices=np.array(ARM_MJ[side]);self.initial=np.asarray(initial,dtype=float).copy()
        if self.initial.shape != (29,) or not np.isfinite(self.initial).all():raise ValueError('invalid initial reference')
        if excursion<=0 or speed<=0 or not np.isfinite([excursion,speed]).all():raise ValueError('invalid feedback bounds')
        self.lo=np.maximum(ik.lo[self.indices],self.initial[self.indices]-excursion)
        self.hi=np.minimum(ik.hi[self.indices],self.initial[self.indices]+excursion)
        self.speed=speed
        if np.any(self.lo>self.hi):raise ValueError('initial reference outside feedback bounds')

    def step(self, reference, actual_pos, actual_rot, target_pos, target_rot, dt):
        ref=np.asarray(reference,dtype=float)
        if not all(np.isfinite(v).all() for v in [ref,actual_pos,actual_rot,target_pos,target_rot]) or not 0<dt<=.1:
            raise ValueError('nonfinite pose or invalid time step')
        idx=self.indices
        if np.any(ref[idx]<self.lo-1e-8) or np.any(ref[idx]>self.hi+1e-8):raise ValueError('reference outside feedback bounds')
        ep=np.asarray(target_pos)-actual_pos;er=rotation_vector(target_rot,actual_rot)
        self.ik.fk(ref,self.side)
        mujoco.mj_comPos(self.ik.model,self.ik.data)
        jp=np.zeros((3,self.ik.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacBody(self.ik.model,self.ik.data,jp,jr,self.ik.palm_bid[self.side])
        cols=self.ik.dadr[idx]
        jac=np.vstack([5.*jp[:,cols],jr[:,cols]])
        error=np.r_[5.*ep,er]
        velocity=np.linalg.solve(jac.T@jac+.02*np.eye(len(idx)),jac.T@error)
        # Limit the whole vector together to preserve its local direction.
        velocity*=min(1.,self.speed/max(np.max(np.abs(velocity)),1e-9))
        candidate=ref.copy();candidate[idx]=np.clip(ref[idx]+velocity*dt,self.lo,self.hi)
        return candidate,dict(position_error=float(np.linalg.norm(ep)),rotation_error=float(np.linalg.norm(er)),
            clipped=bool(np.any(np.abs(candidate[idx]-(ref[idx]+velocity*dt))>1e-9)),
            excursion=float(np.max(np.abs(candidate[idx]-self.initial[idx]))))
