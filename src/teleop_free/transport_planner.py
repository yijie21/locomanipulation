"""Bounded joint-space transport search for diagnostic expert interventions.

The collision model freezes the measured base and target. Every returned edge
has been screened, but this is neither a whole-body nor dynamic certificate.
"""
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class TransportPath:
    ok: bool
    reason: str
    waypoints: list
    direct: bool = False
    checks: int = 0
    elapsed_s: float = 0.


class JointTransportPlanner:
    def __init__(self, guard, indices, lower, upper, *, seed=0,
                 max_nodes=256, timeout=1.5, step=.20, radius=.60):
        if max_nodes < 2 or timeout <= 0 or step <= 0 or radius <= 0:
            raise ValueError('Invalid transport search budget')
        self.guard, self.indices = guard, np.asarray(indices,dtype=int)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.seed, self.max_nodes, self.timeout = seed, max_nodes, timeout
        self.step, self.radius = step, radius

    def plan(self, state, goal, side):
        start=np.asarray(state['joints'],dtype=float).copy()
        goal=np.asarray(goal,dtype=float).copy()
        idx=self.indices;began=time.monotonic();checks=0
        def result(ok,why,path=(),direct=False):
            return TransportPath(ok,why,[q.copy() for q in path],direct,checks,time.monotonic()-began)
        if (goal.shape != start.shape or not np.isfinite([start,goal]).all()
                or np.any(goal[idx]<self.lower[idx]) or np.any(goal[idx]>self.upper[idx])):
            return result(False,'invalid goal')
        fixed=np.ones(len(start),dtype=bool);fixed[idx]=False
        if not np.allclose(start[fixed],goal[fixed],atol=1e-9):
            return result(False,'goal changes uncontrolled joints')
        def edge(a,b):
            nonlocal checks
            checks+=1;st=deepcopy(state);st['joints']=a.tolist()
            return self.guard.check(st,b,side).clear
        if not edge(start,start):return result(False,'initial posture lacks clearance')
        if not edge(goal,goal):return result(False,'goal posture lacks clearance')
        if edge(start,goal):return result(True,'direct path',[start,goal],True)
        lo=np.maximum(self.lower[idx],np.minimum(start[idx],goal[idx])-self.radius)
        hi=np.minimum(self.upper[idx],np.maximum(start[idx],goal[idx])+self.radius)
        rng=np.random.default_rng(self.seed)
        trees=[([start],[-1]),([goal],[-1])]
        def extend(tree,target):
            nodes,parents=tree
            near=int(np.argmin([np.linalg.norm((q-target)[idx]) for q in nodes]))
            delta=target[idx]-nodes[near][idx];length=float(np.linalg.norm(delta))
            if length<1e-9:return near,True
            new=nodes[near].copy();new[idx]+=delta*min(1.,self.step/length)
            if not edge(nodes[near],new):return None,False
            nodes.append(new);parents.append(near)
            return len(nodes)-1,length<=self.step
        def chain(tree,node):
            nodes,parents=tree;out=[]
            while node>=0:out.append(nodes[node]);node=parents[node]
            return out[::-1]
        turn=0
        while sum(len(t[0]) for t in trees)<self.max_nodes and time.monotonic()-began<self.timeout:
            active=turn%2;other=1-active;turn+=1
            target=start.copy()
            target[idx]=rng.uniform(lo,hi)
            node,_=extend(trees[active],target)
            if node is None:continue
            target=trees[active][0][node]
            while sum(len(t[0]) for t in trees)<self.max_nodes and time.monotonic()-began<self.timeout:
                connected,reached=extend(trees[other],target)
                if connected is None:break
                if reached:
                    a=chain(trees[active],node);b=chain(trees[other],connected)
                    path=a+b[-2::-1] if active==0 else b+a[-2::-1]
                    # Greedy shortening still checks every replacement edge.
                    short=[path[0]];i=0
                    while i<len(path)-1:
                        j=i+1
                        for candidate in range(len(path)-1,i+1,-1):
                            if time.monotonic()-began>=self.timeout:break
                            if edge(path[i],path[candidate]):
                                j=candidate;break
                        short.append(path[j]);i=j
                    return result(True,'detour',short)
        return result(False,'search budget exhausted')
