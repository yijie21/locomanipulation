import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free/scalebfm'))
from visual_pelvis_ablation import DiagnosticExecutor,InterventionConfig
from visual_executor import ActionExecutor

HAND={'left':(np.array([0,.2,.1]),np.eye(3))}
def row():
 r=np.zeros(29,dtype=np.float32);r[3:9]=r[12:18]=[1,0,0,0,1,0];r[9:12]=[.3,-.2,.1];r[28]=.0245;return r
class Tests(unittest.TestCase):
 def test_a_and_pretrigger_are_identical_to_original_executor(self):
  for arm in 'ACD':
   a=ActionExecutor(HAND);b=DiagnosticExecutor(HAND,arm)
   for k in range(120):
    p=np.array([k*.001,0,.76]);q=np.array([1.,0,0,0]);r=row() if k%2==0 else None
    x,g=a.step(p,q,r);y,h=b.step(p,q,r)
    self.assertEqual(g,h);np.testing.assert_array_equal(a.row,b.row)
    for key in x:
     for aa,bb in zip(x[key],y[key]):np.testing.assert_array_equal(aa,bb)
 def test_height_override_preserves_right_hand_and_gripper(self):
  cfg=InterventionConfig(start_s=0)
  a=ActionExecutor(HAND);b=DiagnosticExecutor(HAND,'B',cfg)
  for k in range(30):
   p=np.array([0,0,.65]);q=np.array([1.,0,0,0]);r=row() if k%2==0 else None
   x,g=a.step(p,q,r);y,h=b.step(p,q,r)
   self.assertEqual(g,h)
   for aa,bb in zip(x['right_wrist_yaw_link'],y['right_wrist_yaw_link']):np.testing.assert_allclose(aa,bb,atol=1e-7)
   np.testing.assert_array_equal(x['pelvis'][1],y['pelvis'][1])
   self.assertLessEqual(abs(b.v),cfg.speed_m_s+1e-8)
   self.assertAlmostEqual(y['pelvis'][0][1,2]-y['pelvis'][0][0,2],b.v*.02)
  self.assertGreater(y['pelvis'][0][0,2],.65)
 def test_d_holds_world_wrist_and_c_d_share_trigger(self):
  cfg=InterventionConfig(start_s=0)
  c=DiagnosticExecutor(HAND,'C',cfg);d=DiagnosticExecutor(HAND,'D',cfg)
  positions=[]
  for k in range(30):
   p=np.array([k*.001,0,.7]);q=np.array([1.,0,0,0]);r=row() if k%2==0 else None
   c.step(p,q,r);targets,_=d.step(p,q,r)
   if d.trigger is not None:positions.append(targets['right_wrist_yaw_link'][0][0])
  self.assertEqual(c.trigger,d.trigger);self.assertAlmostEqual(c.trigger,.08)
  for p in positions:np.testing.assert_array_equal(p,positions[0])
if __name__=='__main__':unittest.main()
