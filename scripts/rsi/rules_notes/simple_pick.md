## Robot, controller, objects
A Unitree G1 with a three-finger Dex3 hand on each arm, in two models, under the frozen ScaleBFM controller ([envelope]).
XMovePickTeleop, HandoverTeleop and LocomotionPickBetweenTablesTeleop use the model ScaleBFM was trained on. BendPickMP and
TabletopGraspMP use SIMPLE's AMO model of the same robot: its arm joints are position servos (the bed sends them the target
angle, see the fixed switches below), and its finger joints have ten times the damping (0.5 against 0.05) and five times the
armature, so the fingers lag a fast close even in the air, and a close that stops when the fingers fall behind the command can
stop before they touch. SIMPLE drives the AMO hand up to 3-5 N m, the other model's hand up to 0.7 N m. A difference between
the two MP tasks and the other three that the controller does not explain may come from the model (`BED` report, [escalate]).
The right hand picks; in HandoverTeleop the left hand takes the object. How hard fingers hold is how far past the contact the
joints are told to go.
Every target object weighs 0.1 kg whatever its size (the mass is split over the object's convex parts; friction 0.8, stiff
contact). Holding that is easy; keeping the object still while the hand arrives and closes is not: the first finger or palm to touch
pushes or turns it, a fast close flicks it, a tall box tips from a touch high up.
`simple_fake` is simple_q on a fake bed (no simulator): its numbers mean nothing about the robot.

## Perception and the grasp model
Head stereo colour images (depth from them), a segmentation of the target, joint readings, leg odometry. The grasp comes from a grasp model (GraspGen, Dex3 palm poses with a score) on the target's visible points inside the
segmentation: the front and the top, so the middle of those points is not the middle of the object. The seed takes the
`grasp_k`-th candidate in score order that passes the reach tests (`ahead_cos`, `roll_lo` to `roll_hi`, `clear`); nothing in
it looks at where on the object the candidate grips, and the score knows nothing of the weight or the hand's lag. A grip off
the middle of the face the fingers close on is touched by one finger first, which pushes and turns a 0.1 kg object. The card's `grasp chosen` line gives the palm's place from the middle of the seen points in heading axes (ahead, left, up) and
the palm's normal: normal down, the ahead and left parts say how far off the middle it grips; normal sideways, ahead and up.

## What each task asks, and what compliance means
- BendPickMP (8 s): "bend to pick up the cracker box". The right hand takes the box off its support and lifts it; the check is
  a rise of about 9 cm.
- XMovePickTeleop (16 s): "move forward to pick up the {object}". Walk to the table, pick up; rise about 9 cm.
- TabletopGraspMP (16 s): "grasp the object on the tabletop". Rise about 2 cm. The check is weak: it holds when closing
  fingers tip the object up on the table, when the object is knocked into the air, and in a few episodes by itself before the
  robot arrives. None of that is a win.
In the three pick tasks the object must be held: touching a hand and nothing that is neither the robot nor the container, for
at least 0.2 s in a row, the first of those steps inside the limit. Resting on or leaning against a support, or pushed up by
the body, is not held. A step where the check holds without that does not end the episode: the skill goes on, and the round is
won when a held run of 0.2 s begins inside the limit. The record has `t_check` beside `t_success`; `comply.why` is `no_hand`,
`on_support` or `short` (held for less than 0.2 s).
- HandoverTeleop (16 s): "Hand over {object} from right hand to left hand and place it on the container". The check is only a
  contact between the object and the container held for a short time, so an object dropped, pushed or thrown in passes it.
  Compliance: the object was on the right hand in contact with nothing but the robot, then for at least 0.2 s on the left hand
  with no contact with the right hand or anything else, all before the first contact with the container. The left hand alone
  then carries it in.
- LocomotionPickBetweenTablesTeleop (24 s): "pick up the {object} from table1, locomotion to table2, and place on table2". The
  check holds when the object rests on the container within about 12 cm of its centre with no hand on it. Compliance: at some
  moment the object was in a hand touching nothing but the robot (lifted and carried, not slid), and it does not stay in
  contact with a hand at the end.

## The seed and its flow
A straight line of phases: walk to a stance, reach (stepped: behind the pregrasp, the pregrasp, the grasp; each step waits for
the palm to come near the target, and every wait costs clock), close, lift; the two placing tasks carry the object, walk to the
container, lower it, let go and take the hand away. Speeds, waits, tolerances and offsets are parameters. The seed has no hand-over flow; a candidate may add the one file `handover_simple.py` for it, and
it may hold none of the refused names.You see 15 episodes per task.
Fixed switches on, with their measured effect found by hand: `amo_arm_pos` (the AMO arms get the target angle, not a torque;
BendPickMP went from 0 to 6 of 10) and `feed_torque` (leg odometry reads the feet's loads from the torque command, else zero;
PickBetweenTables release error went from about 20 cm to 1 cm). Rounds from before 2026-11-23 were made without them: read
their numbers as from another bed.

## Level 2 here
`stance_d` (m, added to the stand-off from the target or table edge), `stance_t` (m, shift to the robot's left along the
table's edge), `stance_yaw` (deg, added to the heading), `stance_d2`, `stance_t2`, `stance_yaw2` (the same at the placing
stance) and `stance_here` (true: no walk).

## Reading the classes here
- `error`: the round did not run; a bug, fix the code, not a number.
- `timeout`: the check passed after the limit; find the phases that dominate the clock (a wait for an already settled hand, a
  slow leg, a second attempt).
- `fall`: a walk with an arm out, a lean from a reach too far from the stance, a push by the table.
- `perceive`: target or container not found (glossy surface, small object, view blocked by the arm or the load); remedy a better look (stance, heading, clean-up of flying points), not a guess.
- `deliver`: palm more than 2 cm from the grasp or release point, or no reachable candidate left. Tell apart a reach the
  controller did not deliver, a candidate out of the hand's reach (choose by what the arm can deliver) and a target the
  camera read wrongly (the palm arrived where the cloud said).
- `collide`: the target pushed 3 cm or tipped 20 deg before the fingers closed, or the container pushed 3 cm before release:
  the hand's entry path, the idle hand, the body leaning in, a thumb, or the first finger of a grip off the middle.
- `grasp`: closed and the object did not come up or was lost: did the fingers stop on it, was the lift gentle; a candidate
  nearer the middle; on the AMO model, did the fingers reach it before the close stopped.
- `place`: let go and it did not stay: foot height over the rim, how far in, how fast the hand opens and leaves, a tall or light
  object tipping on the near side.
- `comply`: the check held but the instruction was not done (Handover: no transfer between the hands; pick tasks: a lift by
  the body or a support). The cause is in the flow.
