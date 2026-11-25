## Robot, controller, objects
A Unitree G1 with a Dex1 gripper on each wrist: two jaws closing on a pinch centre (the site `left_end_effector` /
`right_end_effector`). The pads are 6.6 cm long and reach about 3.9 cm past the pinch site, so the pinch site stops about 4 cm
short of a bar the jaws close on. Jaws open and close by one command each (0 open .. 1 closed); how hard they hold is the
command's depth. The arms are position controlled and sag under gravity: a joint target is reached only after a few steps, and
the hand ends a few millimetres to centimetres below the aim. The seed integrates the measured hand error for that, and only
once the joint targets have arrived (earlier it overshoots into the thing it reaches for).
There is no ScaleBFM and no keypoint tracker here, so [envelope] does not apply. The controller is BiGym's own: the action is
20-D at 50 Hz through `env.drive(...)`: (vx, vy, pelvis height, wz, the left arm's 7 joint targets, the right arm's 7, the two
grippers). The lower body is the benchmark's GR00T-WBC walking policy: it takes a velocity and a pelvis height and keeps the
robot upright; arms and grippers take absolute joint targets directly. The 21-D pitch tasks are mapped by the server and the
seed does not use them. The arm solution is `env.me.ik`: damped least squares on the arm's 7 joints to a hand target in the
pelvis frame, shoulder roll kept outward (>= 0.12 rad), else the shoulder-yaw link meets the torso.
Measured envelope: walking stops below a pelvis height of about 0.66 m (0.55 and 0.62 move 3 cm in 2 s; 0.68 and 0.74 move
22 cm), so a crouch to reach low things is 0.68 at the lowest if the robot has to walk afterwards. The seed walks at 0.2 m/s and
stops within a few centimetres of where it is told; a walk with the grip closed drags what the hand holds. Standing, the robot
stays. Every speed, distance and wait of a flow is a key of `K` in `skill.py`. A pull that outweighs the walk, a walk too
fast for the crouch, or a hand pushing the drawer while the legs move makes it fall.
Fixed switches: none.

## Perception
None needed and none to be built: the benchmark runs with `target=truth`. `env.view.goal()` gives, in the pelvis frame, the
reach's sphere {side, point} or the drawer's {handle, bar_axis, pull, travel, open_frac}. The head camera exists
(`env.cameras`) and is not used. The package knows nothing of the room beyond the goal; the episodes differ in the robot's
spawn pose and, on the drawer, the opening, so a goal value written as a number is a memorised episode. Offsets, distances and
angles measured from the goal and the robot's own readings (`env.me`: pelvis, hands, arm joints, tilt) are fine.

## What each task asks, and what compliance means
- `reach_target_single`, "Touch the red sphere with your left hand": a 5 cm sphere half a metre ahead of the settled pelvis,
  just out of the arm's reach. The check wants the left pinch centre in the ball for one second and no fall. Compliance needs
  no contact: `success` is the check and no fall.
- `drawer_top_open`, "Open the top drawer of the kitchen cabinet": the check wants the drawer at least 90 % open for one second
  and no fall. The handle is a thin plate (6 mm) standing 2 cm off the drawer face. Compliance, from the server's contact
  record (per control step, every robot body touching a scene body with joints of its own; no floor): a hand (a body whose name
  contains `dex1`: the base link and the two finger links) on the drawer on at least 5 steps (0.1 s), and any other robot body
  (wrist, forearm, elbow, torso, pelvis, legs) on it on fewer than 3 steps. A knee, a hip or a forearm that pushes the drawer
  passes the check and is a loss.
The row carries `success` (compliant; the win), `bench_success` (BiGym's own verdict), `why` (`""`, `not_success`, `fell`,
`no_hand` for a drawer opened with no hand on it, `body_push`, `error`), `hand_steps`, `body_steps`, `body_parts` (the robot
links that touched the drawer).

## The seed and its flow
The seed (`bigym2/seed/skill.py`) solves both tasks on most practice episodes; the loop looks for the starts where it does not
(a re-grasp that takes 990 steps instead of the usual 650, a drawer that stops short of 90 %, a hand outside the sphere). A
straight line of marks (the segments of [segments]). Reach: walk forward until the arm solution reaches the sphere, stop, move
the pinch centre there, hold. Drawer: walk to it, crouch so the shoulder is level with the handle, turn the hand so the jaws
open vertically (above and below the bar), put the pinch centre on the bar, close, walk backwards with the grip closed (arms
fixed in joint space) until the drawer is open, let go; the seed's pull ends at 96 % open.

## Level 2 here
There is none: the robot has no stance keys (a `stance_` key is refused) and where it stands is part of the flow. Levels are
level 1 (parameters, hand path, crouch height, order and timing of the marks, pull speed, how the jaws close, what a step
checks before the next) and level 3 ([escalate]: the walking policy does not walk below a pelvis height, does not hold a posture
under a pull, the arm's joint limits). The guard permits `env.view`, `env.me` and the clock here.

## Reading the classes here
- `error`: an exception in the package, a crash or the server's error (read the row's `error`); a package that raises ends its
  episode.
- `timeout`: the step budget ran out without the check: find the mark the clock went to; a retry that cannot finish in the
  budget left costs the same.
- `fall`: lower the speeds, keep the pelvis higher, release before the legs run out of margin.
- `comply`: the check held and the hands did not (`no_hand`, `body_push`): `body_parts` name the links that touched the drawer;
  bring the gripper (not wrist or forearm) to the handle and keep the arm off the drawer face.
- `short`: ended with the check not held and no budget run out (drawer 80 % open, hand outside the sphere): the last mark
  reached and what the flow did next say where it stopped.
- A re-run is close to deterministic here, yet a different step count in a walk or a re-grasp (990 against 650) shows contact can
  differ ([noise]).
