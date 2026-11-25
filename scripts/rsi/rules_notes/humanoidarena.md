## Robot, controller, objects
A Unitree G1 (29 degrees of freedom) with a three-finger Dex3 hand on each wrist. A hand takes one value (0 open .. 1 closed);
the fingers close by joint targets and the hand is not a pinch tool: the seed carries the box between two palms, squeezing from
the sides. The box is a 21 cm cube on a table whose top is 0.745 m. The arms are position controlled. The simulator (Isaac) is
slow: 7 to 10 control steps a second at 50 Hz on four cores, so an episode takes one to three minutes; the step budget is 1450.
There is no ScaleBFM and [envelope] does not apply. The controller is the native TWIST2 tracker, a learned whole-body policy
that follows a reference motion; the package gives a reference and the tracker turns it into leg and waist motion. What
follows, measured:
- The root step alone does not walk (the tracker follows the reference's legs); `env.gait` supplies a stride's leg joints
  (`gait_cycle.json`: the near-straight strides of TWIST2's example motions averaged, period 1.31 s, 0.80 m/s).
- Standing still while the arms move is NOT stable: with the legs held at the standing pose and only arm targets changing, the
  pelvis wandered 1 to 1.5 m and yawed up to 50 deg within 6 s. This is where the seed's episodes fall or wander; keep the legs
  stepping (the gait) while the arms work. A reaching or squeezing phase is the dangerous one.
- The policy's state reads the first two matrix columns of the root rotation; written as rows, the robot falls within 28
  steps.
- A walk with the gait stops within 2-3 cm of where it is told and steers; a turn in place is slower. The tracker follows the
  reference with a lag, so a hand's target is reached after a few steps.
The action is 40-D through `env.drive(action)`: the reference root's step (dx, dy in the heading frame; 2), its height (1), its
heading as the first two columns of a rotation matrix (rot6d; 6), the 29 canonical joint positions (29) and the two hand values
(2). `env.hold(joints, dxy, yaw, height, hands)` builds a native action that keeps the robot as it is; `env.me` gives the
pelvis, the wrists, the arm Jacobians and `policy_state()`. Every speed, distance and wait of a flow is a key of `K` in
`skill.py`. Fixed switches: none.

## Perception
None needed and none to be built: `target=truth`. `env.view.goal()` gives the box's pose and extent and the shelf layer's height
and rectangle in the world frame. The front camera exists (`env.cameras`) and is not used. The scene (table, box start, shelf)
is the same up to small differences, so a goal value written into code as a number is a memorised episode; offsets measured from
the goal and the robot's own readings are fine. The guard's facade is `env.view.goal()`, `env.me`, `env.drive`, `env.hold`,
`env.gait`, `env.mark`, the clock.

## What the task asks, and what compliance means
`pp_box`, "Pick up the box and place it on the shelf" (the only task; budget 1450 steps). The check: the box's bottom within
0.06 m of 0.648 m and its centre inside the shelf surface's rectangle (so the carry also lowers the box by 0.1 m from the
table). The check needs no contact: a shove, a kick or a body push passes it. Compliance (from the server's contact record, a
robot link pressing on the box with more than 1 N): a hand (a link whose name contains `hand`: base, palm, finger links) on the
box on at least 5 control steps (0.1 s), any other link (wrist, forearm, elbow, shoulder, torso, pelvis, legs) on it on fewer
than 3 steps, no fall, and the box moved at least 0.10 m. The row carries `success` (the win), `bench_success` (the reward
reached 1 before a fall), `why` (`""`, `not_success`, `fell`, `no_record`, `no_hand`, `body_push`, `not_moved`, `error`),
`hand_steps`, `body_steps`, `body_parts`, `obj_move` (the box's travel, m).

## The seed and its flow
The seed (`humanoidarena/seed/skill.py`) is a start, not a solution: it walks to the table and reaches, and loses the box or the
balance after that. Measured on 5 practice episodes (2026-11-24): all reach `at_table`; none places the box; three fall while
carrying. The loop looks for what keeps the robot upright while the arms work. The flow is a line of marks: settle (30 steps),
walk to the table's edge (one thigh's depth short, level with the box, facing it) = `at_table`, reach both wrists to the box's
two sides (a palm's length behind its middle, the arms raised by the Jacobian servo), squeeze in a few centimetres and hold
(`reached`, `squeezed`), lift both wrists (`lifted`), back off, turn to the shelf and carry, lower to the layer's height, step
in, open arms and hands, back away.

## Level 2 here
There is none: no stance keys (a `stance_` key is refused); where the robot stands is part of the flow. Level 1: the walk's
speed, ramp and stop, the reference's arm motion (how fast the arms move, how they are held while the legs stand), the hands'
path to the box, the squeeze depth, lift and carry heights, the order and timing of the marks. Level 3 ([escalate], the tracker
itself): it does not hold the standing pose while the arms move, does not carry the weight, falls at a speed the gait asks for;
give the pelvis's drift and yaw against the command as evidence. `BED` also covers a first frame after a reset that still shows
the last episode.

## Reading the classes here
- `error`: an exception, a crash or the server's error (read `error`); a package that raises ends its episode.
- `timeout`: ran the 1450 steps with the robot standing and the box on the table: find the mark the clock went to.
- `fall`: the arms moved while the legs stood, a carry too fast, a turn with the box held, a step into the shelf: keep the legs
  stepping, slow the carry, widen the margin to the shelf.
- `comply` (`no_hand`, `body_push`, `not_moved`): `body_parts` name the links that touched the box; carry it on the palms and
  keep forearms and torso off it.
- `short`: ended with the check not held and the budget not run out (box set down 5 cm short of the layer, off the rectangle):
  the last mark reached and what the flow did next say where it stopped.
- Isaac's contact and the learned tracker can make the same code from the same start finish differently ([noise]).
