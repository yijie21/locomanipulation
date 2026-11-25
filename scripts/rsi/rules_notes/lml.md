## Robot, controller, objects
A Unitree G1 with a three-finger Dex3 hand on each arm under the frozen ScaleBFM controller ([envelope]); LocoManiLab (`lml_q_bench`: twenty walk/lift/put/move tasks; `lml_fridge_bench`: three door tasks; `lml_drawer_bench`: three drawer tasks). How hard fingers hold is how far past the contact the joints go, and the hand's torque limit is low against a pinch: a palm
under the object carries weight better than a pinch.
The objects are light (friction 0.9): apple 10-15 g, cup 9 or 69 g, bread 26 g, candle 30 g, can 51 g, boxed food 54 g, sponge
22 g, book 115 g, basket 306 g. Holding that is easy; keeping the object where
it is while the hand arrives and closes is not: the first finger or palm to touch it pushes or turns it, a fast close flicks
it, a tall object tips from a touch high up.
A door turns about its hinge: a hand on it follows the arc, and the door must stay past the goal for the hold time.
A drawer slides straight out toward the robot; its handle is a bar low on the front (centre about 0.27 m above the floor),
below where standing hands hang: how the robot stands to get a hand low (the pelvis's drop, the torso's lean: the motor
layer's `posture` and the `pelvis_z` of `reach`) is level 2; what no such stance delivers is level 3 ([escalate]).
Fixed switch `feed_torque` (feet loads from the torque command; without it a moving foot reads as half a pelvis move): by hand
the scrub brush's hand estimate at the end of the reach went from 8 cm off to 1.5 cm.

## Perception
Head camera colour and depth, a segmentation prompted from the instruction's words, grasp candidates on the target's visible
points, joint readings, leg odometry. A task's scene is the same in practice and in the final evaluation with objects moved a few
centimetres, so a position in the room's frame written as a number (stand point, target spot, heading to furniture) is
knowledge of the evaluation, not of the robot: never write one, whatever its source. Offsets measured from what the camera
sees are fine; review keeps this.

## What each task asks, and what compliance means
A win needs `success`, `comply` and `genuine` (the check did not already hold at the start; false counts as `error`). Checks
hold for about a second unless said.
- Walking tasks (ApproachDesk, ApproachToilet, ApproachBed, ApproachFridge, ApproachSofa): base within about 1 m of the desk's
  centre (0.2 s suffices), within 35 cm of the toilet or the bed, or in an area a metre or two across beside the
  fridge or sofa. Compliance: upright when the check holds (`fell`); an earlier fall with a get-up is fine.
- Lifting tasks (LiftSoapBottle, PickBookFromDresser, LiftBasketFromFloor, LiftScrubBrush): the object about 30 cm above its
  start (all but the book also need a hand closed on it). Compliance:
  on a hand and touching nothing but the robot for 0.2 s in a row. `why`: `short`, `on_support` (a hand and
  something else), `no_hand`.
- Putting and moving tasks: MoveCupToCenter (cup 15-30 cm from the desk's centre), MoveCupToCorner (within 12 cm
  of a robot-side corner), PutSpongeInBathtub (in the basin, no hand on it), BoxedFoodIntoBasket and CanIntoBasket (inside and
  touching the basket, no hand within 18 cm), AppleIntoBox (apple's middle in the box; the check allows a hand still holding
  it), BendAndPutAppleOnPlate (on the plate within 13 cm of its centre; bending is not checked), BreadOnPlate (the same),
  CupToSink (in the sink, no hand within 25 cm), BookIntoBasket and CandleToBasket (touching the basket within 12 cm of its
  centre, no hand within 25 cm). Compliance: carried first, on a hand
  and touching nothing but the robot for at least 0.1 s in a row (`not_carried`), and no hand on the object when the check
  holds (`hand_on`). A hand is a wrist or finger link; forearm, torso, legs are the robot.
- Door tasks (the fridge's left door; fraction = angle over 90 deg): OpenFridge (starts at 0.10,
  needs 0.50 or more for a second), CloseFridge (starts 0.30, needs 0.10 or less), ApproachOpenFridge (starts closed, needs
  0.40 with any part of the robot within 0.75 m of the fridge, no hold). Compliance: not fallen (`fell before the door check
  held`); which part moves the door is free.
- Drawer tasks (the dresser's lower-left drawer; fraction over 20 cm of travel): LowDrawerPnP (apple from the dresser top,
  drawer to 0.80 or more for 0.1 s, then apple inside and drawer at 0.20 or less for a second; drawer starts at 0.20, robot
  close), WalkAndOpenLowDrawer (robot within 0.35 m of the drawer, then drawer at 0.80 or more; the robot starts a few metres
  away with furniture between), WalkAndCloseLowDrawer (drawer starts at 0.80, ends at 0.20 or less). Compliance: not fallen and
  `genuine`, and the drawer moved by a hand: a hand among what touched it while it moved and no thigh, knee, shin or foot
  (`drawer moved by a leg`, `drawer not moved by a hand`; forearm or torso touching too is allowed). Crouching is recorded, not
  required. The apple task adds the putting rule for the apple.
The cards' `door:` / `drawer:` line: fraction at start, maximum, minimum, end, when it moved and entered the goal, which parts
touched it, the pelvis's drop (cm), the palm nearest the handle (distance, height above, sideways, fingers closed within 4 cm).

## The seed and its flow
Straight-line phases: walk, reach (stepped, slow near the object), grip, lift; putting tasks carry, lower, let go, take the
hand away. The seed's `articulate`
flow walks when asked and looks but moves no door or drawer: writing that flow is level 1 work. The pool is 30 starts per task
(levels 0-2), you see half.

## Level 2 here
Pick: `stance_d` (m added to the stand-off), `stance_t` (m, to the robot's left), `stance_yaw` (deg),
`stance_here` (true: no walk). Putting and moving tasks, the placing stance: `stance_d2`, `stance_t2`, `stance_stay` (true: no
second walk). The five walking tasks have no stance keys: where a walk ends is the skill's decision (level 1). Doors and drawers: the posture
keys above.

## Reading the classes here
- `error`: no run, or `genuine` false; a bug. `timeout`: the flow was still on its way (a wait, a repeated look). `fall`: also
  a success with `why` `fell`.
- `perceive`: not found; a better look or another phrase from the instruction's words.
- `navigate`: a walk ended outside the check: where it stopped against what the camera saw, the planner's clearance, a cut stop.
- `deliver`, `collide` (also `on_support`), `grasp` (also `no_hand`, `short`, `not_carried`), `place` (also `hand_on`): as usual.
- Door and drawer tasks add: `art_not_reached` (no palm within 5 cm of the handle, nothing touched the door; for drawers a hand,
  forearm or upper arm counts as reached, and a success refused because a leg moved it or no hand did lands here): reach not
  delivered, camera error, or a stance too far or too low. `art_no_move` (a hand got there; the fraction never moved 0.03):
  fingers did not hold, or a pull or push the hinge does not allow. `art_partial` (moved, never reached the goal): slipped off
  the arc, arm out of reach, body in the way. `art_not_held` (left the goal before the hold ended): swung back, or let go early.
- The apple drawer task goes by its steps: apple pick, drawer never at 0.80 (`art_*`), apple lost, drawer left open
  (`art_partial`), apple not inside (`place`), drawer shut under a second (`art_not_held`).
