## Robot, controller, objects
A Unitree G1 with a three-finger Dex3 hand on each arm under the frozen ScaleBFM controller ([envelope]); SIMPLE's
`g1_sonic` in all four tasks. Fingers close by joint targets; how hard they hold is how far past the contact the joints are
told to go, and SIMPLE drives each finger joint up to 0.7 N m. The objects are hinged assets (PartNet-Mobility); each hinge
has damping and dry friction:
- OpenOvenTeleop: a double wall oven with two glass doors, each on a hinge of 0 to 90 deg, damping 1, dry friction 4 N m on
  one door and 2 N m on the other: a door does not move until pushed or pulled harder than that.
- CloseDoorTeleop: a room door, hinge -10 to 80 deg, damping 2, dry friction 1 N m; it starts open at a random angle
  between 0 and 80 deg.
- OpenFaucetTeleop: a lever that turns up to 90 deg either way, damping 0.2, friction 0.3 N m; a small part.
- OpenTrashCanTeleop: a trash can with a hinged lid, 0 to 90 deg.
Every task also places an apple and a few other objects nearby; they play no part. Where each hinge's axis lies and which
way each part swings is not written here: the camera shows it, and a probe measures it.

## Perception
The part is found in the head camera's images: a segmentation model (SAM 3) is prompted with words for the part, and depth
comes from the stereo pair. A handle is small, a glass door shows what is behind it, a door leaf seen edge-on is a thin
strip, and the arm hides the part once the hand is on it. The hinge's place and swing direction are not given: the skill
infers them from what it sees (the part's outline, the frame) or from how the hand moves when it pushes or pulls (the
robot's own joint readings), never from the simulator.

## What each task asks, and what compliance means
The environment's checks read one hinge angle and nothing else. Compliance, from the start until the check first holds:
no part of the robot outside the two hands (the wrist's last link, the palm, the fingers) touches a moving part
(`comply.why` `body_touch`), and a hand touched a moving part and the part moved at least 5 deg toward the goal while a hand
was on it (`no_hand`). A part that swings on after the hand pushed it and let go is fine.
- OpenOvenTeleop: one of the two doors opens to at least 80 of its 90 deg, held for 9 steps.
- CloseDoorTeleop: the door is pushed shut against its stop (past -9 deg), held for 9 steps. 9 s in all.
- OpenFaucetTeleop: the lever turns at least 40 deg either way, held for 4 steps.
- OpenTrashCanTeleop: the lid opens at least 29 deg, held for 15 steps.
The record's `art` (hinge angle: start, furthest toward the goal, end, needed) and `art_touch` (the hand's contact with the
moving part) are the judge's facts, for the cards; the skill never sees them.

## The seed and its flow
The skill is a straight line of phases: walk to a stance in front of the object, reach the right hand to the part, close the
fingers on it (or set the palm against it for a push), move the part, let go and take the hand away (segments: walk, reach,
close, move, release). A step that missed is not followed by the next as if it had worked: check the measured state (palm
error, whether the fingers stopped on something, whether the hand moved when it pulled or pushed) before the next phase, and
give up or retry only within the time limit. Speeds, waits, tolerances, offsets and the path of the move are parameters (the
context's `parameters`, with the skill file's comments). The seed's `ART_TASK` chooses a task's parameters by its name. The
seed's author set it up on development episodes that are in none of the loop's pools; there it won 23 of 30 (oven), 13 of 30
(door), 9 of 30 (faucet) and 22 of 30 (trash can) by the compliance rule.

## Level 2 here
`stance_d` (m, added to the stand-off), `stance_t` (m, shift to the robot's left along the table's edge), `stance_yaw`
(deg, added to the heading), `stance_here` (true: no walk). To learn whether a factor matters, make a hypothesis with a
`delta` that sets it to a few values (there are no design probes or infeasibility claims here; an episode the present
skill cannot do is a failure to explain, and a level 3 request when the controller is the limit).

## Reading the classes here
- `collide`: a robot link outside the hands touched a moving part (the body walked into the door, a knee met an oven door
  swinging toward the robot, a forearm leaned on the lid): change the stance, the arm's path, or where the body is while the
  part moves.
- `deliver`: no hand touched a moving part. Tell apart a reach the controller did not deliver (palm error large though the
  command was fine: a nearer stance, a slower entry), a part the camera located wrongly (the palm arrived where the cloud
  said) and a hand that stopped on the frame or the fixed body.
- `grasp`: a hand touched the part but it moved less than 3 deg toward the goal: the hand slipped, pushed against the hinge or
  the wrong way, or held too weakly against the hinge's friction.
- `short`: the part moved toward the goal but not far enough (the move ended early, the hand lost the part on the arc, the path
  left the arc, time ran out), or the check held and the hand did not make the move.
- `timeout`: the check passed after the limit: find the phases that dominate the clock and shorten what buys nothing (a wait for
  an already settled hand), never a check the next phase relies on.
- `fall`: usually a walk with an arm out, a large lean from a reach too far from the stance, or a pull that drags the body.
