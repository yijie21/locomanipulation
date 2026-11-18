# Two-hand lifts of hand-picked scanned objects and the carton, 30 cm and held: the plan the roles read (logistics_room_gso_bimanual, 2026-11-17)

This run's plan for the scene RSI inner loop's roles. Its section numbers are the ones the role prompts cite (0, 2, 3,
4, 7, 8). Everything but the goal's height and the scene is `gso_bimanual.md`'s: read that plan's sections 2, 3, 7 and
8 for the structure, the motor layer, the measurements and the starts that cannot be done.

## 0. Goal

The robot walks to the pallet platform and lifts a wide object off it with both hands, 30 cm up, level and steady.
This run is one task family, `gso_bimanual_lift`, one task `lift_X` per object of the scene
`logistics_room_gso_bimanual`: the two-hand objects the user picked by hand (user 2026-11-17, five to nine;
`assets/mujoco_scanned_objects/scene_picks.json`, from the catalog's `bimanual` objects of the training split after the
hand choices) and the open carton, placed with a fixed seed on the 0.371 m platform, every object within 0.35 m of an
edge and the carton on the hug's band (0.18-0.31 m in, long side along the edge). The goal:
- `held(X)`: the fingers of both hands on X and X off every surface;
- `lifted(X, 0.30, 60)`: X's lowest point 30 cm or more above the platform top, its upright axis less than 60 deg off
  vertical (`gso_bimanual` asks 20 cm);
- both checked every 0.1 s and holding at every check over the round's last 5 s (`judge_s`): the end of the plan and
  then 2 s standing still (`hold_s`).

Each start keeps 0.18 m free beside the walls the hug squeezes and in front, moves the other objects (`shuffle_others`)
and spawns the robot anywhere in the room it can walk from, 0.5 m or more away.

## 2. Structure

As `gso_bimanual.md` section 2; the start is the main skill library (`sandbox.seed` null).

## 3. What exists below the skills

As `gso_bimanual.md` section 3.

## 4. Decisions

As `gso_bimanual.md` section 4, with the goal of this plan's section 0. If the user finds 30 cm out of reach the
family is rewritten at 20 cm (`gso_family.py --lift 0.20`); a role does not lower it.

## 7. What the layers below deliver (measured)

As `gso_bimanual.md` section 7; none of it was measured at 30 cm or on these objects.

## 8. Starts that cannot be done (D13)

As `gso_bimanual.md` section 8.
