# Planner root waypoint over ZMQ (`WAYPOINT_TARGET_V1`)

The planner ONNX model accepts `has_specific_target`, `specific_target_positions[4,3]` and
`specific_target_headings[4]` (`planner_onnx.md`, Advanced Inputs), but the stock deploy never sets them.
`sonic_waypoint_target.patch` adds an optional planner-message field `target = [dx, dy, dyaw, id]`:

- the offsets are in the planner root's heading frame; when `id` changes, the planner thread converts the offset
  to an absolute planner-frame target from the reference root at the current frame, and holds it;
- while a target is active, the planner replans on its 1 s timer (static modes included) and passes the target
  to the model (all four waypoint frames get the same point); a message without `target` clears it;
- the deploy prints `[waypoint] id=… root=(…) target=(…)`.

Python: `agent_server.frame(..., target=(dx, dy, dyaw, id))`. An unpatched deploy ignores the field.

Diff base: the private bottle build `outputs/bottle_continuous_20260914/deploy` (pose and hand ownership).
It was built as the copy `outputs/deploy_waypoint_20260923`, whose absolute include paths were changed to
point at that copy, using the compile and link commands in its `build_commands.json`. Select it with
`SONIC_DEPLOY_ROOT`. Check: `strings target/release/g1_deploy_onnx_ref | grep '\[waypoint\]'`.

Measured 2026-09-23 in the walk scene (target 0.5 m ahead, SLOW_WALK 0.25, 10 trials):
- the planner's reference moves ~0.44 m, but the body moves only 0.18–0.28 m;
- re-sending the measured remainder does not move the body at all;
- the stops are square, with feet staggered 3.9 ± 2.0 cm, versus ~13 cm after an IDLE brake.

It is not a position controller. It is kept for the record, and no collection line uses it.
