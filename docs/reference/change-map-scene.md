# Scene tasks on ScaleBFM (scene RSI, TODO 27)

The line's plan and decisions: [scene RSI plan](../handoff/scene-rsi-plan-20261016.md). A scene file (ADT apartment on
mars), the ScaleBFM motor layer, tasks judged by goal predicates, and the skills the loop composes and writes.
Use the matching row; expand along actual callers only as the change requires.
Back to [domain navigation](change-map.md). Test names are relative to `tests/`.

| Change | Entry / owner | Configuration or example | Tests |
|---|---|---|---|
| Motor primitives (walk, turn, sidestep, posture, reach, grip, release, hold; mode per primitive) | [motor.py](../../src/locomani/backends/scalebfm/motor.py) | `MotorConfig` in motor.py | `test_scalebfm_motor.py`; live `scripts/experiments/scene_rsi_20261016/motor_check.py` |
| Scene world, start sampling, navigation, one round | [world.py](../../src/locomani/tasks/scene/world.py), [start.py](../../src/locomani/tasks/scene/start.py), [navigate.py](../../src/locomani/tasks/scene/navigate.py), [episode.py](../../src/locomani/tasks/scene/episode.py) | Demo 1 `tasks.jsonl` rows | `test_scene_task_runner.py`; live `scripts/experiments/scene_rsi_20261016/episode_check.py` |
| Skills and goal-to-plan mapping | [skills/](../../src/locomani/tasks/scene/skills/__init__.py) | — | `test_scene_task_runner.py` |
| Goal predicates | [predicates.py](../../src/locomani/scene_discovery/predicates.py) | scene file | `test_scene_discovery.py` |
| Inner loop: pools, rounds, gate, driver, infeasibility claims and parks (D13), sealed pool, pictures and videos | [scene_rsi_20261016/](../../scripts/experiments/scene_rsi_20261016/loop.py) (`family.py`, `scene_round.py`, `batch.py` with the success-rule version, `gate.py`, `loop.py`, `claim_check.py`, `sealed.py`, `visualize.py`) | `families/<family>.json`, `rules.md`, `prompts/` | `test_scene_rsi_loop.py` |
| ADT scene build (MuJoCo; furniture collides with the robot only) | [adt_to_scene.py](../../scripts/adt_to_scene.py) | `configs/scenes.yaml` `adt_apartment_mars` | live rebuild + `motor_check.py pick` |
