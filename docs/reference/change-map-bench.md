# Public benchmarks and the run switches (selfwm TODO 38.11)

A run on a public benchmark is one set of switches: the benchmark, the whole-body controller, where the robot's
knowledge of objects and grasps comes from (simulator truth or perception), perception's input (stereo pair or one
RGB-D camera, with or without depth noise), where the robot believes it is, and the pelvis ScaleBFM is told.
The switch layer is the bench-options branch's (its TODO 49.1), so both branches switch a run the same way; selfwm
adds the controllers it runs (SONIC, AMO, GR00T-WBC) and couplings between axes.
`python -m locomani.bench table` prints what each benchmark supports and what runs here yet. The default combination
(2026-11-24, user): SIMPLE, ScaleBFM, one RGB-D camera. Commands: [benchmark runbook](../selfwm_bench.md).
Back to [domain navigation](change-map.md). Test names are relative to `tests/`.

| Change | Entry / owner | Configuration or example | Tests |
|---|---|---|---|
| Run switches: choices, resolution (code defaults, profile, `--bench-set`), consistency, couplings, signature, record | [options.py](../../src/locomani/bench/options.py) | [configs/bench/](../../configs/bench/simple.yaml) | `test_bench_options.py` |
| A benchmark's capability table (ready / planned per value, with reasons; couplings) | [spec.py](../../src/locomani/bench/spec.py), `bench/<name>/__init__.py` | `python -m locomani.bench table` | `test_bench_options.py` |
| SIMPLE: the switches onto the pick bed's own settings (column, cached look, controller, root) | [bench/simple/](../../src/locomani/bench/simple/__init__.py), [simple_bench.py](../../scripts/experiments/simple_eval_20261112/simple_bench.py) | `--bench-set controller=sonic`, `--bench-set target=truth`, `--bench-set sensor=stereo`; shell runners: `python -m locomani.bench.simple` | `test_bench_options.py` |
| SIMPLE: the RGB-D look's full frame for the path planner (the stereo look's comes from `look_all.sh`) | [look_frame.py](../../scripts/experiments/simple_eval_20261112/look_frame.py) | `outputs/simple_eval_20261112/look_dump_blind/` | the per-episode frame check it prints |
| SIMPLE: the S arm (J checked by the controller's self model, repaired by replanning, given up when it cannot be) | [s_check.py](../../scripts/experiments/simple_eval_20261112/s_check.py), [s_run.sh](../../scripts/experiments/simple_eval_20261112/s_run.sh), `bench_via.sh` step `S` | `<bench dir>/S*/` (state, rounds, choice, via) | `s_check.py state` prints the rebuilt reference against the film's; `s_check.py table` |
| SIMPLE: development episodes (SIMPLE's unused hard levels) in a tree of their own | `SIMPLE_EVAL_OUT`, `SIMPLE_LEVELS` in [episodes.py](../../scripts/experiments/simple_eval_20261112/episodes.py); [s_dev.sh](../../scripts/experiments/simple_eval_20261112/s_dev.sh) | `outputs/simple_dev_20261124/` | the default levels and tree unchanged |
| The controller and the root an entry passes to the self-model environment | [env.py](../../scripts/experiments/selfwm_20261029/env.py) (`Env(scene, controller, root)`) | `REACH_CONTROLLER` / `BFM_ROOT` when not passed | `test_bench_options.py` |
