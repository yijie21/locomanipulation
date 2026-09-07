"""Isaac Sim rendering for MuJoCo-collected episodes.

Physics is always MuJoCo, wall-clock, with SONIC's ZMQ timing untouched. Isaac
only produces pixels, re-rendered from a recorded trajectory after the fact.

Kit boot time is the reason this is two layers:

* :mod:`frames` is pure data — parquet + pose-sidecar join, the 43-vector to
  joint-name map, the MJCF camera spec. numpy and pandas, nothing else, so it
  imports in well under a second and ``compute_cam_traj`` can use it without
  going anywhere near Isaac.
* :mod:`stage` is the Isaac half — stage assembly, state sync, camera mount,
  render step. Every isaacsim/pxr import lives INSIDE a function, because
  ``import isaacsim`` boots Kit and takes minutes.

Which venv: **both steps run under ``paths.RENDER_PY``**. Checked 2026-09-05 --
the sim venv has mujoco but no pandas, and only the render venv carries mujoco
and pandas together, so ``compute_cam_traj`` cannot run in the sim venv despite
what its docstring claimed before the migration.

Nothing here imports the ``simple`` package. See docs/rendering.md.
"""
