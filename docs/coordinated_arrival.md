# Coordinated arrival (experimental, opt-in)

The historical collector remains the default. Select the new strategy with:

```sh
python src/teleop_free/grasp_gym.py \
  --reach-config configs/collection/coordinated_arrival.json \
  --rounds 10 --until-ok 1 --record --targets water_bottle \
  --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20 \
  --spawn-lateral .20 --edge-jitter .20 --manifest /path/to/manifest.jsonl
```

Use the simulation Python environment and start the normal encoder-enabled
collection stack first. Set `SIM_VIDEO_DIR` on that stack to record third-person
MP4s; `SIM_VIS=none` removes debug overlays. Existing strategies/configurations
need no migration; omitting the new config preserves the historical path.

The new controller prepares only the grasping arm during navigation. The other
arm is slightly bent, beside/behind the torso. It carries the owned upper-body
reference into the final geometry-gated approach and then into a simultaneous
crouch and pregrasp transport. It does not impose the historical READY_UPPER
finish hold or the above-pregrasp/up/down itinerary. The artificial lateral
approach detour and repeated planned-stand repair are disabled for this strategy;
random initial conditions and the live depth/cone gate remain.

World-space hand targets are converted through the measured pelvis pose at each
update. Position changes are slew-limited and publish corresponding velocities.
Both arm sweeps are screened against the table and target. Gripper orientation
sets a geometric height floor; approach preparation includes extra clearance for
body motion. Actual pregrasp position/orientation and delivered crouch/depth are
checked before calling the existing entry/close/lift servo. Failures stop motion
and clear upper-body velocity. The manifest records strategy selection and sweep
statistics. Failed attempts remain distinct from successful episodes.

Current scope is Dex1 tabletop grasping. The released planner delivers height
only in static squat mode: this strategy overlaps walking with arm preparation,
then crouching with reaching; it does not claim simultaneous locomotion and squat.
A successful preview is not evidence of robustness across the sampling range or
improved VLA training. Acceptance still requires paired batches and model tests.
