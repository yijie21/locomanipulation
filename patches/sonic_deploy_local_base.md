# The local deploy commits every production binary is built on

`sonic_deploy_local_base.patch` is `git diff 1983e888 4b737df -- gear_sonic_deploy` (less one hunk, below) from the
`lockstep-port` branch of the original author's `third_party/GR00T-WholeBodyControl`: six commits
that were never pushed anywhere. Every deploy binary this repository's results were measured with
contains them. A build that skips them is a different controller, not the same one with fewer
options (ledger `docs/ledger/bottle.md` §38: upstream main plus only the ownership patches grasped
10/17 against 20/23, hips into the table 5/17 against 1/23).

Apply it to upstream **1983e88888217f6c69283cf3a9d1af01e87f07af** (the base the branch was cut
from), first, then the rest of the series in [setup step 3](../docs/setup.md):

| Commit | Change | Why it is needed here |
|---|---|---|
| 810e6de | DDS domain from `SONIC_DDS_DOMAIN` | parallel instances; same change as [`sonic_dds_domain.patch`](sonic_dds_domain.md), which this patch supersedes |
| d2f49c9 | planner-init timeout 5 s → 10 s | the sim's reset gap starves LowState; at 5 s the deploy stops itself |
| cd99cb5 | `SONIC_AUTO_START`, `--default-motion` (from SIMPLE) | every clean evaluation stack; without them the policy never engages and starts from the first clip (a dance) |
| 1573502 | publish the encoder token on `rt/wbc_token` | SIMPLE's teleop recorder; inert for this repository |
| d832834, 4b737df | `--sim-lockstep` and its rejection log | lockstep evaluation (`docs/archive/lockstep.md`); inert unless the flag is given |

cd99cb5 also set the shoulder-roll default angles to 0.0 (SIMPLE's choice). The production binaries
use NVIDIA's ±0.2 -- the original author's tree carries
[`sonic_policy_defaults.patch`](sonic_policy_defaults.patch) to undo it -- so that one hunk of
`policy_parameters.hpp` is left out here, and upstream's ±0.2 stays. Leaving it out rather than
applying the revert as a separate patch is what makes the series re-runnable: with the revert applied,
the base patch no longer reverse-checks as applied.

**Verify** after the whole series (`docs/setup.md` step 3 lists the commands): the patched
`gear_sonic_deploy/src` hashes to the value given there, and the binary carries the markers
`POSE_UPPER_OWNERSHIP_V2`, `PLANNER_HAND_OWNERSHIP_V1`, `SONIC_AUTO_START` and `SONIC_DDS_DOMAIN`.
The binary's own sha256 differs between machines (absolute paths are compiled in); the source hash
does not. Checked 2026-09-25: this series rebuilt the exact sources of the bottle/warehouse binary
`5cab2fb` and compiled with the documented cmake commands.
