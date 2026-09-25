# Patch: SONIC deploy DDS domain from env

> **Superseded 2026-09-25**: this change is part of [`sonic_deploy_local_base.patch`](sonic_deploy_local_base.md)
> (commit 810e6de), which `docs/setup.md` step 3 applies. Do not apply both; this file stays as the record of
> what the change is.

`GR00T-WholeBodyControl/gear_sonic_deploy` is not tracked by this repo (see
`.gitignore`), but it carries one local modification that parallel data
collection depends on. Record it here so it can be re-applied after a fresh
checkout/build.

**File:** `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`

The stock binary hardcodes DDS domain 0, so two sim+deploy stacks on one host
cross-talk over lowstate/lowcmd. The patch reads the domain from
`SONIC_DDS_DOMAIN` (default 0 — byte-identical behaviour when unset):

```cpp
// near the top (around line 49):
#include <cstdlib>

// where ChannelFactory is initialised:
int dds_domain = 0;
if (const char* d = std::getenv("SONIC_DDS_DOMAIN")) dds_domain = std::atoi(d);
ChannelFactory::Instance()->Init(dds_domain, networkInterface);
// was: ChannelFactory::Instance()->Init(0, networkInterface);
```

Rebuild:

```bash
cd gear_sonic_deploy && cmake --build build --target g1_deploy_onnx_ref
```

`teleop_free/start_stack.sh` exports `SONIC_DDS_DOMAIN=$INSTANCE`; the sim side
picks the same domain via `run_sim_locomani.py` (`wbc_config["DOMAIN_ID"]`).

**Never run `gear_sonic_deploy/scripts/install_deps.sh`** — its `apt install
cuda` fallback pulls `cuda-drivers`, which overwrites the host-injected NVIDIA
driver on this Vast instance.
