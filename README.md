# locomanipulation

A Unitree G1 demonstration and VLA training repository. Application code supports
SONIC-based bottle/crate tasks and an in-process ScaleBFM bottle pipeline.

## Development

Start with [AGENTS.md](AGENTS.md), then select the relevant row of the
[change map](docs/reference/change-map.md). It identifies the actual code entry,
configuration owner and tests. For the design principles and module boundaries, read
[the standards](docs/reference/standards.md). When adding or changing a feature, follow
[the change workflow](docs/reference/change-map.md#change-workflow); when adding docs,
use [documentation ownership](docs/README.md#documentation-ownership).

```bash
make check
```

This runs CPU development checks with the installed simulation Python environment.
It does not certify physical behavior or historical experiment results.

## Running the project

- Environment and assets: [setup](docs/setup.md).
- Maintained command entry points: [runbook](docs/reference/runbook.md).
- Task tutorials and historical evidence: [document index](docs/README.md).
- Active work/resources: [STATE](docs/STATE.md); migrations: [ROADMAP](docs/ROADMAP.md).

## Layout

| Path | Purpose |
|---|---|
| `src/locomani/pipeline/` | Run orchestration, conversion, training and evaluation |
| `src/locomani/tasks/` | Task sampling, decisions and acceptance |
| `src/locomani/backends/` | SONIC / ScaleBFM integration |
| `src/locomani/primitives/`, `sim/` | Shared calculations and simulator integration |
| `configs/` | Task/run profiles and GR00T modality definitions |
| `tests/`, `scripts/` | Checks, launchers and reproducible experiment sources |
| `src/gr00t/` | Vendored GR00T code; see its UPSTREAM.md |
| `outputs/`, `third_party/` | Untracked artifacts and external installations |

Earlier README text is preserved in [archive](docs/archive/readme_20260912.md).
Historical success rates are claims tied to particular protocols; see the
[evidence rules](docs/reference/standards.md#evidence) before using them as baselines.
