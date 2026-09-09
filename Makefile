# Developer checks (docs/reference/standards.md). `make check` is what pre-commit runs.
PY_SIM := third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
RUFF   := /venv/main/bin/ruff

.PHONY: check format lint test test-sim gate-bottle gate-crate

check: lint test

format:
	$(RUFF) format .

lint:
	$(RUFF) format --check .
	$(RUFF) check .
	python3 scripts/lint_sizes.py
	python3 scripts/lint_env.py

test:
	$(PY_SIM) -m pytest -x -q tests

test-sim:
	$(PY_SIM) -m pytest -x -q tests --run-sim

# Behaviour gate: run the candidate first (see scripts/gate.py --help), then compare.
gate-bottle:
	python3 scripts/gate.py compare bottle "$(MANIFESTS)" tests/gate/baseline_bottle.json --json outputs/gate/bottle.json
gate-crate:
	python3 scripts/gate.py compare crate "$(MANIFESTS)" tests/gate/baseline_crate.json --json outputs/gate/crate.json
