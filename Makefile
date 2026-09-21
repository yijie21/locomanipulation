# Developer checks; make check and pre-commit share scripts/check_changes.py.
PY_SIM := third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
RUFF   := /venv/main/bin/ruff
TESTS  ?= tests

.PHONY: check check-all check-docs format lint test test-sim gate-bottle gate-crate config-doc run

check:
	python3 scripts/check_changes.py

check-all:
	python3 scripts/check_changes.py --all

check-docs:
	python3 scripts/lint_repository.py --navigation-only --changed-docs

format:
	$(RUFF) format .

lint:
	$(RUFF) format --check .
	$(RUFF) check .
	python3 scripts/lint_sizes.py
	python3 scripts/lint_env.py --strict
	python3 scripts/lint_repository.py --changed-docs
	python3 scripts/lint_mermaid.py
	python3 scripts/gen_config_reference.py --check

# Regenerate docs/reference/config_fields.md from the config dataclasses.
config-doc:
	python3 scripts/gen_config_reference.py

test:
	$(PY_SIM) -m pytest -x -q $(TESTS)

test-sim:
	$(PY_SIM) -m pytest -x -q $(TESTS) --run-sim

# One experiment through every stage, e.g. make run ARGS="configs/experiments/bottle_simple_200.yaml --gpu 1"
run:
	bash scripts/run_overnight.sh $(ARGS)

# Behaviour gate: run the candidate first (see scripts/gate.py --help), then compare.
gate-bottle:
	python3 scripts/gate.py compare bottle "$(MANIFESTS)" tests/gate/baseline_bottle.json --json outputs/gate/bottle.json
gate-crate:
	python3 scripts/gate.py compare crate "$(MANIFESTS)" tests/gate/baseline_crate.json --json outputs/gate/crate.json
