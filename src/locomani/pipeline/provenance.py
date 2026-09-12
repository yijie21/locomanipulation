"""`<out>/config_snapshot.json` -- the provenance every pipeline entry writes
(decision 8): the four config dicts (task/backend/eval/pilot -- `None` for
whichever this entry point does not construct), the git sha, the start time,
and a sha256 digest of every src/configs/scripts source file -- the same
coverage run_learn_pilot.py's retired `provenance.json` had. Replaces that
file; nothing writes `provenance.json` any more.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from locomani import paths


def _digest(p: Path) -> str:
    sha = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def source_digests(root: Path | None = None) -> dict[str, str]:
    root = Path(root or paths.REPO)
    files = sorted(
        p
        for b in ("src/locomani", "configs", "scripts")
        for p in (root / b).rglob("*")
        if p.suffix in (".py", ".json", ".sh")
    )
    return {str(p.relative_to(root)): _digest(p) for p in files}


def file_digests(directory) -> dict[str, str]:
    """{file name: sha256} of every regular file directly under `directory` (a frozen scene dir)."""
    d = Path(directory)
    return {p.name: _digest(p) for p in sorted(d.iterdir()) if p.is_file()} if d.is_dir() else {}


def digest(path) -> str:
    return _digest(Path(path))


def sonic_policy_identity(env: dict | None = None) -> dict:
    """Content identity of the resolved checkpoint, including equal-size weight changes."""
    env = dict(os.environ) if env is None else env
    policy = env.get("SONIC_POLICY", paths.SONIC_POLICY)
    deploy = Path(env.get("SONIC_DEPLOY_ROOT", paths.SONIC_DEPLOY))
    binary = Path(env.get("SONIC_DEPLOY_BIN", "./target/release/g1_deploy_onnx_ref"))
    deps = Path(env.get("DEPS_DIR", str(Path(env.get("LOCOMANI_WORKSPACE", paths.WORKSPACE)) / "deps")))
    if env.get("LOCKSTEP") == "1":
        binary = (
            Path(env.get("LOCKSTEP_DEPLOY_ROOT", str(deps / "gear_sonic_deploy_ours_lockstep")))
            / "target/release/g1_deploy_onnx_ref"
        )
    elif env.get("CLEAN_DEPLOY") == "1":
        binary = (
            Path(env.get("CLEAN_DEPLOY_ROOT", str(deps / "gear_sonic_deploy_ours_lockstep_nvdef")))
            / "target/release/g1_deploy_onnx_ref"
        )
    if binary.is_absolute():
        deploy = binary.parent.parent.parent
    d = deploy / policy
    cfg = d / "observation_config.yaml"
    out = {"policy": policy, "dir": str(d), "exists": d.is_dir()}
    if cfg.is_file():
        out["observation_config_sha256"] = _digest(cfg)
    for name in ("model_encoder.onnx", "model_decoder.onnx"):
        f = d / name
        if f.is_file():
            out[f"{name}_bytes"] = f.stat().st_size
            out[f"{name}_sha256"] = _digest(f)
    return out


def record_deploy(receipt, binary, policy) -> dict:
    """Append content identity from the launcher's final cwd, before starting deploy."""
    from locomani.pipeline.stages import recorded_settings

    cwd = Path.cwd()
    identity = sonic_policy_identity({"SONIC_DEPLOY_ROOT": str(cwd), "SONIC_POLICY": policy})
    executable = Path(binary).resolve()
    identity.update(
        binary=str(executable),
        binary_sha256=_digest(executable),
        cwd=str(cwd),
        time=time.time(),
        settings=recorded_settings(dict(os.environ)),
        source="launcher after resolving deploy cwd; before exec",
    )
    path = Path(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(identity) + "\n")
    return identity


def checkpoint_identity(path) -> dict:
    """Hash every checkpoint file, including sharded weights and normalization data."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    files = {p.relative_to(path).as_posix(): _digest(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError(f"checkpoint directory is empty: {path}")
    return {"path": str(path), "files": files}


def git_sha(root: Path | None = None) -> str:
    root = Path(root or paths.REPO)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()


def write_config_snapshot(
    out_dir,
    *,
    task_cfg=None,
    backend_cfg=None,
    eval_cfg=None,
    pilot_cfg=None,
    started: float | None = None,
    inputs: dict | None = None,
    runtime_env: dict | None = None,
) -> Path:
    """Write `<out_dir>/config_snapshot.json`; returns the path written.

    `inputs`: identity of external inputs the source digests cannot cover -- the frozen scene
    files, the initial token, a reach calibration -- as {name: sha256 or {file: sha256}}
    (2026-09-10 review: the retired provenance.json hashed these; the snapshot had dropped them).
    """
    from locomani.pipeline.stages import recorded_settings

    runtime_env = dict(os.environ) if runtime_env is None else runtime_env
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = out_dir / "config_snapshot.json"
    snap.write_text(
        json.dumps(
            dict(
                identity_scope="pre-launch configuration; deploy_receipts.jsonl records actual launch selection",
                task=task_cfg.to_dict() if task_cfg is not None else None,
                backend=backend_cfg.to_dict() if backend_cfg is not None else None,
                eval=eval_cfg.to_dict() if eval_cfg is not None else None,
                pilot=pilot_cfg.to_dict() if pilot_cfg is not None else None,
                sonic_policy=sonic_policy_identity(runtime_env),
                runtime=recorded_settings(runtime_env or {}),
                git_sha=git_sha(),
                started=started if started is not None else time.time(),
                sources=source_digests(),
                inputs=inputs or {},
            ),
            indent=1,
        )
        + "\n"
    )
    return snap


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record the launcher's final deployment identity")
    parser.add_argument("--deploy-receipt", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    record_deploy(args.deploy_receipt, args.binary, args.policy)
