"""`Stack`: sim + deploy + exporter as supervised children (see the package docstring)."""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from locomani import paths
from locomani.sim.ports import instance_ports

READY_DEPLOY = "Init Done"
READY_DEPLOY_LOCKSTEP = "simulation-clock control thread active"
READY_AUTO_START = "transitioning to CONTROL state"
READY_SIM_IMAGE = "Image publish frequency"
READY_SIM_EPISODE = "[episode] :"  # privileged_pub.EpisodeControlServer is listening
LOWSTATE_TRIP = "LowState"  # the deploy stops itself on a one-loop LowState timeout (memory sonic-deploy-lowstate-trip)
DEPLOY_LIBS = ("zmq-install/lib", "onnxruntime-linux-x64-gpu-1.22.0/lib", "tensorrt-root/lib")


def session_root(instance: int) -> Path:
    """Logs and exporter sessions of one instance (the shell's `locomani_session_root`)."""
    return Path(paths.session_root(instance))


@dataclasses.dataclass(frozen=True)
class StackSpec:
    """Everything one stack needs, spelled out (no inherited shell variables)."""

    instance: int = 0  # port block 5555 + 10 * instance and DDS domain
    gpu: str = "0"  # CUDA_VISIBLE_DEVICES and MUJOCO_EGL_DEVICE_ID for the sim and the deploy
    scene_dir: str = paths.SIM_DIR  # frozen scene directory (scene_locomani.xml, scene_meta.json)
    prompt: str = "walk forward"  # the exporter's task prompt
    record: bool = True  # run the exporter (every stack may record: decision 3)
    policy: str = paths.SONIC_POLICY  # SONIC checkpoint directory relative to the deploy root
    deploy_root: str = paths.SONIC_DEPLOY  # deploy tree (binary, policy/, planner/, reference/)
    deploy_bin: str | None = None  # explicit deploy binary; None = <deploy_root>/target/release/g1_deploy_onnx_ref
    encoder: bool = True  # pass --encoder-file (off = token-only / VLA deployment mode)
    lockstep: bool = False  # --sim-lockstep deploy (simulation-clock control thread)
    auto_start: bool = False  # SONIC_AUTO_START=1 for the child (clean deploy without a planner warm-up)
    default_motion: str | None = None  # --default-motion clip name when the deploy needs one (lockstep/clean)
    deploy_extra_args: tuple[str, ...] = ()
    wrist_cams: bool = True  # exporter --record-wrist-cameras
    exporter_threads: int = 4
    video_dir: str | None = None  # SIM_VIDEO_DIR for the sim's video writer
    image_publish: bool = True  # sim renders and publishes the cameras (backend.render live)
    sim_vis: str = "none"  # SIM_VIS for the sim process
    scene_random: bool = False  # SCENE_RANDOM for the sim process
    dds_domain: int | None = None  # SONIC_DDS_DOMAIN; None = instance
    env: dict = dataclasses.field(
        default_factory=dict
    )  # extra environment for every child (backend knobs the sim/exporter read)
    receipts: str | None = None  # deploy provenance receipt file (LOCOMANI_DEPLOY_RECEIPTS)

    def ports(self) -> dict[str, int]:
        return instance_ports(int(self.instance))

    def scene_path(self) -> Path:
        """The scene directory as an absolute path (children run in other working directories)."""
        p = Path(self.scene_dir)
        return p if p.is_absolute() else Path(paths.REPO) / p

    def settings(self) -> dict[str, str]:
        """Every variable this stack decides for its children -- the only place that decision is made.

        Read it, and you know what the sim, the deploy and the exporter were told; write it next to
        the run (`Stack.up` does) and you can still answer that question afterwards. Chasing one
        variable across a resolver, a spec field and a filter is how `SIM_VIDEO_DIR` reached the
        exporter but not the sim, and the eval stage recorded no video for a day (2026-09-17).
        """
        env = {k: str(v) for k, v in self.ports().items()}
        env.update(
            INSTANCE=str(self.instance),
            SONIC_DDS_DOMAIN=str(self.instance if self.dds_domain is None else self.dds_domain),
            CUDA_VISIBLE_DEVICES=str(self.gpu),
            LOCOMANI_SIM_DIR=str(self.scene_path()),
            SIM_VIS=self.sim_vis,
            SCENE_RANDOM="1" if self.scene_random else "0",
            PYTHONUNBUFFERED="1",
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            SONIC_POLICY=self.policy,
            SONIC_DEPLOY_ROOT=str(self.deploy_root),
        )
        if str(self.gpu).isdigit():
            env["MUJOCO_EGL_DEVICE_ID"] = str(self.gpu)
        if self.video_dir:
            env["SIM_VIDEO_DIR"] = str(self.video_dir)
        if self.auto_start:
            env["SONIC_AUTO_START"] = "1"
        env.update({k: str(v) for k, v in self.env.items()})
        return env

    def child_env(self, base: dict | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        # owned by `settings`, which sets them only conditionally: an inherited value must not leak in
        for name in ("SIM_VIDEO_DIR", "SONIC_DDS_DOMAIN", "SONIC_AUTO_START", "MUJOCO_EGL_DEVICE_ID"):
            env.pop(name, None)
        env.update(self.settings())
        return env


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def _wait(predicate, timeout: float, interval: float = 1.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _log_has(path: Path, text: str, since: int = 0) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as f:
        f.seek(since)
        return text.encode() in f.read()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        # a zombie is not alive
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ")[1][0] != "Z"
    except OSError:
        return False


def instance_busy(instance: int) -> str | None:
    """Why `instance` cannot take a new stack now, or None when it can.

    Three things hold an instance: a stack whose state file still has live children, a starter holding its lock,
    and any program bound to one of its ports -- the Unreal bridge took instance 0's whole block on 2026-09-28,
    and a stack started there came up with a deploy that died on `Address already in use`."""
    state = session_root(instance) / "stack.json"
    if state.exists():
        try:
            pids = json.loads(state.read_text()).get("pids", {})
        except (OSError, ValueError):
            pids = {}
        live = {name: int(pid) for name, pid in pids.items() if _alive(int(pid))}
        if live:
            return f"runs a stack ({live})"
    lock = Path(paths.TMP) / "locomani_locks" / f"instance_{instance}.lock"
    if lock.exists():
        fd = os.open(lock, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "its lock is held by a stack starting up"
        finally:
            os.close(fd)  # closing drops a lock this probe took
    bound = sorted(port for port in instance_ports(int(instance)).values() if _port_open(port))
    return f"ports {bound} are in use" if bound else None


def _kill_group(pid: int, grace: float = 2.0, proc: subprocess.Popen | None = None) -> None:
    """TERM then KILL the whole process group a child was started as the leader of; reap it if it is ours."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            break
        if _wait(lambda: not _alive(pid), grace, 0.1):
            break
    if proc is not None:
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass


def spec_from_config(
    cfg: dict,
    *,
    run_dir,
    gpu: str,
    backend: dict | None = None,
    scene_dir=None,
    video_dir=None,
    record: bool | None = None,
) -> StackSpec:
    """The stack one expanded experiment config asks for (`configs/experiments/*.yaml`).

    `backend` picks the block, because the two stacks a run needs differ exactly there: collection
    runs the deploy WITH its encoder (without it the planner's arm and crouch references are
    dropped), evaluation runs it without (the token stream will not hold otherwise). Passing the
    block instead of reading a fixed key is what keeps that difference visible at the call site.
    """
    from locomani.backends.sonic.config import SonicBackendConfig

    run_dir = Path(run_dir)
    b = SonicBackendConfig.from_dict(dict(backend if backend is not None else cfg["backend"]))
    task, collect = cfg["task"], cfg.get("collect", {})
    # Which deploy tree runs is a property of the machine, not of the experiment: the streamed-pose
    # experts need one built with the hand-ownership patch, and on a box where that build lives
    # outside the tree SONIC_DEPLOY_ROOT points at it. Same precedence the previous path had.
    child: dict[str, str] = {"LOCOMANI_BACKEND_JSON": json.dumps(b.to_dict(), sort_keys=True)}
    if task.get("init_token_file"):
        child["TF_INIT_TOKEN_FILE"] = str(task["init_token_file"])
    return StackSpec(
        instance=int(collect.get("instance", 0)),
        gpu=str(gpu),
        scene_dir=str(scene_dir if scene_dir is not None else run_dir / "scene"),
        prompt=task.get("prompt") or "walk forward",
        record=bool(collect.get("record", True)) if record is None else bool(record),
        policy=os.environ.get("SONIC_POLICY", paths.SONIC_POLICY),
        deploy_root=os.environ.get("SONIC_DEPLOY_ROOT", paths.SONIC_DEPLOY),
        deploy_bin=os.environ.get("SONIC_DEPLOY_BIN") or None,
        encoder=not b.no_encoder,
        lockstep=bool(b.lockstep.enabled),
        auto_start=bool(b.clean_deploy or b.lockstep.enabled),
        # Same rule as `pipeline.stack.start_stack` and `stack.cli`: a deploy nothing warms up with a
        # planner command starts from this clip. This builder dropped it until 2026-09-23, so every
        # clean eval stack from `start_stack.sh eval` started from the deploy's own default instead.
        default_motion="neutral_kick_R_001__A543" if (b.clean_deploy or b.lockstep.enabled) else None,
        wrist_cams=bool(b.wrist_cams),
        video_dir=None if video_dir is None or b.render == "offline" else str(video_dir),
        image_publish=b.render == "live",
        sim_vis=b.sim_vis,
        scene_random=bool(b.scene_random),
        dds_domain=b.sonic_dds_domain,
        env=child,
        receipts=str(run_dir / "deploy_receipts.jsonl"),
    )


class Stack:
    """One instance's sim, deploy and exporter; `up()` blocks until each is ready, `down()` kills by group."""

    CHILDREN = ("sim", "deploy", "exporter")

    def __init__(
        self,
        spec: StackSpec,
        *,
        base_env: dict | None = None,
        record_command=None,
        run_dir=None,
        mode=None,
        settings_file: str = "stack_env.json",
    ):
        self.spec = spec
        self.root = session_root(spec.instance)
        # Where to leave stack_env.json, and -- through the state file -- which run this stack
        # serves: the process that attaches later checks it (`run_context.require_stack_of`).
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.settings_file = settings_file  # one per stack when a run collects in several lanes
        self.mode = mode  # "collect" / "eval" when a stage started it
        self.session: Path | None = None  # the exporter session the running exporter records into
        self.env = spec.child_env(base_env)
        self.pids: dict[str, int] = {}
        self._procs: dict[str, subprocess.Popen] = {}  # handles of the children THIS process started (reaped on stop)
        self._lock_fd = None
        self._record = record_command  # callable(name, argv, env) for the run's commands.jsonl

    def record_settings(self) -> Path | None:
        """Leave what the children were told next to the run, so it stays answerable afterwards."""
        if self.run_dir is None:
            return None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / self.settings_file
        path.write_text(json.dumps(self.spec.settings(), indent=1, sort_keys=True))
        return path

    # -- state file ---------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.root / "stack.json"

    def _save(self) -> None:
        state = dict(
            spec=dataclasses.asdict(self.spec),
            pids=self.pids,
            ports=self.spec.ports(),
            run_dir=None if self.run_dir is None else str(self.run_dir),
            mode=self.mode,
            session=None if self.session is None else str(self.session),
            root=str(self.root),
            owner=os.getpid(),
            started=time.time(),
        )
        self.state_path.write_text(json.dumps(state, indent=1))

    @classmethod
    def attach(cls, instance: int) -> Stack:
        """The live stack of `instance`, from its state file (any process may call this)."""
        path = session_root(instance) / "stack.json"
        if not path.exists():
            raise FileNotFoundError(f"no stack state for instance {instance} ({path})")
        state = json.loads(path.read_text())
        stack = cls(StackSpec(**state["spec"]), run_dir=state.get("run_dir"), mode=state.get("mode"))
        stack.pids = {k: int(v) for k, v in state["pids"].items()}
        stack.session = Path(state["session"]) if state.get("session") else None
        return stack

    # -- lock -----------------------------------------------------------------
    def _require_instance_free(self) -> None:
        """Refuse to start on top of a stack that is still running, whoever started it.

        The flock below lives in the process that called `up()`, so it is released the moment that
        process exits -- which is exactly what `scripts/start_stack.sh` does: start the stack and
        return. The children outlive it, so from 2026-09-21 the question "is this instance taken"
        is answered by the state file's pids, not by who holds the fd.
        """
        path = self.state_path
        if not path.exists():
            return
        try:
            pids = json.loads(path.read_text()).get("pids", {})
        except (OSError, ValueError):
            return  # unreadable state tells us nothing; the flock and the port probes still guard
        live = {name: int(pid) for name, pid in pids.items() if _alive(int(pid))}
        if live:
            raise RuntimeError(
                f"instance {self.spec.instance} already runs a stack ({live}); "
                f"stop it with `python -m locomani.stack.cli down --instance {self.spec.instance}` "
                f"or pick another instance"
            )
        path.unlink()  # every child of the recorded stack is gone: the file is stale

    def _acquire_lock(self) -> None:
        lock_dir = Path(paths.TMP) / "locomani_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_dir / f"instance_{self.spec.instance}.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.read(fd, 200).decode(errors="replace").strip()
            os.close(fd)
            raise RuntimeError(
                f"instance {self.spec.instance} is held by a live stack ({holder}); pick another instance"
            )
        os.ftruncate(fd, 0)
        os.write(fd, f"stack pid {os.getpid()} at {time.strftime('%FT%TZ', time.gmtime())}\n".encode())
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    # -- children ---------------------------------------------------------------
    def _spawn(self, name: str, argv: list[str], *, cwd: str, env: dict) -> int:
        log = self.root / f"{name}.log"
        with log.open("ab") as f:
            f.write(f"\n===== {name} start {time.strftime('%FT%TZ', time.gmtime())} =====\n".encode())
            f.write(("COMMAND " + json.dumps(argv) + "\n").encode())
            # stdin is /dev/null, never the starter's terminal: the deploy's input thread does a
            # blocking read(STDIN_FILENO) every frame for its keyboard shortcuts, and on a tty that
            # read waits for a keypress -- so the 'start' command from the agent was never
            # consumed, the policy never engaged, and every round failed [stand] with the robot
            # limp on the floor (2026-09-25; only when start_stack.sh ran in an interactive shell).
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.pids[name] = proc.pid
        self._procs[name] = proc
        if self._record:
            self._record(f"stack_{name}", argv, env)
        self._save()
        return proc.pid

    def _fail(self, name: str, why: str) -> RuntimeError:
        log = self.root / f"{name}.log"
        tail = ""
        if log.exists():
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        self.down()
        return RuntimeError(f"stack {name}: {why}\n--- {log} ---\n{tail}")

    def _deploy_argv(self) -> tuple[list[str], str, dict]:
        s = self.spec
        deploy_bin = s.deploy_bin or os.path.join(s.deploy_root, "target", "release", "g1_deploy_onnx_ref")
        cwd = str(Path(deploy_bin).resolve().parent.parent.parent) if s.deploy_bin else str(s.deploy_root)
        policy_dir = Path(cwd) / s.policy
        if not (policy_dir / "model_decoder.onnx").exists():
            raise RuntimeError(
                f"{policy_dir}/model_decoder.onnx does not exist; set spec.policy to a checkpoint this tree has"
            )
        argv = [
            deploy_bin,
            "lo",
            f"{s.policy}/model_decoder.onnx",
            "reference/example/",
            "--obs-config",
            f"{s.policy}/observation_config.yaml",
        ]
        if s.encoder:
            argv += ["--encoder-file", f"{s.policy}/model_encoder.onnx"]
        ports = s.ports()
        if not s.lockstep:
            # A --sim-lockstep deploy refuses a planner model: the planner runs on its own wall-clock thread, which
            # the simulation clock never starts (docs/archive/lockstep.md, pitfall 1).
            argv += ["--planner-file", "planner/target_vel/V2/planner_sonic.onnx"]
        argv += [
            "--input-type",
            "zmq_manager",
            "--output-type",
            "all",
        ]
        argv += [
            "--zmq-host",
            "localhost",
            "--zmq-port",
            str(ports["TF_SONIC_PORT"]),
            "--zmq-out-port",
            str(ports["TF_OUT_PORT"]),
            "--disable-crc-check",
        ]
        if s.lockstep:
            argv.append("--sim-lockstep")
        if s.default_motion:
            argv += ["--default-motion", s.default_motion]
        argv += list(s.deploy_extra_args)
        env = dict(self.env)
        deps = os.path.join(paths.WORKSPACE, "deps")
        tensorrt = env.get("TensorRT_ROOT", os.path.join(deps, "TensorRT-10.13.3.9"))
        onnx = env.get("onnxruntime_ROOT", os.path.join(deps, "onnxruntime"))
        cuda = env.get("CUDA_HOME", "/usr/local/cuda")
        libs = [
            f"{tensorrt}/lib",
            f"{onnx}/lib",
            f"{cuda}/lib64",
            f"{s.deploy_root}/thirdparty/unitree_sdk2/thirdparty/lib/x86_64",
        ]
        if s.deploy_bin:
            libs = [os.path.join(os.path.expanduser("~/tools"), d) for d in DEPLOY_LIBS] + libs
        env["LD_LIBRARY_PATH"] = ":".join(libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        env["PATH"] = f"{s.deploy_root}/target/release:" + env.get("PATH", "")
        return argv, cwd, env

    def _start_sim(self) -> None:
        argv = [
            paths.VENV_SIM_PY,
            os.path.join(paths.SRC, "sim", "run_sim_locomani.py"),
            "--instance",
            str(self.spec.instance),
        ]
        argv += ["--interface", "lo", "--no-enable-onscreen"]
        if not self.spec.image_publish:
            # backend.render offline: nothing to render, nothing to wait for; the sim is up once it publishes state
            self._spawn("sim", argv, cwd=paths.SONIC, env=self.env)
            return
        argv += ["--enable-offscreen", "--enable-image-publish"]
        port = self.spec.ports()["TF_CAMERA_PORT"]
        # Refuse a port someone else already holds rather than adopting it. This probe asks
        # "is anything listening", which an orphaned publisher from an aborted run satisfies;
        # the sim's own publisher then dies on bind, the sim keeps running without images, and
        # the expert reports `camera_unavailable` a whole round later (2026-09-17).
        if _port_open(port):
            raise self._fail("sim", f"camera port {port} is already in use; an earlier run left a publisher behind")
        self._spawn("sim", argv, cwd=paths.SONIC, env=self.env)
        if not _wait(lambda: _port_open(port), 90):
            raise self._fail("sim", f"camera port {port} never opened")

    def _start_deploy(self) -> None:
        argv, cwd, env = self._deploy_argv()
        receipts = self.spec.receipts or str(self.root / "deploy_receipts.jsonl")
        prov = [
            paths.VENV_SIM_PY,
            "-m",
            "locomani.pipeline.provenance",
            "--deploy-receipt",
            receipts,
            "--binary",
            argv[0],
            "--policy",
            self.spec.policy,
        ]
        prov_env = dict(
            env, PYTHONPATH=os.path.join(paths.REPO, "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        )
        if subprocess.call(prov, cwd=cwd, env=prov_env):
            raise self._fail("deploy", "provenance receipt refused this binary/policy")
        log = self.root / "deploy.log"
        mark = log.stat().st_size if log.exists() else 0
        self._spawn("deploy", argv, cwd=cwd, env=env)
        ready = READY_DEPLOY_LOCKSTEP if self.spec.lockstep else READY_DEPLOY
        if not _wait(lambda: _log_has(log, ready, mark) or not _alive(self.pids["deploy"]), 1200, 2.0):
            raise self._fail("deploy", f"never reached '{ready}'")
        if not _alive(self.pids["deploy"]):
            raise self._fail("deploy", "exited during init")
        if self.spec.auto_start and not self.spec.lockstep:
            if not _wait(lambda: _log_has(log, READY_AUTO_START, mark), 120, 2.0):
                raise self._fail("deploy", "SONIC_AUTO_START never engaged the policy")

    def _start_exporter(self) -> None:
        s = self.spec
        env = dict(self.env)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[k] = str(s.exporter_threads)
        meta = s.scene_path() / "scene_meta.json"
        if "TF_POSE_OBJECTS" not in env and meta.exists():
            env["TF_POSE_OBJECTS"] = ",".join(json.loads(meta.read_text()).get("objects", {}))
        argv = [
            os.path.join(paths.SONIC, ".venv_data_collection", "bin", "python"),
            os.path.join(paths.SRC, "pipeline", "run_instance_exporter.py"),
        ]
        # The session is NAMED here rather than found afterwards by mtime. The sessions directory
        # is shared by every run on this instance, and the exporter only creates its session once
        # the robot config arrives -- so "the newest one" could be the previous run's
        # (2026-09-23). `convert` pairs session k with manifest k, so the name has to be certain.
        sessions = self.root / "outputs"
        sessions.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y-%m-%d-%H-%M-%S")
        while (sessions / name).exists():
            time.sleep(1.0)
            name = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.session = sessions / name
        argv += ["--task-prompt", s.prompt, "--root-output-dir", str(sessions), "--dataset-name", name]
        argv.append("--no-text-to-speech")
        if s.wrist_cams:
            argv.append("--record-wrist-cameras")
        self._spawn("exporter", argv, cwd=paths.SONIC, env=env)
        time.sleep(8)
        if not _alive(self.pids["exporter"]):
            raise self._fail("exporter", "exited at start")
        if not _wait(lambda: self.session.exists() or not _alive(self.pids["exporter"]), 60, 1.0):
            raise self._fail("exporter", f"never opened its session {self.session}")
        if not _alive(self.pids["exporter"]):
            raise self._fail("exporter", "exited before opening its session")

    # -- lifecycle ------------------------------------------------------------
    def up(self) -> dict[str, str]:
        """Start sim, deploy and (if recording) the exporter; return the environment an agent process needs."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.record_settings()
        self._require_instance_free()
        self._acquire_lock()
        try:
            # sim.log is shared by every run on this instance and only ever grows, so this
            # readiness marker has to be looked for in THIS run's section of it -- the deploy
            # markers below already do. Without the offset the check found a "Image publish
            # frequency" line from a previous run and passed while this sim's publisher was
            # dead, which is exactly the case the message names (2026-09-17: an orphaned
            # publisher from an aborted run held :5575, the port probe in _start_sim saw it
            # listening, and the crate expert only found out a full round later, reporting
            # `camera_unavailable` as if the task had failed).
            sim_log = self.root / "sim.log"
            sim_mark = sim_log.stat().st_size if sim_log.exists() else 0
            self._start_sim()
            self._start_deploy()
            if self.spec.record:
                self._start_exporter()
            if not self.spec.image_publish:
                # backend.render offline publishes no image; the sim is up once its episode server is
                if not _wait(lambda: _log_has(sim_log, READY_SIM_EPISODE, sim_mark), 40, 2.0):
                    raise self._fail("sim", "episode server never came up")
            elif not _wait(lambda: _log_has(sim_log, READY_SIM_IMAGE, sim_mark), 40, 2.0):
                raise self._fail("sim", "has not published a single image (an orphaned publisher may hold the port)")
        except BaseException:
            self._release_lock()
            raise
        self._save()
        return dict(self.env)

    def down(self) -> None:
        """Kill every child by process group, wait for the ports, drop the state file and the lock.

        The children are this object's pids AND the state file's: a child restarted by another process's
        attached Stack (the bottle gym's per-round `deploy.restart()`) is recorded only in the file. Reading
        only this object's pids killed the dead first deploy, deleted the file and orphaned the live one on
        the instance's ports (bottle smoke of 2026-09-28).
        """
        recorded: dict[str, int] = {}
        if self.state_path.exists():
            try:
                recorded = {k: int(v) for k, v in json.loads(self.state_path.read_text()).get("pids", {}).items()}
            except (OSError, ValueError):
                recorded = {}
        for name in reversed(self.CHILDREN):
            for pid in dict.fromkeys(p for p in (self.pids.get(name), recorded.get(name)) if p):
                _kill_group(pid, proc=self._procs.pop(name, None) if pid == self.pids.get(name) else None)
        ports = self.spec.ports()
        _wait(lambda: not any(_port_open(p) for p in ports.values()), 15)
        self.pids = {}
        if self.state_path.exists():
            self.state_path.unlink()
        self._release_lock()

    def restart_deploy(self) -> bool:
        """Kill and relaunch this stack's deploy (the sim and exporter keep running). True when it came back."""
        pid = self.pids.get("deploy")
        if pid:
            _kill_group(pid, proc=self._procs.pop("deploy", None))
        try:
            self._start_deploy()
        except RuntimeError as e:
            print(f"[stack] deploy restart failed: {e}")
            return False
        with (self.root / "deploy.log").open("a") as f:
            f.write("deploy restarted\n")
        return True

    def restart_exporter(self) -> None:
        pid = self.pids.get("exporter")
        if pid:
            _kill_group(pid, proc=self._procs.pop("exporter", None))
        self._start_exporter()

    def health(self) -> dict:
        """Which children are alive, which ports answer, and whether the deploy log shows a LowState trip."""
        ports = self.spec.ports()
        log = self.root / "deploy.log"
        tripped = False
        if log.exists():
            tail = log.read_bytes()[-20000:].decode(errors="replace")
            tripped = LOWSTATE_TRIP in tail and "timeout" in tail.lower()
        return dict(
            alive={name: _alive(pid) for name, pid in self.pids.items()},
            ports={
                name: _port_open(p) for name, p in ports.items() if name in ("TF_CAMERA_PORT", "TF_OUT_PORT")
            },  # TF_SONIC_PORT is bound by the agent
            deploy_lowstate_trip=tripped,
        )

    def __enter__(self) -> Stack:
        self.up()
        return self

    def __exit__(self, *exc) -> None:
        self.down()
