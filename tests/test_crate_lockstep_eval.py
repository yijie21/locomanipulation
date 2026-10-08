"""The crate evaluation's time model (docs/handoff/lockstep-isaac-eval-plan-20260927.md).

Two clocks share one verdict. On the wall clock the loop waits ~166 ms for a picture and a chunk; `gap: continue`
keeps the robot on the running chunk meanwhile (GapFeeder), `hold` reproduces the loop before 2026-09-27. On the
simulation clock (`vla_lockstep`) the world stands still while the policy looks, and only the inference latency
moves it -- as frames of the old chunk, the new chunk adopted where that latency left off.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from locomani.pipeline import camera_delay as C
from locomani.pipeline import eval_crate_closed_loop as driver
from locomani.tasks.crate import vla_lockstep as L
from locomani.tasks.crate import vla_rounds as R


def chunk(tag: int, n: int = 40, grip: float = 0.02):
    """A 40-frame bimanual chunk whose frame j carries token value tag*100 + j."""
    tok = np.stack([np.full(64, tag * 100 + j, dtype=float) for j in range(n)])
    return tok, np.full(n, grip), np.full(n, grip)


def tags(frames):
    return [int(round(float(np.asarray(f[0])[0]))) for f in frames]


# -- the wall clock: what the robot executes while the loop waits ---------------------------------------------


class _Cmd:
    def __init__(self):
        self.sent = []

    def token_frame(self, tok, right_hand=None, left_hand=None, frame_index=0):
        self.sent.append((int(round(float(np.asarray(tok)[0]))), frame_index))


def test_the_gap_feeder_continues_the_running_chunk_and_hands_the_frame_index_back():
    p = SimpleNamespace(cmd=_Cmd())
    feeder = R.GapFeeder(p, dt=0.001, low=-0.06)
    tok, left, right = chunk(1, n=6)
    feeder.start((tok, left, right), 3, frame_index=50)
    time.sleep(0.05)
    nxt = feeder.stop(50)
    sent = p.cmd.sent
    assert [t for t, _ in sent[:3]] == [103, 104, 105]  # frames 3.. of the running chunk
    assert all(t == 105 for t, _ in sent[3:])  # past its end: its last frame, held explicitly
    assert [fi for _, fi in sent] == list(range(50, 50 + len(sent))) and nxt == 50 + len(sent)
    assert feeder.stop(nxt) == nxt  # stopping an idle feeder changes nothing


def test_the_wall_clock_loop_feeds_the_gap_unless_told_to_hold():
    source = (Path(R.__file__)).read_text()
    assert 'if gap == "continue":\n                feeder.start(running[0], running[1], frame_index)' in source
    assert "frame_index = feeder.stop(frame_index)" in source
    assert "running = ((tok, left, right), skip + n)" in source


# -- the simulation clock: one cycle, frame by frame ---------------------------------------------------------


class _Session:
    """Stands in for vla_lockstep.Session: records every step, numbers boundaries (one seq per control tick)."""

    def __init__(self, ticks=2):
        self.init_tok = np.full(64, -1.0)
        self.open_hands = (np.zeros(7), np.zeros(7))
        self.steps, self.stands, self.seq, self.ticks = [], [], 100, ticks
        self.link = _SimLink()

    def step(self, frames, want_obs=False):
        self.steps.append(list(frames))
        self.seq += self.ticks * len(frames)
        return {"priv_seq": self.seq, "images": {"ego_view": None}, "fallen": False}

    def stand(self, ticks, chunk=10):
        self.stands.append(ticks)

    def state_at(self, seq):
        return {"seq": seq, "t": seq * 0.02, "fallen": False, "pelvis": {"pos": [0.0, 0.0, 0.79]}}


class _SimLink:
    """The lockstep sim's `observe`: the state it kept for a seq, and MuJoCo cameras drawn from it."""

    def __init__(self):
        self.asked = []

    def observe(self, seq):
        self.asked.append(seq)
        return {"seq": seq, "t": seq * 0.02, "wall": 1.0, "qpos": np.full(3, float(seq)), "images": {"ego_view": seq}}


class _Obs:
    def __init__(self):
        self.index, self.image_meta, self.built = 0, None, []

    def settle_proprio(self, after):
        self.index += 1
        return {"index": self.index}

    def build(self, images=None, seq=None):
        self.built.append(dict(images=images, seq=seq))
        return {"boundary": seq}


class _Link:
    def __init__(self, grip=0.02):
        self.n, self.grip, self.seeds = 0, grip, []

    def get_chunk_bimanual(self, obs):
        self.seeds.append(obs.get("policy.seed"))
        self.n += 1
        return chunk(self.n, grip=self.grip)


def _round(
    monkeypatch,
    gap,
    frames=2,
    camera=0,
    budget_s=3.0,
    grip=0.02,
    rate=25,
    horizon=16,
    delay=None,
    obs=None,
    image_source="isaac",
):
    monkeypatch.setattr(L, "prelude", lambda *a: ({}, None))
    monkeypatch.setattr(
        R, "evaluate", lambda *a, **k: dict(ok=False, tilt_deg=0.0, touching=[], rise=0.0), raising=True
    )
    p = SimpleNamespace(
        object_pos=lambda key, st: np.zeros(3), pelvis_xy=lambda st: np.zeros(2), pelvis_yaw=lambda st: 0.0
    )
    sess, link = _Session(L.ticks_per_frame(rate)), _Link(grip)
    sess.p = p
    cfg = SimpleNamespace(
        visual=SimpleNamespace(close_target=-0.06, lift_height=0.1, max_tilt_deg=30.0, start_edges=())
    )
    latency = L.Latency({"first": [frames], "steady": [frames], "source": "fixed"}, camera, np.random.default_rng(0))
    result = L.run_round(
        sess,
        None,
        obs or _Obs(),
        link,
        {"target_object": "box"},
        cfg,
        np.random.default_rng(6000),
        budget_s=budget_s,
        execution_horizon=horizon,
        trace_path=None,
        latency=latency,
        gap=gap,
        image_source=image_source,
        seed_policy=True,
        rate=rate,
        camera=None if delay is None else L.Camera(C.parse(delay), 6000),
    )
    return sess, link, result


def test_the_first_chunk_is_computed_on_the_standing_token_then_blended_in(monkeypatch):
    sess, _, _ = _round(monkeypatch, "continue", frames=2)
    first, handover = sess.steps[0], sess.steps[1]
    assert tags(first) == [-1]  # one standing frame: the boundary the first observation is taken at
    assert tags(handover[:2]) == [-1, -1]  # the deploy stands while the first chunk is computed (L = 2)
    assert len(handover) == 2 + L.RATE and tags(handover)[-1] == 100  # a 1 s blend ending on chunk 0's frame 0


def test_a_cycle_continues_the_old_chunk_through_the_latency_and_adopts_the_new_one_there(monkeypatch):
    sess, link, result = _round(monkeypatch, "continue", frames=2)
    cycle1, cycle2 = tags(sess.steps[2]), tags(sess.steps[3])
    assert cycle1 == [100, 101] + list(range(202, 218))  # chunk 0 from its frame 0, then chunk 1 from D=2
    assert cycle2 == [218, 219] + list(range(302, 318))  # no frozen reference and no jump: frames stay continuous
    assert all(s is not None for s in link.seeds)  # every inference seeded: a MuJoCo-image round replays exactly
    assert result["seconds"] == pytest.approx(5 * 18 / L.RATE)  # 3 s budget: five 18-frame cycles


def test_hold_reproduces_the_wall_clock_loop_before_2026_09_27(monkeypatch):
    sess, _, _ = _round(monkeypatch, "hold", frames=4)
    cycle1, cycle2 = tags(sess.steps[2]), tags(sess.steps[3])
    assert cycle1 == [100] * 4 + list(range(204, 220))  # the blend's last frame, clamped 4 frames; adopt at 4
    assert cycle2 == [219] * 4 + list(range(304, 320))


def test_camera_frames_add_to_the_inference_latency(monkeypatch):
    sess, _, _ = _round(monkeypatch, "continue", frames=2, camera=1)
    assert tags(sess.steps[2])[:3] == [100, 101, 102] and tags(sess.steps[2])[3] == 203


def test_a_close_counts_only_in_the_frames_the_policy_actually_adopted(monkeypatch):
    _, _, result = _round(monkeypatch, "continue", grip=0.001)
    assert result["commanded_close"] is True
    _, _, result = _round(monkeypatch, "continue", grip=0.02)
    assert result["commanded_close"] is False


def test_the_latency_distribution_comes_from_wall_clock_round_traces(tmp_path):
    rows = [
        dict(skip=4, latency_s=0.166, image=dict(render_ms=95.0)),
        dict(skip=4, latency_s=0.170, image=dict(render_ms=96.0)),
        dict(skip=5, latency_s=0.190, image=None),
    ]
    (tmp_path / "round_000.json").write_text(json.dumps(dict(trace=rows)))
    wall = L.latency_dist(tmp_path, inference_only=False)
    assert wall["steady"] == [4, 4, 5] and "wall-clock skip" in wall["source"]
    inference = L.latency_dist(tmp_path, inference_only=True)
    assert inference["steady"] == [2, 2, 5]  # 71 ms, 74 ms -> 2 frames; no render recorded -> the whole latency


# -- 50 fps: NVIDIA's client on the simulation clock (DECISIONS 2026-10-07) -----------------------------------


def test_at_50_fps_a_frame_is_one_tick_and_the_first_chunk_takes_over_without_a_blend(monkeypatch):
    sess, _, _ = _round(monkeypatch, "continue", frames=4, rate=50, horizon=20)
    assert L.ticks_per_frame(50) == 1 and L.ticks_per_frame(25) == 2
    handover = sess.steps[1]
    # the standing token while the first chunk is computed, then chunk 0 from its compensated index, 20 frames
    assert tags(handover) == [-1] * 4 + list(range(104, 124))
    cycle1 = tags(sess.steps[2])
    assert cycle1 == [124, 125, 126, 127] + list(range(204, 224))  # the old chunk through the gap, the new from 4


def test_at_50_fps_a_chunk_plays_its_horizon_and_holds_its_last_frame_past_the_end(monkeypatch):
    sess, _, _ = _round(monkeypatch, "continue", frames=30, rate=50, horizon=20)
    handover, cycle1 = tags(sess.steps[1]), tags(sess.steps[2])
    assert handover[30:] == list(range(130, 140)) + [139] * 10  # NVIDIA's index clamps at the horizon's end
    assert cycle1[:30] == [139] * 30 and cycle1[30:] == list(range(230, 240)) + [239] * 10


def test_a_camera_delay_shows_each_observation_an_earlier_state_and_the_proprioception_stays(monkeypatch):
    delay = {"fps": 30, "latency_ms": [20, 30]}
    obs = _Obs()
    sess, _, _ = _round(
        monkeypatch, "continue", frames=4, rate=50, horizon=20, delay=delay, obs=obs, image_source="mujoco"
    )
    boundaries = [101] + [101 + sum(len(s) for s in sess.steps[1 : k + 1]) for k in range(1, len(sess.steps) - 1)]
    camera = L.Camera(C.parse(delay), 6000)
    assert sess.link.asked == [camera.seq(b) for b in boundaries]  # the dataset's draw, keyed by the round's seed
    ages = [(b - a) * 20 for b, a in zip(boundaries, sess.link.asked)]
    latency = camera.draw[0] * 1e3
    assert all(latency <= age < latency + 1e3 / 30 + 20 for age in ages)  # latency + up to a frame + a tick
    # its pictures, not the boundary's
    assert [b["images"] for b in obs.built] == [{"ego_view": s} for s in sess.link.asked]


def test_the_closed_loop_ages_a_picture_exactly_as_the_dataset_re_render_does():
    delay = C.parse({"fps": 30, "latency_ms": [20, 30]})
    tseq = np.arange(300, 900)  # a trace with every tick, as the lockstep sim keeps it
    cam, _ = C.delayed_rows(delay, tseq, np.arange(len(tseq)), "lockstep/6000")
    draw = C.episode_draw(delay, "lockstep/6000")
    assert [int(tseq[c]) for c in cam[20:]] == [C.camera_seq(delay, draw, int(s)) for s in tseq[20:]]


def test_the_mirror_draws_the_state_the_sim_kept_when_the_camera_is_late():
    from locomani.sim.mirror import MirrorClient

    class Mirror(MirrorClient):
        def __init__(self):
            self.states, self.last_error = [], ""

        def render(self, seq=None, state=None):
            self.states.append(state)
            return {"ego_view": np.zeros((1, 1, 3), np.uint8)}, {"seq": state["seq"], "render_ms": 9.0}

    obs = _Obs()
    obs.mirror = Mirror()
    sess = _Session(ticks=1)
    camera = L.Camera(C.parse({"fps": 30, "latency_ms": [20, 30]}), 6000)
    _, fields, _ = L.observe(sess, obs, {"priv_seq": 500}, "isaac", None, camera)
    state = obs.mirror.states[0]
    assert state["seq"] == camera.seq(500) == fields["image"]["seq"] and list(state["qpos"]) == [state["seq"]] * 3
    assert fields["image"]["render_ms"] == 9.0 and fields["boundary_seq"] == 500


def test_the_lockstep_sim_keeps_recent_states_and_renders_an_earlier_one_off_the_live_data():
    import mujoco

    from locomani.backends.sonic import lockstep_sim as S

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="b"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>'
    )
    live = mujoco.MjData(model)
    rendered = []

    class Env:
        mj_model, mj_data = model, live

        def update_render_caches(self):
            rendered.append(self.mj_data is live)
            return {"ego_view_image": np.full((1, 1, 3), int(self.mj_data.qpos[0] * 10), np.uint8)}

    pub = SimpleNamespace(last_seq=None)
    runner = S.LockstepRunner.__new__(S.LockstepRunner)
    runner.env = Env()
    runner.env._privileged_pub = pub
    runner._history, runner._scratch = __import__("collections").OrderedDict(), None
    for seq in range(S.HISTORY_TICKS + 5):
        live.qpos[0] = seq / 10
        pub.last_seq = seq
        runner._remember()
    assert len(runner._history) == S.HISTORY_TICKS and next(iter(runner._history)) == 5
    rep = runner._observe(60)
    assert rep["ok"] and rep["seq"] == 60 and int(rep["images"]["ego_view"][0, 0, 0]) == 60
    assert rendered == [False] and runner.env.mj_data is live and live.qpos[0] == pytest.approx(6.8)
    assert not runner._observe(2)["ok"]  # older than the history: refused, never a wrong picture


def test_a_mirror_renders_the_state_a_request_carries():
    from locomani.sim.mirror import handle

    class View:
        def render(self, qpos):
            return {"ego_view": np.full((1, 1, 3), int(qpos[0]), np.uint8)}

    state = {"seq": 7, "t": 0.14, "wall": time.time(), "qpos": [5.0, 0.0]}
    frames, header = handle(json.dumps({"op": "render", "state": state}).encode(), View(), lambda: None, 2)
    assert header["seq"] == 7 and frames[1] == bytes([5, 5, 5])


def test_a_latency_distribution_from_lockstep_rounds_counts_the_servers_time_at_the_runs_rate(tmp_path):
    rows = [dict(skip=4, infer_s=0.071), dict(skip=4, infer_s=0.095), dict(skip=4, infer_s=0.130)]
    (tmp_path / "round_000.json").write_text(json.dumps(dict(trace=rows)))
    dist = L.latency_dist(tmp_path, inference_only=True, fps=50)
    assert dist["steady"] == [4, 5, 6] and dist["fps"] == 50  # NVIDIA's round(latency x 50)
    with pytest.raises(SystemExit, match="inference-only"):
        L.latency_dist(tmp_path, inference_only=False)
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"first": [2], "steady": [2]}))  # a file from before `fps`: 25 fps frames
    args = SimpleNamespace(dataset_fps=50, latency_dist=str(old), latency_frames=None, camera_delay=None)
    with pytest.raises(SystemExit, match="counts frames at 25"):
        L.run_batch(args, None, None, None)


def test_the_evaluator_plays_50_fps_and_ages_pictures_on_the_lockstep_clock_only(tmp_path):
    from locomani.pipeline.eval_config import EvalConfig

    delay = json.dumps({"fps": 30, "latency_ms": [20, 30]})
    cfg = tmp_path / "eval.json"
    EvalConfig(client="lockstep", latency_frames=4, execution_horizon=20).dump(cfg)
    args = driver.parse_args(
        _ckpt(tmp_path) + ["--eval-config", str(cfg), "--dataset-fps", "50", "--camera-delay", delay]
    )
    driver.check_timing(args)
    argv = driver.lockstep_argv(args)
    assert argv[argv.index("--dataset-fps") + 1] == 50 and argv[argv.index("--camera-delay") + 1] == delay
    blocking = tmp_path / "blocking.json"
    EvalConfig(client="blocking").dump(blocking)
    for extra, match in (
        (["--dataset-fps", "50"], "only the lockstep client"),
        (["--camera-delay", delay], "lockstep clock only"),
    ):
        with pytest.raises(SystemExit, match=match):
            driver.check_timing(driver.parse_args(_ckpt(tmp_path) + ["--eval-config", str(blocking), *extra]))
    both = tmp_path / "both.json"
    EvalConfig(client="lockstep", latency_frames=4, camera_frames=1).dump(both)
    with pytest.raises(SystemExit, match="both delay the pictures"):
        driver.check_timing(driver.parse_args(_ckpt(tmp_path) + ["--eval-config", str(both), "--camera-delay", delay]))
    with pytest.raises(SystemExit):
        R.main(["--manifest", "/dev/null", "--initial-token-file", "x", "--dataset-fps", "50"])


# -- the evaluator: lockstep is a client it implements, and the stack it attaches to must match ---------------


def _ckpt(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "config.json").write_text("{}")
    return ["--checkpoint", str(ckpt), "--scene", str(tmp_path), "--out", str(tmp_path / "o")]


def test_the_evaluator_delivers_a_lockstep_block_to_the_rounds(tmp_path):
    from locomani.pipeline.eval_config import EvalConfig

    cfg = tmp_path / "eval.json"
    EvalConfig(client="lockstep", latency_frames=2, camera_frames=1, image_source="isaac").dump(cfg)
    args = driver.parse_args(_ckpt(tmp_path) + ["--eval-config", str(cfg)])
    assert (args.client, args.gap, args.latency_frames, args.camera_frames) == ("lockstep", "continue", 2, 1)
    assert driver.lockstep_argv(args) == ["--camera-frames", 1, "--dataset-fps", 25, "--latency-frames", 2]
    assert not {"client", "gap", "latency_frames", "latency_dist", "camera_frames"} & set(args.ignored_eval_fields)

    blocking = tmp_path / "blocking.json"
    EvalConfig(client="blocking", gap="hold").dump(blocking)
    args = driver.parse_args(_ckpt(tmp_path) + ["--eval-config", str(blocking)])
    assert (args.client, args.gap, driver.lockstep_argv(args)) == ("blocking", "hold", [])

    no_latency = tmp_path / "no_latency.json"
    EvalConfig(client="lockstep", latency_frames=None).dump(no_latency)
    with pytest.raises(SystemExit):
        driver.parse_args(_ckpt(tmp_path) + ["--eval-config", str(no_latency)])


def test_the_eval_stack_runs_lockstep_exactly_when_the_client_does():
    from locomani.pipeline.eval_config import EvalConfig
    from locomani.tasks.crate import prepare

    lockstep = prepare.eval_backend(EvalConfig(client="lockstep", latency_frames=2))
    blocking = prepare.eval_backend(EvalConfig(client="blocking"))
    assert lockstep["lockstep"]["enabled"] is True and lockstep["clean_deploy"] is True
    assert blocking["lockstep"]["enabled"] is False and blocking["clean_deploy"] is True


def test_the_rounds_refuse_latency_flags_on_the_wall_clock():
    with pytest.raises(SystemExit):
        R.main(["--manifest", "/dev/null", "--initial-token-file", "x", "--latency-frames", "2"])


# -- the sim and the mirror: a frozen boundary is renderable by seq ------------------------------------------


def test_an_offline_lockstep_sim_returns_no_images_and_names_its_boundary():
    from locomani.backends.sonic.lockstep_sim import LockstepRunner

    offline = SimpleNamespace(env=SimpleNamespace(update_render_caches=lambda: None))
    assert LockstepRunner._render(offline) is None
    live = SimpleNamespace(env=SimpleNamespace(update_render_caches=lambda: {"ego_view_image": np.zeros((2, 2, 3))}))
    assert set(LockstepRunner._render(live)) == {"ego_view"}
    assert LockstepRunner._priv_seq(SimpleNamespace(env=SimpleNamespace(_privileged_pub=None))) is None
    pub = SimpleNamespace(last_seq=41)
    assert LockstepRunner._priv_seq(SimpleNamespace(env=SimpleNamespace(_privileged_pub=pub))) == 41


def test_a_frozen_sim_repeats_its_newest_state_for_a_late_subscriber():
    import mujoco

    from locomani.sim.privileged_pub import PrivilegedPublisher, PrivilegedSubscriber

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 0.8"><freejoint/><geom size="0.1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    pub = PrivilegedPublisher(model, data, {}, port=port)
    try:
        assert pub.last_seq is None
        pub.publish()  # nobody listens yet: this one is lost
        sub = PrivilegedSubscriber(port=port)
        got, end = None, time.time() + 5
        while got is None and time.time() < end:
            pub.resend()
            got = sub.latest(50)
        assert got is not None and got["seq"] == 0 == pub.last_seq  # the same state, not a new one
        sub.close()
    finally:
        pub.close()


def test_the_observation_waits_for_the_deploys_last_tick_and_asks_the_mirror_for_the_boundary():
    from locomani.backends.sonic.vla_obs import ObsBuilder

    class State:
        def __init__(self, msgs):
            self.msgs = list(msgs)

        def get_msg(self):
            return self.msgs.pop(0) if self.msgs else None

    class Mirror:
        def __init__(self):
            self.asked = []

        def render(self, seq=None):
            self.asked.append(seq)
            return {"ego_view": np.zeros((1, 1, 3), np.uint8)}, {"seq": seq}

    obs = ObsBuilder.__new__(ObsBuilder)
    obs.cam, obs.mirror, obs._last_img, obs._last_proprio, obs.image_meta = None, Mirror(), None, None, None
    obs.state = State([{"index": 7}, {"index": 8}, {"index": 9}])
    assert obs.settle_proprio(7, quiet=0.005)["index"] == 9  # the newest, once nothing newer comes
    with pytest.raises(RuntimeError, match="never moved past"):
        obs.settle_proprio(9, timeout=0.05)
    assert obs._render(41) and obs.mirror.asked == [41] and obs.image_meta == {"seq": 41}
