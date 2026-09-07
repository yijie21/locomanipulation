"""Opt-in crouch prefix commitment using proprioception and predicted tokens.

No object pose, distance, simulator time, or privileged camera labels. This
is an experimental execution policy, not a learned phase classifier. Future
tokens are decoded at a fixed measured history, not rolled out in physics.
"""
from collections import deque
import math
import os
import threading
import time

import numpy as np


class CommitWindow:
    def __init__(self, hz=25):
        self.hz = hz
        self.until = 0.

    def arm(self, observation_time, first_crouch, horizon):
        # Keep the predicted transition plus up to four follow-through frames.
        self.until = observation_time + min(horizon, first_crouch + 5) / self.hz

    def frozen(self, observation_time, latency_frames):
        # Absolute deadline: another replan must not restart the countdown.
        return max(latency_frames, math.ceil(max(0., self.until - observation_time) * self.hz))


class TransitionCommitment:
    def __init__(self, hz, onnx_path):
        from sonic_decoder import SonicDecoder
        self.window = CommitWindow(hz)
        self.history = deque(maxlen=100)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.walked = False
        self.events = []
        self.errors = []
        self.completed = False
        self.last_trigger = -1e9
        self.decoder = SonicDecoder(onnx_path, num_threads=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
        subscriber = ZMQStateSubscriber(port=int(os.environ.get('TF_OUT_PORT', '5557')))
        try:
            while not self.stop.is_set():
                state = subscriber.get_msg()
                if state is not None:
                    with self.lock: self.history.append((time.time(), state))
                self.stop.wait(.002)
        except Exception as exc:
            self.errors.append(repr(exc))
        finally: subscriber.close()

    def consider(self, observation_time, token):
        from scipy.spatial.transform import Rotation
        from sonic_decoder import DEFAULT_ANGLES, ACTION_SCALE, MUJOCO_TO_ISAACLAB as ORDER
        with self.lock:
            history = [(t, s) for t, s in self.history if t <= observation_time]
        if len(history) < 30 or observation_time - history[-1][0] > .1:
            return
        states = [s for _, s in history]
        if not np.all(np.diff([s['index'] for s in states[-30:]]) == 1): return
        q = np.array([s['body_q'] for s in states])
        dq = np.array([s['body_dq'] for s in states])
        # Require an observed locomotion period before allowing commitment;
        # initial standing at spawn alone cannot trigger it.
        if np.sqrt(np.mean(dq[:, :12] ** 2, axis=1)).max() > .8:
            self.walked = True
        if self.completed or not self.walked: return
        if np.min(q[-1, [3, 9]]) > .6:
            self.completed = True
            self.window.until = 0.
            self.events.append(dict(event='crouch_observed', wall=observation_time))
            return
        stationary = (np.sqrt(np.mean(dq[-25:, :12] ** 2)) < .35
                      and np.linalg.norm(states[-1]['base_ang_vel']) < .3)
        if not stationary or np.max(q[-1, [3, 9]]) > .45: return
        if observation_time < self.window.until or observation_time - self.last_trigger < 2.: return
        a = np.array([s['last_action'] for s in states])
        gravity = Rotation.from_quat(np.array([s['base_quat'] for s in states]),
                                     scalar_first=True).inv().apply(np.tile([0., 0., -1.], (len(states), 1)))
        pre = dict(q=(q-DEFAULT_ANGLES)[:, ORDER], dq=dq[:, ORDER],
                   a=((a-DEFAULT_ANGLES)/ACTION_SCALE)[:, ORDER],
                   omega=np.array([s['base_ang_vel'] for s in states]), g=gravity)
        i = len(states)-1
        actual = self.decoder.joint_targets(self.decoder.obs_at(pre, i-1, states[-2]['token_state']))[0]
        calibration = float(np.degrees(abs(actual-a[-1])).mean())
        if calibration > .1:
            self.errors.append(f'decoder calibration {calibration:.6f} deg')
            return
        decoded = self.decoder.joint_targets(np.array([self.decoder.obs_at(pre, i, t) for t in token]))
        # BOTH knees must bend for three consecutive future frames: an
        # isolated swing-leg command is not sufficient evidence of crouching.
        flex = np.all(decoded[:, [3, 9]] > a[-1, [3, 9]] + .25, axis=1)
        starts = [j for j in range(10, len(flex)-2) if np.all(flex[j:j+3])]
        if not starts: return
        self.window.arm(observation_time, starts[0], len(token))
        self.last_trigger = observation_time
        event = dict(event='crouch_prefix_committed', wall=observation_time,
                     first_frame=starts[0], until=self.window.until,
                     calibration_deg=calibration)
        self.events.append(event)
        print(f'[transition-commit] {event}', flush=True)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
