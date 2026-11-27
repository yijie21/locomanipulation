"""selfmodel_hardgrasp_20261125 (BiGym): M2 online, numpy only (BiGym's venv has no torch).

M2 is bigym_endpoint_20261125's model (outputs/bigym_20261124/selfmodel/model_early/via_point.pt, trained on
selfmodel/data_early: bigym_data.py build --end early), exported by m2_export.py into m2_weights.npz; band.npz is the
training paths' (bucket < 8) 0.5-99.5 % band of the 29 body joints at the command (bigym_data.joint_range's).

One reach's row is built exactly as bigym_data.py build --end early builds a bench row from a film, but at the command:
the reference's future "sent" values are the refs about to be sent (refs[k] becomes the film's sent at i_cmd + 1 + k,
the arm then holding refs[-1]); the times are the film's own clock (k * DT). Then via_point.features and the members'
mean, as via_point.predict.

    m = M2(weights_npz, band_npz)
    out = m.predict(film, i_cmd, refs, ctrl0, dur, h_cmd)     # at the command, before refs[0] is stepped
       film    bigym_bench.Run.film (dict of per-step lists or arrays) with "jnt" (Kin.jnt) in it, or pass jnt=
       i_cmd   the film index of the state at the command (the last entry before refs[0] is sent), >= 10
       refs    (N, 7) right-arm joint targets, one per 20 ms control step
       ctrl0   the arm's sent targets at i_cmd (film["sent"][i_cmd])
       dur     the reach's duration as bigym_bench.joint_refs / line_refs report it (info["dur_s"], rounded to 0.01 s:
               the value the training rows used)
       h_cmd   the base height command asked at the command if crouching (> 0, e.g. 0.62), else -1
    -> dict(off (Q+1, 24, 3) heading frame, p_head (3,) the settled point's offset (row Q) averaged over the 3 palm
       spheres, heading frame; p_world (3,) the same turned into the world (about z only); ref_palm_end_head /
       ref_palm_end_world (3,) the reference's last palm (3 spheres' mean) in the heading frame / world; x the row;
       Rh, org the heading frame (world = Rh @ head + org))
    a = m.label(film, i_cmd, n_reach, dur)       # after the episode: the actual settled offset, heading frame (3,)
       n_reach  the number of reach steps sent (len(refs)); the film must run TAIL_S = 0.5 s past the reach's end
                with the arm holding refs[-1] (the window is i_cmd .. i_cmd + n_reach + 25, as build --end early)
    n29, n7, names = m.out_of_band(film, i_cmd)  # body joints at the command outside the training band

The prediction error of a reach is |out["p_head"] - m.label(...)| (both heading frame at the command).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bigym_20261124"))
sys.path.insert(0, str(HERE.parent / "selfmodel_via_20261120"))
import bigym_data as BD  # noqa: E402
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402

Q, TAIL, DT = VM.Q, VM.TAIL, BD.DT
N_TAIL = int(round(BD.TAIL_S / DT))  # 25 control steps: the window's 0.5 s past the reach


def _silu(h: np.ndarray) -> np.ndarray:
    return h / (1.0 + np.exp(-np.clip(h, -60.0, 60.0)))


class M2:
    def __init__(self, weights_npz, band_npz):
        z = np.load(weights_npz)
        self.n_members, self.n_layers = int(z["n_members"]), int(z["n_layers"])
        self.W = [[z[f"W{i}_{j}"] for j in range(self.n_layers)] for i in range(self.n_members)]
        self.b = [[z[f"b{i}_{j}"] for j in range(self.n_layers)] for i in range(self.n_members)]
        self.mu_x, self.sd_x = z["mu_x"].astype(np.float32), z["sd_x"].astype(np.float32)
        self.mu_y, self.sd_y = z["mu_y"].astype(np.float32), float(z["sd_y"])
        self.pt_md5 = str(z["pt_md5"])
        b = np.load(band_npz)
        self.lo, self.hi, self.band_jnt = b["lo"], b["hi"], [str(x) for x in b["jnt"]]
        self.arm7 = np.array([self.band_jnt.index(n) for n in BD.ARM])
        self.body = None  # BD.Body, made on first use

    # ---------------------------------------------------------------------------------------------------- network
    def forward(self, F: np.ndarray) -> np.ndarray:
        """via_point.predict's members' mean on features F (rows, din) -> (rows, 72) offsets, metres."""
        X = ((np.asarray(F, np.float32) - self.mu_x) / self.sd_x).astype(np.float32)
        out = 0.0
        for i in range(self.n_members):
            h = X
            for j in range(self.n_layers):
                h = h @ self.W[i][j].T + self.b[i][j]
                if j < self.n_layers - 1:
                    h = _silu(h)
            out = out + (self.mu_y + self.sd_y * h)
        return out / self.n_members

    # ------------------------------------------------------------------------------------------------------ helpers
    def _bind(self, film, jnt):
        if self.body is None:
            self.body = BD.Body()
            link = list(self.body.sph.links)
            self.key = np.array([link.index(x) for x in VM.KEY_LINKS])
            self.hand2 = np.array([link.index("hand_palm_link") + 1, link.index("hand_index_1_link")])
            self.palm = np.array([x == "hand_palm_link" for x in link])
        jnt = [str(x) for x in (jnt if jnt is not None else film["jnt"])]
        self.body.bind(jnt)
        bidx = [i for i, n in enumerate(jnt) if "dex1" not in n]
        assert len(bidx) == 29 and [jnt[i] for i in bidx] == self.band_jnt, "the film's body joints differ from M2's"
        return jnt, bidx, [jnt.index(n) for n in BD.ARM]

    @staticmethod
    def _heading(root0):
        p0 = np.asarray(root0[:3], float)
        R0 = BD.quat_R(root0[3:7])
        yaw = math.atan2(R0[1, 0], R0[0, 0])
        Rh = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        return Rh, np.array([p0[0], p0[1], 0.0])

    @staticmethod
    def _state(film, i: int, body: list) -> np.ndarray:
        """bigym_data.state on a film of per-step entries."""
        root = np.asarray(film["root"][i], float)
        Rp = BD.quat_R(root[3:7])
        return np.r_[np.asarray(film["qj"][i])[body], np.asarray(film["dqj"][i])[body], Rp.T @ np.array([0.0, 0.0, -1.0]),
                     np.asarray(film["root_v"][i], float)[3:6], root[2]].astype(np.float32)  # fmt: skip

    def _times(self, film, i_cmd: int, n: int) -> np.ndarray:
        """The film's clock from the command: entry i_cmd + j at (k_cmd + j) * DT - k_cmd * DT (Run.record's t)."""
        k0 = int(round(float(film["t"][i_cmd]) / DT))
        return np.array([(k0 + j) * DT for j in range(n)]) - float(film["t"][i_cmd])

    # -------------------------------------------------------------------------------------------------- the model
    def predict(self, film, i_cmd: int, refs, ctrl0, dur: float, h_cmd: float, jnt=None) -> dict:
        assert i_cmd >= 10, "the state 0.2 s before the command is needed"
        jnt, body, arm = self._bind(film, jnt)
        B = self.body
        refs = np.asarray(refs, float)
        root0, q0 = np.asarray(film["root"][i_cmd]).astype(float), np.asarray(film["qj"][i_cmd]).astype(float)
        Rh, org = self._heading(root0)

        def head(X):
            return (np.asarray(X, float) - org) @ Rh

        sents = [np.asarray(ctrl0, float)] + list(refs) + [refs[-1]] * 2  # held after the reach
        T = self._times(film, i_cmd, len(sents))
        XR = []
        qr = q0.copy()
        for s in sents:
            qr = q0.copy()
            qr[arm] = s
            XR.append(B.pose(root0, qr)[0])
        _, ee_e, Rw_e = B.pose(root0, qr)  # the reference's last pose
        XR = head(np.array(XR))
        u = np.linspace(0.0, 1.0, Q)
        ref = VM.interp(T, XR, u * dur)
        S3 = np.stack([self._state(film, i, body) for i in (i_cmd, i_cmd - 5, i_cmd - 10)])
        jaw = head(film["ee_p"][i_cmd])
        Rj = Rh.T @ BD.quat_R(film["ee_q"][i_cmd])
        Rg = Rh.T @ Rw_e
        crouch = h_cmd > 0
        x = np.r_[S3.ravel(), jaw, Rj[:, 0], Rj[:, 1], head(ee_e), Rg[:, 0], Rg[:, 1], np.zeros(48 + 96 + 16), 0.0, BD.SPEED,
                  float(crouch), max(h_cmd, 0.0), 1.0, dur, ref[::3, self.key].ravel()].astype(np.float32)  # fmt: skip
        Tt = dict(x=x[None], ref=ref[None].astype(np.float32), key=self.key, hand2=self.hand2)
        off = self.forward(VP.features(Tt, np.arange(1))).reshape(Q + 1, 24, 3).astype(float)
        p_head = off[Q][self.palm].mean(0)
        ref_end = ref[-1][self.palm].mean(0)
        return dict(off=off, p_head=p_head, p_world=Rh @ p_head, ref_palm_end_head=ref_end,
                    ref_palm_end_world=Rh @ ref_end + org, x=x, ref=ref, Rh=Rh, org=org)  # fmt: skip

    def label(self, film, i_cmd: int, n_reach: int, dur: float, jnt=None) -> np.ndarray:
        """bigym_data build --end early's off[Q] averaged over the 3 palm spheres: the actual palm's mean over the last
        TAIL 10 Hz ticks of the window (to n_reach + 0.5 s after the command) minus the reference's at dur (the film's
        own sent values), heading frame at the command."""
        jnt, body, arm = self._bind(film, jnt)
        B = self.body
        i_end = i_cmd + n_reach + N_TAIL
        assert len(film["t"]) > i_end, f"the film ends {len(film['t']) - 1 - i_cmd} steps after the command, {n_reach + N_TAIL} needed"
        root0, q0 = np.asarray(film["root"][i_cmd]).astype(float), np.asarray(film["qj"][i_cmd]).astype(float)
        Rh, org = self._heading(root0)
        t = np.array([float(film["t"][i]) for i in range(i_cmd, i_end + 1)])
        T = t - t[0]
        Xa, XR = [], []
        for i in range(i_cmd, i_end + 1):
            Xa.append(B.pose(np.asarray(film["root"][i], float), np.asarray(film["qj"][i]))[0])
            qr = q0.copy()
            qr[arm] = film["sent"][i]
            XR.append(B.pose(root0, qr)[0])
        Xa, XR = (np.array(Xa) - org) @ Rh, (np.array(XR) - org) @ Rh
        t10 = 0.1 * np.arange(int(np.floor(T[-1] / 0.1 + 1e-9)) + 1)
        X10 = VM.interp(T, Xa, t10)
        ref_end = VM.interp(T, XR, np.array([dur]))[0]
        return (X10[-TAIL:].mean(0) - ref_end)[self.palm].mean(0)

    def out_of_band(self, film, i_cmd: int, jnt=None):
        jnt, body, arm = self._bind(film, jnt)
        s0 = np.asarray(film["qj"][i_cmd])[body]
        out = (s0 < self.lo) | (s0 > self.hi)
        return int(out.sum()), int(out[self.arm7].sum()), [self.band_jnt[i] for i in np.where(out)[0]]
