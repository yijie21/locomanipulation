"""The privileged stream carries the complete qpos, identical to the state-trace row of the same seq.

The Isaac mirror renders from this field; the consistency gate re-renders from the trace row with
the seq the mirror reported. Both only mean the same picture if the two rows are the same numbers.
"""

import socket
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

from locomani.sim.privileged_pub import PrivilegedPublisher, PrivilegedSubscriber, StateTrace

XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8"><freejoint/><geom size="0.1"/>
      <body name="arm"><joint name="elbow" type="hinge"/><geom size="0.05" pos="0.2 0 0"/></body>
    </body>
  </worldbody>
</mujoco>
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_published_qpos_is_the_trace_row_of_the_same_seq():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[:] = [0.1, -0.2, 0.8, 0.9238795325112867, 0.0, 0.0, 0.3826834323650898, 1.0 / 3.0]
    mujoco.mj_forward(model, data)
    port = free_port()
    pub = PrivilegedPublisher(model, data, {}, port=port)
    sub = PrivilegedSubscriber(port=port)
    with tempfile.TemporaryDirectory() as folder:
        trace = StateTrace(folder)
        trace.start("ep")
        try:
            got, end = None, time.time() + 5
            while got is None and time.time() < end:  # PUB/SUB drops what is sent before the subscriber joins
                seq = pub.publish()
                trace.frame(data, seq)
                got = sub.latest(50)
            trace.stop()
        finally:
            pub.close()
        assert got is not None
        rows = np.load(Path(folder) / "ep_state_trace.npz")
        row = int(np.flatnonzero(rows["seq"] == got["seq"])[0])
        np.testing.assert_array_equal(np.asarray(got["qpos"]), rows["qpos"][row])  # exact, not approximate
        assert len(got["qpos"]) == model.nq
