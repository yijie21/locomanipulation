"""Optional per-row source receipt; does not alter dataset rows or scheduling.

Camera timestamps are publisher wall time, not exposure timestamps. In the
ROS-free stack proprio.ros_timestamp is exporter receipt time, not physical
sample time. These measurements detect stale/repeated sources but cannot
prove exact sensor-action synchronization.
"""

import json
import time
from pathlib import Path


def install(collector_class, output_path):
    original = collector_class._add_data_frame_sonic
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = destination.open("a", buffering=1)
    counters = {}

    def observed(self, t_start):
        episode = int(self.current_episode_index)
        state = self.latest_proprio_msg
        images = self.latest_image_msg
        stamp = time.time()
        receipt = dict(
            episode_index=episode,
            row_index=counters.get(episode, 0),
            row_build_wall=stamp,
            row_build_monotonic=time.monotonic(),
            state_index=int(state["index"]) if "index" in state else None,
            state_receipt_or_ros_wall=float(state.get("ros_timestamp", 0.0)),
            image_publish_wall={k: float(v) for k, v in images.get("timestamps", {}).items()},
        )
        result = original(self, t_start)
        priv = getattr(self, "_latest_priv", None) or {}
        receipt["priv_seq"] = priv.get("seq")
        receipt["consumed_hand_action"] = {
            side: list(map(float, state[f"last_{side}_hand_action"]))
            for side in ("left", "right")
            if f"last_{side}_hand_action" in state
        }
        stream.write(json.dumps(receipt) + "\n")
        counters[episode] = receipt["row_index"] + 1
        return result

    collector_class._add_data_frame_sonic = observed
    return stream
