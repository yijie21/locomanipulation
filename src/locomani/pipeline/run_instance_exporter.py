"""Run the upstream exporter with all ports bound to the selected instance.

Upstream exposes camera/state/pose ports as CLI arguments but hard-codes the
keyboard subscriber at 5580. Override that constructor in this process only.
Source scripts/paths.sh before launching so TF_* ports are consistent.
"""

import os
import sys

# Launched by scripts/stack/start_stack.sh with the data-collection venv and no PYTHONPATH:
# put src/ on the path so `locomani.pipeline.collection_frame_audit` resolves (the pre-refactor
# file imported its sibling by bare name; caught by the 2026-09-10 GPU gate, exporter never started).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gear_sonic.utils.data_collection import keyboard_subscriber  # noqa: E402

from locomani.sim import ports

OriginalSubscriber = keyboard_subscriber.ZMQKeyboardSubscriber


class InstanceKeyboardSubscriber(OriginalSubscriber):
    def __init__(self, port=None, host="localhost"):
        super().__init__(port=ports.keyboard() if port is None else port, host=host)


if __name__ == "__main__":
    keyboard_subscriber.ZMQKeyboardSubscriber = InstanceKeyboardSubscriber
    for flag, variable in [
        ("--camera-port", "TF_CAMERA_PORT"),
        ("--sonic-zmq-port", "TF_SONIC_PORT"),
        ("--state-zmq-port", "TF_OUT_PORT"),
    ]:
        if flag not in sys.argv:
            sys.argv.extend([flag, str(ports.port(variable))])
    from gear_sonic.scripts import run_data_exporter as exporter

    from locomani.pipeline.recorded_hand_actions import install as install_hands

    install_hands(exporter.GrootDataCollector)
    stream = None
    if os.environ.get("TF_FRAME_AUDIT_PATH"):
        from locomani.pipeline.collection_frame_audit import install

        stream = install(exporter.GrootDataCollector, os.environ["TF_FRAME_AUDIT_PATH"])
    config = exporter.tyro.cli(exporter.SonicDataExporterConfig)
    if config.dataset_name is None:
        config.dataset_name = exporter.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    try:
        exporter.main(config)
    finally:
        if stream is not None:
            stream.close()
