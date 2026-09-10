"""Run the upstream exporter with all ports bound to the selected instance.

Upstream exposes camera/state/pose ports as CLI arguments but hard-codes the
keyboard subscriber at 5580. Override that constructor in this process only.
Source scripts/paths.sh before launching so TF_* ports are consistent.
"""

import os
import runpy
import sys

# Launched by scripts/stack/start_stack.sh with the data-collection venv and no PYTHONPATH:
# put src/ on the path so `locomani.pipeline.collection_frame_audit` resolves (the pre-refactor
# file imported its sibling by bare name; caught by the 2026-09-10 GPU gate, exporter never started).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gear_sonic.utils.data_collection import keyboard_subscriber  # noqa: E402

OriginalSubscriber = keyboard_subscriber.ZMQKeyboardSubscriber


class InstanceKeyboardSubscriber(OriginalSubscriber):
    def __init__(self, port=None, host="localhost"):
        super().__init__(port=int(os.environ.get("TF_KEYBOARD_PORT", "5580")) if port is None else port, host=host)


if __name__ == "__main__":
    keyboard_subscriber.ZMQKeyboardSubscriber = InstanceKeyboardSubscriber
    for flag, variable, default in [
        ("--camera-port", "TF_CAMERA_PORT", 5555),
        ("--sonic-zmq-port", "TF_SONIC_PORT", 5556),
        ("--state-zmq-port", "TF_OUT_PORT", 5557),
    ]:
        if flag not in sys.argv:
            sys.argv.extend([flag, os.environ.get(variable, str(default))])
    if not os.environ.get("TF_FRAME_AUDIT_PATH"):
        runpy.run_module("gear_sonic.scripts.run_data_exporter", run_name="__main__")
    else:
        from gear_sonic.scripts import run_data_exporter as exporter

        from locomani.pipeline.collection_frame_audit import install

        stream = install(exporter.GrootDataCollector, os.environ["TF_FRAME_AUDIT_PATH"])
        config = exporter.tyro.cli(exporter.SonicDataExporterConfig)
        if config.dataset_name is None:
            config.dataset_name = exporter.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        try:
            exporter.main(config)
        finally:
            stream.close()
