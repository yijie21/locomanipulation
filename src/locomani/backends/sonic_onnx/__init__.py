"""SONIC v1.1 run in this process from its ONNX files (planner, encoder, decoder), on our MuJoCo sim.

`runtime` holds the policy (`SonicPolicy`: the deploy's teleop mode -- VR 3-point wrists and head for the upper body,
the velocity planner for the legs, walking and pelvis height) and the sim (`SonicSim`: `scalebfm.runtime.Sim` with
SONIC's gains); `motor` holds `SonicMotor`, the `scalebfm.motor.Motor` interface on top of them. The ZMQ / deploy based
SONIC code is `backends.sonic`; this package does not talk to the C++ deploy.
"""
