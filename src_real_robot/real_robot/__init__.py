"""GR00T + SONIC inference client for the real Unitree G1 (Dex1 grippers, three RealSense cameras).

The structure is NVIDIA's `gear_sonic/scripts/run_vla_inference.py` (GR00T-WholeBodyControl v1.1, see
UPSTREAM.md): an asynchronous inference thread, a 50 Hz publish loop with latency-compensated chunk indexing, and
protocol-v4 token messages to the SONIC C++ deploy. What differs is the observation and action format, which is
this repository's dataset format (`locomani.schema`), and the Dex1 gripper (`real_robot.gripper`).
Design: docs/handoff/real-robot-sim2real-plan-20261006.md.
"""
