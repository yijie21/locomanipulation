"""What the robot knows about the world and about itself, behind one interface per question (TODO 49, 2026-11-24).

    view.py       View: where the target is, how to grasp it, where the table edge and the container are. Two
                  families: TruthView reads the benchmark's privileged state (option target=truth); CameraView reads
                  a Sensor, segments with SAM 3 and plans grasps on the points (target=perception).
    sensor.py     Frame and Sensor: one colour image with metric depth in the camera's frame, from a stereo pair
                  through a stereo model (sensor=stereo) or from a rendered depth image, optionally with a
                  RealSense-like noise model (sensor=rgbd, depth_noise=none|realsense).
    self_pose.py  SelfPose: where the pelvis is in the robot's own frame, from the simulator (self_pose=truth) or
                  from leg odometry (self_pose=odometry).

A skill package sees these interfaces and nothing of the simulator; the only way truth reaches a skill is a
TruthView or a truth SelfPose that the run's options chose.
"""
