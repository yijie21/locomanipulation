"""Construct the selected bottle controller; round sequencing does not select classes."""

from locomani.tasks.bottle.reach_config import ReachConfig


def make_grasp_policy(robot, task, config: ReachConfig, arm_ramp, execution):
    """Build one controller using the resolved strategy and unchanged dependencies.

    Imports stay at construction time: inspecting configuration needs no simulator.
    The unconfigured legacy mode has no coordinated controller.
    """
    if config.strategy == "whole_body":
        from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

        return WholeBodyGraspPolicy(robot, task, config, arm_ramp, execution)
    if config.strategy == "legacy":
        return None
    if config.strategy == "ready":
        from locomani.tasks.bottle.ready_grasp_policy import ReadyGraspPolicy as controller
    elif config.strategy == "recovery":
        from locomani.tasks.bottle.recoverable_grasp_policy import RecoverableGraspPolicy as controller
    elif config.strategy == "coupled":
        from locomani.tasks.bottle.coupled_grasp_policy import CoupledGraspPolicy as controller
    elif config.strategy == "image_servo":
        from locomani.tasks.bottle.image_servo_policy import ImageServoPolicy as controller
    elif config.strategy == "teleop":
        from locomani.tasks.bottle.teleop_grasp_policy import TeleopGraspPolicy as controller
    else:
        raise ValueError(f"unsupported grasp strategy {config.strategy!r}")
    return controller(robot, task, config, arm_ramp, execution)
