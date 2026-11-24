# ruff: noqa: I001, E401, E402, E501, E701, E702, E741
"""LocoManiLab's own environment behind a socket, driven by OUR controller's joint targets.

Runs in third_party/LocoManiLab/.venv (LocoManiLabConfig.lml_python) with PYTHONPATH=<g1_backend>:<humanoid_benchmark> as a
standalone file: it imports nothing of locomani (the benchmark's venv has no locomani), so what it needs of the bed comes as
arguments (TODO 49.4; before: LML, LML_LOOP in the environment):
    MUJOCO_GL=egl python server.py <socket path> <out dir> <LocoManiLab root> <seed floor> <radii 0|1>
<seed floor>: the least `jitter seed + episode` a reset may make a start state of (LocoManiLabConfig.seed_floor): the wall that
keeps the benchmark's own episodes (jitter seed 0, episodes 0-19) out whatever a client asks; it is never under 20.
Env construction, task overrides, reset and per-episode jitter reuse LocoManiLab's own functions in
src/locomanilab/runtime/sim_client.py (backend scalebfm, --fixed-scene, jitter radii 0.05 / 0.02,
default_rng(jitter_seed + episode)), so (jitter_seed=0, episode=i) is official evaluation episode i.

Hook: the robot's `control` method (called by the official env.step at each of the 10 physics substeps,
right after sim.forward()) is replaced by a PD law on our targets. The normal env.step runs unchanged around it
(substeps, _update_observables, _post_action counters, success check); the ScaleBFM composite controller is never
called, so it writes no torque.

Messages (tuples) -> replies (dicts): ("tasks",) ("reset", task, episode, jitter_seed, cameras[, robot_radius, object_radius])
("step", q_des, kp, kd) ("eyes",) ("quit",).
"""

import json
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener

import numpy as np

sock_path, out_dir, LML, SEED_FLOOR, NEED_RADII = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
    sys.argv[5] == "1",
)
EVAL_EPISODES = 20  # the benchmark's own episodes: jitter seed 0, episodes 0-19
if SEED_FLOOR < EVAL_EPISODES:
    raise SystemExit(f"seed floor {SEED_FLOOR} would admit the benchmark's own episodes")
sys.path.insert(0, LML + "/src")
from locomanilab.runtime import sim_client as sc  # noqa: E402
import mujoco  # noqa: E402
import robosuite  # noqa: E402
import robocasa  # noqa: E402,F401
import yaml  # noqa: E402

TASKS = yaml.safe_load(open(LML + "/src/locomanilab/benchmark/tasks.yaml"))
DEFAULTS, ROWS = TASKS["defaults"], TASKS["tasks"]
CAM = sc.EGO_CAMERA


def loop_wall(episode, jitter_seed, robot_radius=None, object_radius=None) -> None:
    """A start state is made only from a practice seed with both radii given. The server is the only code that can make one, so
    this is the wall whatever the client asks (the benchmark's own episodes are seeds 0-19). The old wall was on only when the
    server inherited LML_LOOP=1; it is always on now, with the floor from its arguments."""
    if int(jitter_seed) + int(episode) < SEED_FLOOR:
        raise PermissionError(
            f"seed wall: jitter seed {int(jitter_seed)} + episode {int(episode)} is under {SEED_FLOOR}"
        )
    if NEED_RADII and (robot_radius is None or object_radius is None):
        raise PermissionError("seed wall: a reset needs both radii")


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


# ----------------------------------------------------------------------------------------------------------------------
# Normalised task spec: what each task class's success check actually reads. Built from the source of `_check_success`
# and the functions it relies on (humanoid_benchmark/robocasa/environments/...), one builder per defining class.
# Thresholds are read from the live env (class constants / spec values), never typed in here.
# ----------------------------------------------------------------------------------------------------------------------
def _jv(x):
    return jsonable(x)


def _th(E, *names):
    out = {}
    for n in names:
        try:
            v = getattr(E, n)
        except Exception:
            continue
        if v is None or callable(v):
            continue
        out[n] = _jv(v)
    return out


def _bname(E, bid):
    return mujoco.mj_id2name(E.sim.model._model, mujoco.mjtObj.mjOBJ_BODY, int(bid))


def _tree(E, root_name):
    m = E.sim.model._model
    r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
    ids = {r}
    for b in range(1, m.nbody):
        if int(m.body_parentid[b]) in ids:
            ids.add(b)
    return [n for n in (_bname(E, b) for b in sorted(ids)) if n]  # unnamed bodies have no name to give


def _free_root(E, name):
    """The body that carries the free joint of the thing named `name` (itself, an ancestor, or a descendant)."""
    m = E.sim.model._model
    r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    has_free = lambda b: any(
        int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
        if m.body_jntadr[b] >= 0
    )
    if has_free(r):
        return name
    b = r
    while b > 0:
        b = int(m.body_parentid[b])
        if has_free(b):
            return _bname(E, b)
    for b in range(1, m.nbody):
        p, ok = b, False
        while p > 0:
            if p == r:
                ok = True
                break
            p = int(m.body_parentid[p])
        if ok and has_free(b):
            return _bname(E, b)
    return name


def _region(E, region, z_tol=None):
    c, yaw = region.world_pose(E.sim)
    h = np.asarray(region.half_size, dtype=float)
    return dict(
        centre=_jv(c),
        half=_jv(h),
        yaw=float(yaw),
        z=[float(c[2] - h[2]), float(c[2] + h[2])] if len(h) > 2 else [float(c[2]), float(c[2])],
    )


def _fix(E, f):
    return f.body_name if f is not None and hasattr(f, "body_name") else None


def _hold_s(E, secs):
    return max(1, int(round(float(secs) * E.control_freq)))


def _spec_for(E, owner):
    """-> dict(object, target, rule, thresholds, support, extra) for the class that defines _check_success."""
    O = lambda n: E.objects[n].root_body
    obj = {}
    tgt = dict(body=None, region=None)
    th = {}
    sup = None
    ex = {}
    rule = ""
    skill_obj = None
    if owner == "HolodeckApproachDesk":
        d = E.desk_fixture
        tgt.update(body=_fix(E, d), xy=_jv(np.asarray(d.center[:2])), radius=float(E.success_radius))
        th = _th(E, "success_radius", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        rule = "robot base xy within success_radius of desk fixture centre xy for HOLD_STEPS steps"
        ex = dict(
            nav=dict(
                target_body=_fix(E, d),
                target_xy=_jv(np.asarray(d.center[:2])),
                tolerance=float(E.success_radius),
                hold_steps=int(E.HOLD_STEPS),
                metric="euclid xy robot base to fixture centre",
            )
        )
    elif owner == "ExactSceneNavigation":
        f = E.target_fixture
        tgt.update(body=_fix(E, f), distance=float(E._navigation_success_distance))
        th = _th(E, "_navigation_success_distance", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        rule = "robot-to-target-fixture collision-geometry distance (max 1.0 m probe) <= _navigation_success_distance for HOLD_STEPS steps"
        ex = dict(
            nav=dict(
                target_body=_fix(E, f),
                target_xy=None,
                tolerance=float(E._navigation_success_distance),
                hold_steps=int(E.HOLD_STEPS),
                metric="geom distance robot bodies to fixture bodies (not the robot base xy)",
            )
        )
    elif owner == "RoomNavMixin":
        f = getattr(E, "target_fixture", None)
        tgt.update(body=_fix(E, f), xy=_jv(E._navigation_target_xy), half_xy=_jv(E._navigation_target_tolerance))
        th = _th(E, "_navigation_target_tolerance", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        rule = "robot base xy inside the axis-aligned box target_xy +- tolerance (per axis, world frame) for HOLD_STEPS steps"
        ex = dict(
            nav=dict(
                target_body=_fix(E, f),
                target_xy=_jv(E._navigation_target_xy),
                tolerance=_jv(E._navigation_target_tolerance),
                hold_steps=int(E.HOLD_STEPS),
                metric="per-axis |robot_xy - target_xy| <= tolerance",
            )
        )
    elif owner == "HolodeckMoveCupToCenter":
        skill_obj = O("cup")
        sp = E.task_spec
        tgt.update(
            region=dict(
                centre=_jv(E._target_base_xy()),  # the subclass's (MoveCupToCorner: the robot-side corner on its right)
                half=_jv(E._target_tolerance()),
                z=None,
                frame="world xy, per axis, on the cup ROOT body xy",
            )
        )
        if hasattr(E, "_corner_target_base_xys"):  # either robot-side corner counts, +-half in the desk's own axes
            tgt["region"]["corners"] = _jv(E._corner_target_base_xys)
        sup = _fix(E, getattr(E, "desk_fixture", None))
        th = _th(E, "hold_steps")
        th.update(target_tolerance=_jv(E._target_tolerance()))
        rule = "cup root xy within +-target_tolerance of target_base_xy AND cup in contact with desk bodies, for hold_steps steps (no grasp/release test)"
    elif owner == "HolodeckPlaceCupOnPlacemat":
        skill_obj = O("cup")
        tgt.update(body=O("placemat"))
        th = _th(E, "hold_steps", "PLACEMAT_EDGE_TOLERANCE", "SUPPORT_Z_TOLERANCE")
        rule = "cup bbox xy inside placemat bbox xy (placemat frame, +PLACEMAT_EDGE_TOLERANCE) and cup in contact with placemat (contact test skipped only for base class scene train_117); hold_steps steps; no release test"
    elif owner == "HolodeckPlaceBoxOnDesk":
        skill_obj = O("box")
        ids = getattr(E, "_desk_body_ids", None)
        tgt.update(body=None, bodies=[_bname(E, i) for i in sorted(ids)] if ids is not None else None)
        th = _th(E, "hold_steps")
        rule = "box (any of its bodies) in contact with any desk body for hold_steps steps; no release or pose test"
    elif owner == "HolodeckPlaceBoxOnFloor":
        skill_obj = O("box")
        tgt.update(body=None, floor_geom_id=int(E._box_floor_geom_id))
        th = _th(E, "hold_steps", "CONTACT_MARGIN")
        rule = (
            "box in contact with the floor geom (contact dist <= CONTACT_MARGIN) for hold_steps steps; no release test"
        )
    elif owner in ("HolodeckPickBookFromDresser", "HolodeckPickBoxFromFloor", "HolodeckPickCupFromDesk"):
        if owner.endswith("Dresser"):
            nm = E.task_spec.get("target_body") or O("book")
            z0 = "_book_initial_z"
            d = "lift_delta"
        elif owner.endswith("Floor"):
            nm = O("box")
            z0 = "_box_initial_z"
            d = "MIN_LIFT_DELTA"
        else:
            nm = O("cup")
            z0 = "_cup_initial_z"
            d = "lift_delta"
        skill_obj = nm
        th = _th(E, z0, d, "hold_steps")
        tgt.update(z_min=float(getattr(E, z0) + getattr(E, d)))
        rule = f"object root z > {z0} + {d} for hold_steps steps; NO grasp, hand or release condition (any way of lifting counts)"
    elif owner == "_BathroomNativeLiftTask":
        skill_obj = E.source_fixture.body_name
        th = _th(E, "MIN_LIFT_DELTA", "GRASP_HEIGHT_ALLOWANCE", "HOLD_STEPS", "_initial_source_z")
        th["hold_steps"] = int(E.HOLD_STEPS)
        tgt.update(z_min=float(E._initial_source_z + E.MIN_LIFT_DELTA))
        rule = "hand closed (>=3 of 6 finger joints >=20% of range) AND touching the object: first seen near reset height (<= reset z + GRASP_HEIGHT_ALLOWANCE), then grasped with root z >= reset z + MIN_LIFT_DELTA, for HOLD_STEPS steps"
        sup = _fix(E, getattr(E, "work_surface_fixture", None))
    elif owner in ("_BathroomCandleStoolTask", "IThorBedroomLiftBasketFromFloor", "IThorBedroomLiftBasket"):
        skill_obj = O(E.TARGET_NAME)
        z0 = "_initial_source_z" if owner.startswith("_Bath") else "_lift_initial_z"
        th = _th(E, z0, "MIN_LIFT_DELTA", "GRASP_HEIGHT_ALLOWANCE", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        tgt.update(z_min=float(getattr(E, z0) + E.MIN_LIFT_DELTA))
        rule = "hand closed AND touching the object (g1_hand_grasping_object): first seen near reset height (<= z0 + GRASP_HEIGHT_ALLOWANCE), then grasped with root z >= z0 + MIN_LIFT_DELTA, for HOLD_STEPS steps"
    elif owner == "KitchenLiftObjectMixin":
        skill_obj = O(E.TARGET_NAME)
        th = _th(E, "_lift_initial_z", "min_lift_delta", "required_lift_hold_steps", "lift_height")
        th["hold_steps"] = int(E.required_lift_hold_steps)
        tgt.update(z_min=float(E._lift_initial_z + E.min_lift_delta))
        rule = "object root z >= reset z + min_lift_delta for required_lift_hold_steps steps; NO grasp condition"
    elif owner == "RoomLiftMixin":
        skill_obj = O(E.TARGET_NAME)
        th = _th(E, "_lift_initial_z", "MIN_LIFT_DELTA", "HOLD_SECONDS")
        th["hold_steps"] = _hold_s(E, E.HOLD_SECONDS)
        tgt.update(z_min=float(E._lift_initial_z + E.MIN_LIFT_DELTA))
        rule = "object root z >= reset z + MIN_LIFT_DELTA for HOLD_SECONDS*control_freq steps; NO grasp condition"
    elif owner == "_BathroomSpongeInBathtubTask":
        skill_obj = O(E.SOURCE_NAME)
        bf = getattr(E, "bathtub_fixture", None)
        tgt.update(
            body=_fix(E, bf),
            region=dict(
                centre_xy=_jv(E.basin_center_xy), half_xy=_jv(E.basin_half_xy), z=_jv(E.basin_z_range), frame="world"
            ),
        )
        th = _th(E, "BASIN_XY_TOLERANCE", "BASIN_Z_TOLERANCE", "PENETRATION_TOLERANCE", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        rule = "source bbox centre xy inside basin box (+tol) and bbox z within basin z range (+tol), shallow bathtub contact (no contact deeper than PENETRATION_TOLERANCE), no illegal penetration, and no hand contact (released), HOLD_STEPS steps"
    elif owner == "SurfaceObjectInReceptacle":
        skill_obj = O(E.SOURCE_NAME)
        tgt.update(body=O(E.RECEPTACLE_NAME))
        th = _th(
            E,
            "RECEPTACLE_INTERIOR_SCALE",
            "RECEPTACLE_XY_CONTAINMENT",
            "RECEPTACLE_RIM_CLEARANCE",
            "RECEPTACLE_BOTTOM_TOLERANCE",
            "RELEASE_DISTANCE",
            "HOLD_STEPS",
        )
        th["hold_steps"] = int(E.HOLD_STEPS)
        rule = "source touches receptacle AND source bbox inside receptacle interior (xy ellipse scaled by RECEPTACLE_INTERIOR_SCALE in receptacle frame, bottom below rim-RECEPTACLE_RIM_CLEARANCE and above floor-RECEPTACLE_BOTTOM_TOLERANCE) AND no end-effector site within RELEASE_DISTANCE of the object root, HOLD_STEPS steps"
    elif owner == "IThorKitchenPackProduceIntoBox":
        skill_obj = O(E.TARGET_NAME)
        tgt.update(body=E.usd_box.root_body)
        th = _th(E, "success_hold_seconds", "TARGET_XML")
        th["hold_steps"] = _hold_s(E, E.success_hold_seconds)
        th["box_tolerance_m"] = 0.005
        rule = (
            "object bbox centre inside the box volume (walls/floor geoms, 5 mm tol) AND object touches box AND not grasped (hand closed+contact)"
            if E.TARGET_XML is not None
            else "object ROOT position inside the box inner collision volume (floor to lowest wall top); no contact, grasp or release test"
        ) + "; success_hold_seconds*control_freq steps"
    elif owner == "IThorKitchenAppleOnPlate":
        skill_obj = O(E.SOURCE_NAME)
        tgt.update(body=O("plate"))
        th = _th(E, "APPLE_ON_PLATE_XY_TOL", "success_hold_seconds", "USE_BASE_SUCCESS_HOLD")
        th["hold_steps"] = _hold_s(E, E.success_hold_seconds)
        rule = "object touches plate AND root-body xy distance to plate root xy < APPLE_ON_PLATE_XY_TOL; no release test; success_hold_seconds*control_freq steps (counter advanced in the base _post_action only when USE_BASE_SUCCESS_HOLD, see post_action_owner)"
    elif owner == "IThorKitchenBowlToSink":
        skill_obj = O(E.SOURCE_NAME)
        tgt.update(body=_fix(E, getattr(E, "sink_fixture", None)), region=_region(E, E.sink_region))
        th = _th(E, "SINK_Z_TOL", "SINK_XY_TOL", "success_hold_seconds")
        th["hold_steps"] = _hold_s(E, E.success_hold_seconds)
        th["release_distance"] = 0.25
        rule = "object bbox inside sink region (xy in region frame +SINK_XY_TOL, z within region +SINK_Z_TOL) AND no end-effector site within 0.25 m of the object root, success_hold_seconds*control_freq steps"
    elif owner == "IThorLivingRoomRemoteToBasket":
        skill_obj = O(E.REMOTE_NAME)
        tgt.update(body=O(E.BASKET_NAME))
        th = _th(E, "success_xy_tolerance", "release_distance", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        rule = "object touches basket AND root xy distance to basket root xy < success_xy_tolerance AND no end-effector site within release_distance of the object root, HOLD_STEPS steps"
    elif owner == "HolodeckIceCubeInGlass":
        skill_obj = O("ice_cube")
        m = E.sim.model._model
        gid = [
            i
            for i in range(m.ngeom)
            if "cup" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            and "reg_bbox" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        ][0]
        tgt.update(
            body=m.body(int(m.geom_bodyid[gid])).name,
            region=dict(
                centre=_jv(E.sim.data._data.geom_xpos[gid]),
                half=_jv(m.geom_size[gid]),
                frame="world, axis aligned |ice_root - centre| <= half",
                geom=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid),
            ),
        )
        th = _th(E, "hold_steps")
        rule = "ice cube root position inside the glass reg_bbox geom box (axis-aligned in world) for hold_steps steps; no release test"
    elif owner == "HolodeckSugarCubeInJar":
        skill_obj = O("sugar_cube")
        tgt.update(body=O("jar"))
        th = _th(E, "JAR_BBOX_TOLERANCE", "hold_steps")
        rule = "all 8 sugar bbox corners inside the jar bbox in the jar root frame (xy box and diamond, z range, +JAR_BBOX_TOLERANCE) for hold_steps steps; no release test"
    elif owner == "HolodeckPushChairToDesk":
        skill_obj = E.chair_body
        th = _th(E, "hold_steps", "_pull_origin_xy")
        th["MIN_PULL_DISPLACEMENT"] = float(type(E).MIN_PULL_DISPLACEMENT)
        tgt.update(origin_xy=_jv(E._pull_origin_xy), min_displacement=th["MIN_PULL_DISPLACEMENT"])
        rule = "chair body xy displaced >= MIN_PULL_DISPLACEMENT (euclid) from its reset xy for hold_steps steps; no direction, hand or desk test"
    elif owner == "RoomMultiLevelFixturePnP":
        skill_obj = O(E.SOURCE_NAME)
        tgt.update(body=_fix(E, getattr(E, "multilevel_fixture", None)), region=_region(E, E.target_region))
        th = _th(E, "RELEASE_DISTANCE", "TARGET_LEVEL_Z_TOLERANCE", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        th["target_center_xy_tolerance"] = _jv(E.scene_task_spec()["target_center_xy_tolerance"])
        rule = "object bbox centre xy inside target region (+target_center_xy_tolerance) and bbox bottom within TARGET_LEVEL_Z_TOLERANCE of the region centre z, object touches the shelf, no end-effector within RELEASE_DISTANCE of the object root, HOLD_STEPS steps"
    elif owner == "_BathroomNativeUprightTask":
        skill_obj = E.source_fixture.body_name
        tgt.update(body=_fix(E, E.work_surface_fixture), region=_region(E, E.work_surface_region))
        sup = _fix(E, E.work_surface_fixture)
        th = _th(
            E,
            "UPRIGHT_COSINE",
            "SUPPORT_PLANE_TOLERANCE",
            "CONTACT_PENETRATION_TOLERANCE",
            "_target_upright_axis",
            "HOLD_STEPS",
        )
        th["hold_steps"] = int(E.HOLD_STEPS)
        rule = "object upright (sign*R[2,axis] >= UPRIGHT_COSINE) AND bbox inside work-surface region xy AND bbox bottom within SUPPORT_PLANE_TOLERANCE of the surface plane AND in contact with the surface (no contact deeper than CONTACT_PENETRATION_TOLERANCE) AND no illegal penetration AND no hand contact, HOLD_STEPS steps"
    elif owner == "_BathroomNativePickPlaceTask":
        skill_obj = E.source_fixture.body_name
        tgt.update(body=_fix(E, E.bin_fixture), region=_region(E, E.bin_region))
        th = _th(E, "BIN_CENTER_XY_TOLERANCE", "BIN_CENTER_Z_TOLERANCE", "BIN_MAX_RESET_DISPLACEMENT", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        rule = "object centre xy inside the bin region shrunk by BIN_CENTER_XY_TOLERANCE and z within region z +-BIN_CENTER_Z_TOLERANCE, touching the bin, no hand contact, bin upright, and bin centre moved <= BIN_MAX_RESET_DISPLACEMENT from reset, HOLD_STEPS steps"
    elif owner == "IThorKitchenCupStack":
        names = list(E.CUP_NAMES)
        skill_obj = None
        ex = dict(object_candidates=[O(n) for n in names], symmetric=True)
        th = _th(
            E,
            "STACK_XY_TOL",
            "CUP_SCALE",
            "MIN_STACK_Z_GAP",
            "MAX_STACK_Z_GAP",
            "AXIS_UP_DOT",
            "BOTTOM_AXIS_UP_DOT",
            "stack_hold_steps",
        )
        th["hold_steps"] = int(E.stack_hold_steps)
        rule = "the two cups are stacked in either order (upper centre above lower by min_z_gap..max_z_gap, xy within STACK_XY_TOL*CUP_SCALE), both upright (z-axis dot >= AXIS_UP_DOT, lower >= BOTTOM_AXIS_UP_DOT) and touching; symmetric: either cup may be the moved one; no release test"
    elif owner == "IThorKitchenPanTableToStove":
        skill_obj = O("pan")
        tgt.update(region=_region(E, E.target_region))
        th = _th(E, "pan_base_local_xy", "pan_base_radius", "support_z_tolerance", "success_hold_seconds")
        th["hold_steps"] = _hold_s(E, E.success_hold_seconds)
        rule = "pan base circle (pan_base_local_xy in the pan frame, radius pan_base_radius) inside target region xy, pan bottom within support_z_tolerance of the region z, and pan not grasped (hand closed+contact), success_hold_seconds*control_freq steps"
    elif owner == "IThorKitchenStandCannedDrinkUpright":
        skill_obj = O("canned_drink")
        tgt.update(region=_region(E, E.work_surface_region))
        sup = None
        th = _th(E, "upright_cosine", "support_z_tolerance", "release_distance", "required_success_hold_steps")
        th["hold_steps"] = int(E.required_success_hold_steps)
        rule = "can root z-axis cosine >= upright_cosine AND bbox inside work-surface region xy with bottom within support_z_tolerance AND in contact with the surface fixture AND no end-effector within release_distance of the can root, required_success_hold_steps steps"
    elif owner == "IThorLivingRoomPourBlockFromBasket":
        skill_obj = O(E.TARGET_NAME)
        tgt.update(body=O(E.BASKET_NAME))
        th = _th(E, "OUTSIDE_MARGIN", "_hold_steps_required")
        th["hold_steps"] = int(E._hold_steps_required)
        rule = "block bbox fully outside the basket bbox on ANY one axis (basket frame) by OUTSIDE_MARGIN (this includes above or below the basket, so lifting the block clear also counts); no hand test; _hold_steps_required steps"
    elif owner == "IThorLivingRoomPourIceFromCup":
        skill_obj = O(E.ICE_NAME)
        tgt.update(body=O(E.CUP_NAME))
        th = _th(E, "ICE_REGION_MARGIN", "_hold_steps_required")
        th["hold_steps"] = int(E._hold_steps_required)
        rule = "all ice bbox corners outside the cup interior cylinder (radius size0+ICE_REGION_MARGIN, half height size1+ICE_REGION_MARGIN); no hand test; _hold_steps_required steps"
    elif owner == "IThorLivingRoomPushBasketball":
        skill_obj = O(E.TARGET_NAME)
        th = _th(E, "PUSH_DISTANCE", "MAX_HEIGHT_CHANGE", "HOLD_SECONDS")
        th["hold_steps"] = _hold_s(E, E.HOLD_SECONDS)
        tgt.update(origin_xy=_jv(E._push_initial_pos[:2]), min_displacement=float(E.PUSH_DISTANCE))
        rule = "a hand has touched the ball at some point (latched) AND ball xy displacement from reset >= PUSH_DISTANCE AND |dz| <= MAX_HEIGHT_CHANGE, HOLD_SECONDS*control_freq steps"
    elif owner == "IThorLivingRoomPushBlockToTargetRegion":
        skill_obj = O(E.TARGET_NAME)
        tgt.update(
            region=dict(
                centre_xy=_jv(E.target_region_pos[:2]),
                half_xy=_jv(E.target_region_half_xy),
                yaw=float(E.target_region_yaw),
                frame="world centre, region-rotated half extents, ALL bbox corners inside",
            )
        )
        th = _th(E, "MAX_LIFT_HEIGHT", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        sup = _fix(E, getattr(E, "work_surface_fixture", None))
        rule = "block touched by a hand at some point (latched) AND never lifted above reset bottom + MAX_LIFT_HEIGHT (latched) AND all bbox corners inside the target region xy AND block on the work-surface region (z tol 0.05), HOLD_STEPS steps"
    elif owner == "IThorLivingRoomStackBookOnBook":
        skill_obj = O(E.MOVING_BOOK_NAME)
        tgt.update(body=O(E.BASE_BOOK_NAME))
        th = _th(E, "XY_TOLERANCE", "ORIENTATION_COSINE", "release_distance", "HOLD_STEPS")
        th["hold_steps"] = int(E.HOLD_STEPS)
        rule = "books touch AND bbox-centre xy distance <= XY_TOLERANCE AND moving book bottom >= base book top - 0.03 AND face alignment |dot z-axes| >= ORIENTATION_COSINE AND no end-effector within release_distance of the moving book AND base book still on the work surface (z tol 0.05, contact), HOLD_STEPS steps"
    elif owner == "_HolodeckShelfFolderMixin":
        skill_obj = O("folder")
        sp = E.task_spec
        th = _th(E, "hold_steps", "MODE")
        if E.MODE == "insert":
            tgt.update(
                region=dict(
                    centre=_jv(sp["region_center"]),
                    half=_jv(sp["region_half_size"]),
                    frame="world axis aligned, folder bbox CENTRE inside",
                )
            )
            rule = "folder bbox centre inside the world box region_center +- region_half_size AND folder touches the shelf, hold_steps steps; no release test"
        else:
            rule = "folder has NO contact with the shelf bodies for hold_steps steps (the folder starts in the shelf; lifting it off in any way counts, and so does knocking it clear); no hand, height or release test"
    else:
        return None
    return dict(object=skill_obj, target=tgt, rule=rule, thresholds=th, support=sup, extra=ex)


def _defining_class(E, name):
    for c in type(E).__mro__:
        if name in c.__dict__:
            return c.__name__
    return None


def _articulation_spec(E):
    """The articulated tasks (fork: bed2 the three fridge doors, bed3 the three low-drawer tasks): the joint, its type and
    range, the moving body and its handle body, the start fraction and the goal the benchmark's check holds. None for any
    other task. Families: "articulation" (open or close one joint), "navigation_articulation" (ApproachOpenFridge),
    "articulation_pnp" (LowDrawerPnP: open it, put the apple in, close it; object_body = the apple's root, target.body =
    the drawer's body, the stages' thresholds under `stages`)."""
    name_ = type(E).__name__
    if not hasattr(E, "target_semantic_joint") or not ("Fridge" in name_ or "LowDrawer" in name_):
        return None
    m = E.sim.model._model
    d = E.sim.data._data
    jname = E.target_joint
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    door = int(m.jnt_bodyid[j])
    tree = {door}
    for b in range(m.nbody):  # the moving body's subtree
        p = b
        while p > 0 and p not in tree:
            p = int(m.body_parentid[p])
        if p in tree:
            tree.add(b)
    handle, n_handle = None, 0
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in tree:
            continue
        mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) if m.geom_dataid[g] >= 0 else ""
        if "handle" in f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''} {mesh or ''}".lower():
            cnt = sum(1 for h in range(m.ngeom) if int(m.geom_bodyid[h]) == b)
            if handle is None or cnt > n_handle:
                handle, n_handle = b, cnt
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))  # noqa: E731
    adr = int(m.jnt_qposadr[j])
    sem = E.target_semantic_joint
    closed, opened = (
        float(sem.closed_qpos),
        float(sem.open_qpos),
    )  # fraction = clip((q - closed) / (open - closed), 0, 1)
    close = E.BEHAVIOR == "close"
    hold = int(E.HOLD_STEPS)
    nav = name_ == "IThorKitchenApproachOpenFridge"
    pnp = "LowDrawerPnP" in name_
    drawer = "Drawer" in name_
    thr = float(E.SUCCESS_CLOSE_FRACTION if close else E.SUCCESS_OPEN_FRACTION)
    fraction = float(np.clip((d.qpos[adr] - closed) / (opened - closed), 0.0, 1.0))
    spec = dict(
        family="navigation_articulation" if nav else "articulation_pnp" if pnp else "articulation",
        skill=E.ATOMIC_SKILL,
        joint=jname,
        joint_type={int(mujoco.mjtJoint.mjJNT_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge"}.get(
            int(m.jnt_type[j]), "other"
        ),
        joint_range=[float(x) for x in m.jnt_range[j]],
        closed_qpos=closed,
        open_qpos=opened,
        qpos_adr=adr,
        door_body=name(door),
        handle_body=None if handle is None else name(handle),
        init_fraction=fraction,
        goal=dict(op="<=" if close else ">=", threshold=thr, hold_steps=0 if nav else hold),
        target=dict(body=name(door) if pnp else None, region=None),
        object_body=None,
        hand_drives=drawer,  # the drawer tasks' compliance: the motion must come from a hand
        rule="joint fraction clip((q - closed) / (open - closed), 0, 1) "
        + ("<=" if close else ">=")
        + f" {thr} for {0 if nav else hold} consecutive steps"
        + (" and robot-fridge geom distance <= max_dist" if nav else ""),
    )
    if nav:
        spec["max_dist"] = float(E.SUCCESS_DISTANCE)
    if drawer and not pnp:  # the walk-and-crouch tasks: stage 1 is a 5-step stay within approach_distance of the drawer
        spec["approach_distance"] = float(E._get_task_spec()["approach_distance"])
    if pnp:
        apple = E.objects["apple"].root_body
        spec.update(
            object_body=apple,
            goal=dict(op="<=", threshold=float(E.SUCCESS_CLOSE_FRACTION), hold_steps=hold),
            stages=[
                dict(name="open", op=">=", threshold=float(E.SUCCESS_OPEN_FRACTION), hold_steps=5),
                dict(
                    name="put_and_close",
                    op="<=",
                    threshold=float(E.SUCCESS_CLOSE_FRACTION),
                    hold_steps=hold,
                    also="apple centre inside the drawer's receptacle site box",
                ),
            ],  # fmt: skip
            rule="stage 1: fraction >= %.2f for 5 steps; then stage 2: apple bbox centre inside the drawer's receptacle site box AND fraction <= %.2f for %d consecutive steps"
            % (float(E.SUCCESS_OPEN_FRACTION), float(E.SUCCESS_CLOSE_FRACTION), hold),
        )
    return spec


def build_spec(E):
    try:
        art = _articulation_spec(E)
        if art is not None:
            return jsonable(art)
        meta = E.get_ep_meta()
        owner = _defining_class(E, "_check_success")
        r = _spec_for(E, owner)
        if r is None:
            return {"unmapped": f"no builder for the class {owner} that defines _check_success"}
        o = r["object"]
        sup_attr = None
        if r["support"] is None:  # the class's own notion of what the object stands on, if it keeps one
            for a in (
                "work_surface_fixture",
                "lift_fixture",
                "support_fixture",
                "table_fixture",
                "low_table_fixture",
                "base_support_fixture",
                "desk_fixture",
            ):
                f = getattr(E, a, None)
                if f is not None and hasattr(f, "body_name"):
                    r["support"], sup_attr = f.body_name, a
                    break
        fr = _free_root(E, o) if o else None
        spec = dict(
            family=meta.get("task_family"),
            skill=meta.get("atomic_skill"),
            check_owner=owner,
            post_action_owner=_defining_class(E, "_post_action"),
            predicate_owner=_defining_class(E, "_raw_place_success") or _defining_class(E, "_raw_success"),
            object_body=fr,
            object_bodies=_tree(E, fr) if fr else None,
            object_named_root=o,
            target=r["target"],
            rule=r["rule"],
            thresholds=r["thresholds"],
            support_body=r["support"],
            support_attr=sup_attr,
        )
        spec.update(r["extra"])
        return jsonable(spec)
    except Exception as e:  # never guess: report why
        return {"unmapped": f"builder failed: {type(e).__name__}: {e}"}


class Env:
    def __init__(self, task, cameras):
        row = ROWS[task]
        self.task, self.cameras, self.row = task, cameras, row
        a = sc.parse_args(
            [
                "--backend",
                "scalebfm",
                "--env",
                row["env"],
                "--scene-id",
                str(row["scene_id"]),
                "--fixed-scene",
                "--scene-seed",
                str(DEFAULTS["scene_seed"]),
                "--inference-backend",
                "onnx-cpu",
            ]
        )
        self.args = a
        cfg = sc.build_controller_config(a)
        for line in sc.apply_task_overrides(a.env, a.scene_id):
            print(f"[server] task override: {line}", flush=True)
        self.env = robosuite.make(
            env_name=a.env,
            robots=a.robot,
            controller_configs=cfg,
            has_renderer=False,
            has_offscreen_renderer=cameras,
            use_camera_obs=cameras,
            camera_names=[CAM] if cameras else None,
            camera_heights=240,
            camera_widths=320,
            camera_depths=cameras,
            render_camera=None,
            control_freq=50,
            ignore_done=True,
            renderer="mujoco",
            seed=0,
            scene_ids=a.scene_id,
            fixed_scene_seed=a.scene_seed,
        )
        self.base = getattr(self.env, "unwrapped", self.env)
        m = self.m
        self.jid = m.actuator_trnid[:, 0].copy()
        self.qadr = m.jnt_qposadr[self.jid].copy()
        self.dadr = m.jnt_dofadr[self.jid].copy()
        self.gear = m.actuator_gear[:, 0].copy()
        self.cr = m.actuator_ctrlrange.copy()
        self.target = None
        self.steps = 0
        self.obs = None
        self.zero = np.zeros(self.env.action_dim)
        xml = f"{out_dir}/{task}/exported.xml"
        os.makedirs(os.path.dirname(xml), exist_ok=True)
        open(xml, "w").write(self.base.model.get_xml())
        self.xml = xml

    # env.reset() may rebuild the sim, so never cache the MjModel / MjData handles
    @property
    def m(self):
        return self.base.sim.model._model

    @property
    def d(self):
        return self.base.sim.data._data

    def install_hook(self):
        robot = self.env.robots[0]
        if getattr(robot.control, "_lml_hook", False):
            return

        def control(action, policy_step=False):
            q_des, kp, kd = self.target
            q, v = self.d.qpos[self.qadr], self.d.qvel[self.dadr]
            tau = kp * (q_des - q) - kd * v
            self.d.ctrl[:] = np.clip(tau / self.gear, self.cr[:, 0], self.cr[:, 1])

        control._lml_hook = True
        robot.control = control

    def reset(self, episode, jitter_seed, robot_radius=None, object_radius=None):
        """robot_radius / object_radius (m): the jitter disks; None is the official one (tasks.yaml defaults). Larger
        disks go through the same utility, which keeps a draw only when nothing collides and the task's placements
        stay valid, and leaves the nominal pose after 32 failed draws."""
        loop_wall(episode, jitter_seed, robot_radius, object_radius)
        a = self.args
        a.reset_jitter, a.reset_object_jitter, a.jitter_seed = (
            DEFAULTS["reset_jitter"] if robot_radius is None else float(robot_radius),
            DEFAULTS["reset_object_jitter"] if object_radius is None else float(object_radius),
            jitter_seed,
        )
        t = time.time()
        obs = self.env.reset()
        q0 = self.d.qpos.copy()
        obs = sc._apply_reset_jitter(self.env, self.base, a, obs, episode=episode)
        m, moved = self.m, {"robot_cm": 0.0, "object_cm": 0.0}
        for j in range(m.njnt):  # how far the jitter moved the robot and the farthest-moved free object
            if m.jnt_type[j] == 0:
                k = int(m.jnt_qposadr[j])
                d = float(100 * np.linalg.norm(self.d.qpos[k : k + 2] - q0[k : k + 2]))
                key = (
                    "robot_cm"
                    if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).endswith("floating_base_joint")
                    else "object_cm"
                )
                moved[key] = max(moved[key], d)
        self.jitter = dict(moved, robot_radius=a.reset_jitter, object_radius=a.reset_object_jitter)
        self.obs = obs
        self.sha = sc._initial_state_digest(self.base)["sha256"]
        self.steps = 0
        q = self.d.qpos[self.qadr].copy()
        self.target = (q, np.zeros(len(q)), np.zeros(len(q)))
        self.install_hook()
        self.reset_s = time.time() - t

    def source_info(self):
        info = {}
        b = self.base
        for name in sorted(dir(b)):
            if name.startswith("__") or not any(
                k in name.lower() for k in ("fixture", "target", "receptacle", "source", "region_refs")
            ):
                continue
            try:
                v = getattr(b, name)
            except Exception:
                continue
            if callable(v):
                continue
            if hasattr(v, "body_name"):
                info[name] = {"body_name": v.body_name}
            elif isinstance(v, dict):
                items = {k: x.body_name for k, x in v.items() if hasattr(x, "body_name")}
                if items:
                    info[name] = items
            elif isinstance(v, (int, float, str, bool, np.ndarray, list, tuple)) and not name.startswith("_get"):
                try:
                    info[name] = jsonable(v)
                except Exception:
                    pass
        for name in ("MIN_LIFT_DELTA", "HOLD_STEPS", "GRASP_HEIGHT_ALLOWANCE", "HOLD_SECONDS", "_initial_source_z"):
            if hasattr(b, name):
                info[name] = jsonable(getattr(b, name))
        return info

    def reset_reply(self):
        m = self.m
        joints = [
            dict(
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                type=int(m.jnt_type[j]),
                qpos_adr=int(m.jnt_qposadr[j]),
                dof_adr=int(m.jnt_dofadr[j]),
            )
            for j in range(m.njnt)
        ]
        acts = [
            dict(
                joint=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)),
                ctrlrange=self.cr[i].tolist(),
                gear=float(self.gear[i]),
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            )
            for i, j in enumerate(self.jid)
        ]
        try:
            meta = jsonable(self.base.get_ep_meta())
        except Exception as e:
            meta = {"error": str(e)}
        return dict(
            xml=self.xml,
            joints=joints,
            qpos=self.d.qpos.copy(),
            qvel=self.d.qvel.copy(),
            actuators=acts,
            max_steps=int(self.row["max_steps"]),
            prompt=self.row["prompt"],
            meta=meta,
            source=self.source_info(),
            spec=build_spec(self.base),
            jitter=self.jitter,
            sha=self.sha,
            reset_s=self.reset_s,
            genuine_min_steps=int(self.row["genuine_min_steps"]),
        )

    def step(self, q_des, kp, kd):
        self.target = tuple(np.asarray(x, dtype=np.float64).reshape(-1) for x in (q_des, kp, kd))
        self.obs = self.env.step(self.zero)[0]
        self.steps += 1
        return dict(
            qpos=self.d.qpos.copy(),
            qvel=self.d.qvel.copy(),
            success=bool(self.base._check_success()),
            steps=self.steps,
            truncated=self.steps >= int(self.row["max_steps"]),
        )

    def eyes(self):
        from robosuite.utils.camera_utils import get_real_depth_map

        o = self.obs
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
        depth = get_real_depth_map(self.base.sim, np.asarray(o[f"{CAM}_depth"]))[::-1, :, 0]
        return dict(
            rgb=np.ascontiguousarray(o[f"{CAM}_image"][::-1]),
            depth=np.ascontiguousarray(depth).astype(np.float32),
            fovy=float(self.m.cam_fovy[cid]),
            cam_pos=self.d.cam_xpos[cid].copy(),
            cam_mat=self.d.cam_xmat[cid].reshape(3, 3).copy(),
        )


def main():
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    lis = Listener(sock_path, family="AF_UNIX")
    print("[server] listening", sock_path, flush=True)
    cur = None
    while True:
        conn = lis.accept()
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            try:
                if msg[0] == "quit":
                    conn.send({"ok": True})
                    return
                if msg[0] == "tasks":
                    rep = {
                        "tasks": [
                            dict(
                                name=k,
                                **{
                                    f: v[f]
                                    for f in (
                                        "env",
                                        "scene_id",
                                        "type",
                                        "split",
                                        "episodes",
                                        "max_steps",
                                        "genuine_min_steps",
                                        "prompt",
                                    )
                                },
                            )
                            for k, v in ROWS.items()
                        ]
                    }
                elif msg[0] == "reset":
                    _, task, episode, jseed, cams, *radii = msg
                    cams = bool(cams)
                    loop_wall(episode, jseed, *(radii + [None, None])[:2])  # before any environment is built
                    if cur is None or cur.task != task or cur.cameras != cams:
                        if cur is not None:
                            cur.env.close()
                        cur = Env(task, cams)
                    cur.reset(int(episode), int(jseed), *radii)
                    rep = cur.reset_reply()
                elif msg[0] == "step":
                    rep = cur.step(msg[1], msg[2], msg[3])
                elif msg[0] == "eyes":
                    rep = cur.eyes()
                else:
                    rep = {"error": f"unknown message {msg[0]}"}
            except Exception:
                rep = {"error": traceback.format_exc()}
            conn.send(rep)


if __name__ == "__main__":
    main()
