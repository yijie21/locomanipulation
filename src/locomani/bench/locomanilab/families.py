"""The 26 tasks the bed runs and what each family of them switches on (TODO 49.4; replaces LOOP_TASKS, the three `missing()` case
lists of run_lml_q.sh / run_lml_bed2.sh / run_lml_bed3.sh, and the per-bed ART tuples).

One bed for all tasks: the old beds were three copies of one file set (the fridge and drawer beds each added their own articulation
code). What differs between tasks is only data: which practice pool the task's rows are in, whether the robot looks at an object
before the round (the look pass), which judge family its success check belongs to, and whether the articulated recorder runs.

group      tasks   pool (config field)           judge families (spec["family"] of specs.json)
lml_q      20      practice_dir                  navigation, lift, pick, pick_place, receptacle, composite
fridge      3      practice_dir_fridge           articulation, navigation_articulation
drawer      3      practice_dir_drawer           articulation, articulation_pnp
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskInfo:
    group: str  # "lml_q" | "fridge" | "drawer"
    kind: str  # what the instruction asks, as view `parse` names it: navigation | pick | place | articulate
    looks: bool  # the look pass (the target's seen points + GraspGen-X grasps) is needed before the round


PRACTICE_FIELD = dict(lml_q="practice_dir", fridge="practice_dir_fridge", drawer="practice_dir_drawer")


def _t(group: str, kind: str, looks: bool = True) -> TaskInfo:
    return TaskInfo(group, kind, looks)


TASKS: dict[str, TaskInfo] = {
    # the plan's 20 "atomic, seen" tasks
    "HolodeckApproachDesk_scene117": _t("lml_q", "navigation", False),
    "IThorBathroomApproachToilet_scene401": _t("lml_q", "navigation", False),
    "IThorBedroomApproachBed_scene329": _t("lml_q", "navigation", False),
    "IThorKitchenApproachFridge_scene13": _t("lml_q", "navigation", False),
    "IThorLivingRoomApproachSofa_scene209": _t("lml_q", "navigation", False),
    "IThorBathroomLiftSoapBottle_scene401": _t("lml_q", "pick"),
    "HolodeckPickBookFromDresser_scene117": _t("lml_q", "pick"),
    "IThorBedroomLiftBasketFromFloor_scene329": _t("lml_q", "pick"),
    "IThorBathroomLiftScrubBrush_scene401": _t("lml_q", "pick"),
    "HolodeckMoveCupToCenter_scene117": _t("lml_q", "place"),
    "HolodeckMoveCupToCorner_scene117": _t("lml_q", "place"),
    "IThorBathroomPutSpongeInBathtub_scene401": _t("lml_q", "place"),
    "IThorBedroomBoxedFoodIntoBasket_scene329": _t("lml_q", "place"),
    "IThorBedroomCanIntoBasket_scene329": _t("lml_q", "place"),
    "IThorKitchenAppleIntoBox_scene13": _t("lml_q", "place"),
    "IThorKitchenBendAndPutAppleOnPlate_scene13": _t("lml_q", "place"),
    "IThorKitchenBreadOnPlate_scene13": _t("lml_q", "place"),
    "IThorKitchenCupToSink_scene13": _t("lml_q", "place"),
    "IThorLivingRoomBookIntoBasket_scene209": _t("lml_q", "place"),
    "IThorLivingRoomCandleToBasket_scene209": _t("lml_q", "place"),
    # the three fridge-door tasks (nothing to look at before the round: the door's handle is looked at in the episode)
    "IThorKitchenOpenFridge_scene13": _t("fridge", "articulate", False),
    "IThorKitchenCloseFridge_scene13": _t("fridge", "articulate", False),
    "IThorKitchenApproachOpenFridge_scene13": _t("fridge", "articulate", False),
    # the three low-drawer tasks (the pick-and-place looks at its apple)
    "IThorLivingRoomLowDrawerPnP_scene209": _t("drawer", "articulate", True),
    "IThorLivingRoomWalkAndOpenLowDrawer_scene209": _t("drawer", "articulate", False),
    "IThorLivingRoomWalkAndCloseLowDrawer_scene209": _t("drawer", "articulate", False),
}

# judge families (spec["family"]) that are articulated: the recorder of the moving part runs, the judge is strict about falling
ART_FAMILIES = ("articulation", "navigation_articulation", "articulation_pnp")


def info(task: str) -> TaskInfo:
    try:
        return TASKS[task]
    except KeyError:
        raise KeyError(f"{task!r} is not one of the bed's {len(TASKS)} tasks") from None


def practice_dir(cfg, task: str) -> str:
    """The folder of the task's practice file `<Task>.jsonl`."""
    return getattr(cfg, PRACTICE_FIELD[info(task).group])
