"""grasp_gym.py's per-round loop, split out of its main() (refactor step 4b,
pure extraction; decision 9's file-size cap).

`round_loop()` reads `grasp_gym.RESTART_DEPLOY` and calls `grasp_gym.fresh_deploy`
/ `grasp_gym.run_round` / `grasp_gym.instruction` through a deferred import of
grasp_gym.py (not a module-level one, since grasp_gym.py imports this
function back -- a module-level import in the other direction would make
the two files a load-order-sensitive circular import). `RESTART_DEPLOY` is
one of grasp_gym.py's monkeypatchable globals (a caller can assign
`grasp_gym.RESTART_DEPLOY = False`), so it has to be read fresh off the
module each round, not captured once; `EDGE_MIN`/`EDGE_MAX`/`CAN_INBOARD`/
`RAISE_LEAD` are plain constants there but get the same qualified treatment
for consistency and to avoid a second, differently-shaped circular import.

`round_loop(args, setup)` takes `setup` (a `grasp_gym._SceneSetup`) duck-typed
rather than importing its class, for the same reason.
"""

from __future__ import annotations

import json
import time

import numpy as np

from locomani import paths
from locomani.backends.sonic.primitives import LocomotionMode, wrap


def round_loop(args, setup) -> None:
    """The per-round loop -- unpacks `setup` back into the same local names
    the inline body used, so nothing below this line changed."""
    from locomani.tasks.bottle import grasp_gym

    fixed_cases = setup.fixed_cases
    sampled_cases = setup.sampled_cases
    reach_config = setup.reach_config
    edge_range = setup.edge_range
    off_lo = setup.off_lo
    off_hi = setup.off_hi
    sidecar = setup.sidecar
    rng = setup.rng
    meta = setup.meta
    pool = setup.pool
    surf_a = setup.surf_a
    half = setup.half
    cxy = setup.cxy
    surf_b = setup.surf_b
    rects = setup.rects
    edge_dist = setup.edge_dist
    rm = setup.rm
    plans = setup.plans
    p = setup.p
    reset = setup.reset
    robot_nom = setup.robot_nom
    from locomani.pipeline.experiment_cases import save_cases

    goal_key, place_config, goal_prompt, goal_xy = None, None, None, None
    if args.task_config_obj is not None:
        task_cfg = args.task_config_obj
        goal_key, place_config = (task_cfg.goal or None), task_cfg.place
        goal_prompt = task_cfg.prompt
        goal_xy = np.asarray(meta["objects"][goal_key]["init_pos"][:2], dtype=float) if goal_key else None
        if goal_key and goal_key not in meta["objects"]:
            raise SystemExit(f"--task-config goal {goal_key!r} is not in the scene ({sorted(meta['objects'])})")
        print(f"[gym] goal: {goal_key or 'none (lift and hold)'}")

    wins = 0
    played = 0
    deck: list[str] = []
    for rnd in range(args.rounds):
        # --until-ok collects a fixed number of SUCCESSES, which is what a
        # dataset is counted in; --rounds stays the attempt cap so a run
        # whose success rate collapses still ends.
        if args.until_ok is not None and wins >= args.until_ok:
            break
        played += 1
        t0 = time.time()
        # -- sample --------------------------------------------------------
        # Draw the round's target first: everything after it -- the edge to
        # hug, the grip height, the stand candidates, the squat -- is a
        # property of WHICH object this round is about.
        #
        # A shuffled deck drawn without replacement, not iid uniform. What
        # this feeds is a dataset, and a dataset wants BALANCE: an iid draw
        # over four targets left one of them unseen for the first eight
        # rounds of a twelve-round batch, which at collection scale is a
        # class the model gets much less of for no reason anyone chose. The
        # deck guarantees every target appears once per cycle and still
        # randomises the order within it.
        if fixed_cases is not None:
            case = fixed_cases[rnd]
            obj_key = case["target"]
            pl = plans[obj_key]
            can_nom = pl.nom
            task = grasp_gym.instruction(pl.phrase)
            can_xy, stand, rel = (np.asarray(case[k], dtype=float) for k in ("can_xy", "stand", "rel"))
            psi, spawn_off, spawn_lat, dpsi = (case[k] for k in ("psi", "spawn_off", "spawn_lat", "dpsi"))
            squat_hint, appearance = case["squat_hint"], case["appearance"]
        else:
            if not deck:
                deck = list(pool)
                rng.shuffle(deck)
            obj_key = deck.pop()
            pl = plans[obj_key]
            can_nom, rel_cands, squat_hint = pl.nom, pl.rel_cands, pl.squat_hint
            ax, sgn = pl.ax, pl.sgn
            task = grasp_gym.instruction(pl.phrase)
            # One integer per episode decides the whole look of the scene, so the
            # manifest can record it and a replay can reproduce the exact frame.
            appearance = int(rng.integers(0, 2**31 - 1))
            # The target lives in a band hugging its nominal edge (radial 5-10 cm
            # inboard, +-args.edge_jitter along it): a pelvis 10 cm OUTSIDE the edge
            # then reaches it at rel depth 0.15-0.24 — comfortably inside the
            # map instead of at its rim, which is what lets the no-contact
            # stand-off exist at all.
            for try_i in range(500):
                can_xy = can_nom.copy()
                can_xy[ax] = cxy[ax] + sgn * (half[ax] - rng.uniform(*grasp_gym.CAN_INBOARD))
                can_xy[1 - ax] = can_nom[1 - ax] + rng.uniform(-args.edge_jitter, args.edge_jitter)
                if goal_key and abs(can_xy[1 - ax] - goal_xy[1 - ax]) < place_config.goal_gap:
                    continue  # pick-and-place: the edge jitter must not put the target against the container
                from locomani.tasks.bottle.ready_grasp_policy import sample_stand as sample_comfortable_stand

                rel, squat_hint = sample_comfortable_stand(pl, rng, reach_config)
                # Face the table from outside, along the can's inward edge
                # normal, +-25 deg (a uniform world yaw almost never yields a
                # stand outside the table).
                normal = np.zeros(2)
                normal[ax] = -sgn
                psi = float(np.arctan2(normal[1], normal[0]) + rng.uniform(-0.14, 0.14))
                c, s = np.cos(psi), np.sin(psi)
                stand = can_xy - np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
                # Prefer slack over the bare floor: a stand SAMPLED at edge
                # 0.17 exactly leaves the +-4 cm execution scatter nowhere to
                # land (the edge-hold retry loop — a yaw -90 draw spent six
                # re-approaches hovering at 0.11-0.15). First 300 tries demand
                # 3 cm of headroom; only then fall back to the bare band.
                edge_lo = grasp_gym.EDGE_MIN + (0.03 if try_i < 300 else 0.0)
                if not (edge_lo <= edge_dist(stand) <= grasp_gym.EDGE_MAX):
                    continue
                # Where the robot starts. Sampled INSIDE the rejection loop
                # because the clearance check below depends on all three: a
                # spawn 3 m out with 0.35 m of lateral offset crosses different
                # furniture than one 1 m out dead on the line.
                spawn_lat = float(rng.uniform(-args.spawn_lateral, args.spawn_lateral))
                fv2 = np.array([np.cos(psi), np.sin(psi)])
                perp = np.array([-fv2[1], fv2[0]])
                if edge_range is None:
                    spawn_off = float(rng.uniform(off_lo, off_hi))
                else:
                    # Bisect for the offset that lands the spawn the requested
                    # distance from the table. edge_dist grows monotonically as
                    # the robot backs off along -fv2, so 60 halvings of [0, 6]
                    # resolve it far below any distance that matters.
                    want = float(rng.uniform(*edge_range))
                    lo_off, hi_off = 0.0, 6.0
                    for _ in range(60):
                        mid = 0.5 * (lo_off + hi_off)
                        if edge_dist(stand - mid * fv2 + spawn_lat * perp) < want:
                            lo_off = mid
                        else:
                            hi_off = mid
                    spawn_off = 0.5 * (lo_off + hi_off)
                spawn_pt = stand - spawn_off * fv2 + spawn_lat * perp
                jitter = float(np.deg2rad(rng.uniform(-args.spawn_yaw_err, args.spawn_yaw_err)))
                if args.face_object:
                    # Face the object FROM THE SPAWN, then jitter. dpsi stays the
                    # quantity the episode turns out at the spawn, so everything
                    # downstream (the turn, its guards, the manifest) is unchanged.
                    to_obj = can_xy - spawn_pt
                    dpsi = float(wrap(np.arctan2(to_obj[1], to_obj[0]) - psi) + jitter)
                else:
                    dpsi = jitter
                # The WALK must clear the table too: the batch's "fell during
                # the walk-up" class was the robot brushing the table edge on
                # the way in (video: hip catches the rim, robot tips backwards
                # onto the tabletop). Check the ACTUAL two legs, not one
                # straight line: with a lateral offset the robot goes
                # spawn -> lead_tgt (a diagonal) -> stand (on the line), and
                # the diagonal is the segment that swings widest.
                lead_pt = stand - (0.30 + grasp_gym.RAISE_LEAD) * fv2
                legs = [(spawn_pt, lead_pt), (lead_pt, stand)]
                n_chk = max(2, int(np.ceil(spawn_off / 0.25)) + 1)
                pts = [q for a, b in legs for q in np.linspace(a, b, n_chk, endpoint=False)]
                if all(edge_dist(q) >= grasp_gym.EDGE_MIN for q in pts):
                    break
            else:
                print("EPISODE FAILED [gym] no feasible sample found")
                continue
        case_id = fixed_cases[rnd]["case_id"] if fixed_cases is not None else f"{args.seed}:{rnd}"
        sampled_cases.append(
            dict(
                case_id=case_id,
                target=obj_key,
                appearance=int(appearance),
                can_xy=can_xy.tolist(),
                stand=stand.tolist(),
                rel=rel.tolist(),
                psi=float(psi),
                spawn_off=float(spawn_off),
                spawn_lat=float(spawn_lat),
                dpsi=float(dpsi),
                squat_hint=None if squat_hint is None else float(squat_hint),
            )
        )
        if fixed_cases is not None and "object_yaw" in case:
            sampled_cases[-1]["object_yaw"] = case["object_yaw"]
        if args.sample_only:
            continue
        if goal_key:
            # Pick-and-place: the instruction names the container too (BottleTaskConfig.prompt).
            task = goal_prompt
        side = "right"
        goal = f"{args.until_ok} ok" if args.until_ok is not None else f"{args.rounds}"
        print(
            f"[gym] round {rnd + 1}/{goal} (ok {wins}): {obj_key} "
            f"{np.round(can_xy, 2)}"
            f" rel [{rel[0]:.2f} {rel[1]:+.2f}] yaw {np.degrees(psi):.0f}"
            f" side {side} | spawn {spawn_off:.2f}m lat {spawn_lat:+.2f}m "
            f'yaw_err {np.degrees(dpsi):+.0f}deg | "{task}"'
        )

        # -- reset + settle -----------------------------------------------
        # A stand-stage failure may reflect navigation or initialization.
        # Keep legacy retries available, but fixed-case acceptance can disable
        # them rather than hide a failed first attempt behind reset success.
        for attempt in range(1 + args.stand_retries):
            if grasp_gym.RESTART_DEPLOY and (rnd > 0 or attempt > 0):
                # A fresh deploy per round (and per stand retry): the
                # nominal-spawn calibration already ran on this one, so
                # round 1 right after a stack boot skips the restart.
                # (A warm-up stand after the restart was tried and REVERTED:
                # it made 9/24 rounds fail to rise at all — start_control
                # right after "Init Done" while the engines still lazy-load
                # leaves the deploy's stand controller broken. The walk-up
                # falls it targeted turned out to be table brushes, fixed by
                # the walk-line clearance in the sampler.)
                grasp_gym.fresh_deploy(p)
            if args.dump_sample:
                json.dump(
                    dict(
                        target=obj_key,
                        appearance=appearance,
                        d_can=list(map(float, can_xy - can_nom)),
                        d_robot=list(map(float, stand - robot_nom)),
                        psi=float(psi),
                        side=side,
                        rel=list(map(float, rel)),
                        spawn_off=float(spawn_off),
                        spawn_lat=float(spawn_lat),
                        dpsi=float(dpsi),
                        squat_hint=(None if squat_hint is None else float(squat_hint)),
                        stand=list(map(float, stand)),
                        robot_nom=list(map(float, robot_nom)),
                    ),
                    open(args.dump_sample, "w"),
                    indent=1,
                )
            if sidecar:
                sidecar.start(rnd, attempt)
            round_result = grasp_gym.run_round(
                p,
                reset,
                meta,
                obj_key,
                surf_a,
                edge_dist,
                can_xy - can_nom,
                stand - robot_nom,
                psi,
                side,
                rel,
                rm,
                stand,
                squat_hint,
                spawn_off=spawn_off,
                spawn_lat=spawn_lat,
                dpsi=dpsi,
                appearance=appearance,
                record=args.record,
                view_gate_x=args.view_gate_x,
                view_gate_y=args.view_gate_y,
                one_shot=args.one_shot,
                mid_band=args.mid_band,
                reach_config=reach_config,
                object_yaw=(case.get("object_yaw") if fixed_cases is not None else None),
                goal_key=goal_key,
                place_config=place_config,
                round_id=rnd,
            )
            ok, why, produced = round_result
            if args.record and produced and ok:
                # 'c' finishes RECORDING -> NEED_TO_SAVE and the exporter's
                # own loop writes the episode. Failures need no branch here
                # any more: run_round discards them itself, before the retry
                # below can toggle the exporter out of phase.
                p.cmd.key("c")
                # facing MATTERS here: an IDLE hold without one commands the
                # default world +x, which pirouetted the robot away from the
                # approach heading right after the back-off -- the "grasp,
                # retreat, then spin round" seen in every stream video. The
                # episode itself was never contaminated (the 'c' above ends
                # it first; fast250 tails hold yaw to 7-16 deg), but the spin
                # wastes 2-4 s of settling and reads as part of the task.
                p.cmd.hold(
                    4.0, mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0)
                )  # let the write finish
            if sidecar:
                sidecar.stop()
            if args.manifest:
                # One line per ATTEMPT, not per round: a stand retry that got
                # as far as recording leaves an episode in the session, so
                # the converter's join has to see it too. It filters on
                # `produced`; the rest is for reading per-target success.
                with open(args.manifest, "a") as f:
                    # The scene's actual dimensions, not the seed that
                    # produced them: they are what an analysis wants, they
                    # survive a change to the sampling ranges, and they cost
                    # nothing since the scene already declares them.
                    o = meta["objects"][obj_key]
                    # The INITIAL CONDITIONS, not just the outcome. Without
                    # these a finished dataset cannot be replayed: an episode
                    # says what happened but not what it started from, so a
                    # closed-loop policy cannot be put on the same scene a
                    # recorded episode used, and "same initial conditions" is
                    # the only way to compare two policies without paying for
                    # the variance of independent draws. Cheap to write, and
                    # impossible to recover afterwards -- the collection that
                    # produced the first 200-episode bottle set omitted them
                    # and the holdout episodes' scenes are simply gone.
                    # World-frame, which is what EpisodeControlClient.reset
                    # takes: object xy, the robot's spawn xy, and its yaw.
                    fv2 = np.array([np.cos(psi), np.sin(psi)])
                    spawn_xy = stand - spawn_off * fv2 + spawn_lat * np.array([-fv2[1], fv2[0]])
                    rec_line = dict(
                        case_id=case_id,
                        round=rnd,
                        attempt=attempt,
                        target=obj_key,
                        task=task,
                        ok=bool(ok),
                        why=why,
                        appearance=appearance,
                        init=dict(
                            obj_xy=[round(float(v), 5) for v in can_xy],
                            robot_xy=[round(float(v), 5) for v in spawn_xy],
                            stand_xy=[round(float(v), 5) for v in stand],
                            yaw=round(float(psi + dpsi), 5),
                            psi=round(float(psi), 5),
                            side=side,
                            spawn_off=round(float(spawn_off), 4),
                            spawn_lat=round(float(spawn_lat), 4),
                        ),
                        table_top=round(float(surf_a["top"]), 4),
                        obj_height=round(float(o["height"]), 4),
                        obj_diameter=round(float(o["box"]["half_extent"][0]) * 2, 4),
                        produced=bool(args.record and produced),
                        # The park distances this dataset was missing: without
                        # them the cliff analysis has to be re-mined from pose
                        # sidecars.
                        park_first=round_result.execution["metrics"].get("first_stop"),
                        park=round_result.execution["metrics"].get("park"),
                        transition=round_result.execution["transition"],
                        execution=round_result.execution,
                        seconds=round(time.time() - t0, 1),
                    )
                    f.write(json.dumps(rec_line) + chr(10))
            if ok or not why.startswith("[stand]") or attempt >= args.stand_retries:
                break
            p.log(f"[gym] stand-stage retry ({why})")
        dt = time.time() - t0
        if ok:
            wins += 1
            print(f"EPISODE OK in {dt:.0f}s")
        else:
            print(f"EPISODE FAILED {why} in {dt:.0f}s")
        reset.end_video()
    if args.write_cases:
        save_cases(args.write_cases, sampled_cases, paths.SCENE_META)

    print(f"[gym] {wins}/{played} OK")
