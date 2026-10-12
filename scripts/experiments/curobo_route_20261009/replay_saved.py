"""Item 4 of docs/handoff/single-arm-curobo-route-plan-20261009.md: replay the GraspNet comparison's saved CuRobo requests.

Each saved request (outputs/runs/gnb_obj_NNN{,_fixed}/grasp_planned/round_01_curobo/curobo_request.json, 119) is cut to its
first goal -- the single-arm main line plans one pre-grasp at a time -- and sent twice, the second pass in reverse
order so every request meets a different predecessor. `--service new` talks to primitives/curobo_service.py
(configured once: the first request's joints, limits and start as the retract posture, which is what the old server
used for every request), `--service old` to the GraspNet experiment's curobo_server.py. Writes per-request results and
a summary: success agreement between the passes, chosen-path difference, call time.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/curobo_route_20261009/replay_saved.py --service new --port 5795 outputs/curobo_route_20261009/new.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import zmq

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from locomani.primitives.arm_route import ArmRouteClient, Box, MeshObstacle  # noqa: E402


def saved_requests():
    # the two formal batches only (gnb_obj_NNN, 10-08, and gnb_obj_NNN_fixed, 10-09), not the debug runs beside them
    paths = sorted(
        p
        for pattern in ("gnb_obj_[0-9][0-9][0-9]", "gnb_obj_[0-9][0-9][0-9]_fixed")
        for p in glob.glob(str(REPO / "outputs/runs" / pattern / "grasp_planned/round_01_curobo/curobo_request.json"))
    )
    return [(str(Path(p).relative_to(REPO)), json.loads(Path(p).read_text())) for p in paths]


def obstacles(req):
    t = req["table"]
    return [
        Box("table", (*t["center"], *t.get("quat", [1, 0, 0, 0])), tuple(2 * h for h in t["half"])),
        MeshObstacle(
            "target", (0, 0, 0, 1, 0, 0, 0), np.asarray(req["object"]["vertices"]), np.asarray(req["object"]["faces"])
        ),
    ]


class OldServer:
    def __init__(self, port):
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.connect(f"tcp://127.0.0.1:{port}")

    def plan(self, req):
        one = dict(req, pregrasp=req["pregrasp"][:1], grasp=req["grasp"][:1])
        t0 = time.time()
        self.sock.send_string(json.dumps(one))
        rep = json.loads(self.sock.recv())
        return dict(
            ok=bool(rep["ok"]),
            success=(rep.get("success") or [bool(rep["ok"])])[:1],
            path=rep.get("pregrasp_path") or [],
            seconds=time.time() - t0,
            error=rep.get("status") if str(rep.get("status", "")).startswith("error") else "",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", choices=("new", "old"), required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="first N requests only")
    args = ap.parse_args()
    reqs = saved_requests()
    if args.limit:
        reqs = reqs[: args.limit]
    first = reqs[0][1]
    if args.service == "new":
        client = ArmRouteClient(args.port, plan_timeout_s=120)
        t0 = time.time()
        rep = client.configure(first["names"], first["lower"], first["upper"], first["start"])
        print(f"configure: built {rep['built']} {time.time() - t0:.1f} s", flush=True)

        def plan(req):
            t = time.time()
            r = client.plan(req["start"], req["pregrasp"][:1], obstacles(req))
            return dict(
                ok=r.ok,
                success=list(r.success),
                path=r.path.tolist(),
                seconds=time.time() - t,
                error="",
                rebuilt=r.rebuilt,
                poisoning=r.poisoning,
            )
    else:
        old = OldServer(args.port)
        plan = old.plan
    passes = []
    for order in (range(len(reqs)), reversed(range(len(reqs)))):
        res = {}
        for i in order:
            res[i] = plan(reqs[i][1])
            print(
                f"{reqs[i][0].split('/')[2]}: ok {res[i]['ok']} {res[i]['seconds']:.1f} s rebuilt {res[i].get('rebuilt')} "
                f"poisoning {res[i].get('poisoning')} {res[i]['error']}",
                flush=True,
            )
        passes.append(res)
    rows, agree, diffs = [], 0, []
    for i, (name, _) in enumerate(reqs):
        a, b = passes[0][i], passes[1][i]
        pa, pb = np.asarray(a["path"]), np.asarray(b["path"])
        same_ok = a["success"] == b["success"]
        agree += same_ok
        d = float(np.abs(pa - pb).max()) if a["ok"] and b["ok"] and pa.shape == pb.shape else None
        if d is not None:
            diffs.append(d)
        rows.append(
            dict(
                request=name,
                ok=(a["ok"], b["ok"]),
                seconds=(a["seconds"], b["seconds"]),
                shapes=(pa.shape, pb.shape),
                maxdiff=d,
            )
        )
    both_ok = sum(1 for r in rows if all(r["ok"]))
    secs = [s for r in rows for s in r["seconds"]]
    summary = dict(
        service=args.service,
        requests=len(reqs),
        success_pass1=sum(passes[0][i]["ok"] for i in range(len(reqs))),
        success_pass2=sum(passes[1][i]["ok"] for i in range(len(reqs))),
        success_agree=agree,
        both_ok=both_ok,
        same_length=sum(1 for r in rows if all(r["ok"]) and r["shapes"][0] == r["shapes"][1]),
        maxdiff_median=float(np.median(diffs)) if diffs else None,
        maxdiff_max=float(max(diffs)) if diffs else None,
        within_tol=sum(1 for d in diffs if d <= 0.02),
        rebuilds=[(reqs[i][0], k) for k, res in enumerate(passes) for i in res if res[i].get("rebuilt")],
        poisoning=[(reqs[i][0], k) for k, res in enumerate(passes) for i in res if res[i].get("poisoning")],
        seconds_median=float(np.median(secs)),
        seconds_p90=float(np.percentile(secs, 90)),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, default=str))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
