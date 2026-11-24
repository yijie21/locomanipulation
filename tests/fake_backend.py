"""A fake second backend for the loop's backend seam (scripts/rsi/backend.py, interface of section B): it records its
calls and writes nothing but what the interface asks. Registered as RSI_BACKEND_PATH=fake=fake_backend."""

from pathlib import Path

CALLS: list = []
HERE = Path(__file__).resolve().parent
CAPS = {"claims": False, "design_probes": False, "diag_redraw": False, "sentinel_exact": False, "min_net": 2}


def load(name):
    CALLS.append(("load", name))
    return dict(name=name, backend="fake", tasks=["XTask"], rows={"XTask": dict(id="XTask", goal=["g"])}, pools={})


def draw(fam, pool, out, n=None, tasks=None):
    CALLS.append(("draw", pool, n, tasks))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    k = n or 2
    for i in range(k):
        (out / f"XTask_s{i}.json").write_text('{"task": "XTask", "episode_id": "XTask/Q/%d"}' % i)
    return dict(written=k, skipped=[], unavailable={}, next_seed=k)


def run_jobs(jobs, family, pool, procs, log, gpus, stop=None, where="probe"):
    CALLS.append(("run_jobs", family, where, len(jobs)))
    return {k: [] for k in range(len(jobs))}


def signature(family, where):
    CALLS.append(("signature", family, where))
    return f"fake-{where}"


def cards(rounds, out):
    CALLS.append(("cards", len(rounds)))
    out = Path(out)
    (out / "cards.md").write_text("# fake cards\n")
    return out / "cards.md"


def layout(family):
    return dict(skills=HERE / "fake_skills", rules=HERE / "fake_rules.md", prompts=HERE / "fake_prompts", plan=None,
                leak=[r"released_episodes"], class_suggest={}, caps=dict(CAPS))  # fmt: skip


def sandbox(dest, family):
    CALLS.append(("sandbox", str(dest), family if isinstance(family, str) else family.get("name")))
