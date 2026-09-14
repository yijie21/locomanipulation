# Repository maintenance roadmap

Delivery and evidence: [maintenance completion](handoff/maintenance-completion.md).
Earlier research plans remain in [the archive](archive/roadmap_20260912.md); their
claims and priorities require evidence before reuse.

| Work | Status and acceptance evidence |
|---|---|
| Short entry/navigation | Delivered: known local edits start at their owner; map/standards/history loaded when relevant |
| Configuration path | Delivered: explicit bottle strategy and precedence; immutable runtime choices; real client and subprocess delivery tests |
| Runner/environment convergence | Delivered: bottle/crate share stack resolution and stage records; actual-shell admission and failed-start cleanup tests |
| Backend/task separation | Delivered: backend-to-task exception list empty; bottle execution stays with bottle; generic arm dependencies explicit |
| Runtime/acceptance | Delivered: typed completed RoundResult and pure acceptance; task-specific phase inputs remain local |
| Test assets | Delivered: explicit portable geometry fixture; full CPU checks pass without generated outputs/scene |
| Schema/provenance | Delivered: shared stored/model field order; server identity checked; final deployment cwd/model/binary hashes recorded per launch |
| Evidence discipline | Delivered for this maintenance: pinned comparisons/raw logs and claim scopes; historical conclusions/retractions remain unverified |
| Agent navigation | Delivered: two fresh-context probe edits with read logs, patches and passing tests; no inferred token-reduction percentage |
| Runtime validation | Executed and audited: inference equality PASS; crate pilot flow PASS; bottle A/B and A/A gates FAIL, physical equivalence unverified |

The user excluded crate physical success/reproducibility as a blocker on 2026-09-12.
Crate scene redesign remains separate future work; this delivery does not tune its
control or acceptance thresholds. Size advisories locate remaining complex control
routines; they are not a claim that every function has been rewritten.

Current workflow changes and remaining guideline work: [2026-09-14 delivery](handoff/agent-workflow-20260914.md).
