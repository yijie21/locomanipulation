# Change map

Start at a specified file or known owner for a local edit. Use this router for new
features, unclear ownership or changes across modules. Open the matching domain,
then only its relevant row; follow actual callers and dependencies as needed.

| Change domain | Code, configuration and relevant tests |
|---|---|
| Bottle grasp, pick-and-place, ScaleBFM | [Bottle map](change-map-bottle.md) |
| Bimanual crate, waist / low reach, assisted visual expert | [Crate map](change-map-crate.md) |
| SONIC, stack lifecycle, dataset, training/evaluation, shared config, repository checks | [Runtime map](change-map-runtime.md) |
| Scene assets, cameras, Isaac/MuJoCo rendering and job shutdown | [Rendering map](change-map-rendering.md) |

## Making a change

1. Find the owner and its existing interface. If no domain fits, use the
   [module ownership rules](standards.md#module-ownership). Read standards when
   changing interfaces, dependency direction or configuration formats.
2. Extend that interface; keep helpers with their owner. Configuration belongs in
   the owning config and travels explicitly to its actual consumer. Check delivery
   and precedence, not just whether a field parses.
3. Use relevant tests during development. Regenerate field documentation with
   `make config-doc` when config changes. Choose one final validation entry using
   the [session workflow](../agents/session-workflow.md#verification); use the
   [live-validation requirements](standards.md#checks) for physical behavior changes.
4. Update the affected domain row when an entry, config owner or test changes.
   Add a row for a new capability in that domain; add a domain only for a distinct
   ownership boundary. Maintain the [owning document](../README.md#documentation-ownership)
   alongside the change. Report actual evidence and limits using the
   [completion criteria](../agents/session-workflow.md#completion).

Domain maps are navigation, not experimental certification. Do not copy parameter
defaults into them; use the linked config. Launch commands: [runbook](runbook.md).
