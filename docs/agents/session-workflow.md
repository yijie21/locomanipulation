# Session workflow (token discipline)

Cost is context length x turns. The conversation is the most expensive place to keep
anything; files are free (loaded once, cached). So: keep the conversation short-lived,
keep the state in files, and start fresh often.

## Where each kind of context lives

| kind | file | who updates |
|---|---|---|
| rules, conventions, pitfalls | `CLAUDE.md`, `docs/reference.md` | rarely, when a rule changes |
| durable non-obvious facts | `~/.claude/projects/.../memory/*.md` (auto-loaded) | the agent, as it learns them |
| what is happening NOW, next step, blockers | `docs/STATE.md` (one screen) | the agent at every checkpoint |
| decisions handed to another session | `docs/handoff/*.md` + SendMessage | the agent, once per handoff |
| results | `outputs/**` + manifest/run.log | the runs themselves |

## One session = one task

1. **Start**: `claude`, then the first message is `读 docs/STATE.md,继续 <task>`. Do not paste
   history; do not ask it to summarise the project.
2. **Work**: bulk reads/searches/analysis go to subagents (`explorer`, `mechanic`, `implementer`,
   or `Agent`), which return conclusions only. Tool output > 4 KB goes to a file, read bounded
   excerpts. Long runs go to `nohup` + a background wait, never a live tail in the conversation.
3. **Checkpoint** (each milestone, and always before stopping): update `docs/STATE.md`
   (what changed, what is running, next step, open questions), write new memories, commit if
   the user wants commits.
4. **End**: `/clear` (not `/compact`). The next session starts from step 1.

## When the context still grows

- Run `/context` when it feels slow: if MCP tool definitions or CLAUDE.md dominate, disable
  unused MCP servers in settings.
- If a task must outlive one context window, `/compact` at ~50% with an instruction
  (`/compact 只保留结论、文件路径和下一步`), never at 95%.
- Parallel sessions (main line vs branch) each pay their own context; batch cross-session
  questions into one message.

## STATE.md template

```
# STATE (updated <date> by <session role>)
## Now
- <one line per active thread; what is running where, on which GPU/instance>
## Next
- <ordered, 1-3 items>
## Blocked / waiting on
- <who/what>
## Parked (do not forget)
- <link to the doc that holds the plan>
## Recent results (last 3)
- <date> <run dir> <number>
```
