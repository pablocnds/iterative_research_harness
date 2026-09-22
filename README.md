# Iterative Research Harness

A small persistent planning → experiment → paper review research loop. One Python
daemon owns project lifecycle, model sessions, sandboxes, evidence, and all
mutating control. The CLI is only a client of its HTTP-over-Unix-socket API.

External agent capabilities use MCP. At project creation the operator chooses
configured, trusted servers; the daemon exposes each allowed MCP tool directly
to the appropriate role. There is no plugin marketplace, automatic matching,
generic tool-dispatch wrapper, GUI, or observer agent.

## Design

- The daemon and model-driven processes run as the restricted `rh-agent` user.
  `omega` is an operator through the `irh-worker` group and the daemon socket.
- SQLite is the only authority. Events are append-only; status and bounded
  event queries do not rebuild a JSON ledger.
- Upper and iteration sessions persist separately outside agent-visible project
  files. A timed-out iteration keeps its session and workspace for a bounded
  paper-only write-up. A later revision is a new iteration with selected inputs.
- Selected resources use reflink copies when supported. Completed workspaces
  are archived; only paper.md and selected, SHA-256-registered supplementary
  files enter the upper agent's publication view.
- Base prompts contain only cross-domain research behavior. Operator-selected
  prompts, skills, and external-tool configuration are frozen for execution at
  project creation; the public audit snapshot remains separate.
- MCP arguments and results retain their declared schemas. Model-correctable
  errors remain tool results; configured incidents pause the whole daemon run.

See [architecture](docs/architecture.md), [tool integration](docs/tools.md), and
[operations](docs/operations.md).

## Development

```sh
uv sync
uv run ruff check .
uv run pytest -q
```

The tests exercise the lifecycle, real Bubblewrap containment, the API, direct
MCP discovery/calls, Runpod argument binding, manual intervention, workspaces,
and artifact hashing without making paid provider calls.
