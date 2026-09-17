# External tools

## MCP boundary

Global configuration defines trusted MCP servers. A project explicitly selects
server IDs when it is created. For each phase the daemon connects to those
servers, lists their tools, filters them through the configured role allowlist,
and gives the model first-class names such as `runpod__wait`. The original input
schema, description, structured result, and error status are preserved.

The small harness layer around MCP contains only:

- a stable server ID and configuration/version identifier;
- stdio command or Streamable HTTP URL;
- role-specific tool allowlists and concise optional guidance;
- namespacing to prevent collisions;
- optional `pause_on_calls` for a successful tool that requests intervention;
- an optional hidden monitor tool whose non-null blocked field pauses the run.

MCP server instructions do not silently become general agent instructions. Use
the concise configured guidance when a model needs usage advice. Configuration,
resolved server identity, exposed names, and the exact project prompt/skill text
are recorded for later audit. Secrets do not belong in project snapshots.

## Manual intervention

`~/llms/manual_intervention_tool` is a stateless stdio MCP server exposing only
`request_manual_intervention(reason, requested_action)`. It sends no alert and
does not control the daemon. The harness configuration lists that native tool in
`pause_on_calls`; after a successful call the harness records an
`operator_intervention_required` event and pauses the project.

## Runpod

`irh-runpod-mcp` is a thin MCP facade over the standalone tool's private Unix
socket. It exposes the standalone operations directly: `run`, `status`, `wait`,
`cancel`, and `replace`. Project, session, and Pod identifiers are not model
arguments. One harness project uses one durable Runpod session and therefore the
same retained Pod semantics. Job IDs remain explicit for stale-call protection
and auditability.

The adapter freezes named inputs from the current iteration workspace into the
standalone service's controller-owned input root before submission. Verified
outputs collected by that service are reflinked into `gpu-outputs/` in the
iteration workspace, where they can be registered as ordinary harness evidence.
Provider credentials, provisioning, SSH, billing, migration, transfer retries,
and incident resolution remain exclusively in `runpod_agent_tool`.

The hidden `host_status` MCP tool is polled by the daemon. A durable Runpod
incident or loss of its required monitor pauses the full research workflow; it
is never presented as an ordinary agent-fixable failure. Incident resolution is
not in the role allowlist and remains an operator action against the standalone
service.

The example adapter configuration is in `config/runpod-tool.json`. Its
`input-root` and `output-root` must exactly match the standalone service config
and be writable/readable respectively by `rh-agent` (normally via
`irh-worker`). Add that object to the main configuration's `tools` array before
creating a project with `--tool runpod`.

