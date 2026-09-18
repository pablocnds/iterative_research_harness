# Architecture

## One authoritative daemon

`irh-daemon` is the sole writer and lifecycle authority. It hosts a local HTTP
API on a Unix socket, runs at most one research project at a time, and owns the
upper → iteration → review loop. CLI commands never open SQLite directly.

SQLite stores current projects and iterations, explicit attempt records,
artifacts, notes, and an append-only event stream. Event reads are bounded by a
cursor and limit. Active-time accounting is checkpointed every 30 seconds and
again at phase exit; no full ledger projection is rewritten on that path.

The project directory contains operator resources, iteration definitions,
archived workspaces, snapshots, and artifact evidence. Canonical model sessions
are under the controller-private tree and are not mounted into agent sandboxes.
Runtime workspaces are moved rather than copied when an iteration finishes.

## Trust boundaries

The daemon, MCP adapters, and configured MCP servers are trusted code. The
model, retrieved text, and experiment code are not.

The daemon runs as the dedicated `rh-agent` Unix account. `omega` receives
operator access through the `irh-worker` group, not by running the daemon as the
operator. Upper-agent commands run in Bubblewrap with a read-only project mount,
a writable scratch directory, no host home, and no network namespace. Iteration
commands see only their writable workspace/scratch and an explicit system
toolchain.

Iteration networking is configurable. `none` removes networking. `host` permits
ordinary network access while retaining filesystem, process, device, capability,
and Unix-socket containment; it does not filter private IP ranges. This is an
explicit current limitation, not a claim equivalent to a network firewall.

## Agent loop

Both roles receive a small research-wide prompt: distinguish evidence from
inference, identify units and uncertainty, and run the smallest useful test
before expensive work. The upper role reviews actual evidence and uses `decide`.
The iteration role executes one definition and uses `submit_results`.

Sessions are stored as JSON and resumed for the upper agent and a controlled
iteration continuation. When estimated context approaches the configured
window, the model creates a compact continuation summary while retaining recent
turns. Tool calls and per-request model timing/token usage are recorded as
append-only SQLite events. Project status exposes aggregate token usage without
exposing the private session transcript. Reported token rates are effective
tokens per full request second, including provider queueing and prompt handling;
they are not a claim about raw decoder throughput.

An unclean daemon loss turns an executing attempt into an explicit interrupted
attempt and pauses for upper review. A controlled pause preserves the active
workspace and session. Restarting a paused project always requires an explicit
resume request.
