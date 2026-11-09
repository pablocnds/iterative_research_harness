# Architecture

## One authoritative daemon

`irh-daemon` is the sole writer and lifecycle authority. It hosts a local HTTP
API on a Unix socket, runs at most one research project at a time, and owns the
planning → execution → paper review loop. CLI commands never open SQLite directly.

SQLite stores current projects and iterations, explicit attempt and execution
records, a durable message inbox, pause provenance, accepted phase results,
artifacts, notes, and an append-only event stream. Event reads are bounded by a
cursor and limit. Active-time accounting is checkpointed every 30 seconds and
again at phase exit; no full ledger projection is rewritten on that path. The
phase clock can pause through trusted MCP progress markers while an agent is
blocked by unmetered external infrastructure.

Each project has a public audit snapshot and a separate private execution
snapshot. The latter freezes its prompts, selected skill text, and complete
external-tool configuration so a paused project resumes with the same inputs
after daemon or configuration updates. Credential file contents remain external
and are read when the tool connection opens.

The project directory contains operator resources, archived workspaces, and a
separate publication view. Canonical model sessions are under the
controller-private tree and are not mounted into agent sandboxes. Runtime
workspaces are moved rather than copied when an iteration finishes.

## Trust boundaries

The daemon, MCP adapters, and configured MCP servers are trusted code. The
model, retrieved text, and experiment code are not.

The daemon runs as the dedicated `rh-agent` Unix account. `omega` receives
operator access through the `irh-worker` group, not by running the daemon as the
operator. Initial planning can inspect the project; paper reviews instead mount
only `publications/` read-only, never raw iteration workspaces. The reviewer has
read, note, time, and decision tools, not a shell or external research tools.
Iteration commands see only their writable workspace/scratch and an explicit
system toolchain.

Iteration networking is configurable. `none` removes networking. `host` permits
ordinary network access while retaining filesystem, process, device, capability,
and Unix-socket containment; it does not filter private IP ranges. This is an
explicit current limitation, not a claim equivalent to a network firewall.

## Agent loop

Both roles receive a small research-wide prompt: distinguish evidence from
inference, identify units and uncertainty, and run the smallest useful test
before expensive work. The iteration agent owns execution and writes a concise
`paper.md` with Methodology, Results, and Conclusions, including honest partial
or failed results. `submit_results` names only its outcome and selected
supplementary files; the paper is the substantive report. The upper agent
reviews that package, plans the next iteration, or requests a targeted revision
with `decide`.

Review commits its conclusion and proposed next action before dispatching
another iteration. The `ready` phase holds that proposal. A bounded execution
ends there after its allowed iteration cycles have been reviewed, leaving the
project available for another supervisor-authorized run. Revisions count as
new iterations. Pauses and resumes preserve the execution ID and its allowance.

Steering and periodic time reminders enter a shared durable inbox. The agent
appends them at complete conversation boundaries before token counting. Receipt
IDs stored outside compacted conversation history reconcile delivery after a
crash between the session save and SQLite acknowledgement. Cooperative pause
uses its own signal: finish and save the current operation, explicitly close
unstarted batch calls, commit an accepted terminal result, and acknowledge the
pause before further execution. Cancellation remains a separate interruption
path for hard deadlines and failures.

Synthesis papers are immutable publication artifacts stored in `syntheses/`,
independent of iteration ownership. They reuse the artifact registry and file
hashing/staging machinery. The supervisor supplies their content and source
references and owns research-line organization. Existing `publications/`
directories remain the restricted per-iteration review view.

Sessions are stored as JSON. Planning and each upper review have separate
sessions. A timed-out iteration continues in its existing session, preserving
its working context and original system prompt, but only bounded file listing
and read-only file access, `write_paper`,
`time_remaining`, and `submit_results` remain model-visible. Its configurable
write-up grace defaults to one hour and is accounted separately from execution
time; if exhausted without a paper, the project pauses for the operator. Review
begins with a short index of the paper and supplementary package. Before each
model request, the harness asks the model's
`/v1/chat/completions/input_tokens` endpoint for the complete prompt size. The
existing context threshold is unchanged; when reached, the model creates a
compact continuation summary while retaining recent turns. A compaction boundary
is moved to the assistant call that began a tool batch, so tool results are never
retained without their originating call. Tool results are
included only once, and large ones are saved to the agent's readable files with
a bounded preview and path in the conversation. Tool calls and per-request model
timing, token usage, and prompt-cache usage are recorded as append-only SQLite
events. A non-pausing warning event identifies a warm continuation that does not
substantially reuse the preceding prompt. Project status exposes aggregate token
usage without exposing the private session transcript. Reported token rates are effective
tokens per full request second, including provider queueing and prompt handling;
they are not a claim about raw decoder throughput.

An unclean daemon loss moves an executing attempt without an accepted submission
to the same paper-only write-up path and pauses for explicit operator resume.
Accepted submissions are journaled before returning their terminal tool result,
so recovery can finalize them without another model call. Accepted papers enter a
durable internal finalization phase; archiving, publication, and artifact
registration are idempotently completed after a restart. A controlled pause
preserves the active workspace and session. Phase timeout or cancellation also
cancels any in-flight model request. Restarting a paused project always requires
an explicit resume request.
