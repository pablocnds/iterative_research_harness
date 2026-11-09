# Operating the harness from a supervisor

The harness owns execution, checkpoints, budgets, and evidence. The supervisor
chooses research directions, sends guidance, decides when another execution is
useful, and writes synthesis papers. Research lines and their organization stay
in the supervisor. A publication here is an archived research artifact; creating
one does not send it to any external service.

All examples use `PROJECT_ID` for an existing project. Commands print JSON.
The default socket is `/var/lib/iterative-research-harness/daemon.sock`; override
it with `irh --socket PATH ...` or `IRH_SOCKET`. The HTTP examples below use paths
relative to that socket's `/v1` API.

## 1. Steering

Queue guidance without interrupting the current operation:

```sh
irh steer PROJECT_ID --actor supervisor \
  --message "Validate the baseline before adding another model." \
  --idempotency-key baseline-guidance-1
irh messages PROJECT_ID
```

The daemon stores the message immediately. At the next safe turn boundary, after
the preceding tool batch is saved and before the next model request, it appends
an attributed user message to the conversation. The original system prompt stays
unchanged. A paused or inactive project retains messages until execution resumes.
Messages cannot be queued for a completed project.

`--target next` is the default and delivers to whichever agent next reaches a
turn boundary. Use `--target upper` for planning/review guidance or
`--target iteration` for experiment execution. A role-specific message stays
pending until that role runs. Each message is delivered to one session; this is
not a broadcast to every future iteration.

`messages` includes `id`, `target`, `actor`, `created_at`, `delivered_at`, and
`session`. Delivery means the message is in the persisted conversation, not a
claim that its instructions were followed. Delivery receipts survive session
compaction and prevent duplicate appends after a crash between saving a session
and acknowledging delivery in SQLite.

API:

```http
POST /v1/projects/PROJECT_ID/messages
Content-Type: application/json

{"text":"Validate the baseline first.","actor":"supervisor","target":"upper","idempotency_key":"baseline-guidance-1"}
```

`GET` on the same path lists messages. Retrying a message with the same
project-scoped idempotency key returns the original message. Reusing that key
with different text, actor, or target returns HTTP 409. Text is limited to 32,000
characters. No configuration is required.

## 2. Cooperative pause

```sh
irh pause PROJECT_ID --actor supervisor --reason "Inspect intermediate evidence"
irh status PROJECT_ID
```

Pause requests return promptly. Status exposes `control.state` as
`waiting_to_pause` until the current model request or foreground tool operation
finishes and its response is saved. `control.current_operation` identifies the
recorded model request or tool call when one is in progress.

The harness then starts no further tool operations. Unstarted calls in an
existing batch receive explicit "not executed" results. If a terminal tool
already accepted a paper or decision, that result is committed before pausing.
The active workspace and conversation remain resumable.

Wait for acknowledgement from the CLI:

```sh
irh pause PROJECT_ID --actor supervisor --reason "Inspect evidence" \
  --wait --wait-timeout 120
irh run PROJECT_ID --resume
```

Omit `--wait-timeout` to wait indefinitely. A client wait timeout exits with an
error but leaves the request pending. It never cancels the operation. Repeating
a pause is idempotent and preserves the original request and resume phase.

`control.pause_request` records the requester, reason, request time, and
acknowledgement time. `actor` is caller-supplied attribution under the existing
Unix-socket access control, rather than authenticated individual identity.
Harness-triggered pauses use `harness`; a controlled daemon shutdown uses
`daemon`. If the project completes while pause is pending, completion is
preserved and the request is acknowledged.

API:

```http
POST /v1/projects/PROJECT_ID/pause
Content-Type: application/json

{"actor":"supervisor","reason":"Inspect intermediate evidence"}
```

Poll `GET /v1/projects/PROJECT_ID` until `control.state` stops being
`waiting_to_pause`; check `project.phase` to distinguish pause from completion.
There is no automatic 30-second cancellation fallback. Existing operation
timeouts, active-time limits, infrastructure incidents, process termination,
and the systemd shutdown deadline remain separate interruption mechanisms.
Cooperative pause does not cancel detached remote jobs. No configuration is
required; pause before updating or restarting the deployed daemon.

## 3. Bounded executions

Run exactly one iteration cycle, including its paper, finalization, and review:

```sh
irh run PROJECT_ID --max-iterations 1 --actor supervisor
irh status PROJECT_ID
```

An execution has a durable run ID returned by `run`. Status includes its
`max_iterations`, `iterations_completed`, state, elapsed running time, and
completion reason under `control.run`.

When the allowance is used, the run ends with state `completed` and
`completion_reason: iteration_limit_reached`. The project remains in phase
`ready`, and `control.pending_decision` contains the reviewed proposal. The next
iteration is not created or staged. If the reviewer concludes the project,
the project becomes `complete` and the run ends with `project_complete` instead.

A revision counts as a new iteration. Continuing a paused iteration does not
consume another slot. If an execution begins with an unfinished iteration or
uncommitted review, finishing that cycle counts toward its allowance. A timeout
still follows the existing paper-only write-up and review workflow; exhausted
write-up grace pauses the execution rather than skipping review.

After inspecting a completed bounded run, start another execution:

```sh
irh run PROJECT_ID --max-iterations 1 --actor supervisor
```

This creates a new run ID and consumes the saved proposal. To change that
proposal first, queue steering for `upper` (or `next`) before starting the run;
the upper agent revisits it before dispatch. Iteration-specific guidance is
delivered to the iteration agent instead.

`irh run PROJECT_ID --resume` continues a cooperatively paused or recovered run
with the same ID and allowance. Omit `--max-iterations` on resume, or repeat the
same limit. Changing an existing run's limit on resume is rejected. A completed
bounded run uses a new `run` command without `--resume`. Omitting the limit on a
new run retains the original continuous-loop behavior.

API:

```http
POST /v1/projects/PROJECT_ID/run
Content-Type: application/json

{"max_iterations":1,"actor":"supervisor"}
```

Resume uses `{"resume":true}`. Iteration limits must be positive integers.
No global configuration is required. Project and phase time budgets continue
to apply independently.

## 4. Time reminders

Reminders are enabled automatically every 30 minutes of execution wall time.
No CLI operation is needed. They include elapsed running time, charged active
time, remaining project budget, iteration, phase, and the last recorded tool
completion when available. They do not infer milestone completion or invent
progress percentages.

Wall time includes infrastructure waits that are excluded from the active-time
budget, and excludes periods when the execution is paused. The run's clock and
next reminder threshold persist across phases and resumes. Checkpoints occur
at most 30 seconds apart and again at controlled exit; an unclean process loss
can lose the final uncheckpointed wall-time interval.

Reminders use the same inbox and conversation boundary as steering. During a
long operation they remain pending. Multiple overdue reminders coalesce into
one current reminder, so the agent does not receive a backlog. Inspect them
with `irh messages PROJECT_ID` and the normal event feed.

To change the cadence, set this top-level daemon configuration field:

```json
{"reminder_seconds": 1800}
```

Set it to `0` to disable future reminders. Existing queued messages remain.
The setting takes effect after a daemon restart and is global, not frozen in
the project's prompt snapshot. Reminders do not make additional model calls.

## 5. Publication artifacts

The supervisor writes a self-contained Markdown synthesis using selected
iterations, then submits it to the harness:

```sh
irh publication-create PROJECT_ID \
  --title "Baseline findings across three iterations" \
  --paper /path/to/synthesis.md \
  --source PROJECT_ID:1 --source PROJECT_ID:2 --source PROJECT_ID:3 \
  --supplement tables/results.csv=/path/to/results.csv \
  --actor supervisor
irh publications PROJECT_ID
irh artifacts PROJECT_ID
irh publication-export PROJECT_ID PUBLICATION_ID --output findings.zip
```

Sources may refer to finalized iterations in other projects as well. The
owning project can already be complete; publication creation does not resume
experimentation. The daemon validates source existence and finalization, then
stores the paper, selected UTF-8 text supplements, and a manifest with source
artifact hashes, title, creator, timestamps, and file hashes. It uses the same
artifact registry and staging/hash conventions as iteration reports, with
`owner_kind: publication` and a publication ID instead of a dummy iteration.

The ZIP contains `paper.md`, `manifest.json`, and `supplements/`. It includes
only the publication's own files; source workspaces and private conversations
are not needed or copied automatically. Put essential evidence, tables,
methods, limitations, negative results, and reusable conclusions in the paper
or its supplements. The supervisor is responsible for substantive synthesis;
the harness does not judge the paper's quality or generate it automatically.
This initial submission API accepts text packages: paper up to 4 MiB, at most
64 supplements, and at most 8 MiB total UTF-8 content.

Publications are immutable through the API. To revise one, create another
with `--supersedes PUBLICATION_ID`; the previous package remains available.
Exports refuse to overwrite an existing destination file.

API submission:

```http
POST /v1/projects/PROJECT_ID/publications
Content-Type: application/json

{"title":"Combined findings","paper":"# Findings\n...","actor":"supervisor","sources":[{"project_id":"PROJECT_ID","iteration":1},{"project_id":"OTHER_PROJECT_ID","iteration":2}],"supplements":{"results.csv":"method,result\nbaseline,partial\n"}}
```

Retrieval endpoints:

| Operation | Path |
| --- | --- |
| List manifests | `GET /v1/projects/{project}/publications` |
| Read one manifest | `GET /v1/projects/{project}/publications/{publication}` |
| Export ZIP | `GET /v1/projects/{project}/publications/{publication}/export` |
| Read a package file | `GET /v1/projects/{project}/publications/{publication}/files/paper.md` |
| Read a supplement | `GET /v1/projects/{project}/publications/{publication}/files/supplements/results.csv` |

No configuration or external publishing account is required. Research-line
membership and cross-line discovery belong to the supervisor. Existing
`publications/0001/...` directories remain iteration review packets for
compatibility; synthesis packages live under `syntheses/{publication_id}/`.

## Updating an existing installation

Pause the active project and wait for acknowledgement before using the normal
`deploy/update.sh` workflow. SQLite schema upgrades run automatically when the
new daemon opens the store; existing projects, iteration artifacts, and saved
sessions are preserved. Older session JSON files need no manual edits.

The artifact schema now supports publication ownership and nullable iteration
numbers. API clients should inspect `owner_kind` and `owner_id`; iteration
artifacts retain their existing `iteration` and relative `path` values. An older
daemon does not understand the new schema, so reverting executable code alone
is not a supported database downgrade.

The local controller task manager is now named `ExecutionManager`; the old
`Supervisor` import remains as a compatibility alias. Accepted terminal results
are journaled in SQLite before lifecycle transitions, and review commits its
proposal before a separate dispatch checkpoint. A recovered accepted paper is
finalized rather than asking the agent to submit it again.
