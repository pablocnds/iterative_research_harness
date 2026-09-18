# Installation and operation

## One-time privileged bootstrap

The root-only bootstrap removes the known v0.1 installation, installs immutable
release trees under `/srv`, creates the new data directory, installs the systemd
unit, and starts the daemon as `rh-agent`. It does not delete the v0.1 source
checkout under `~/llms`.

First copy `config/example.json` to a private local file and set the actual model
endpoint and model ID. Then run:

```sh
sudo /home/omega/llms/iterative_research_harness/deploy/bootstrap-root.sh \
  --config /absolute/path/to/config.local.json \
  --model-key /absolute/path/to/model-api-key \
  --exa-key /absolute/path/to/exa-api-key
```

The script removes only these old deployment paths:

- `/opt/iterative-research-harness`
- `/var/lib/iterative-research-harness`
- `/home/rh-agent/.config/irh`
- the old `irh@` and GPU-watchdog units
- `/usr/local/bin/irh` and its old narrow sudoers file

The new service is `irh-daemon.service`. Its Unix socket and data files are
accessible to `omega` through `irh-worker`; model-driven sandboxes cannot see
them.

## Recurring updates without sudo

Pause any active project, then run:

```sh
/home/omega/llms/iterative_research_harness/deploy/update.sh
/home/omega/llms/manual_intervention_tool/deploy/update.sh
```

Harness updates build an immutable release, atomically switch `/srv/irh/current`,
and ask the old daemon to exit with its supervised-restart status. systemd then
starts the new release. The API rejects restart while a project is active.
Manual-tool updates take effect the next time an MCP process is opened. A broken
service that cannot answer its control API still needs the operator to ask an
administrator to restart or repair the system unit.

## CLI and API

The CLI defaults to the group-protected Unix socket:

```sh
irh health
irh create --name "Study" --goal "Evaluate the supplied hypothesis" \
  --tool manual --iteration-skill /path/to/methods.md
irh add-resource PROJECT_ID /path/to/data --name dataset
irh run PROJECT_ID
irh status PROJECT_ID
irh events PROJECT_ID --after 0 --limit 200
irh artifacts PROJECT_ID
irh pause PROJECT_ID
irh run PROJECT_ID --resume
```

The HTTP API uses the same operations under `/v1`. It is intentionally local in
this version; Unix-socket permissions provide access control. There is no GUI,
assistant agent, network listener, or independent mutation path.

Project status includes aggregate model-token usage and current/latest request
activity. The bounded event feed includes `model_request_started` and
`model_request_finished` records with request latency and effective token rates,
so a separate local observer can poll by cursor without reading private session
files or the SQLite database directly.

Status and logs need no sudo:

```sh
irh health
systemctl status irh-daemon.service
journalctl -u irh-daemon.service
```

Paths supplied to `add-resource` and project skill options must be readable by
`rh-agent`. Place operator-provided files in a shared group-readable location
when the operator home directory itself is not traversable by that account.
