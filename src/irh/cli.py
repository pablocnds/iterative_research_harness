from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_SOCKET = "/var/lib/iterative-research-harness/daemon.sock"


class API:
    def __init__(self, socket_path: str):
        self.client = httpx.Client(
            transport=httpx.HTTPTransport(uds=socket_path), base_url="http://irh", timeout=60
        )

    def call(self, method: str, path: str, value: object | None = None) -> Any:
        response = self.client.request(method, path, json=value)
        payload = response.json()
        if response.is_error:
            raise RuntimeError(payload.get("message", json.dumps(payload)))
        return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Control the iterative research harness daemon")
    result.add_argument("--socket", default=os.environ.get("IRH_SOCKET", DEFAULT_SOCKET))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("health")
    commands.add_parser("restart-daemon")
    commands.add_parser("list")
    create = commands.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--goal", required=True)
    create.add_argument("--guidelines-file")
    create.add_argument("--budget-hours", type=float)
    tool_selection = create.add_mutually_exclusive_group()
    tool_selection.add_argument("--tool", action="append", help="Override configured default tools")
    tool_selection.add_argument("--no-tools", dest="tool", action="store_const", const=[])
    create.add_argument("--upper-skill", action="append", default=[])
    create.add_argument("--iteration-skill", action="append", default=[])
    status = commands.add_parser("status")
    status.add_argument("project_id")
    run = commands.add_parser("run")
    run.add_argument("project_id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-iterations", type=int)
    run.add_argument("--actor", default="operator")
    pause = commands.add_parser("pause")
    pause.add_argument("project_id")
    pause.add_argument("--actor", default="operator")
    pause.add_argument("--reason", default="Pause requested")
    pause.add_argument("--wait", action="store_true")
    pause.add_argument(
        "--wait-timeout",
        type=float,
        default=0,
        help="Seconds to wait; zero waits indefinitely. Never forces cancellation.",
    )
    steer = commands.add_parser("steer")
    steer.add_argument("project_id")
    steer.add_argument("--message", required=True)
    steer.add_argument("--actor", default="operator")
    steer.add_argument("--target", choices=["next", "upper", "iteration"], default="next")
    steer.add_argument("--idempotency-key")
    messages = commands.add_parser("messages")
    messages.add_argument("project_id")
    publications = commands.add_parser("publications")
    publications.add_argument("project_id")
    publication = commands.add_parser("publication-create")
    publication.add_argument("project_id")
    publication.add_argument("--title", required=True)
    publication.add_argument("--paper", required=True)
    publication.add_argument("--source", action="append", required=True, help="PROJECT_ID:ITERATION")
    publication.add_argument("--supplement", action="append", default=[], help="PACKAGE_NAME=LOCAL_PATH")
    publication.add_argument("--actor", default="supervisor")
    publication.add_argument("--supersedes")
    export = commands.add_parser("publication-export")
    export.add_argument("project_id")
    export.add_argument("publication_id")
    export.add_argument("--output", required=True)
    resource = commands.add_parser("add-resource")
    resource.add_argument("project_id")
    resource.add_argument("source")
    resource.add_argument("--name", required=True)
    events = commands.add_parser("events")
    events.add_argument("project_id")
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--limit", type=int, default=200)
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("project_id")
    return result


def main() -> None:
    args = parser().parse_args()
    api = API(args.socket)
    try:
        if args.command == "health":
            value = api.call("GET", "/v1/health")
        elif args.command == "restart-daemon":
            value = api.call("POST", "/v1/admin/restart", {})
        elif args.command == "list":
            value = api.call("GET", "/v1/projects")
        elif args.command == "create":
            guidelines = Path(args.guidelines_file).read_text() if args.guidelines_file else ""
            value = api.call(
                "POST",
                "/v1/projects",
                {
                    "name": args.name,
                    "goal": args.goal,
                    "guidelines": guidelines,
                    "budget_seconds": args.budget_hours * 3600 if args.budget_hours else None,
                    "tools": args.tool,
                    "upper_skills": args.upper_skill,
                    "iteration_skills": args.iteration_skill,
                },
            )
        elif args.command == "status":
            value = api.call("GET", f"/v1/projects/{args.project_id}")
        elif args.command == "run":
            value = api.call(
                "POST",
                f"/v1/projects/{args.project_id}/run",
                {"resume": args.resume, "max_iterations": args.max_iterations, "actor": args.actor},
            )
        elif args.command == "pause":
            value = api.call(
                "POST", f"/v1/projects/{args.project_id}/pause", {"actor": args.actor, "reason": args.reason}
            )
            deadline = time.monotonic() + args.wait_timeout if args.wait_timeout > 0 else None
            while args.wait and value["state"] == "waiting_to_pause":
                if deadline is not None and time.monotonic() >= deadline:
                    raise RuntimeError("Still waiting to pause; request remains pending")
                time.sleep(0.5)
                value = api.call("GET", f"/v1/projects/{args.project_id}")["control"]
        elif args.command == "steer":
            value = api.call(
                "POST",
                f"/v1/projects/{args.project_id}/messages",
                {
                    "text": args.message,
                    "actor": args.actor,
                    "target": args.target,
                    "idempotency_key": args.idempotency_key,
                },
            )
        elif args.command in ("messages", "publications"):
            value = api.call("GET", f"/v1/projects/{args.project_id}/{args.command}")
        elif args.command == "publication-create":
            sources = []
            for source in args.source:
                project_id, number = source.rsplit(":", 1)
                sources.append({"project_id": project_id, "iteration": int(number)})
            supplements = {}
            for item in args.supplement:
                name, path = item.split("=", 1)
                if name in supplements:
                    raise ValueError(f"Duplicate supplement: {name}")
                supplements[name] = Path(path).read_text()
            value = api.call(
                "POST",
                f"/v1/projects/{args.project_id}/publications",
                {
                    "title": args.title,
                    "paper": Path(args.paper).read_text(),
                    "sources": sources,
                    "supplements": supplements,
                    "actor": args.actor,
                    "supersedes": args.supersedes,
                },
            )
        elif args.command == "publication-export":
            response = api.client.get(
                f"/v1/projects/{args.project_id}/publications/{args.publication_id}/export"
            )
            response.raise_for_status()
            with Path(args.output).open("xb") as stream:
                stream.write(response.content)
            value = {"output": args.output}
        elif args.command == "add-resource":
            value = api.call(
                "POST",
                f"/v1/projects/{args.project_id}/resources",
                {"source": str(Path(args.source).resolve()), "name": args.name},
            )
        elif args.command == "events":
            value = api.call(
                "GET", f"/v1/projects/{args.project_id}/events?after={args.after}&limit={args.limit}"
            )
        elif args.command == "artifacts":
            value = api.call("GET", f"/v1/projects/{args.project_id}/artifacts")
        else:  # pragma: no cover
            raise AssertionError(args.command)
        print(json.dumps(value, indent=2))
    except (OSError, httpx.HTTPError, RuntimeError, ValueError) as error:
        print(f"irh: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
