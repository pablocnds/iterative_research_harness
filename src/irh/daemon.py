from __future__ import annotations

import argparse
import asyncio
import grp
import os
import pwd
import socket

import uvicorn

from .api import create_app
from .config import load_config
from .controller import Supervisor
from .store import Store
from .workspace import WorkspaceManager


async def serve(config_path: str | None) -> None:
    config = load_config(config_path)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    if current_user != config.runtime_user:
        raise RuntimeError(f"Daemon must run as {config.runtime_user}, not {current_user}")
    os.umask(0o007)
    config.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o770)
    config.socket_path.unlink(missing_ok=True)
    store = Store(config.data_root)
    recovered = store.recover_after_restart()
    workspaces = WorkspaceManager(store)
    for project_id in recovered:
        project = store.get_project(project_id)
        if not project["current_iteration"]:
            continue
        iteration = store.get_iteration(project_id, project["current_iteration"])
        runtime_workspace = store.runtime_dir(project_id) / "iteration" / "workspace"
        if iteration["status"] == "interrupted" and runtime_workspace.is_dir():
            archive = workspaces.archive(project_id, project["current_iteration"])
            store.event(
                project_id,
                "interrupted_workspace_archived",
                {"iteration": project["current_iteration"], "path": str(archive)},
            )
    supervisor = Supervisor(config, store)
    restart_requested = False
    server: uvicorn.Server | None = None

    def request_restart() -> None:
        nonlocal restart_requested
        restart_requested = True

        async def stop_after_response() -> None:
            await asyncio.sleep(0.25)
            assert server is not None
            server.should_exit = True

        asyncio.create_task(stop_after_response())

    app = create_app(config, store, supervisor, request_restart)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(config.socket_path))
    listener.listen(128)
    listener.setblocking(False)
    group_id = grp.getgrnam(config.operator_group).gr_gid
    os.chown(config.socket_path, -1, group_id)
    os.chmod(config.socket_path, 0o660)
    server = uvicorn.Server(uvicorn.Config(app, log_level="info", access_log=False))
    try:
        await server.serve(sockets=[listener])
    finally:
        await supervisor.close()
        store.close()
        listener.close()
        config.socket_path.unlink(missing_ok=True)
    if restart_requested:
        raise SystemExit(75)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the iterative research harness daemon")
    parser.add_argument("--config")
    args = parser.parse_args()
    try:
        asyncio.run(serve(args.config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
