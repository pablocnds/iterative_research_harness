from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from .config import SandboxConfig
from .types import Role


class Sandbox:
    def __init__(self, role: Role, workspace: Path, scratch: Path, project: Path, config: SandboxConfig):
        self.role = role
        self.workspace = workspace.resolve()
        self.scratch = scratch.resolve()
        self.project = project.resolve()
        self.config = config
        for path in (
            self.workspace,
            self.scratch,
            self.scratch / "home",
            self.scratch / "cache",
            self.scratch / "tmp",
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _args(self, command: str) -> list[str]:
        bwrap = str(self.config.bubblewrap)
        args = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-user-try",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        for path in (
            "/etc/ssl",
            "/etc/ca-certificates",
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/passwd",
            "/etc/group",
            "/etc/ld.so.cache",
            "/etc/localtime",
        ):
            if Path(path).exists():
                args.extend(["--ro-bind", path, path])
        if self.role == "upper":
            args.extend(["--unshare-net", "--ro-bind", str(self.project), "/project"])
            cwd = "/scratch"
        else:
            if self.config.iteration_network == "none":
                args.append("--unshare-net")
            args.extend(["--bind", str(self.workspace), "/workspace"])
            cwd = "/workspace"
        args.extend(
            [
                "--bind",
                str(self.scratch),
                "/scratch",
                "--setenv",
                "PATH",
                "/usr/bin:/bin",
                "--setenv",
                "HOME",
                "/scratch/home",
                "--setenv",
                "XDG_CACHE_HOME",
                "/scratch/cache",
                "--setenv",
                "TMPDIR",
                "/scratch/tmp",
                "--setenv",
                "LANG",
                "C.UTF-8",
                "--setenv",
                "LC_ALL",
                "C.UTF-8",
                "--setenv",
                "TZ",
                "UTC",
                "--setenv",
                "PYTHONNOUSERSITE",
                "1",
                "--setenv",
                "GIT_CONFIG_NOSYSTEM",
                "1",
                "--setenv",
                "GIT_TERMINAL_PROMPT",
                "0",
                "--chdir",
                cwd,
                "--",
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                command,
            ]
        )
        return args

    async def run(self, command: str, timeout: float | None = None) -> dict[str, object]:
        process = await asyncio.create_subprocess_exec(
            *self._args(command),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            raise TimeoutError("Sandbox command timed out")
        except asyncio.CancelledError:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            raise
        limit = self.config.command_output_bytes
        clipped = len(output) > limit
        if clipped:
            output = output[-limit:]
        return {
            "exit_code": process.returncode,
            "output": output.decode(errors="replace"),
            "truncated": clipped,
        }
