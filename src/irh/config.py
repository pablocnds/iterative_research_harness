from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    api_key_file: Path
    context_window: int = Field(default=131_072, gt=8_192)
    max_output_tokens: int = Field(default=16_384, gt=0)
    request_timeout_seconds: float = Field(default=3_600, gt=0)
    temperature: float = 1.0
    extra_body: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_window(self) -> ModelConfig:
        if self.max_output_tokens >= self.context_window / 2:
            raise ValueError("max_output_tokens must leave at least half the context for input")
        return self


class StdioTransport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class HttpTransport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["http"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    header_files: dict[str, Path] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_headers(self) -> HttpTransport:
        overlap = self.headers.keys() & self.header_files.keys()
        if overlap:
            raise ValueError(f"HTTP header configured twice: {min(overlap)}")
        return self


class MonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    interval_seconds: float = Field(default=5, ge=1)
    blocked_field: str = "incident"
    pause_on_error: bool = True


class ExternalToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    transport: StdioTransport | HttpTransport = Field(discriminator="type")
    roles: dict[Literal["upper", "iteration"], list[str]]
    guidance: dict[Literal["upper", "iteration"], str] = Field(default_factory=dict)
    pause_on_calls: list[str] = Field(default_factory=list)
    monitor: MonitorConfig | None = None
    configuration_id: str


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bubblewrap: Path = Path("/usr/bin/bwrap")
    iteration_network: Literal["host", "none"] = "host"
    command_output_bytes: int = Field(default=262_144, ge=4096)


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_root: Path
    socket_path: Path
    runtime_user: str = "rh-agent"
    operator_group: str = "irh-worker"
    model: ModelConfig
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    tools: list[ExternalToolConfig] = Field(default_factory=list)
    project_seconds: int = Field(default=604_800, gt=0)
    iteration_seconds: int = Field(default=28_800, gt=0)
    max_iteration_seconds: int = Field(default=86_400, gt=0)
    writeup_seconds: int = Field(default=3_600, gt=0)
    upper_seconds: int = Field(default=7_200, gt=0)

    @model_validator(mode="after")
    def validate_config(self) -> HarnessConfig:
        if self.iteration_seconds > self.max_iteration_seconds:
            raise ValueError("iteration_seconds exceeds max_iteration_seconds")
        ids = [tool.id for tool in self.tools]
        if len(ids) != len(set(ids)):
            raise ValueError("external tool ids must be unique")
        return self

    def tool(self, tool_id: str) -> ExternalToolConfig:
        for tool in self.tools:
            if tool.id == tool_id:
                return tool
        raise ValueError(f"Unknown configured tool: {tool_id}")


def default_config_path() -> Path:
    return Path(os.environ.get("IRH_CONFIG", "/home/rh-agent/.config/irh/config.json"))


def load_config(path: Path | str | None = None) -> HarnessConfig:
    config_path = Path(path) if path else default_config_path()
    raw = json.loads(config_path.read_text())
    config = HarnessConfig.model_validate(raw)
    config.data_root = config.data_root.resolve()
    config.socket_path = config.socket_path.resolve()
    config.model.api_key_file = config.model.api_key_file.resolve()
    for tool in config.tools:
        if isinstance(tool.transport, HttpTransport):
            tool.transport.header_files = {
                name: secret.resolve() for name, secret in tool.transport.header_files.items()
            }
    return config


def read_secret(path: Path) -> str:
    stat = path.stat()
    # The operator group may update credentials without sudo. Other users must
    # never receive access.
    if not path.is_file() or stat.st_mode & 0o037:
        raise ValueError(f"Secret must be a private regular file (mode 0600 or 0640): {path}")
    value = path.read_text().strip()
    if not value or "\n" in value:
        raise ValueError(f"Invalid secret file: {path}")
    return value
