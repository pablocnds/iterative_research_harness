from __future__ import annotations

from pathlib import Path

import pytest

from irh.config import HarnessConfig


@pytest.fixture
def config(tmp_path: Path) -> HarnessConfig:
    secret = tmp_path / "model.key"
    secret.write_text("test-key")
    secret.chmod(0o600)
    return HarnessConfig.model_validate(
        {
            "data_root": str(tmp_path / "data"),
            "socket_path": str(tmp_path / "daemon.sock"),
            "runtime_user": "rh-agent",
            "operator_group": "irh-worker",
            "model": {
                "base_url": "http://127.0.0.1:1",
                "model": "test-model",
                "api_key_file": str(secret),
                "context_window": 32768,
                "max_output_tokens": 4096,
            },
            "iteration_seconds": 60,
            "max_iteration_seconds": 120,
            "upper_seconds": 60,
            "project_seconds": 600,
        }
    )
