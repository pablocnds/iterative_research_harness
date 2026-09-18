from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from httpx import Timeout
from jsonschema import validate
from openai import APIError, AsyncOpenAI

from .config import HarnessConfig, read_secret
from .errors import InterventionRequired
from .store import Store
from .tooling import AgentTool
from .types import Role


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o770)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")))
    os.replace(temporary, path)


class AgentSession:
    def __init__(
        self,
        config: HarnessConfig,
        store: Store,
        project_id: str,
        role: Role,
        session_path: Path,
        system_prompt: str,
        tools: list[AgentTool],
        abort: asyncio.Event,
        terminal: Callable[[], bool],
    ):
        self.config = config
        self.store = store
        self.project_id = project_id
        self.role = role
        self.session_path = session_path
        self.system_prompt = system_prompt
        self.tools = {tool.name: tool for tool in tools}
        self.abort = abort
        self.terminal = terminal
        self.client = AsyncOpenAI(
            base_url=config.model.base_url.rstrip("/") + "/v1",
            api_key=read_secret(config.model.api_key_file),
            max_retries=0,
            timeout=Timeout(
                config.model.request_timeout_seconds,
                connect=10,
                write=60,
                pool=60,
            ),
        )
        self.messages: list[dict[str, Any]] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "model": config.model.model, "compactions": 0}
        if session_path.exists():
            saved = json.loads(session_path.read_text())
            self.messages = saved.get("messages", [])
            self.usage.update(saved.get("usage", {}))

    def save(self) -> None:
        atomic_json(
            self.session_path,
            {"system_prompt": self.system_prompt, "messages": self.messages, "usage": self.usage},
        )

    async def _await_or_abort(self, awaitable: Any) -> Any:
        work = asyncio.create_task(awaitable)
        stopped = asyncio.create_task(self.abort.wait())
        done, _ = await asyncio.wait((work, stopped), return_when=asyncio.FIRST_COMPLETED)
        if stopped in done and self.abort.is_set():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            raise asyncio.CancelledError("Agent phase was stopped")
        stopped.cancel()
        await asyncio.gather(stopped, return_exceptions=True)
        return await work

    async def _complete(self, request_kind: str, **request: Any) -> Any:
        request_id = uuid.uuid4().hex
        started = asyncio.get_running_loop().time()
        self.store.event(
            self.project_id,
            "model_request_started",
            {
                "role": self.role,
                "request_id": request_id,
                "request_kind": request_kind,
                "model": self.config.model.model,
            },
        )
        try:
            response = await self._await_or_abort(self.client.chat.completions.create(**request))
        except APIError as error:
            self.store.event(
                self.project_id,
                "model_request_finished",
                {
                    "role": self.role,
                    "request_id": request_id,
                    "request_kind": request_kind,
                    "model": self.config.model.model,
                    "ok": False,
                    "elapsed_seconds": asyncio.get_running_loop().time() - started,
                    "error_type": type(error).__name__,
                },
            )
            raise InterventionRequired(
                "model",
                {"type": type(error).__name__, "error": str(error)},
            ) from error
        except BaseException as error:
            self.store.event(
                self.project_id,
                "model_request_finished",
                {
                    "role": self.role,
                    "request_id": request_id,
                    "request_kind": request_kind,
                    "model": self.config.model.model,
                    "ok": False,
                    "elapsed_seconds": asyncio.get_running_loop().time() - started,
                    "error_type": type(error).__name__,
                },
            )
            raise
        elapsed = asyncio.get_running_loop().time() - started
        response_usage = response.usage
        input_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(response_usage, "completion_tokens", 0) or 0)
        self.usage["input_tokens"] += input_tokens
        self.usage["output_tokens"] += output_tokens
        self.save()
        self.store.event(
            self.project_id,
            "model_request_finished",
            {
                "role": self.role,
                "request_id": request_id,
                "request_kind": request_kind,
                "model": self.config.model.model,
                "ok": True,
                "elapsed_seconds": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "effective_input_tokens_per_second": input_tokens / elapsed if elapsed > 0 else None,
                "effective_output_tokens_per_second": output_tokens / elapsed if elapsed > 0 else None,
                "effective_total_tokens_per_second": (input_tokens + output_tokens) / elapsed
                if elapsed > 0
                else None,
                "cumulative_input_tokens": self.usage["input_tokens"],
                "cumulative_output_tokens": self.usage["output_tokens"],
            },
        )
        return response

    def _estimated_tokens(self) -> int:
        return sum(len(json.dumps(message, ensure_ascii=False)) for message in self.messages) // 4

    async def compact_if_needed(self) -> None:
        reserve = min(16_384, self.config.model.context_window // 4)
        threshold = self.config.model.context_window - self.config.model.max_output_tokens - reserve
        if self._estimated_tokens() < threshold or len(self.messages) < 10:
            return
        older, recent = self.messages[:-8], self.messages[-8:]
        response = await self._complete(
            request_kind="compaction",
            model=self.config.model.model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this agent history for exact continuation. Preserve decisions, evidence paths, measurements, failures, open work, and tool state. Do not invent facts.",
                },
                {"role": "user", "content": json.dumps(older, ensure_ascii=False)},
            ],
            max_tokens=min(4096, self.config.model.max_output_tokens),
            temperature=0,
            extra_body=self.config.model.extra_body or None,
        )
        summary = response.choices[0].message.content or "No summary was returned."
        self.messages = [{"role": "user", "content": "Prior-session summary:\n" + summary}, *recent]
        self.usage["compactions"] += 1
        self.save()
        self.store.event(self.project_id, "session_compacted", {"role": self.role})

    async def run(self, prompt: str) -> dict[str, Any]:
        self.messages.append({"role": "user", "content": prompt})
        self.save()
        definitions = [tool.openai_definition() for tool in self.tools.values()]
        for _step in range(256):
            if self.abort.is_set():
                raise asyncio.CancelledError("Agent phase was stopped")
            await self.compact_if_needed()
            response = await self._complete(
                request_kind="agent",
                model=self.config.model.model,
                messages=[{"role": "system", "content": self.system_prompt}, *self.messages],
                tools=definitions,
                tool_choice="auto",
                max_tokens=self.config.model.max_output_tokens,
                temperature=self.config.model.temperature,
                extra_body=self.config.model.extra_body or None,
            )
            message = response.choices[0].message
            assistant: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant["tool_calls"] = [call.model_dump(mode="json") for call in message.tool_calls]
            self.messages.append(assistant)
            self.save()
            if not message.tool_calls:
                followup = (
                    "Continue, then call decide to end this phase; plain text does not commit a lifecycle decision."
                    if self.role == "upper"
                    else "Continue working, then call submit_results; plain text does not end the iteration."
                )
                self.messages.append({"role": "user", "content": followup})
                self.save()
                continue
            for call in message.tool_calls:
                name = call.function.name
                started = asyncio.get_running_loop().time()
                ok = False
                self.store.event(
                    self.project_id,
                    "tool_started",
                    {
                        "role": self.role,
                        "call_id": call.id,
                        "tool": name,
                        "arguments_json": call.function.arguments or "{}",
                    },
                )
                try:
                    if self.terminal():
                        raise RuntimeError("A terminal tool already ended this phase")
                    tool = self.tools.get(name)
                    if tool is None:
                        raise ValueError(f"Unknown tool: {name}")
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Tool arguments are not valid JSON: {error}") from error
                    validate(arguments, tool.input_schema)
                    result = await tool.handler(arguments)
                    ok = not result.is_error
                    payload = result.text
                    if result.structured is not None:
                        payload = json.dumps(
                            {
                                "content": result.text,
                                "structured": result.structured,
                                "is_error": result.is_error,
                            },
                            separators=(",", ":"),
                        )
                except InterventionRequired:
                    raise
                except Exception as error:  # noqa: BLE001 - tool failures are model-readable results
                    payload = json.dumps({"is_error": True, "error": str(error)}, separators=(",", ":"))
                finally:
                    self.store.event(
                        self.project_id,
                        "tool_finished",
                        {
                            "role": self.role,
                            "call_id": call.id,
                            "tool": name,
                            "ok": ok,
                            "seconds": asyncio.get_running_loop().time() - started,
                        },
                    )
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})
                self.save()
            if self.terminal():
                return self.usage
        raise RuntimeError("Agent exceeded the per-phase tool-turn limit")
