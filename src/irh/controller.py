from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .agent import AgentSession
from .config import HarnessConfig
from .errors import ConflictError, InterventionRequired
from .internal_tools import phase_tools
from .sandbox import Sandbox
from .store import Store
from .tooling import MCPHub
from .types import Decision, InputSelection, IterationReport, Role
from .workspace import WorkspaceManager

COMMON_PROMPT = """You are a research agent in a persistent, goal-directed project.
Use the provided tools according to their descriptions. Tool errors are evidence: inspect them and correct the call when possible.
Distinguish observations, inferences, and hypotheses. Cite the artifact or source behind important claims. Give units and aggregation methods for measurements, and identify uncertainty or conflicting evidence.
Before substantial time or resource use, choose the smallest useful test, state what outcome would change the plan, and check feasibility. Replan when an assumption fails.
Do not invent results, measurements, files, or citations. The harness owns lifecycle, time limits, and durable evidence. A terminal tool ends your phase."""

UPPER_PROMPT = """Your role is researcher, planner, and reviewer. Inspect actual iteration evidence before accepting conclusions.
Project history is mounted read-only at /project. Preserve durable cross-phase findings with record_note.
Define attainable iterations with a hypothesis and enough context to execute them. Select inputs explicitly; workspaces are not inherited automatically.
Call decide exactly once when ready to start/revise an iteration or conclude the project."""

ITERATION_PROMPT = """Your role is experimenter and executor. Work toward the supplied iteration definition in /workspace.
Record reproducible commands and measurements, and preserve useful evidence as workspace files. End when the subgoal is achieved or realistically infeasible.
Call submit_results exactly once with results, conclusions, metrics, and relative artifact paths."""


class Controller:
    def __init__(self, config: HarnessConfig, store: Store, project_id: str):
        self.config = config
        self.store = store
        self.project_id = project_id
        self.abort = asyncio.Event()
        self.workspaces = WorkspaceManager(store)
        self._operator_pause = False

    def stop(self) -> None:
        self._operator_pause = True
        self.abort.set()

    def _tool_configs(self) -> list[Any]:
        project = self.store.get_project(self.project_id)
        enabled = project["snapshot"].get("enabled_tools", [])
        return [self.config.tool(tool_id) for tool_id in enabled]

    def _skills(self, role: Role) -> str:
        project = self.store.get_project(self.project_id)
        skills = project["snapshot"].get("skills", {}).get(role, [])
        if not skills:
            return ""
        return "\n\nSelected project skills:\n" + "\n\n".join(
            f"--- {skill['name']} ---\n{skill['text']}" for skill in skills
        )

    async def _charge_loop(self, role: Role, done: asyncio.Event) -> None:
        charged_at = time.monotonic()
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=30)
            except TimeoutError:
                now = time.monotonic()
                self.store.charge(self.project_id, role, now - charged_at)
                charged_at = now
        now = time.monotonic()
        self.store.charge(self.project_id, role, now - charged_at)

    async def _phase(
        self,
        role: Role,
        seconds: float,
        workspace: Path,
        scratch: Path,
        system_prompt: str,
        prompt: str,
        session_path: Path,
        choose: Any = None,
        submit: Any = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + seconds
        sandbox = Sandbox(
            role, workspace, scratch, self.store.project_dir(self.project_id), self.config.sandbox
        )
        context = {
            "project_id": self.project_id,
            "role": role,
            "workspace": str(workspace),
            "iteration": str(self.store.get_project(self.project_id)["current_iteration"]),
        }
        done = asyncio.Event()
        charge = asyncio.create_task(self._charge_loop(role, done))
        self.store.event(self.project_id, "agent_started", {"role": role, "seconds": seconds})
        try:
            async with MCPHub(
                self._tool_configs(), role, context, self.store, self.project_id, self.abort
            ) as hub:
                self.store.event(
                    self.project_id, "tool_configuration_resolved", {"role": role, "servers": hub.snapshot}
                )
                prompt_with_guidance = system_prompt
                if hub.guidance:
                    prompt_with_guidance += "\n\nEnabled external-tool guidance:\n" + hub.guidance
                choose_callback = (
                    (lambda value: setattr(choose, "value", value)) if choose is not None else None
                )
                submit_callback = (
                    (lambda value: setattr(submit, "value", value)) if submit is not None else None
                )
                internal = phase_tools(
                    role,
                    sandbox,
                    self.store,
                    self.project_id,
                    deadline,
                    choose_callback,
                    submit_callback,
                )
                terminal = (
                    (lambda: choose is not None and choose.value is not None)
                    if role == "upper"
                    else (lambda: submit is not None and submit.value is not None)
                )
                session = AgentSession(
                    self.config,
                    self.store,
                    self.project_id,
                    role,
                    session_path,
                    prompt_with_guidance,
                    [*internal, *hub.model_tools],
                    self.abort,
                    terminal,
                )
                try:
                    async with asyncio.timeout(seconds):
                        usage = await session.run(prompt)
                except asyncio.CancelledError:
                    if hub.incident:
                        raise hub.incident
                    raise
                if hub.incident:
                    raise hub.incident
                return usage
        finally:
            done.set()
            await charge
            self.store.event(self.project_id, "agent_stopped", {"role": role})

    async def run(self, resume: bool = False) -> None:
        project = self.store.get_project(self.project_id)
        if project["phase"] == "complete":
            raise ConflictError("Project is complete")
        if project["phase"] == "paused":
            if not resume:
                raise ConflictError("Project is paused; explicitly resume it")
            self.store.resume(self.project_id)
        self.store.event(self.project_id, "controller_started", {"resume": resume})
        try:
            while not self.abort.is_set():
                project = self.store.get_project(self.project_id)
                if project["phase"] in ("complete", "paused"):
                    break
                if project["phase"] == "executing":
                    await self._iterate()
                else:
                    await self._upper()
        except InterventionRequired as error:
            self.store.event(
                self.project_id,
                "operator_intervention_required",
                {"source": error.source, "detail": error.detail},
            )
            current = self.store.get_project(self.project_id)
            self.store.pause(self.project_id, str(error), resume_phase=current["phase"])
        except asyncio.CancelledError:
            current = self.store.get_project(self.project_id)
            self.store.pause(
                self.project_id,
                "Operator requested pause" if self._operator_pause else "Controller task was cancelled",
                resume_phase=current["phase"],
            )
        except Exception as error:
            current = self.store.get_project(self.project_id)
            self.store.pause(self.project_id, f"Controller error: {error}", resume_phase=current["phase"])
            raise
        finally:
            self.store.event(self.project_id, "controller_stopped", {})

    async def _upper(self) -> None:
        project = self.store.get_project(self.project_id)
        exhausted = project["spent_seconds"] >= project["budget_seconds"]
        seconds = (
            self.config.upper_seconds
            if exhausted
            else min(self.config.upper_seconds, project["budget_seconds"] - project["spent_seconds"])
        )
        if seconds <= 0 and not exhausted:
            self.store.pause(self.project_id, "Project active-time budget exhausted", resume_phase="review")
            return
        runtime = self.store.runtime_dir(self.project_id)
        scratch = runtime / "upper"
        scratch.mkdir(parents=True, exist_ok=True, mode=0o700)

        class Choice:
            value: Decision | None = None

        choice = Choice()
        latest = (
            self.store.get_iteration(self.project_id, project["current_iteration"])
            if project["current_iteration"]
            else None
        )
        history = self.store.history(self.project_id)
        history_summary = [
            {
                "number": item["number"],
                "status": item["status"],
                "subgoal": item["definition"]["subgoal"],
                "parent": item["definition"].get("parent"),
                "reviewed": item["review"] is not None,
            }
            for item in history
        ]
        packet = {
            "project": {key: value for key, value in project.items() if key != "snapshot"},
            "history": history_summary,
            "latest": latest,
            "artifacts": self.store.artifacts(self.project_id),
            "notes": self.store.notes(self.project_id),
            "instruction": (
                "The execution budget is exhausted. Perform a final evidence review and conclude, or explain what requires more budget."
                if exhausted
                else "Review any unreviewed iteration evidence, then choose the most useful next step."
            ),
        }
        prompt = COMMON_PROMPT + "\n" + UPPER_PROMPT + self._skills("upper")
        try:
            await self._phase(
                "upper",
                max(seconds, min(300, self.config.upper_seconds)) if exhausted else seconds,
                scratch,
                scratch,
                prompt,
                json.dumps(packet, separators=(",", ":")),
                self.store.private_dir(self.project_id) / "sessions" / "upper.json",
                choose=choice,
            )
        except TimeoutError:
            self.store.pause(self.project_id, "Upper-agent phase timed out", resume_phase=project["phase"])
            return
        if choice.value is None:
            self.store.pause(
                self.project_id, "Upper agent returned without a decision", resume_phase=project["phase"]
            )
            return
        if exhausted and choice.value.action != "stop":
            self.store.pause(
                self.project_id, "Project budget exhausted after final review", resume_phase="review"
            )
            return
        self.store.apply_decision(self.project_id, choice.value, self.config.max_iteration_seconds)

    async def _iterate(self) -> None:
        project = self.store.get_project(self.project_id)
        number = project["current_iteration"]
        iteration = self.store.get_iteration(self.project_id, number)
        if iteration["status"] == "defined":
            definition = iteration["definition"]
            selections = [InputSelection.model_validate(value) for value in definition.get("inputs", [])]
            workspace = self.workspaces.stage(self.project_id, number, selections)
            self.store.start_attempt(self.project_id, number)
        elif iteration["status"] == "running" and iteration["resume_allowed"]:
            workspace = self.store.runtime_dir(self.project_id) / "iteration" / "workspace"
            if not workspace.is_dir():
                raise ConflictError("Resumable iteration workspace is missing")
            self.store.start_attempt(self.project_id, number)
        else:
            raise ConflictError(f"Iteration cannot run from {iteration['status']}")
        remaining_iteration = iteration["definition"]["time_limit_seconds"] - iteration["active_seconds"]
        remaining_project = project["budget_seconds"] - project["spent_seconds"]
        seconds = min(remaining_iteration, remaining_project)
        if seconds <= 0:
            archive = self.workspaces.archive(self.project_id, number)
            self.store.finish_iteration(
                self.project_id, number, "timeout", None, "Active-time budget exhausted"
            )
            return

        class Submission:
            value: IterationReport | None = None

        submission = Submission()
        prompt = COMMON_PROMPT + "\n" + ITERATION_PROMPT + self._skills("iteration")
        definition_prompt = json.dumps(iteration["definition"], separators=(",", ":"))
        try:
            await self._phase(
                "iteration",
                seconds,
                workspace,
                self.store.runtime_dir(self.project_id) / "iteration" / "scratch",
                prompt,
                definition_prompt,
                self.store.private_dir(self.project_id) / "sessions" / f"iteration-{number}.json",
                submit=submission,
            )
            status = submission.value.outcome if submission.value else "failed"
            error = None if submission.value else "Iteration agent returned without submit_results"
        except TimeoutError:
            status, error = "timeout", "Iteration active-time limit reached"
        archive = self.workspaces.archive(self.project_id, number)
        try:
            self.workspaces.register_report_artifacts(self.project_id, number, archive, submission.value)
        except Exception as artifact_error:  # noqa: BLE001 - convert registration failures to lifecycle state
            submission.value = None
            status, error = "failed", f"Artifact registration failed: {artifact_error}"
        self.store.finish_iteration(self.project_id, number, status, submission.value, error)


class Supervisor:
    """Own active controllers; all mutating run control passes through this object."""

    def __init__(self, config: HarnessConfig, store: Store):
        self.config = config
        self.store = store
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.controllers: dict[str, Controller] = {}
        self.lock = asyncio.Lock()

    async def start(self, project_id: str, resume: bool = False) -> dict[str, Any]:
        async with self.lock:
            active = [key for key, task in self.tasks.items() if not task.done()]
            if active:
                raise ConflictError(f"A project is already active: {active[0]}")
            project = self.store.get_project(project_id)
            if project["phase"] == "complete":
                raise ConflictError("Project is complete")
            if project["phase"] == "paused" and not resume:
                raise ConflictError("Project is paused; explicitly resume it")
            if project["phase"] != "paused" and resume:
                raise ConflictError("Project is not paused; omit --resume")
            controller = Controller(self.config, self.store, project_id)
            task = asyncio.create_task(controller.run(resume), name=f"irh:{project_id}")
            self.controllers[project_id] = controller
            self.tasks[project_id] = task
            task.add_done_callback(lambda finished, key=project_id: self._finished(key, finished))
            return {"accepted": True, "project_id": project_id}

    def _finished(self, project_id: str, task: asyncio.Task[None]) -> None:
        self.controllers.pop(project_id, None)
        if not task.cancelled():
            task.exception()

    async def pause(self, project_id: str) -> dict[str, Any]:
        controller = self.controllers.get(project_id)
        task = self.tasks.get(project_id)
        if controller is None or task is None or task.done():
            project = self.store.get_project(project_id)
            self.store.pause(project_id, "Operator requested pause", resume_phase=project["phase"])
            return {"accepted": True, "state": "paused"}
        controller.stop()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=30)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return {"accepted": True, "state": "paused"}

    async def close(self) -> None:
        for controller in list(self.controllers.values()):
            controller.stop()
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
