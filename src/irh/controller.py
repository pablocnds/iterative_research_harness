from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from .agent import AgentSession, atomic_json
from .config import ExternalToolConfig, HarnessConfig
from .errors import ConflictError, InterventionRequired
from .internal_tools import phase_tools
from .sandbox import Sandbox
from .store import Store
from .tooling import MCPHub
from .types import Decision, InputSelection, IterationReport, Role
from .workspace import WorkspaceManager

logger = logging.getLogger(__name__)

COMMON_PROMPT = """You are a research agent in a persistent, goal-directed project.
Use the provided tools according to their descriptions. Tool errors are evidence: inspect them and correct the call when possible.
Distinguish observations, inferences, and hypotheses. Cite the artifact or source behind important claims. Give units and aggregation methods for measurements, and identify uncertainty or conflicting evidence.
Before substantial time or resource use, choose the smallest useful test, state what outcome would change the plan, and check feasibility. Replan when an assumption fails.
Do not invent results, measurements, files, or citations. The harness owns lifecycle, time limits, and durable evidence. A terminal tool ends your phase."""

UPPER_PROMPT = """Your role is researcher and planner. Project history is mounted read-only at /project.
Define attainable iterations with a hypothesis and enough context to execute them. Select inputs explicitly; workspaces are not inherited automatically.
Call decide exactly once when ready to start/revise an iteration or conclude the project."""

ITERATION_PROMPT = """Your role is experimenter and executor. Work toward the supplied iteration definition in /workspace.
Record reproducible commands and measurements in workspace files. Produce a concise paper.md with Methodology, Results, and Conclusions sections; report failures and incomplete work sincerely. Keep paper.md under 4 MiB and select at most 64 supplementary files, each under 2 GiB and at most 8 GiB total.
Call submit_results with the paper outcome and supplementary paths when finished."""

WRITEUP_PROMPT = """Execution time has ended. Continue from this same session and finish a sincere, concise paper.md using only existing evidence. State missing results, failed checks, interrupted calls, and pending jobs plainly; do not claim unverified outcomes. You may read existing files, write only paper.md, and submit it with at most 64 selected supplementary files; paper.md must stay under 4 MiB, each supplement under 2 GiB, and the package under 8 GiB. No further experiments are available."""

REVIEW_PROMPT = """Your role is to plan and review the iteration paper and its selected supplementary files in /project. The raw iteration workspace is not available. Assess what the paper supports, request a targeted revision or more evidence if needed, or define the next bounded iteration. For a revision, select needed files from the packet's revision_inputs as definition.inputs; source values are project-root paths and must retain the exact revision_inputs prefix, which differs from read_file paths. The full workspace may be selected without inspecting it. Do not reconstruct the whole experiment from source files. Call decide once when ready."""


class ActiveTimeBudget:
    """A phase clock that can pause while an agent is blocked by infrastructure."""

    def __init__(self, seconds: float):
        self.initial = seconds
        self._remaining = seconds
        self._last = time.monotonic()
        self._pause_depth = 0
        self._changed = asyncio.Event()

    def _settle(self) -> None:
        now = time.monotonic()
        if self._pause_depth == 0:
            self._remaining -= now - self._last
        self._last = now

    def remaining(self) -> float:
        self._settle()
        return max(0.0, self._remaining)

    def consumed(self) -> float:
        return min(self.initial, max(0.0, self.initial - self.remaining()))

    def pause(self) -> None:
        self._settle()
        self._pause_depth += 1
        self._changed.set()

    def resume(self) -> None:
        if self._pause_depth == 0:
            return
        self._settle()
        self._pause_depth -= 1
        self._last = time.monotonic()
        self._changed.set()

    async def wait_until_exhausted(self) -> None:
        while self.remaining() > 0:
            self._changed.clear()
            if self._pause_depth:
                await self._changed.wait()
                continue
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=self.remaining())
            except TimeoutError:
                pass


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
        execution = self.store.execution_snapshot(self.project_id)
        frozen = execution.get("tools")
        if isinstance(frozen, list):
            return [ExternalToolConfig.model_validate(value) for value in frozen]
        project = self.store.get_project(self.project_id)
        enabled = execution.get("enabled_tools", project["snapshot"].get("enabled_tools", []))
        return [self.config.tool(tool_id) for tool_id in enabled]

    def _base_prompts(self) -> dict[str, str]:
        execution = self.store.execution_snapshot(self.project_id)
        prompts = execution.get("base_prompts", {})
        return {
            "common": prompts.get("common", COMMON_PROMPT),
            "upper": prompts.get("upper", UPPER_PROMPT),
            "iteration": prompts.get("iteration", ITERATION_PROMPT),
            "writeup": prompts.get("writeup", WRITEUP_PROMPT),
            "review": prompts.get("review", REVIEW_PROMPT),
        }

    def _skills(self, role: Role) -> str:
        project = self.store.get_project(self.project_id)
        execution = self.store.execution_snapshot(self.project_id)
        skills = execution.get("skills", project["snapshot"].get("skills", {})).get(role, [])
        if not skills:
            return ""
        return "\n\nSelected project skills:\n" + "\n\n".join(
            f"--- {skill['name']} ---\n{skill['text']}" for skill in skills
        )

    def _session_path(self, name: str) -> Path:
        directory = self.store.private_dir(self.project_id) / "sessions"
        canonical = directory / f"{name}.json"
        if name == "upper" and not canonical.exists():
            legacy = directory / "upper.json.json"
            if legacy.is_file():
                return legacy
        return canonical

    async def _charge_loop(self, role: Role, done: asyncio.Event, budget: ActiveTimeBudget) -> None:
        charged = budget.consumed()
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=30)
            except TimeoutError:
                current = budget.consumed()
                self.store.charge(self.project_id, role, current - charged)
                charged = current
        current = budget.consumed()
        self.store.charge(self.project_id, role, current - charged)

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
        mode: str = "normal",
        project_view: Path | None = None,
    ) -> dict[str, Any]:
        budget = ActiveTimeBudget(seconds)
        sandbox = Sandbox(
            role,
            workspace,
            scratch,
            project_view or self.store.project_dir(self.project_id),
            self.config.sandbox,
        )
        context = {
            "project_id": self.project_id,
            "role": role,
            "workspace": str(workspace),
            "iteration": str(self.store.get_project(self.project_id)["current_iteration"]),
        }
        done = asyncio.Event()
        charge = asyncio.create_task(
            self._charge_loop("writeup" if mode == "writeup" else role, done, budget)
        )
        self.store.event(self.project_id, "agent_started", {"role": role, "mode": mode, "seconds": seconds})
        try:
            async with MCPHub(
                [tool for tool in self._tool_configs() if mode == "normal" or tool.monitor is not None],
                role,
                context,
                self.store,
                self.project_id,
                self.abort,
                budget.pause,
                budget.resume,
            ) as hub:
                self.store.event(
                    self.project_id, "tool_configuration_resolved", {"role": role, "servers": hub.snapshot}
                )
                prompt_with_guidance = system_prompt
                if mode == "normal" and hub.guidance:
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
                    budget.remaining,
                    choose_callback,
                    submit_callback,
                    mode=mode,
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
                    workspace,
                    session_path,
                    prompt_with_guidance,
                    [*internal, *(hub.model_tools if mode == "normal" else [])],
                    self.abort,
                    terminal,
                    project_view=sandbox.project,
                    report_only=mode == "writeup",
                )
                session_task = asyncio.create_task(session.run(prompt))
                expiry_task = asyncio.create_task(budget.wait_until_exhausted())
                try:
                    completed, _pending = await asyncio.wait(
                        (session_task, expiry_task), return_when=asyncio.FIRST_COMPLETED
                    )
                    if session_task in completed:
                        usage = await session_task
                    else:
                        session_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await session_task
                        raise TimeoutError
                except asyncio.CancelledError:
                    if hub.incident:
                        raise hub.incident
                    raise
                finally:
                    expiry_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await expiry_task
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
                elif project["phase"] == "writeup":
                    await self._writeup()
                elif project["phase"] == "finalizing":
                    self._finalize(project["current_iteration"])
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
                "paper": (
                    f"{item['number']:04d}/evidence/{item['report']['paper']}"
                    if item["report"] and item["report"].get("paper")
                    else None
                ),
            }
            for item in history
        ]
        instruction = (
            "The execution budget is exhausted. Perform a final evidence review and conclude, or explain what requires more budget."
            if exhausted
            else "Review any unreviewed iteration evidence, then choose the most useful next step."
        )
        project_record = {key: value for key, value in project.items() if key != "snapshot"}
        notes = self.store.notes(self.project_id)
        if latest is None:
            packet = {
                "project": project_record,
                "history": history_summary,
                "latest": None,
                "notes": notes,
                "instruction": instruction,
            }
            session_name = "upper"
            project_view = self.store.project_dir(self.project_id)
        else:
            project_view = self.store.project_dir(self.project_id) / "publications"
            project_view.mkdir(parents=True, exist_ok=True, mode=0o770)
            for name, value in (
                ("project.json", project_record),
                ("history.json", history_summary),
                ("notes.json", notes),
            ):
                atomic_json(project_view / name, value)
            publication = project_view / f"{latest['number']:04d}"
            if not publication.exists():
                publication.mkdir(parents=True, mode=0o770)
                (publication / "notice.md").write_text(
                    "No iteration paper is available. This iteration ended before the paper workflow was installed "
                    "or before a valid paper was submitted. Request a targeted revision if more evidence is needed.\n"
                )
                atomic_json(publication / "manifest.json", {"paper": None, "supplementary": []})
            else:
                publication.chmod(0o770)
            packet = {
                "project": {
                    "name": project["name"],
                    "goal_preview": project["goal"][:1000],
                    "goal_truncated": len(project["goal"]) > 1000,
                },
                "latest": {
                    "number": latest["number"],
                    "status": latest["status"],
                    "error": latest["error"],
                    "subgoal_preview": latest["definition"]["subgoal"][:500],
                    "report_outcome": (latest["report"] or {}).get("outcome"),
                    "paper": history_summary[-1]["paper"],
                    "manifest": f"{latest['number']:04d}/manifest.json",
                },
                "review_files": {
                    "project": "project.json",
                    "history": "history.json",
                    "notes": "notes.json",
                },
                "revision_inputs": {
                    "published": f"publications/{latest['number']:04d}/evidence",
                    "full_workspace_if_needed": f"iterations/{latest['number']:04d}/workspace",
                },
                "instruction": instruction,
            }
            session_name = f"upper-review-{latest['number']}"
        prompts = self._base_prompts()
        prompt = (
            prompts["common"]
            + "\n"
            + (prompts["review"] if latest else prompts["upper"])
            + self._skills("upper")
        )
        try:
            await self._phase(
                "upper",
                max(seconds, min(300, self.config.upper_seconds)) if exhausted else seconds,
                scratch,
                scratch,
                prompt,
                json.dumps(packet, separators=(",", ":")),
                self._session_path(session_name),
                choose=choice,
                mode="paper_review" if latest else "normal",
                project_view=project_view,
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
        self.store.apply_decision(
            self.project_id,
            choice.value,
            self.config.iteration_seconds,
            self.config.max_iteration_seconds,
        )

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
            self.store.begin_writeup(self.project_id, number, "Active-time budget exhausted")
            return

        class Submission:
            value: IterationReport | None = None

        submission = Submission()
        prompts = self._base_prompts()
        prompt = prompts["common"] + "\n" + prompts["iteration"] + self._skills("iteration")
        definition_prompt = json.dumps(iteration["definition"], separators=(",", ":"))
        try:
            await self._phase(
                "iteration",
                seconds,
                workspace,
                self.store.runtime_dir(self.project_id) / "iteration" / "scratch",
                prompt,
                definition_prompt,
                self._session_path(f"iteration-{number}"),
                submit=submission,
            )
        except TimeoutError:
            self.store.begin_writeup(self.project_id, number, "Iteration active-time limit reached")
            return
        if submission.value is None:
            self.store.begin_writeup(self.project_id, number, "Iteration ended without submitting a paper")
            return
        self._finish_with_paper(number, submission.value, submission.value.outcome, None)

    def _finish_with_paper(
        self, number: int, report: IterationReport, status: str, error: str | None
    ) -> None:
        self.store.begin_finalization(self.project_id, number, status, report, error)
        self._finalize(number)

    def _finalize(self, number: int) -> None:
        iteration = self.store.get_iteration(self.project_id, number)
        if iteration["status"] != "finalizing" or not iteration["report"]:
            raise ConflictError(f"Iteration cannot finalize from {iteration['status']}")
        if not iteration["final_status"]:
            raise ConflictError("Finalizing iteration has no target status")
        report = IterationReport.model_validate(iteration["report"])
        archive = self.workspaces.archive(self.project_id, number)
        self.workspaces.publish_report(self.project_id, number, archive, report)
        self.workspaces.register_report_artifacts(self.project_id, number, archive, report)
        self.store.finish_iteration(
            self.project_id,
            number,
            iteration["final_status"],
            report,
            iteration["error"],
        )

    async def _writeup(self) -> None:
        project = self.store.get_project(self.project_id)
        number = project["current_iteration"]
        iteration = self.store.get_iteration(self.project_id, number)
        if iteration["status"] != "writeup":
            raise ConflictError(f"Iteration cannot write up from {iteration['status']}")
        workspace = self.store.runtime_dir(self.project_id) / "iteration" / "workspace"
        if not workspace.is_dir():
            raise ConflictError("Write-up workspace is missing")
        seconds = self.config.writeup_seconds - iteration["writeup_seconds"]
        if seconds <= 0:
            self.store.pause(self.project_id, "Write-up grace exhausted", resume_phase="writeup")
            return

        class Submission:
            value: IterationReport | None = None

        submission = Submission()
        prompts = self._base_prompts()
        prompt = f"{prompts['writeup']}\nExecution ended because: {iteration['error']}"
        try:
            await self._phase(
                "iteration",
                seconds,
                workspace,
                self.store.runtime_dir(self.project_id) / "iteration" / "scratch",
                prompts["common"] + "\n" + prompts["writeup"],
                prompt,
                self._session_path(f"iteration-{number}"),
                submit=submission,
                mode="writeup",
            )
        except TimeoutError:
            self.store.pause(self.project_id, "Write-up grace exhausted", resume_phase="writeup")
            return
        if submission.value is None:
            self.store.pause(self.project_id, "Iteration returned without a paper", resume_phase="writeup")
            return
        reason = iteration["error"]
        status = "interrupted" if reason and "daemon stopped" in reason else "timeout"
        self._finish_with_paper(number, submission.value, status, reason)


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
        if self.tasks.get(project_id) is task:
            self.tasks.pop(project_id, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "Controller task failed for project %s",
                    project_id,
                    exc_info=(type(error), error, error.__traceback__),
                )

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
