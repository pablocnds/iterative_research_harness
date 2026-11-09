from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["upper", "iteration"]
ReportPath = Annotated[str, Field(min_length=1, max_length=1024)]


class InputSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        min_length=1,
        description="Project-root source path. During paper review, copy the exact prefix supplied in revision_inputs; read_file paths are not source paths.",
    )
    destination: str = Field(min_length=1, description="Relative path in the new iteration workspace.")


class IterationDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoal: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    introduction: str = Field(min_length=1)
    requirements: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)
    parent: int | None = Field(
        ge=1,
        description="Always include this key: the reviewed iteration number for action='revise', or null for action='iterate'.",
    )
    time_limit_seconds: float | None = Field(default=None, gt=0)
    inputs: list[InputSelection] = Field(default_factory=list)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["iterate", "revise", "stop"]
    reason: str = Field(min_length=1)
    definition: IterationDefinition | None = None
    stop_reason: Literal["achieved", "infeasible", "insufficient_resources", "not_worth_pursuing"] | None = (
        None
    )

    @model_validator(mode="after")
    def validate_action(self) -> Decision:
        if self.action in ("iterate", "revise") and self.definition is None:
            raise ValueError("iterate/revise requires definition")
        if self.action == "stop" and self.stop_reason is None:
            raise ValueError("stop requires stop_reason")
        if self.action == "revise" and (self.definition is None or self.definition.parent is None):
            raise ValueError("revise requires definition.parent")
        return self


class IterationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal["achieved", "infeasible", "partial"]
    paper: Literal["paper.md"] = "paper.md"
    artifacts: list[ReportPath] = Field(default_factory=list, max_length=64)


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    guidelines: str = ""
    budget_seconds: float | None = Field(default=None, gt=0)
    tools: list[str] = Field(default_factory=list)
    upper_skills: list[str] = Field(default_factory=list)
    iteration_skills: list[str] = Field(default_factory=list)


class ProjectRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resume: bool = Field(default=False, strict=True)
    max_iterations: int | None = Field(default=None, gt=0, strict=True)
    actor: str = Field(default="operator", min_length=1, max_length=200)


class SteeringMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=32_000)
    actor: str = Field(default="operator", min_length=1, max_length=200)
    target: Literal["next", "upper", "iteration"] = "next"
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class PauseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actor: str = Field(default="operator", min_length=1, max_length=200)
    reason: str = Field(default="Pause requested", min_length=1, max_length=2000)


class PublicationSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str
    iteration: int = Field(gt=0, strict=True)


class PublicationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=500)
    paper: str = Field(min_length=1, max_length=4 * 1024 * 1024)
    actor: str = Field(default="supervisor", min_length=1, max_length=200)
    sources: list[PublicationSource] = Field(min_length=1, max_length=256)
    supplements: dict[str, str] = Field(default_factory=dict, max_length=64)
    supersedes: str | None = None


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    structured: Any = None
    is_error: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)
