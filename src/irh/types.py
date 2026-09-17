from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["upper", "iteration"]


class InputSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    destination: str = Field(min_length=1)


class IterationDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoal: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    introduction: str = Field(min_length=1)
    requirements: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)
    parent: int | None = Field(default=None, ge=1)
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

    outcome: Literal["achieved", "infeasible"]
    results: str = Field(min_length=1)
    conclusions: str = Field(min_length=1)
    metrics: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    code_revision: str | None = None
    environment: str | None = None


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    guidelines: str = ""
    budget_seconds: float | None = Field(default=None, gt=0)
    tools: list[str] = Field(default_factory=list)
    upper_skills: list[str] = Field(default_factory=list)
    iteration_skills: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    structured: Any = None
    is_error: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)
