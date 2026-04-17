from __future__ import annotations

from dataclasses import dataclass, field

from vision_rlm.env.schema import Budget, Note


@dataclass
class PlannerState:
    question: str
    page_candidates: list[dict[str, object]] = field(default_factory=list)
    memory: list[Note] = field(default_factory=list)
    remaining_budget: Budget | None = None
    recent_history: list[dict[str, object]] = field(default_factory=list)
