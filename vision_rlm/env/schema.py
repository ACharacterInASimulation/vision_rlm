from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionType(str, Enum):
    RETRIEVE_PAGES = "RETRIEVE_PAGES"
    OPEN_PAGE = "OPEN_PAGE"
    RANK_REGIONS = "RANK_REGIONS"
    INSPECT_REGION = "INSPECT_REGION"
    COMPUTE = "COMPUTE"
    WRITE_NOTE = "WRITE_NOTE"
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Budget:
    visual_tokens: int
    pages_opened: int
    regions_inspected: int
    tool_steps: int


@dataclass(frozen=True)
class EvidenceRef:
    page_id: str
    region_id: str | None = None


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Note:
    note_id: str
    fact: str
    evidence: EvidenceRef
    evidence_type: str
    confidence: float
