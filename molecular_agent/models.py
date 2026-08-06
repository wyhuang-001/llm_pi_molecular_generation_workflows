from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SITE_EVIDENCE = {"edit_site_environment", "edit_site_geometry"}
REQUIRED_EVIDENCE = SITE_EVIDENCE


@dataclass
class ToolObservation:
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    evidence: set[str] = field(default_factory=set)


@dataclass
class AgentState:
    task: str
    max_context_rounds: int
    observations: list[ToolObservation] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    evidence: set[str] = field(default_factory=set)
    call_signatures: set[str] = field(default_factory=set)

    @property
    def missing_evidence(self) -> list[str]:
        return sorted(SITE_EVIDENCE - self.evidence)

    @property
    def ready(self) -> bool:
        return not self.missing_evidence

    def compact_view(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "round": len(self.observations),
            "max_rounds": self.max_context_rounds,
            "covered_evidence": sorted(self.evidence),
            "missing_site_evidence": self.missing_evidence,
            "decisions": self.decisions,
            "observations": [
                {
                    "tool": item.tool,
                    "arguments": item.arguments,
                    "result": item.result,
                    "evidence": sorted(item.evidence),
                }
                for item in self.observations
            ],
        }
