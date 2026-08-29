from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SITE_EVIDENCE = {"edit_site_environment", "edit_site_geometry", "candidate_geometry"}
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
    docking_history: list[dict[str, Any]] = field(default_factory=list)
    candidate_history: list[dict[str, Any]] = field(default_factory=list)
    exploration_attempts: list[dict[str, Any]] = field(default_factory=list)
    unmodifiable_targets: list[dict[str, Any]] = field(default_factory=list)
    tool_rejections: list[dict[str, Any]] = field(default_factory=list)
    site_strategy: dict[str, Any] | None = None
    active_target: dict[str, Any] | None = None
    site_search: dict[str, dict[str, Any]] = field(default_factory=dict)
    convergence: dict[str, Any] = field(default_factory=lambda: {
        "status": "not_started",
        "converged": False,
        "llm_controls_termination": True,
        "stop_authority": "llm_or_hard_safety_limit",
        "best_attempt": None,
        "best_quality": None,
        "non_improving_attempts": 0,
    })

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
            "docking_history": self.docking_history,
            "candidate_history": self.candidate_history,
            "exploration_attempts": self.exploration_attempts,
            "unmodifiable_targets": self.unmodifiable_targets,
            "tool_rejections": self.tool_rejections,
            "site_strategy": self.site_strategy,
            "active_target": self.active_target,
            "site_search": self.site_search,
            "convergence": self.convergence,
        }
