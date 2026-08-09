from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from .adapters import NotConfiguredAdapter, configured_adapters
from .editing import EditResult, apply_substituent, write_sdf
from .models import AgentState, ToolObservation
from .structure import ComplexContext
from .tools import ToolRegistry


class DecisionClient(Protocol):
    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class Workflow:
    def __init__(
        self,
        task_path: Path,
        client: DecisionClient,
        run_dir: Path,
        config_path: Path | None = None,
    ):
        self.context = ComplexContext(task_path)
        self.client = client
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tools = ToolRegistry(self.context)
        if config_path is not None:
            self.docking_adapter, self.rbfe_adapter = configured_adapters(
                config_path.resolve(), self.run_dir
            )
        else:
            self.docking_adapter = NotConfiguredAdapter("docking")
            self.rbfe_adapter = NotConfiguredAdapter("rbfe")
        self.state = AgentState(
            task=self.context.task["task"],
            max_context_rounds=int(self.context.task.get("max_context_rounds", 8)),
        )

    def _signature(self, tool: str, arguments: dict[str, Any]) -> str:
        data = json.dumps(
            {"tool": tool, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _query_payload(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "mode": "context_collection",
            "state": self.state.compact_view(),
            "structure_input": {
                "complex_path": str(self.context.complex_path),
                "ligand_selector": self.context.ligand_selector,
                "ligand_source": self.context.ligand_source,
            },
            "tool_catalog": self.tools.catalog(),
            "host_ready": self.state.ready,
            "instruction": (
                "Choose what knowledge is needed next. Query any available tool, propose a missing "
                "tool, or return READY when your understanding is sufficient and the final edit site "
                "has site-specific evidence. Do not follow a fixed query sequence."
            ),
        }
        if extra:
            payload.update(extra)
        return payload

    def _record_decision(self, decision: dict[str, Any]) -> int:
        self.state.decisions.append(decision)
        decision_number = len(self.state.decisions)
        self._write_json(
            f"decision-{decision_number:02d}.json",
            {"decision": decision, "state": self.state.compact_view()},
        )
        return decision_number

    def _execute_query(self, decision: dict[str, Any]) -> None:
        tool = decision.get("tool")
        arguments = decision.get("arguments")
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            raise RuntimeError("QUERY requires string tool and object arguments")
        signature = self._signature(tool, arguments)
        if signature in self.state.call_signatures:
            raise RuntimeError(f"Duplicate tool call blocked: {tool} {arguments}")
        result, evidence = self.tools.execute(tool, arguments)
        self.state.call_signatures.add(signature)
        self.state.evidence.update(evidence)
        self.state.observations.append(
            ToolObservation(tool=tool, arguments=arguments, result=result, evidence=evidence)
        )
        self._write_json(
            f"observation-{len(self.state.observations):02d}.json",
            self.state.compact_view(),
        )

    def _handle_decision(self, decision: dict[str, Any]) -> str:
        decision_number = self._record_decision(decision)
        action = decision.get("action")
        if action == "PROPOSE_TOOL":
            proposal = {
                "decision": decision,
                "state": self.state.compact_view(),
                "status": "awaiting_host_review",
            }
            self._write_json(f"tool-proposal-{decision_number:02d}.json", proposal)
            raise RuntimeError(
                "LLM proposed a new tool; proposal saved for host review before execution"
            )
        if action == "QUERY":
            self._execute_query(decision)
        elif action != "READY":
            raise RuntimeError(f"Invalid LLM action: {action!r}")
        return action

    def collect_context(self) -> dict[str, Any]:
        while len(self.state.observations) < self.state.max_context_rounds:
            decision = self.client.complete_json(self._query_payload())
            action = self._handle_decision(decision)
            if action == "READY":
                self._validate_design(decision)
                self._write_json("context-final.json", self.state.compact_view())
                return decision
        raise RuntimeError(
            "Context budget exhausted before a valid READY decision; missing site evidence: "
            + ", ".join(self.state.missing_evidence)
        )

    def _retry_ready_decision(
        self, previous_design: dict[str, Any], rejection: dict[str, Any]
    ) -> dict[str, Any]:
        while len(self.state.observations) < self.state.max_context_rounds:
            decision = self.client.complete_json(
                self._query_payload(
                    {
                        "mode": "edit_retry",
                        "previous_design": previous_design,
                        "rejection": rejection,
                        "instruction": (
                            "The previous candidate was rejected by deterministic chemistry or clash "
                            "checks. You may QUERY a new fact or return READY. If you return READY, "
                            "include understanding, edit_atom_index, edit_hypothesis, and fragment_smiles."
                        ),
                    }
                )
            )
            action = self._handle_decision(decision)
            if action == "READY":
                return decision
        raise RuntimeError("Context budget exhausted while selecting an edit retry")

    def _validate_design(self, decision: dict[str, Any]) -> None:
        required = ("understanding", "edit_atom_index", "edit_hypothesis", "fragment_smiles")
        missing = [key for key in required if key not in decision]
        if missing:
            raise RuntimeError(f"READY decision missing fields: {missing}")
        index = decision["edit_atom_index"]
        if not isinstance(index, int) or not 0 <= index < self.context.ligand.GetNumAtoms():
            raise RuntimeError(f"Invalid edit_atom_index: {index!r}")
        atom = self.context.ligand.GetAtomWithIdx(index)
        if atom.GetTotalNumHs() < 1:
            raise RuntimeError(f"Selected edit atom {index} has no replaceable hydrogen")
        environment_sites = {
            item.arguments.get("atom_index")
            for item in self.state.observations
            if item.tool == "get_atom_environment"
        }
        geometry_sites = {
            item.arguments.get("atom_index")
            for item in self.state.observations
            if item.tool == "check_growth_space"
        }
        candidate_geometry = {
            (item.arguments.get("atom_index"), item.arguments.get("fragment_smiles"))
            for item in self.state.observations
            if item.tool == "validate_candidate_geometry"
        }
        fragment = decision["fragment_smiles"]
        if (
            index not in environment_sites
            or index not in geometry_sites
            or (index, fragment) not in candidate_geometry
        ):
            raise RuntimeError(
                f"Selected edit {index} + {fragment} lacks environment, growth-space, "
                "and candidate-geometry evidence"
            )

    def design(self, first_decision: dict[str, Any]) -> dict[str, Any]:
        decision = first_decision
        attempts = int(self.context.task.get("max_edit_attempts", 4))
        history = []
        for attempt in range(1, attempts + 1):
            self._validate_design(decision)
            try:
                result = apply_substituent(
                    self.context.ligand,
                    decision["edit_atom_index"],
                    decision["fragment_smiles"],
                    self.context.protein_atoms,
                    seed=17,
                )
                if result is not None:
                    attempt_path = self.run_dir / f"edit-attempt-{attempt:02d}.sdf"
                    write_sdf(result, attempt_path, name=f"edit-attempt-{attempt:02d}")
                report = {
                    "attempt": attempt,
                    "decision": decision,
                    "validation": result.report,
                    "candidate_path": str(attempt_path) if result is not None else None,
                }
            except Exception as error:
                result = None
                report = {
                    "attempt": attempt,
                    "decision": decision,
                    "validation": {"status": "rejected", "error": str(error)},
                    "candidate_path": None,
                }
            history.append(report)
            self._write_json(f"edit-attempt-{attempt:02d}.json", report)
            if result is not None and result.report["status"] == "accepted":
                candidate_path = self.run_dir / f"candidate-{attempt:02d}.sdf"
                write_sdf(result, candidate_path, name=f"candidate-{attempt:02d}")
                reference_path = self.run_dir / "reference-ligand.sdf"
                write_sdf(
                    EditResult(self.context.ligand, {"status": "reference"}),
                    reference_path,
                    name="reference-ligand",
                )
                receptor_path = self.context.write_receptor_pdb(
                    self.run_dir / "receptor-protein-only.pdb"
                )
                docking = self.docking_adapter.run(
                    candidate_path=candidate_path,
                    receptor_path=receptor_path,
                )
                rbfe = self.rbfe_adapter.run(
                    candidate_path=candidate_path,
                    receptor_path=receptor_path,
                    reference_path=reference_path,
                    docking_result=docking,
                )
                return {
                    "status": "candidate_accepted",
                    "candidate_path": str(candidate_path),
                    "reference_path": str(reference_path),
                    "attempts": history,
                    "docking": docking,
                    "rbfe": rbfe,
                    "fep": rbfe,
                }
            if attempt < attempts:
                decision = self._retry_ready_decision(decision, report["validation"])
        return {"status": "no_candidate_accepted", "attempts": history}

    def run(self) -> dict[str, Any]:
        first_decision = self.collect_context()
        result = self.design(first_decision)
        final = {"state": self.state.compact_view(), "result": result}
        self._write_json("result.json", final)
        return final

    def _write_json(self, name: str, value: dict[str, Any]) -> None:
        (self.run_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class ScriptedDemoClient:
    """Deterministic smoke-test client; it does not contain experimental SAR answers."""

    def __init__(self):
        self.step = 0

    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        missing = payload["state"]["missing_site_evidence"]
        if "edit_site_environment" in missing:
            return {
                "action": "QUERY",
                "question": "What surrounds phenyl atom 10?",
                "tool": "get_atom_environment",
                "arguments": {"atom_index": 10, "radius": 5.0},
                "expected_evidence": "edit site environment",
            }
        if "edit_site_geometry" in missing:
            return {
                "action": "QUERY",
                "question": "Is there outward space at atom 10?",
                "tool": "check_growth_space",
                "arguments": {"atom_index": 10, "distance": 2.0},
                "expected_evidence": "growth space",
            }
        if "candidate_geometry" in missing:
            return {
                "action": "QUERY",
                "question": "Does fluorination at atom 10 pass the exact candidate geometry check?",
                "tool": "validate_candidate_geometry",
                "arguments": {"atom_index": 10, "fragment_smiles": "[*:1]F"},
                "expected_evidence": "candidate geometry",
            }
        return {
            "action": "READY",
            "understanding": "The scaffold has a tested local edit site with a measured protein environment and growth probe.",
            "edit_atom_index": 10,
            "edit_hypothesis": "Add a small substituent at the tested phenyl position.",
            "fragment_smiles": "[*:1]F",
            "knowledge_gaps": ["No affinity prediction has been performed."],
        }
