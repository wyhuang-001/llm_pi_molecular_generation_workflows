from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Protocol

from rdkit import Chem

from .adapters import NotConfiguredAdapter, configured_adapters
from .editing import EditResult, apply_transformation, write_sdf
from .fragment_library import FragmentLibrary
from .models import AgentState, ToolObservation
from .structure import ComplexContext
from .tools import ToolRegistry


class DecisionClient(Protocol):
    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class ReadyDecisionError(RuntimeError):
    def __init__(
        self,
        decision: dict[str, Any],
        error: str,
        failure_class: str = "invalid_ready",
        instruction: str = "Return READY with concrete valid fields; do not use placeholders.",
    ):
        self.rejection = {
            "status": "rejected",
            "failure_class": failure_class,
            "decision": decision,
            "error": error,
            "recommended_queries": [],
            "instruction": instruction,
        }
        super().__init__(error)


class DuplicateToolCallError(RuntimeError):
    def __init__(self, rejection: dict[str, Any]):
        self.rejection = rejection
        super().__init__(rejection["error"])


class ReadyEvidenceError(ReadyDecisionError):
    def __init__(self, transformation: dict[str, Any], missing: list[str], queries: list[dict[str, Any]]):
        knowledge_evidence = {
            "selected fragment library record",
            "selected fragment properties",
            "selected fragment spatial profile",
        }
        requires_llm_review = bool(knowledge_evidence.intersection(missing))
        self.rejection = {
            "status": "rejected",
            "failure_class": "ready_evidence_missing",
            "transformation": transformation,
            "missing_evidence": missing,
            "recommended_queries": queries,
            "requires_llm_review": requires_llm_review,
            "instruction": (
                "The transformation is not rejected chemically. Execute the listed fragment knowledge "
                "queries, inspect their results, and then choose whether to validate and resubmit this "
                "transformation or select a better evidence-backed alternative."
                if requires_llm_review else
                "The transformation is not rejected chemically. Execute the listed missing evidence "
                "queries, then return READY with the same transformation only after all evidence exists."
            ),
        }
        RuntimeError.__init__(
            self,
            f"Selected transformation {transformation} lacks required evidence: "
            + ", ".join(missing),
        )


class Workflow:
    def __init__(
        self,
        task_path: Path,
        client: DecisionClient,
        run_dir: Path,
        config_path: Path | None = None,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.context = ComplexContext(task_path)
        self.client = client
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress = progress
        self.parent_candidates: dict[int, Chem.Mol] = {
            0: Chem.Mol(self.context.ligand)
        }
        self.parent_metadata: dict[int, dict[str, Any]] = {
            0: {"attempt": 0, "generation": 0, "target_type": None, "target_id": None}
        }
        library_path = self.context.task.get("fragment_library_path")
        if library_path:
            library_path = (self.context.input_dir / str(library_path)).resolve()
        self.tools = ToolRegistry(
            self.context,
            FragmentLibrary(library_path) if library_path else None,
            parent_resolver=self._resolve_parent_candidate,
        )
        if config_path is not None:
            self.docking_adapter, self.rbfe_adapter = configured_adapters(
                config_path.resolve(), self.run_dir, progress=self._emit
            )
        else:
            self.docking_adapter = NotConfiguredAdapter("docking")
            self.rbfe_adapter = NotConfiguredAdapter("rbfe")
        self.state = AgentState(
            task=self.context.task["task"],
            max_context_rounds=int(self.context.task.get("max_context_rounds", 8)),
        )
        self.reference_docking_result: dict[str, Any] | None = None
        self._design_phase = False

    def _resolve_parent_candidate(self, parent_attempt: int | None) -> Chem.Mol:
        attempt = 0 if parent_attempt is None else int(parent_attempt)
        try:
            return Chem.Mol(self.parent_candidates[attempt])
        except KeyError as error:
            raise ValueError(f"Unknown parent_attempt: {attempt}") from error

    def _parent_metadata_for(self, parent_attempt: int | None) -> dict[str, Any]:
        attempt = 0 if parent_attempt is None else int(parent_attempt)
        try:
            return dict(self.parent_metadata[attempt])
        except KeyError as error:
            raise ValueError(f"Unknown parent_attempt: {attempt}") from error

    def _emit(self, event: str, details: dict[str, Any] | None = None) -> None:
        if self.progress:
            self.progress(event, details or {})

    @staticmethod
    def _llm_safe_value(value: Any, key: str | None = None) -> Any:
        omitted_keys = {
            "source_molecule_ids", "command", "command_display", "stdout", "stderr",
            "raw_http_body", "poses",
        }
        if isinstance(value, dict):
            return {
                item_key: Workflow._llm_safe_value(item_value, item_key)
                for item_key, item_value in value.items()
                if item_key not in omitted_keys
            }
        if isinstance(value, list):
            limit = 20 if key == "fragments" else 50
            return [Workflow._llm_safe_value(item) for item in value[:limit]]
        if isinstance(value, str) and len(value) > 4000:
            return value[:4000] + "... [truncated for LLM context]"
        return value

    @staticmethod
    def _compact_transformation(transformation: dict[str, Any] | None) -> dict[str, Any]:
        transformation = transformation or {}
        fields = (
            "operation", "edit_atom_index", "replacement_site_id", "fragment_id",
            "fragment_smiles", "cut_bond", "parent_attempt", "generation",
            "replace_existing_substituent",
        )
        return {
            key: transformation[key]
            for key in fields
            if transformation.get(key) is not None
        }

    @classmethod
    def _transformation_key(cls, transformation: dict[str, Any] | None) -> str:
        normalized = cls._exploration_transformation(transformation or {})
        # Fragment IDs preserve provenance but do not make the same chemical edit
        # a distinct transformation. Match duplicate-protection semantics.
        normalized.pop("fragment_id", None)
        normalized.pop("generation", None)
        cut_bond = cls._normalize_cut_bond(normalized.get("cut_bond"))
        if cut_bond is not None:
            normalized["cut_bond"] = list(cut_bond)
        return json.dumps(normalized, sort_keys=True)

    def _compact_observation_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Keep actionable tool evidence while omitting bulky raw tool payloads."""
        compact = self._tool_result_summary(result)
        for key in (
            "sites", "fragments", "candidates", "rejected", "residues", "contacts",
            "direction_profiles", "representative_conformer_atoms", "properties",
            "transformation", "candidate", "parent", "property_delta",
            "replacement_site", "recommended_queries",
        ):
            if key not in result:
                continue
            value = result[key]
            if key == "transformation" and isinstance(value, dict):
                compact[key] = self._compact_transformation(value)
            else:
                compact[key] = self._llm_safe_value(value, key)
        return compact

    def _llm_observation_view(self) -> list[dict[str, Any]]:
        # Context collection is already bounded by max_context_rounds. During the
        # design loop retain baseline evidence, active-target evidence, and a
        # recent window; complete observations remain in the audit files.
        if not self._design_phase:
            selected = list(self.state.observations)
        else:
            baseline_tools = {
                "get_ligand_info", "get_pocket_residues", "detect_basic_interactions",
                "get_edit_site_candidates", "list_fragment_replacement_sites",
                "assess_edit_sites",
            }
            active = self.state.active_target or {}
            active_type = active.get("target_type")
            active_id = active.get("target_id")

            def relevant(item: ToolObservation) -> bool:
                if item.tool in baseline_tools:
                    return True
                arguments = item.arguments or {}
                result = item.result or {}
                if active_type == "atom":
                    return active_id in {
                        arguments.get("atom_index"), arguments.get("edit_atom_index"),
                        arguments.get("target_id"), result.get("atom_index"), result.get("target_id"),
                    }
                if active_type == "replacement_site":
                    return active_id in {
                        arguments.get("replacement_site_id"), arguments.get("target_id"),
                        result.get("replacement_site_id"), result.get("target_id"),
                    }
                return False

            selected = [item for item in self.state.observations if relevant(item)]
            selected.extend(self.state.observations[-12:])

        latest: dict[str, tuple[int, ToolObservation]] = {}
        positions = {id(item): index for index, item in enumerate(self.state.observations)}
        for item in selected:
            latest[self._signature(item.tool, item.arguments)] = (positions[id(item)], item)
        ordered = [item for _index, item in sorted(latest.values(), key=lambda pair: pair[0])]
        if len(ordered) > 32:
            ordered = ordered[-32:]
        return [
            {
                "tool": item.tool,
                "arguments": item.arguments,
                "result": self._compact_observation_result(item.result),
                "evidence": sorted(item.evidence),
            }
            for item in ordered
        ]

    def _compact_docking_history(self) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in self.state.docking_history:
            comparison = item.get("comparison") or {}
            metrics = comparison.get("metrics") or {}
            compact_metrics = {}
            for name, metric in metrics.items():
                if not isinstance(metric, dict):
                    continue
                delta = metric.get("delta_candidate_minus_reference") or {}
                compact_metrics[name] = {
                    "direction": metric.get("direction"),
                    "delta_mean": delta.get("mean"),
                    "delta_stddev": delta.get("stddev"),
                    "candidate_better_seed_count": metric.get("candidate_better_seed_count"),
                    "candidate_better_seed_fraction": metric.get("candidate_better_seed_fraction"),
                }
            pose = item.get("pose_consensus") or {}
            interactions = item.get("interaction_consensus") or {}
            compact.append({
                "attempt": item.get("attempt"),
                "design_region": item.get("design_region"),
                "transformation": self._compact_transformation(item.get("transformation")),
                "status": item.get("status"),
                "primary_metric": item.get("primary_metric"),
                "delta_candidate_minus_reference": item.get("delta_candidate_minus_reference"),
                "seed_stddev": item.get("seed_stddev"),
                "seed_win_fraction": item.get("seed_win_fraction"),
                "quality": item.get("quality"),
                "raw_quality_from_mean": item.get("raw_quality_from_mean"),
                "is_new_best": item.get("is_new_best"),
                "best_quality_so_far": item.get("best_quality_so_far"),
                "comparison": {"status": comparison.get("status"), "metrics": compact_metrics},
                "pose_consensus": {
                    "stable": pose.get("stable"),
                    "mean_pairwise_rmsd": pose.get("mean_pairwise_rmsd"),
                    "max_pairwise_rmsd": pose.get("max_pairwise_rmsd"),
                    "largest_consistent_cluster_fraction": pose.get("largest_consistent_cluster_fraction"),
                },
                "interaction_consensus": {
                    "gained_consensus_residues": interactions.get("gained_consensus_residues", []),
                    "lost_consensus_residues": interactions.get("lost_consensus_residues", []),
                },
            })
        return compact

    def _compact_candidate_history(self) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in self.state.candidate_history:
            validation = item.get("validation") or {}
            docking = item.get("docking") or {}
            compact.append({
                "attempt": item.get("attempt"),
                "candidate_id": item.get("candidate_id"),
                "parent_attempt": item.get("parent_attempt"),
                "generation": item.get("generation", 1),
                "candidate_path": item.get("candidate_path"),
                "record_type": item.get("record_type"),
                "transformation": self._compact_transformation(item.get("transformation")),
                "validation": {
                    key: validation.get(key)
                    for key in (
                        "status", "failure_class", "canonical_smiles", "property_delta",
                        "severe_clash_count", "formal_charge", "heavy_atoms", "molecular_weight",
                    )
                    if key in validation
                },
                "docking": {
                    "status": docking.get("status"),
                    "entered_docking": docking.get("entered_docking"),
                    "completed": docking.get("completed"),
                    "primary_metric": docking.get("primary_metric"),
                    "primary_metric_summary": docking.get("primary_metric_summary"),
                    "pose_consensus": {
                        key: (docking.get("pose_consensus") or {}).get(key)
                        for key in ("stable", "mean_pairwise_rmsd", "max_pairwise_rmsd")
                        if key in (docking.get("pose_consensus") or {})
                    },
                    "interaction_consensus": {
                        key: (docking.get("interaction_consensus") or {}).get(key, [])
                        for key in ("gained_consensus_residues", "lost_consensus_residues")
                    },
                },
            })
        return compact

    def _compact_convergence(self) -> dict[str, Any]:
        return {
            key: self.state.convergence.get(key)
            for key in (
                "status", "converged", "llm_controls_termination", "stop_authority",
                "best_attempt", "best_quality", "non_improving_attempts", "termination_reason",
            )
            if key in self.state.convergence
        }

    def _llm_state_view(self) -> dict[str, Any]:
        return {
            "task": self.state.task,
            "round": len(self.state.observations),
            "max_rounds": self.state.max_context_rounds,
            "covered_evidence": sorted(self.state.evidence),
            "missing_site_evidence": self.state.missing_evidence,
            "decisions": self._llm_safe_value(self.state.decisions[-12:]),
            "observations": self._llm_observation_view(),
            "tool_rejections": self._llm_safe_value(self.state.tool_rejections[-8:]),
            "active_target": self._llm_safe_value(self.state.active_target),
            "site_search": self._llm_safe_value(self.state.site_search),
            "convergence": self._compact_convergence(),
        }

    @staticmethod
    def _tool_result_summary(result: dict[str, Any]) -> dict[str, Any]:
        summary = {
            key: result[key]
            for key in (
                "status", "count", "radius", "cutoff", "atom_index", "element",
                "replaceable_hydrogens", "minimum_clearance", "severe_clash_count",
                "failure_class", "error", "canonical_smiles", "heavy_atoms",
                "molecular_weight", "formal_charge", "replacement_site_id",
                "max_probe_distance", "probe_count", "fragment_id", "fragment_smiles",
                "size_class", "chemical_tag", "chemical_tags", "allowed_operations",
                "conformer_count", "max_attachment_distance", "maximum_forward_extent",
                "maximum_radial_extent", "radius_of_gyration",
                "site_count", "accepted_count", "rejected_count", "target_type", "target_id",
                "filtered_compatible_records", "operation_compatible_records",
            )
            if key in result
        }
        for key in (
            "residues", "contacts", "protein_atoms", "fragments", "sites",
            "size_class_counts", "chemical_tag_counts",
            "direction_profiles", "representative_conformer_atoms",
        ):
            value = result.get(key)
            if isinstance(value, list):
                summary[f"{key}_count"] = len(value)
        return summary

    def _signature(self, tool: str, arguments: dict[str, Any]) -> str:
        normalized_arguments = dict(arguments)
        if tool == "validate_candidate_geometry":
            normalized_arguments.setdefault("operation", "replace_hydrogen")
            if "atom_index" in normalized_arguments and "edit_atom_index" not in normalized_arguments:
                normalized_arguments["edit_atom_index"] = normalized_arguments.pop("atom_index")
        data = json.dumps(
            {"tool": tool, "arguments": normalized_arguments},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def _exploration_transformation(transformation: dict[str, Any]) -> dict[str, Any]:
        operation = transformation.get("operation", "replace_hydrogen")
        normalized = {
            "operation": operation,
            "fragment_smiles": transformation.get("fragment_smiles"),
        }
        parent_attempt = transformation.get("parent_attempt")
        if isinstance(parent_attempt, int) and parent_attempt > 0:
            normalized["parent_attempt"] = parent_attempt
        generation = transformation.get("generation")
        if isinstance(generation, int) and generation >= 0:
            normalized["generation"] = generation
        fragment_id = transformation.get("fragment_id")
        if isinstance(fragment_id, str) and fragment_id:
            normalized["fragment_id"] = fragment_id
        if operation == "replace_fragment":
            normalized["replacement_site_id"] = transformation.get("replacement_site_id")
            cut_bond = transformation.get("cut_bond")
            if isinstance(cut_bond, (list, tuple)) and len(cut_bond) == 2:
                normalized["cut_bond"] = list(cut_bond)
            edit_index = transformation.get("edit_atom_index")
            if isinstance(edit_index, int):
                normalized["edit_atom_index"] = edit_index
        else:
            normalized["edit_atom_index"] = transformation.get(
                "edit_atom_index", transformation.get("atom_index")
            )
        return normalized

    def _record_exploration_attempt(
        self,
        transformation: dict[str, Any],
        status: str,
        source: str,
        *,
        attempt: int | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        normalized = self._exploration_transformation(transformation)
        operation = normalized.get("operation", "replace_hydrogen")
        target_type = "replacement_site" if operation == "replace_fragment" else "atom"
        target_id = (
            normalized.get("replacement_site_id")
            if target_type == "replacement_site"
            else normalized.get("edit_atom_index")
        )
        record = {
            "event": len(self.state.exploration_attempts) + 1,
            "source": source,
            "status": status,
            "target_type": target_type,
            "target_id": target_id,
            "family": self._modification_family(normalized),
            "transformation": normalized,
        }
        if attempt is not None:
            record["attempt"] = attempt
        parent_attempt = normalized.get("parent_attempt")
        if isinstance(parent_attempt, int) and parent_attempt > 0:
            record["parent_attempt"] = parent_attempt
            record["generation"] = normalized.get("generation", 1)
        if reason:
            record["reason"] = reason
        self.state.exploration_attempts.append(record)
        return record

    @staticmethod
    def _update_exploration_attempt(
        record: dict[str, Any],
        status: str,
        *,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        record["status"] = status
        if reason:
            record["reason"] = reason
        if details:
            record["details"] = details

    def _query_payload(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "mode": "context_collection",
            "state": self._llm_state_view(),
            "structure_input": {
                "complex_path": str(self.context.complex_path),
                "ligand_selector": self.context.ligand_selector,
                "ligand_source": self.context.ligand_source,
            },
            "tool_catalog": self.tools.catalog(),
            "host_ready": self.state.ready,
            "instruction": (
                "Choose what knowledge is needed next. When site_strategy_required is enabled, first use "
                "the chemical tools to inspect the ligand, pocket, interactions, and replacement sites. Call "
                "get_edit_site_candidates to obtain one structured host-supported site dossier, then call "
                "assess_edit_sites only as a QUERY decision: action must be QUERY, tool must be "
                "assess_edit_sites, and sites plus global_rationale must be nested inside arguments. Include an "
                "evidence-backed priority and site_type for each plausible host target. The host will lock the "
                "highest-priority open target during design. Use the chemical "
                "tools to compare replace_hydrogen and replace_fragment, query spatial facts when needed, then "
                "return READY only when the evidence supports the selected operation and final edit site."
            ),
        }
        payload["optimization_context"] = self._optimization_context()
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
        action = decision.get("action")
        details: dict[str, Any] = {"decision": decision_number, "action": action}
        if action == "QUERY":
            details.update({"tool": decision.get("tool"), "arguments": decision.get("arguments")})
        elif action == "QUERY_BATCH":
            details["queries"] = decision.get("queries")
        elif action == "READY":
            details.update({
                "operation": decision.get("operation", "replace_hydrogen"),
                "edit_atom_index": decision.get("edit_atom_index"),
                "replacement_site_id": decision.get("replacement_site_id"),
                "cut_bond": decision.get("cut_bond"),
                "fragment_id": decision.get("fragment_id"),
                "fragment_smiles": decision.get("fragment_smiles"),
                "hypothesis": decision.get("edit_hypothesis"),
            })
        elif action == "MARK_UNMODIFIABLE":
            details.update({
                "target_type": decision.get("target_type"),
                "target_id": decision.get("target_id"),
                "scope": decision.get("scope"),
                "family": decision.get("family"),
                "reason": decision.get("reason"),
            })
        elif action == "STOP":
            details["reason"] = decision.get("reason")
        self._emit("decision", details)
        return decision_number

    def _auto_complete_ready_evidence(self, rejection: dict[str, Any]) -> None:
        requires_review = bool(rejection.get("requires_llm_review"))
        for query in rejection.get("recommended_queries", []):
            if not isinstance(query, dict):
                continue
            tool = query.get("tool")
            arguments = query.get("arguments")
            if not isinstance(tool, str) or not isinstance(arguments, dict):
                continue
            if requires_review and tool == "validate_candidate_geometry":
                continue
            signature = self._signature(tool, arguments)
            if signature in self.state.call_signatures:
                continue
            try:
                self._execute_query({"action": "QUERY", "tool": tool, "arguments": arguments})
            except Exception as error:
                auto_rejection = {
                    "status": "rejected",
                    "failure_class": "auto_evidence_query_failed",
                    "tool": tool,
                    "arguments": arguments,
                    "error": str(error),
                }
                self.state.tool_rejections.append(auto_rejection)
                self._emit("ready_evidence_query_failed", auto_rejection)

    def _existing_observation_summary(
        self, tool: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        signature = self._signature(tool, arguments)
        for item in self.state.observations:
            if self._signature(item.tool, item.arguments) == signature:
                return {
                    "tool": item.tool,
                    "arguments": item.arguments,
                    "evidence": sorted(item.evidence),
                    "result": self._tool_result_summary(item.result),
                }
        return None

    def _duplicate_tool_rejection(
        self,
        tool: str,
        arguments: dict[str, Any],
        source: str = "previous_observation",
    ) -> dict[str, Any]:
        existing = self._existing_observation_summary(tool, arguments)
        return {
            "status": "rejected",
            "failure_class": "duplicate_tool_call",
            "tool": tool,
            "arguments": arguments,
            "duplicate_source": source,
            "existing_observation": existing,
            "error": (
                "Duplicate tool call blocked; this exact tool call was already executed and was not run again: "
                f"{tool} {arguments}"
            ),
            "instruction": (
                "Do not repeat this tool and arguments. The existing observation is authoritative. "
                "Return a genuinely new unexecuted QUERY, a chemically distinct READY transformation, "
                "or STOP."
            ),
        }

    def _execute_query(self, decision: dict[str, Any]) -> None:
        tool = decision.get("tool")
        arguments = decision.get("arguments")
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            raise RuntimeError("QUERY requires string tool and object arguments")
        signature = self._signature(tool, arguments)
        if signature in self.state.call_signatures:
            rejection = self._duplicate_tool_rejection(tool, arguments)
            self.state.tool_rejections.append(rejection)
            self._emit("tool_call_reused", rejection)
            return
        if tool == "assess_edit_sites" and not any(
            item.tool == "get_edit_site_candidates" for item in self.state.observations
        ):
            raise RuntimeError(
                "assess_edit_sites requires a prior get_edit_site_candidates observation so priorities and "
                "site types are grounded in one host-supported site dossier"
            )
        if tool == "assess_edit_sites" and self._design_phase and self.state.exploration_attempts:
            raise RuntimeError(
                "assess_edit_sites is only allowed before design attempts; the active site strategy cannot "
                "be reordered after local search has started"
            )
        if tool == "generate_site_candidate_batch" and self._design_phase:
            active = self.state.active_target
            proposed = {
                "target_type": arguments.get("target_type"),
                "target_id": arguments.get("target_id"),
            }
            if active is not None and proposed != {
                "target_type": active.get("target_type"),
                "target_id": active.get("target_id"),
            }:
                raise RuntimeError(
                    "generate_site_candidate_batch may only target the current active prioritized site"
                )
        self._emit("tool_started", {"tool": tool, "arguments": arguments})
        result, evidence = self.tools.execute(tool, arguments)
        self.state.call_signatures.add(signature)
        self.state.evidence.update(evidence)
        self.state.observations.append(
            ToolObservation(tool=tool, arguments=arguments, result=result, evidence=evidence)
        )
        if tool == "assess_edit_sites" and result.get("status") == "complete":
            self.state.site_strategy = result
            self.state.active_target = None
            self.state.site_search = {}
            self._refresh_site_search()
            self._write_json("site-strategy.json", result)
            self._emit("site_strategy_updated", {
                "site_count": result.get("site_count"),
                "active_target": self.state.active_target,
            })
        if tool == "validate_candidate_geometry":
            transformation = result.get("transformation") or arguments
            status = "geometry_accepted" if result.get("status") == "accepted" else "geometry_rejected"
            self._record_exploration_attempt(
                transformation,
                status,
                "validate_candidate_geometry",
                reason=result.get("error") if result.get("status") != "accepted" else None,
            )
        if tool == "generate_site_candidate_batch" and result.get("status") == "complete":
            self._record_candidate_batch_exploration(result)
        self._write_json(
            f"observation-{len(self.state.observations):02d}.json",
            self.state.compact_view(),
        )
        self._emit("tool_completed", {
            "tool": tool,
            "evidence": sorted(evidence),
            "result": self._tool_result_summary(result),
        })

    def _record_candidate_batch_exploration(self, result: dict[str, Any]) -> None:
        """Record batch-prescreened transformations without turning them into docking evidence."""
        target_type = result.get("target_type")
        target_id = result.get("target_id")
        for item in (result.get("candidates") or []):
            transformation = dict(item.get("transformation") or {})
            if not transformation:
                transformation = {
                    "operation": "replace_hydrogen" if target_type == "atom" else "replace_fragment",
                    "fragment_id": item.get("fragment_id"),
                    "fragment_smiles": item.get("fragment_smiles"),
                }
                if target_type == "atom":
                    transformation["edit_atom_index"] = target_id
                else:
                    transformation["replacement_site_id"] = target_id
            self._record_exploration_attempt(
                transformation,
                "batch_geometry_accepted",
                "candidate_batch",
            )
        for item in (result.get("rejected") or []):
            transformation = dict(item.get("transformation") or {})
            if not transformation:
                transformation = {
                    "operation": "replace_hydrogen" if target_type == "atom" else "replace_fragment",
                    "fragment_id": item.get("fragment_id"),
                    "fragment_smiles": item.get("fragment_smiles"),
                }
                if target_type == "atom":
                    transformation["edit_atom_index"] = target_id
                else:
                    transformation["replacement_site_id"] = target_id
            self._record_exploration_attempt(
                transformation,
                "geometry_rejected",
                "candidate_batch",
                reason=item.get("error") or item.get("failure_class"),
            )

    @staticmethod
    def _unwrap_decision(decision: Any) -> Any:
        """Accept a single common response envelope around the workflow decision."""
        current = decision
        for _ in range(2):
            if not isinstance(current, dict) or "action" in current:
                return current
            wrapped = next(
                (
                    current[key]
                    for key in ("answer", "decision", "response")
                    if isinstance(current.get(key), dict)
                ),
                None,
            )
            if wrapped is None:
                return current
            current = wrapped
        return current

    @staticmethod
    def _has_valid_action(decision: Any) -> bool:
        return isinstance(decision, dict) and decision.get("action") in {
            "QUERY",
            "QUERY_BATCH",
            "READY",
            "MARK_UNMODIFIABLE",
            "PROPOSE_TOOL",
            "STOP",
        }

    @staticmethod
    def _contains_transformation_fields(decision: Any) -> bool:
        """Return whether an invalid decision expresses transformation intent.

        Completeness is deliberately not checked here. Missing or invalid
        transformation fields belong to the normal READY validation path.
        """
        if not isinstance(decision, dict):
            return False
        if "operation" in decision:
            return True
        has_fragment = "fragment_smiles" in decision or "fragment_id" in decision
        has_anchor = "replacement_site_id" in decision or "edit_atom_index" in decision
        return has_fragment and has_anchor

    @staticmethod
    def _optional_identifier(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if not normalized or normalized.lower() in {"optional", "none", "null", "n/a"}:
            return None
        return normalized

    def _transformation(self, decision: dict[str, Any]) -> dict[str, Any]:
        operation = decision.get("operation", "replace_hydrogen")
        fragment_id = self._optional_identifier(decision.get("fragment_id"))
        parent_attempt = decision.get("parent_attempt")
        if parent_attempt is not None and (
            not isinstance(parent_attempt, int) or parent_attempt < 1
        ):
            raise RuntimeError("parent_attempt must be a positive attempt number")
        if parent_attempt is not None:
            self._parent_metadata_for(parent_attempt)
        transformation = {
            "operation": operation,
            "fragment_smiles": decision.get("fragment_smiles"),
            "parent_attempt": parent_attempt,
            "generation": (
                self._parent_metadata_for(parent_attempt).get("generation", 0) + 1
                if parent_attempt is not None else 1
            ),
            "replace_existing_substituent": parent_attempt is not None,
        }
        if fragment_id:
            transformation["fragment_id"] = fragment_id
            record = self.tools.fragment_library.get(fragment_id)
            library_operation = "substitute" if operation == "replace_hydrogen" else operation
            if not self.tools.fragment_library.allows_operation(record, library_operation):
                raise RuntimeError(
                    f"Fragment {fragment_id} does not allow operation {library_operation}"
                )
            if not transformation["fragment_smiles"]:
                transformation["fragment_smiles"] = record["smiles"]
            elif not self.tools.fragment_library.smiles_equivalent(
                transformation["fragment_smiles"], record["smiles"]
            ):
                raise RuntimeError(
                    f"fragment_id {fragment_id} does not match fragment_smiles"
                )
            else:
                transformation["fragment_smiles"] = record["smiles"]
            transformation["library_record"] = record
        if operation == "replace_fragment":
            site_id = decision.get("replacement_site_id")
            if not isinstance(site_id, str):
                raise RuntimeError(
                    "replace_fragment requires replacement_site_id from "
                    "list_fragment_replacement_sites; direct cut_bond input is not accepted"
                )
            site = self.tools.resolve_replacement_site(site_id)
            transformation["replacement_site_id"] = site_id
            transformation["replacement_site"] = site
            transformation["cut_bond"] = site["cut_bond"]
            transformation["edit_atom_index"] = site["retained_atom_index"]
        else:
            transformation["edit_atom_index"] = decision.get("edit_atom_index")
        if parent_attempt is None:
            transformation.pop("parent_attempt", None)
            transformation.pop("replace_existing_substituent", None)
        if not isinstance(transformation.get("fragment_smiles"), str):
            raise RuntimeError("READY requires fragment_smiles or a valid fragment_id")
        return transformation

    @staticmethod
    def _normalize_cut_bond(value: Any) -> tuple[int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        if not all(isinstance(item, int) for item in value):
            return None
        return tuple(sorted(value))

    @classmethod
    def _same_transformation(cls, left: dict[str, Any], right: dict[str, Any]) -> bool:
        return cls._transformation_key(left) == cls._transformation_key(right)

    def _transformation_was_attempted(self, transformation: dict[str, Any]) -> bool:
        return any(
            self._same_transformation(item.get("transformation", {}), transformation)
            and (
                item.get("source") == "design"
                or item.get("status") in {
                    "geometry_rejected",
                    "duplicate_structure",
                    "docking_failed",
                }
            )
            for item in self.state.exploration_attempts
        ) or any(
            self._same_transformation(item.get("transformation", {}), transformation)
            for item in self.state.candidate_history
        )

    def _unmodifiable_scope(
        self, target_type: str, target_id: Any, family: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            item for item in self.state.unmodifiable_targets
            if item.get("target_type") == target_type
            and item.get("target_id") == target_id
            and (
                item.get("scope") == "site"
                if family is None
                else item.get("scope") == "site" or item.get("family") == family
            )
        ]

    def _is_unmodifiable(
        self, target_type: str, target_id: Any, family: str | None = None
    ) -> bool:
        return bool(self._unmodifiable_scope(target_type, target_id, family))

    def _distinct_target_transformations(
        self, target_type: str, target_id: Any
    ) -> set[str]:
        transformations: set[str] = set()
        for item in self.state.exploration_attempts:
            if item.get("source") == "MARK_UNMODIFIABLE":
                continue
            transformation = item.get("transformation") or {}
            operation = transformation.get("operation", "replace_hydrogen")
            matches = (
                target_type == "atom"
                and operation == "replace_hydrogen"
                and transformation.get("edit_atom_index") == target_id
                or target_type == "replacement_site"
                and operation == "replace_fragment"
                and transformation.get("replacement_site_id") == target_id
            )
            if matches:
                transformations.add(self._transformation_key(transformation))
        return transformations

    def _record_unmodifiable(self, decision: dict[str, Any]) -> bool:
        target_type = decision.get("target_type")
        target_id = decision.get("target_id")
        scope = decision.get("scope")
        family = decision.get("family")
        reason = decision.get("reason")
        valid_types = {"atom", "replacement_site"}
        valid_families = {"halogen", "non_halogen", "fragment_replacement"}
        if target_type not in valid_types:
            raise RuntimeError("MARK_UNMODIFIABLE target_type must be atom or replacement_site")
        if target_type == "atom":
            if not isinstance(target_id, int) or target_id not in {
                atom.GetIdx() for atom in self.context.ligand.GetAtoms()
                if atom.GetAtomicNum() > 1 and atom.GetTotalNumHs() > 0
            }:
                raise RuntimeError("MARK_UNMODIFIABLE atom target_id must be a heavy atom with hydrogen")
        else:
            known_sites = {
                site["replacement_site_id"]
                for site in self.tools.list_fragment_replacement_sites(limit=100).get("sites", [])
            }
            if not isinstance(target_id, str) or target_id not in known_sites:
                raise RuntimeError("MARK_UNMODIFIABLE replacement_site target_id is not host-enumerated")
        if scope not in {"site", "family"}:
            raise RuntimeError("MARK_UNMODIFIABLE scope must be site or family")
        if scope == "family" and family not in valid_families:
            raise RuntimeError("MARK_UNMODIFIABLE family must be halogen, non_halogen, or fragment_replacement")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError("MARK_UNMODIFIABLE requires a non-empty reason")
        existing = next(
            (
                item for item in self.state.unmodifiable_targets
                if item.get("target_type") == target_type
                and item.get("target_id") == target_id
                and (
                    item.get("scope") == "site"
                    or scope == "family"
                    and item.get("scope") == "family"
                    and item.get("family") == family
                )
            ),
            None,
        )
        if existing is not None:
            self._refresh_site_search()
            active = self.state.active_target
            active_label = (
                f"{active.get('target_type')}:{active.get('target_id')}"
                if active else "none"
            )
            rejection = {
                "status": "reused",
                "failure_class": "duplicate_unmodifiable_declaration",
                "decision": decision,
                "existing_declaration": existing,
                "active_target": active,
                "error": (
                    f"{target_type}:{target_id} is already closed by an accepted "
                    "MARK_UNMODIFIABLE declaration"
                ),
                "instruction": (
                    "Do not repeat the closed target declaration. Its prior closure remains authoritative. "
                    f"Continue with the current active target {active_label}, using a new QUERY, READY "
                    "transformation, or a valid closure for that active target."
                ),
            }
            self.state.tool_rejections.append(rejection)
            self._emit("unmodifiable_declaration_reused", rejection)
            return False
        policy = self._search_policy()
        if policy["site_lock_enabled"] and self._design_phase:
            self._refresh_site_search()
            active = self.state.active_target
            if active is None:
                raise RuntimeError("MARK_UNMODIFIABLE requires an active prioritized target")
            if (
                target_type != active.get("target_type")
                or target_id != active.get("target_id")
            ):
                raise RuntimeError(
                    "MARK_UNMODIFIABLE may only close the current active prioritized target: "
                    f"{active.get('target_type')}:{active.get('target_id')}"
                )
        # The LLM closes a site only after reviewing tool and docking evidence.
        record = {
            "event": len(self.state.unmodifiable_targets) + 1,
            "target_type": target_type,
            "target_id": target_id,
            "scope": scope,
            "family": family if scope == "family" else None,
            "reason": reason.strip(),
        }
        self.state.unmodifiable_targets.append(record)
        attempt_record = self._record_exploration_attempt(
            {
                "operation": "replace_fragment" if target_type == "replacement_site" else "replace_hydrogen",
                "replacement_site_id": target_id if target_type == "replacement_site" else None,
                "edit_atom_index": target_id if target_type == "atom" else None,
                "fragment_smiles": None,
            },
            "llm_unmodifiable",
            "MARK_UNMODIFIABLE",
            reason=reason.strip(),
        )
        if scope == "family" and isinstance(family, str):
            attempt_record["family"] = family
        self._refresh_site_search()
        return True

    @staticmethod
    def _docking_feedback_summary(docking: dict[str, Any]) -> dict[str, Any]:
        comparison = docking.get("comparison") or {}
        return {
            "status": docking.get("status"),
            "seed_count": docking.get("seed_count"),
            "seeds": docking.get("seeds"),
            "pose_count": docking.get("pose_count"),
            "pose_count_per_seed": docking.get("pose_count_per_seed"),
            "total_pose_count": docking.get("total_pose_count"),
            "metrics": comparison.get("metrics"),
            "pose_consensus": docking.get("pose_consensus"),
            "interaction_consensus": docking.get("interaction_consensus"),
            "candidate_per_seed": {
                seed: {
                    "status": result.get("status"),
                    "top_pose": result.get("top_pose"),
                    "pose_selection": result.get("pose_selection"),
                    "audit_path": result.get("audit_path"),
                }
                for seed, result in (docking.get("candidate_per_seed") or {}).items()
            },
            "reference_per_seed": {
                seed: {
                    "status": result.get("status"),
                    "top_pose": result.get("top_pose"),
                    "pose_selection": result.get("pose_selection"),
                    "audit_path": result.get("audit_path"),
                }
                for seed, result in ((docking.get("reference_baseline") or {}).get("per_seed") or {}).items()
            },
            "limitation": (
                "This summary supports ranking and pose-consistency review. Full commands, logs, "
                "poses, and raw output remain in the referenced audit files."
            ),
        }

    def _search_policy(self) -> dict[str, Any]:
        configured = self.context.task.get("search_policy") or {}
        mode = str(configured.get("mode", "family_coverage"))
        return {
            "mode": mode,
            "site_lock_enabled": bool(configured.get("site_lock_enabled", False)),
            "site_strategy_required": bool(configured.get("site_strategy_required", False)),
            "minimum_prioritized_sites": int(configured.get("minimum_prioritized_sites", 1)),
            "local_patience": int(configured.get("local_patience", 3)),
        }

    @staticmethod
    def _target_key(target_type: str, target_id: Any) -> str:
        return f"{target_type}:{target_id}"

    @staticmethod
    def _transformation_target(transformation: dict[str, Any]) -> dict[str, Any]:
        if transformation.get("operation") == "replace_fragment":
            return {
                "target_type": "replacement_site",
                "target_id": transformation.get("replacement_site_id"),
            }
        return {
            "target_type": "atom",
            "target_id": transformation.get("edit_atom_index"),
        }

    def _strategy_sites(self) -> list[dict[str, Any]]:
        strategy = self.state.site_strategy or {}
        sites = strategy.get("sites") or []
        return sorted(
            (dict(item) for item in sites if isinstance(item, dict)),
            key=lambda item: int(item.get("priority", 10**9)),
        )

    def _site_exploration_records(
        self, target_type: str, target_id: Any
    ) -> list[dict[str, Any]]:
        """Return unique host exploration records for one target.

        Geometry screening and candidate-batch results are local search evidence,
        but they are not docking evidence. Keep the latest record for a
        transformation so a batch prescreen followed by docking counts once.
        """
        records: dict[str, dict[str, Any]] = {}
        for item in self.state.exploration_attempts:
            if item.get("source") == "MARK_UNMODIFIABLE":
                continue
            if item.get("target_type") != target_type or item.get("target_id") != target_id:
                continue
            transformation = item.get("transformation") or {}
            records[self._transformation_key(transformation)] = item
        return list(records.values())

    def _refresh_site_search(self) -> None:
        policy = self._search_policy()
        if not policy["site_lock_enabled"] or not self.state.site_strategy:
            self.state.active_target = None
            return
        settings = self._optimization_settings()
        minimum_improvement = settings["minimum_improvement"]
        refreshed: dict[str, dict[str, Any]] = {}
        for site in self._strategy_sites():
            target_type = site.get("target_type")
            target_id = site.get("target_id")
            key = self._target_key(target_type, target_id)
            attempts = self._site_exploration_records(target_type, target_id)
            docking = [
                item for item in self.state.docking_history
                if self._transformation_target(item.get("transformation") or {}) == {
                    "target_type": target_type,
                    "target_id": target_id,
                }
            ]
            families = sorted({
                self._local_modification_family(item.get("transformation") or {})
                for item in attempts
            })
            best_quality = None
            best_attempt = None
            non_improving = 0
            for item in docking:
                quality = item.get("quality")
                if not isinstance(quality, (int, float)):
                    non_improving += 1
                    continue
                if best_quality is None or float(quality) > best_quality + minimum_improvement:
                    best_quality = float(quality)
                    best_attempt = item.get("attempt")
                    non_improving = 0
                else:
                    if float(quality) > best_quality:
                        best_quality = float(quality)
                        best_attempt = item.get("attempt")
                    non_improving += 1
            if not docking:
                non_improving = len(attempts)
            explicitly_closed = self._is_unmodifiable(target_type, target_id)
            initial_status = site.get("search_status", "active")
            patience_reached = non_improving >= policy["local_patience"]
            previous = self.state.site_search.get(key) or {}
            status = (
                "closed"
                if explicitly_closed or initial_status == "hard-reject"
                else "pending"
            )
            refreshed[key] = {
                **site,
                "key": key,
                "status": status,
                "attempt_count": len(attempts),
                "attempt_count_definition": (
                    "unique host-recorded transformations, including geometry screening and batch evidence; "
                    "this is not a docking count"
                ),
                "geometry_accepted": sum(item.get("status") in {"geometry_accepted", "batch_geometry_accepted", "docked"} for item in attempts),
                "geometry_rejected": sum(item.get("status") == "geometry_rejected" for item in attempts),
                "docking_count": len(docking),
                "families": families,
                "best_quality": best_quality,
                "best_attempt": best_attempt,
                "non_improving_attempts": non_improving,
                "initial_search_status": initial_status,
                "local_patience": policy["local_patience"],
                "patience_reached": patience_reached,
                "previous_status": previous.get("status"),
            }
        active = next(
            (item for item in refreshed.values() if item["status"] == "pending"),
            None,
        )
        if active is not None:
            active["status"] = "active"
            self.state.active_target = {
                key: active[key]
                for key in (
                    "target_type", "target_id", "priority", "site_type", "rationale",
                    "search_status",
                )
                if key in active
            }
        else:
            self.state.active_target = None
        self.state.site_search = refreshed

    def _site_lock_rejection(self, transformation: dict[str, Any]) -> dict[str, Any] | None:
        policy = self._search_policy()
        if not policy["site_lock_enabled"] or not self._design_phase:
            return None
        self._refresh_site_search()
        if policy["site_strategy_required"] and not self.state.site_strategy:
            return {
                "status": "rejected",
                "failure_class": "site_strategy_missing",
                "instruction": (
                    "Before READY, inspect the ligand, pocket, interactions, and replacement sites, then call "
                    "assess_edit_sites with prioritized host target IDs and LLM-assigned site types."
                ),
            }
        strategy_count = len(self._strategy_sites())
        if strategy_count < policy["minimum_prioritized_sites"]:
            return {
                "status": "rejected",
                "failure_class": "site_strategy_too_small",
                "instruction": (
                    f"The site strategy contains {strategy_count} targets but at least "
                    f"{policy['minimum_prioritized_sites']} are required. Call assess_edit_sites again with "
                    "a broader evidence-backed prioritized target set."
                ),
            }
        active = self.state.active_target
        if active is None:
            return None
        proposed = self._transformation_target(transformation)
        active_target = {"target_type": active["target_type"], "target_id": active["target_id"]}
        if proposed == active_target:
            parent_attempt = transformation.get("parent_attempt")
            if parent_attempt is not None:
                parent = self._parent_metadata_for(parent_attempt)
                parent_target = {
                    "target_type": parent.get("target_type"),
                    "target_id": parent.get("target_id"),
                }
                if parent_target not in ({"target_type": None, "target_id": None}, active_target):
                    return {
                        "status": "rejected",
                        "failure_class": "parent_site_mismatch",
                        "parent_attempt": parent_attempt,
                        "parent_target": parent_target,
                        "proposed_target": proposed,
                        "instruction": (
                            "Local optimization must keep the parent candidate and child transformation on the "
                            "same active target. Choose a parent from the active target or omit parent_attempt "
                            "for a generation-1 edit of the original ligand."
                        ),
                    }
            return None
        active_status = self.state.site_search.get(
            self._target_key(active["target_type"], active["target_id"]), {}
        )
        return {
            "status": "rejected",
            "failure_class": "site_lock_violation",
            "active_target": active,
            "active_site_search": active_status,
            "proposed_target": proposed,
            "instruction": (
                "Continue the active target with a chemically distinct, tool-supported candidate batch or transformation. "
                "The host advances to the next prioritized site only after the LLM explicitly closes the active target "
                "with an evidence-backed MARK_UNMODIFIABLE decision. Patience is advisory and does not force a site switch."
            ),
        }

    def _adaptive_target_summaries(
        self, global_search: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        global_search = global_search or self._global_search_coverage()
        summaries: list[dict[str, Any]] = []
        targets = [
            ("atom", index, global_search["atom_clearance"].get(str(index)))
            for index in global_search["editable_hydrogen_atoms"]
        ] + [
            ("replacement_site", site_id, None)
            for site_id in global_search["replacement_sites"]
        ]
        for target_type, target_id, clearance in targets:
            records = (
                global_search["attempted_atoms"].get(str(target_id), [])
                if target_type == "atom"
                else global_search["attempted_replacement_sites"].get(target_id, [])
            )
            transformations = [
                item.get("transformation") or {} for item in records
                if item.get("transformation")
            ]
            docking = []
            for item in self.state.docking_history:
                transformation = item.get("transformation") or {}
                matches = (
                    target_type == "atom"
                    and transformation.get("edit_atom_index") == target_id
                    or target_type == "replacement_site"
                    and transformation.get("replacement_site_id") == target_id
                )
                if matches:
                    docking.append(item)
            scored = [item for item in docking if item.get("raw_quality_from_mean") is not None]
            best = max(
                (item for item in scored if item.get("quality") is not None),
                key=lambda item: float(item["quality"]),
                default=None,
            )
            strategy_item = next(
                (
                    item for item in self._strategy_sites()
                    if item.get("target_type") == target_type
                    and item.get("target_id") == target_id
                ),
                None,
            )
            summaries.append({
                "target_type": target_type,
                "target_id": target_id,
                "priority": strategy_item.get("priority") if strategy_item else None,
                "site_type": strategy_item.get("site_type") if strategy_item else None,
                "site_rationale": strategy_item.get("rationale") if strategy_item else None,
                "clearance": clearance,
                "closed": (
                    target_id in global_search["closed_atoms"]
                    if target_type == "atom"
                    else target_id in global_search["closed_replacement_sites"]
                ),
                "distinct_transformations": len({
                    json.dumps(self._exploration_transformation(item), sort_keys=True)
                    for item in transformations
                }),
                "transformations": self._llm_safe_value(transformations[-8:]),
                "geometry_accepted": sum(item.get("status") in {"geometry_accepted", "batch_geometry_accepted", "docked"} for item in records),
                "geometry_rejected": sum(item.get("status") == "geometry_rejected" for item in records),
                "geometry_feasible_not_docked": self._geometry_feasible_not_docked(target_type, target_id),
                "docking_count": len(docking),
                "docking_results": [
                    {
                        "attempt": item.get("attempt"),
                        "delta_candidate_minus_reference": item.get("delta_candidate_minus_reference"),
                        "seed_win_fraction": item.get("seed_win_fraction"),
                        "seed_stddev": item.get("seed_stddev"),
                        "quality": item.get("quality"),
                        "pose_stable": (item.get("pose_consensus") or {}).get("stable"),
                    }
                    for item in docking[-8:]
                ],
                "best_docking": (
                    {
                        "attempt": best.get("attempt"),
                        "delta_candidate_minus_reference": best.get("delta_candidate_minus_reference"),
                        "quality": best.get("quality"),
                        "seed_win_fraction": best.get("seed_win_fraction"),
                        "pose_stable": (best.get("pose_consensus") or {}).get("stable"),
                    }
                    if best else None
                ),
            })
        return summaries

    def _geometry_feasible_not_docked(self, target_type: str, target_id: Any) -> list[dict[str, Any]]:
        """Return host-accepted transformations that have not reached GNINA."""
        docked_keys = {
            self._transformation_key(item.get("transformation"))
            for item in self.state.docking_history
        }
        batch_details: dict[str, dict[str, Any]] = {}
        for observation in self.state.observations:
            if observation.tool != "generate_site_candidate_batch":
                continue
            result = observation.result or {}
            if result.get("target_type") != target_type or result.get("target_id") != target_id:
                continue
            for item in result.get("candidates") or []:
                transformation = item.get("transformation") or {}
                batch_details[self._transformation_key(transformation)] = {
                    "canonical_smiles": item.get("canonical_smiles"),
                    "fragment_properties": self._llm_safe_value(item.get("fragment_properties")),
                }
        candidates: dict[str, dict[str, Any]] = {}
        accepted_statuses = {"batch_geometry_accepted", "geometry_accepted"}
        for item in self.state.exploration_attempts:
            if item.get("target_type") != target_type or item.get("target_id") != target_id:
                continue
            if item.get("status") not in accepted_statuses:
                continue
            transformation = item.get("transformation") or {}
            key = self._transformation_key(transformation)
            if key in docked_keys:
                continue
            entry = self._compact_transformation(transformation)
            entry.update(batch_details.get(key, {}))
            candidates[key] = entry
        return list(candidates.values())

    def _compact_exploration_attempts(self) -> list[dict[str, Any]]:
        compact = []
        for item in self.state.exploration_attempts:
            compact.append({
                "event": item.get("event"),
                "attempt": item.get("attempt"),
                "source": item.get("source"),
                "status": item.get("status"),
                "target_type": item.get("target_type"),
                "target_id": item.get("target_id"),
                "family": item.get("family"),
                "transformation": self._compact_transformation(item.get("transformation")),
                "reason": item.get("reason"),
            })
        return compact[-80:]

    @staticmethod
    def _compact_global_search(global_search: dict[str, Any]) -> dict[str, Any]:
        attempted_targets = []
        for target_id, records in (global_search.get("attempted_atoms") or {}).items():
            attempted_targets.append({
                "target_type": "atom",
                "target_id": int(target_id),
                "attempt_count": len(records),
                "families": sorted({item.get("family") for item in records if item.get("family")}),
            })
        for target_id, records in (global_search.get("attempted_replacement_sites") or {}).items():
            attempted_targets.append({
                "target_type": "replacement_site",
                "target_id": target_id,
                "attempt_count": len(records),
                "families": sorted({item.get("family") for item in records if item.get("family")}),
            })
        return {
            key: global_search.get(key)
            for key in (
                "complete", "closed_atoms", "closed_replacement_sites", "open_targets",
                "missing_edit_atoms", "missing_atom_coverage", "missing_replacement_sites",
                "missing_replacement_coverage", "missing_global_families",
                "missing_target_diversity", "pending_obligations", "policy",
            )
        } | {"attempted_target_summaries": attempted_targets}

    def _optimization_context(self) -> dict[str, Any]:
        global_search = self._global_search_coverage()
        return {
            "reference_baseline": self._llm_safe_value(self.reference_docking_result),
            "available_parents": [
                self._llm_safe_value(metadata)
                for attempt, metadata in sorted(self.parent_metadata.items())
                if attempt > 0
            ],
            "convergence": self._compact_convergence(),
            "global_search": self._compact_global_search(global_search),
            "pending_obligations": self._llm_safe_value(global_search["pending_obligations"]),
            "attempted_transformations": [
                self._compact_transformation(item.get("transformation"))
                for item in self.state.exploration_attempts
                if item.get("source") == "design"
            ][-80:],
            "exploration_attempts": self._compact_exploration_attempts(),
            "unmodifiable_targets": self._llm_safe_value(self.state.unmodifiable_targets[-30:]),
            "search_policy": self._search_policy(),
            "site_strategy": self._llm_safe_value(self.state.site_strategy),
            "active_target": self._llm_safe_value(self.state.active_target),
            "site_search": self._llm_safe_value(self.state.site_search),
            "adaptive_target_summaries": self._adaptive_target_summaries(global_search),
            "candidate_history": self._compact_candidate_history()[-40:],
            "docking_history": self._compact_docking_history()[-40:],
            "instruction": (
                "Prior candidates, exploration attempts, and docking results are authoritative feedback. "
                "Do not return an attempted or rejected transformation again; a new READY decision must "
                "be chemically distinct. This is an evidence-driven search with no fixed per-site attempt floor. "
                "For every target, read adaptive_target_summaries and inspect the "
                "local chemical environment, existing interactions, attachment direction, clearance, fragment "
                "properties, fragment 3D profile, docking scores, seed consistency, pose consensus, interaction "
                "changes, and the trend of prior transformations. Search the fragment library for chemically "
                "appropriate alternatives instead of repeatedly using the smallest generic fragment. For "
                "replace_fragment, only use a fragment returned by search_fragment_library and obtain its "
                "get_fragment_spatial_profile before validate_candidate_geometry. Continue a target while a "
                "new chemically distinct, evidence-backed option could improve or meaningfully validate the "
                "local trend. When site_lock_enabled is true, active_target is authoritative: do not switch "
                "targets until it is explicitly closed with MARK_UNMODIFIABLE; patience is only a review signal. "
                "Use generate_site_candidate_batch when several compatible fragments should be compared. After "
                "the active target has been explored sufficiently and no credible option remains, "
                "explicitly use MARK_UNMODIFIABLE with scope site and an evidence-based reason. STOP is allowed "
                "only after every target is adaptively explored and explicitly closed, or the hard safety limit "
                "is reached."
            ),
        }

    def _repair_decision(
        self, decision: dict[str, Any], payload: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        """Ask for a complete decision without guessing missing tool fields.

        Transformation-shaped responses receive a targeted READY-schema repair.
        Registered tool names used as actions receive a targeted QUERY-schema
        repair. Persistent malformed tool decisions are rejected; only legacy
        transformation-shaped READY responses may use the separate action
        normalization path, with all semantic fields still model-owned.
        """
        decision = self._unwrap_decision(decision)
        attempts = 0
        while not self._has_valid_action(decision) and attempts < 2:
            attempts += 1
            registered_tool_action = (
                decision.get("action")
                if isinstance(decision, dict)
                and decision.get("action") in self.tools.catalog()
                else None
            )
            contains_transformation = self._contains_transformation_fields(decision)
            if registered_tool_action:
                mode = "tool_schema_repair"
                instruction = (
                    f"Your previous response incorrectly used the registered tool name "
                    f"{registered_tool_action!r} as the action. Tool names are never legal action values. "
                    "Return exactly one QUERY object: set action to QUERY, set tool to the registered tool "
                    "name, and move every tool input field under arguments. Preserve the supplied tool "
                    "arguments exactly. Do not return the tool name in action and do not omit the QUERY "
                    "wrapper."
                )
            elif contains_transformation:
                mode = "ready_schema_repair"
                instruction = (
                    "Your previous response expressed a molecular transformation but was not a "
                    "valid workflow decision. Return READY with top-level action, understanding, "
                    "edit_hypothesis, and the complete transformation. Preserve every valid supplied "
                    "transformation field and fill any missing required fields. For replace_fragment "
                    "use replacement_site_id; for replace_hydrogen use edit_atom_index. Return exactly "
                    "one JSON object."
                )
            else:
                mode = "decision_repair"
                instruction = (
                    "Return exactly one complete JSON object with a top-level string action. "
                    "The action must be one of QUERY, QUERY_BATCH, READY, MARK_UNMODIFIABLE, STOP, or "
                    "PROPOSE_TOOL; it must never be a registered tool name. Use QUERY with question, tool, "
                    "and arguments; QUERY_BATCH with a queries array; READY with a complete transformation; "
                    "MARK_UNMODIFIABLE with a precise target, scope, and reason; STOP with a reason; or "
                    "PROPOSE_TOOL. Do not return bare tool arguments or explanatory prose."
                )
            self._write_json(
                f"invalid-decision-{len(self.state.decisions) + attempts:02d}.json",
                {
                    "phase": phase,
                    "repair_mode": mode,
                    "invalid_decision": decision,
                    "instruction": "The previous response was not a valid workflow decision and was not executed.",
                },
            )
            repair_payload = {
                **payload,
                "mode": mode,
                "invalid_decision": decision,
                "instruction": instruction,
            }
            if registered_tool_action:
                repair_payload["query_template"] = {
                    "action": "QUERY",
                    "question": f"Execute {registered_tool_action} using the supplied evidence-backed arguments.",
                    "tool": registered_tool_action,
                    "arguments": {
                        key: value
                        for key, value in decision.items()
                        if key != "action"
                    },
                    "expected_evidence": (
                        "site_strategy"
                        if registered_tool_action == "assess_edit_sites"
                        else "host tool result"
                    ),
                }
            elif contains_transformation:
                repair_payload["ready_template"] = {
                    **decision,
                    "action": "READY",
                    "understanding": "<one sentence on why this edit targets the pocket>",
                    "edit_hypothesis": "<one sentence on the intended structural change>",
                    "knowledge_gaps": ["<optional>"],
                }
            decision = self._unwrap_decision(self.client.complete_json(repair_payload))
        if not self._has_valid_action(decision):
            if self._contains_transformation_fields(decision):
                decision = self._normalize_transformation_action(decision, phase)
            else:
                raise RuntimeError(
                    f"LLM failed to return a valid workflow decision after {attempts} repair attempts: {decision!r}"
                )
        return decision

    def _normalize_transformation_action(
        self, decision: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        """Route a transformation-shaped response through normal READY validation."""
        normalized = {**decision, "action": "READY"}
        self._write_json(
            f"normalized-transformation-decision-{len(self.state.decisions):02d}.json",
            {
                "phase": phase,
                "original_decision": decision,
                "normalized_decision": normalized,
                "reason": (
                    "The model repeatedly expressed a transformation without a valid action. "
                    "Only action was normalized; READY semantics were not synthesized."
                ),
            },
        )
        self._emit("transformation_action_normalized", {
            "phase": phase,
            "operation": normalized.get("operation"),
            "fragment_smiles": normalized.get("fragment_smiles"),
            "replacement_site_id": normalized.get("replacement_site_id"),
            "edit_atom_index": normalized.get("edit_atom_index"),
        })
        return normalized

    def _record_duplicate_tool_rejection(self, message: str) -> None:
        rejection = {
            "status": "rejected",
            "failure_class": "duplicate_tool_call",
            "error": message,
            "instruction": (
                "This exact tool call was already executed. Read the existing observation in "
                "state.observations and choose a different unexecuted query, or return READY "
                "with a chemically distinct transformation. Do not repeat the same arguments."
            ),
        }
        self.state.tool_rejections.append(rejection)
        self._emit("tool_call_rejected", rejection)

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
        elif action == "MARK_UNMODIFIABLE":
            if not self._record_unmodifiable(decision):
                return "MARK_UNMODIFIABLE_REUSED"
        elif action == "QUERY_BATCH":
            queries = decision.get("queries")
            if not isinstance(queries, list) or not queries:
                raise RuntimeError("QUERY_BATCH requires a non-empty queries array")

            # Filter duplicates per item so new independent queries in the same
            # batch still execute. Existing observations remain authoritative.
            executable: list[dict[str, Any]] = []
            skipped: list[dict[str, Any]] = []
            signatures: set[str] = set()
            for query in queries:
                if not isinstance(query, dict):
                    raise RuntimeError("Each QUERY_BATCH item must be an object")
                tool, arguments = query.get("tool"), query.get("arguments")
                if not isinstance(tool, str) or not isinstance(arguments, dict):
                    raise RuntimeError("Each QUERY_BATCH item requires string tool and object arguments")
                if tool not in self.tools.catalog():
                    raise RuntimeError(f"Unknown tool in QUERY_BATCH: {tool}")
                signature = self._signature(tool, arguments)
                if signature in self.state.call_signatures or signature in signatures:
                    skipped.append(self._duplicate_tool_rejection(
                        tool,
                        arguments,
                        source="same_batch" if signature in signatures else "previous_observation",
                    ))
                    continue
                signatures.add(signature)
                executable.append({"action": "QUERY", **query})

            for rejection in skipped:
                self.state.tool_rejections.append(rejection)
                self._emit("tool_call_reused", rejection)
            if not executable:
                return action
            if len(self.state.observations) + len(executable) > self.state.max_context_rounds:
                raise RuntimeError("QUERY_BATCH exceeds the remaining context-tool budget")
            for query in executable:
                self._execute_query(query)
        elif action not in {"READY", "STOP"}:
            raise RuntimeError(f"Invalid LLM action: {action!r}")
        return action

    def _handle_with_recovery(
        self,
        decision: dict[str, Any],
        payload: dict[str, Any],
        phase: str,
    ) -> tuple[dict[str, Any], str]:
        """Recover malformed or repeated tool decisions without losing valid state."""
        last_message = "tool decision error"
        last_failure_class = "invalid_tool_decision"
        for _ in range(3):
            action = decision.get("action")
            try:
                return decision, self._handle_decision(decision)
            except RuntimeError as error:
                message = str(error)
                last_message = message
                if action not in {"QUERY", "QUERY_BATCH", "MARK_UNMODIFIABLE"}:
                    raise
                is_duplicate_tool = isinstance(error, DuplicateToolCallError)
                if is_duplicate_tool:
                    last_failure_class = "duplicate_tool_call"
                    rejection = error.rejection
                elif action == "MARK_UNMODIFIABLE":
                    last_failure_class = "invalid_unmodifiable_decision"
                    active = self.state.active_target
                    active_label = (
                        f"{active.get('target_type')}:{active.get('target_id')}"
                        if active else "none"
                    )
                    rejection = {
                        "status": "rejected",
                        "failure_class": last_failure_class,
                        "decision": decision,
                        "active_target": active,
                        "error": message,
                        "instruction": (
                            "Correct the MARK_UNMODIFIABLE declaration. In site-lock mode it may only "
                            f"close the current active target {active_label}, and only after the local "
                            "attempt and chemical-family gate is complete. Otherwise return a new QUERY "
                            "or READY transformation for the active target."
                        ),
                    }
                else:
                    last_failure_class = "invalid_tool_decision"
                    rejection = {
                        "status": "rejected",
                        "failure_class": last_failure_class,
                        "error": message,
                        "instruction": (
                            "The tool decision was malformed or failed validation. Return a corrected QUERY "
                            "or QUERY_BATCH with each item containing a string tool and object arguments."
                        ),
                    }
                self.state.tool_rejections.append(rejection)
                self._emit("decision_rejected", rejection)
                recovery_payload = {
                    **payload,
                    "state": self._llm_state_view(),
                    "mode": (
                        "decision_recovery"
                        if action == "MARK_UNMODIFIABLE" else "tool_call_recovery"
                    ),
                    "decision_rejection": rejection,
                    "instruction": rejection["instruction"],
                }
                decision = self._repair_decision(
                    self.client.complete_json(recovery_payload),
                    recovery_payload,
                    phase,
                )
        if last_failure_class == "duplicate_tool_call":
            raise RuntimeError(f"LLM did not recover from duplicate tool calls: {last_message}")
        raise RuntimeError(f"LLM did not recover from rejected decisions: {last_message}")

    def collect_context(self) -> dict[str, Any]:
        ready_evidence_retries = 0
        no_progress_decisions = 0
        self._emit("context_collection_started", {
            "max_tool_calls": self.state.max_context_rounds,
            "fragment_library": str(self.tools.fragment_library.path),
            "fragment_count": len(self.tools.fragment_library.records),
        })
        while len(self.state.observations) < self.state.max_context_rounds:
            payload = self._query_payload()
            observation_count = len(self.state.observations)
            decision = self._repair_decision(
                self.client.complete_json(payload), payload, "context_collection"
            )
            decision, action = self._handle_with_recovery(
                decision, payload, "context_collection"
            )
            if action == "READY":
                no_progress_decisions = 0
                try:
                    self._validate_design(decision)
                except (ReadyEvidenceError, ReadyDecisionError) as error:
                    ready_evidence_retries += 1
                    self.state.tool_rejections.append(error.rejection)
                    self._emit("ready_decision_rejected", error.rejection)
                    if isinstance(error, ReadyEvidenceError):
                        before_observations = len(self.state.observations)
                        self._auto_complete_ready_evidence(error.rejection)
                        if (
                            len(self.state.observations) > before_observations
                            and not error.rejection.get("requires_llm_review")
                        ):
                            try:
                                self._validate_design(decision)
                            except (ReadyEvidenceError, ReadyDecisionError) as retry_error:
                                self.state.tool_rejections.append(retry_error.rejection)
                                self._emit("ready_decision_rejected", retry_error.rejection)
                            else:
                                self._write_json("context-final.json", self.state.compact_view())
                                self._emit("context_collection_completed", {
                                    "tool_calls": len(self.state.observations),
                                    "covered_evidence": sorted(self.state.evidence),
                                })
                                return decision
                    if ready_evidence_retries > 4:
                        raise RuntimeError(
                            "LLM repeatedly returned invalid READY decisions without recovering"
                        ) from error
                    continue
                self._write_json("context-final.json", self.state.compact_view())
                self._emit("context_collection_completed", {
                    "tool_calls": len(self.state.observations),
                    "covered_evidence": sorted(self.state.evidence),
                })
                return decision
            if action == "MARK_UNMODIFIABLE":
                no_progress_decisions = 0
                continue
            if action == "STOP":
                raise RuntimeError(
                    f"LLM stopped before selecting a valid design: {decision.get('reason', '')}"
                )
            if len(self.state.observations) == observation_count:
                no_progress_decisions += 1
                if no_progress_decisions >= 4:
                    raise RuntimeError(
                        "Context decision budget exhausted after repeated no-information decisions"
                    )
            else:
                no_progress_decisions = 0
        raise RuntimeError(
            "Context budget exhausted before a valid READY decision; missing site evidence: "
            + ", ".join(self.state.missing_evidence)
        )

    def _retry_ready_decision(
        self, previous_design: dict[str, Any], rejection: dict[str, Any]
    ) -> dict[str, Any]:
        no_progress_retries = 0
        duplicate_ready_retries = 0
        max_no_progress_retries = 12  # Only consecutive no-progress decisions count.
        while len(self.state.observations) < self.state.max_context_rounds:
            observation_count = len(self.state.observations)
            exploration_count = len(self.state.exploration_attempts)
            unmodifiable_count = len(self.state.unmodifiable_targets)
            if no_progress_retries >= max_no_progress_retries:
                raise RuntimeError(
                    "LLM exceeded the edit-retry no-progress limit without selecting a new transformation"
                )
            payload = self._query_payload(
                    {
                        "mode": "edit_retry",
                        "previous_design": previous_design,
                        "rejection": rejection,
                        "optimization_context": self._optimization_context(),
                        "instruction": (
                            "The previous candidate was rejected by deterministic chemistry/clash checks "
                            "or was evaluated by docking. Inspect the supplied rejection, candidate_history, "
                            "and docking_history. All prior transformations are forbidden. You may QUERY a "
                            "new fact or return READY with a revised site or fragment. "
                            "If you return READY, include understanding, edit_hypothesis, operation, "
                            "edit_atom_index for replace_hydrogen or replacement_site_id for "
                            "replace_fragment, and fragment_id or fragment_smiles."
                        ),
                    }
                )
            decision = self._repair_decision(
                self.client.complete_json(payload), payload, "edit_retry"
            )
            decision, action = self._handle_with_recovery(
                decision, payload, "edit_retry"
            )
            if action == "READY":
                try:
                    transformation = self._transformation(decision)
                except Exception as error:
                    rejection = ReadyDecisionError(decision, str(error)).rejection
                    self.state.tool_rejections.append(rejection)
                    self._emit("ready_decision_rejected", rejection)
                    no_progress_retries += 1
                    continue
                if self._transformation_was_attempted(transformation):
                    duplicate_ready_retries += 1
                    if duplicate_ready_retries > 3:
                        raise RuntimeError(
                            "LLM repeatedly returned an already evaluated transformation"
                        )
                    rejection = {
                        "status": "rejected",
                        "failure_class": "duplicate_transformation",
                        "transformation": transformation,
                        "previous_failure_class": rejection.get("failure_class"),
                        "instruction": (
                            "This transformation was already evaluated. Use candidate_history and "
                            "docking_history; return a chemically distinct transformation or QUERY an "
                            "unexecuted fact. Do not submit this transformation again."
                        ),
                    }
                    self.state.tool_rejections.append(rejection)
                    self._emit("transformation_rejected", rejection)
                    no_progress_retries += 1
                    continue
                try:
                    self._validate_design(decision)
                except (ReadyEvidenceError, ReadyDecisionError) as error:
                    rejection = error.rejection
                    self.state.tool_rejections.append(rejection)
                    self._emit("ready_decision_rejected", rejection)
                    if isinstance(error, ReadyEvidenceError):
                        before_observations = len(self.state.observations)
                        self._auto_complete_ready_evidence(rejection)
                        if (
                            len(self.state.observations) > before_observations
                            and not rejection.get("requires_llm_review")
                        ):
                            try:
                                self._validate_design(decision)
                            except ReadyEvidenceError as retry_error:
                                rejection = retry_error.rejection
                                self.state.tool_rejections.append(rejection)
                                self._emit("ready_decision_rejected", rejection)
                            except ReadyDecisionError as retry_error:
                                rejection = retry_error.rejection
                                self.state.tool_rejections.append(retry_error.rejection)
                                self._emit("ready_decision_rejected", retry_error.rejection)
                            else:
                                return decision
                        else:
                            no_progress_retries += 1
                    else:
                        no_progress_retries += 1
                    continue
                return decision
            if action == "MARK_UNMODIFIABLE":
                no_progress_retries = 0
                continue
            if action == "STOP":
                stop_rejection = self._stop_gate_rejection()
                if stop_rejection is None:
                    return decision
                self.state.tool_rejections.append(stop_rejection)
                self._emit("stop_rejected", stop_rejection)
                rejection = stop_rejection
                no_progress_retries += 1
                continue
            current_progress = (
                len(self.state.observations),
                len(self.state.exploration_attempts),
                len(self.state.unmodifiable_targets),
            )
            if current_progress > (
                observation_count,
                exploration_count,
                unmodifiable_count,
            ):
                no_progress_retries = 0
            else:
                no_progress_retries += 1
        raise RuntimeError("Context budget exhausted while selecting an edit retry")

    def _validate_design(self, decision: dict[str, Any]) -> None:
        required = ("understanding", "edit_hypothesis")
        missing = [key for key in required if key not in decision]
        if missing:
            raise ReadyDecisionError(
                decision,
                f"READY decision missing fields: {missing}",
                instruction="Return READY with understanding and edit_hypothesis.",
            )
        try:
            transformation = self._transformation(decision)
        except Exception as error:
            raise ReadyDecisionError(decision, str(error)) from error
        site_lock_rejection = self._site_lock_rejection(transformation)
        if site_lock_rejection is not None:
            raise ReadyDecisionError(
                decision,
                site_lock_rejection["instruction"],
                failure_class=site_lock_rejection["failure_class"],
                instruction=site_lock_rejection["instruction"],
            )
        operation = transformation["operation"]
        if operation not in {"replace_hydrogen", "replace_fragment"}:
            raise ReadyDecisionError(decision, f"Unsupported READY operation: {operation!r}")
        index = transformation.get("edit_atom_index")
        parent_attempt = transformation.get("parent_attempt")
        validation_parent = self._resolve_parent_candidate(parent_attempt)
        if not isinstance(index, int) or not 0 <= index < validation_parent.GetNumAtoms():
            raise ReadyDecisionError(decision, f"Invalid edit_atom_index: {index!r}")
        if operation == "replace_hydrogen":
            atom = validation_parent.GetAtomWithIdx(index)
            if (
                atom.GetTotalNumHs() < 1
                and not transformation.get("replace_existing_substituent")
            ):
                raise ReadyDecisionError(
                    decision,
                    f"Selected edit atom {index} has no replaceable hydrogen for replace_hydrogen",
                    instruction=(
                        "Choose a heavy atom with a replaceable hydrogen and provide a concrete "
                        "fragment_smiles; do not use placeholder atom indices or SMILES."
                    ),
                )
        environment_sites = {
            (item.arguments.get("atom_index"), item.arguments.get("parent_attempt"))
            for item in self.state.observations
            if item.tool == "get_atom_environment"
        }
        geometry_sites = {
            (item.arguments.get("atom_index"), item.arguments.get("parent_attempt"))
            for item in self.state.observations
            if item.tool == "check_growth_space"
        }
        candidate_geometry = []
        for item in self.state.observations:
            if item.tool != "validate_candidate_geometry" or item.result.get("status") != "accepted":
                continue
            args = item.result.get("transformation") or item.arguments
            observed = {
                "operation": args.get("operation", "replace_hydrogen"),
                "edit_atom_index": args.get("edit_atom_index", args.get("atom_index")),
                "replacement_site_id": args.get("replacement_site_id"),
                "cut_bond": args.get("cut_bond"),
                "fragment_smiles": args.get("fragment_smiles"),
                "fragment_id": args.get("fragment_id"),
                "parent_attempt": args.get("parent_attempt"),
                "replace_existing_substituent": args.get("replace_existing_substituent"),
            }
            candidate_geometry.append(observed)
        exact_geometry = any(
            self._same_transformation(observed, transformation)
            for observed in candidate_geometry
        )
        replacement_sites_queried = any(
            item.tool == "list_fragment_replacement_sites"
            for item in self.state.observations
        )
        missing_evidence = []
        recommended_queries = []
        if (index, parent_attempt) not in environment_sites:
            missing_evidence.append("edit-site environment")
            environment_arguments = {"atom_index": index, "radius": 5.0}
            if parent_attempt is not None:
                environment_arguments["parent_attempt"] = parent_attempt
            recommended_queries.append({
                "tool": "get_atom_environment",
                "arguments": environment_arguments,
            })
        if operation == "replace_hydrogen" and (index, parent_attempt) not in geometry_sites:
            missing_evidence.append("growth space")
            growth_arguments = {"atom_index": index, "distance": 2.0}
            if parent_attempt is not None:
                growth_arguments["parent_attempt"] = parent_attempt
            recommended_queries.append({
                "tool": "check_growth_space",
                "arguments": growth_arguments,
            })
        if operation == "replace_fragment" and not replacement_sites_queried:
            missing_evidence.append("host-enumerated replacement site")
            recommended_queries.append({
                "tool": "list_fragment_replacement_sites",
                "arguments": {"limit": 50},
            })
        if self._search_policy()["mode"] == "adaptive":
            fragment_id = transformation.get("fragment_id")
            fragment_smiles = transformation.get("fragment_smiles")

            def smiles_match(value: Any) -> bool:
                return bool(
                    isinstance(value, str)
                    and isinstance(fragment_smiles, str)
                    and self.tools.fragment_library.smiles_equivalent(value, fragment_smiles)
                )

            library_match = any(
                (
                    item.tool == "search_fragment_library"
                    and any(
                        isinstance(fragment, dict)
                        and (
                            fragment.get("fragment_id") == fragment_id
                            or smiles_match(fragment.get("smiles"))
                        )
                        for fragment in (item.result.get("fragments") or [])
                    )
                )
                or (
                    item.tool == "get_fragment_record"
                    and isinstance(item.result, dict)
                    and item.result.get("fragment_id") == fragment_id
                )
                for item in self.state.observations
            )
            if operation == "replace_fragment" and not library_match:
                missing_evidence.append("selected fragment library record")
                recommended_queries.append({
                    "tool": "search_fragment_library",
                    "arguments": {
                        "query": "",
                        "operation": "replace_fragment",
                        "limit": 50,
                    },
                })
            properties_match = any(
                item.tool == "get_fragment_properties"
                and (
                    smiles_match(item.arguments.get("smiles"))
                    or smiles_match((item.result or {}).get("canonical_smiles"))
                )
                for item in self.state.observations
            ) or library_match
            if not properties_match:
                missing_evidence.append("selected fragment properties")
                recommended_queries.append({
                    "tool": "get_fragment_properties",
                    "arguments": {"smiles": fragment_smiles},
                })
            profile_match = any(
                item.tool == "get_fragment_spatial_profile"
                and (
                    item.arguments.get("fragment_id") == fragment_id
                    or smiles_match(item.arguments.get("fragment_smiles"))
                    or smiles_match((item.result or {}).get("fragment_smiles"))
                )
                for item in self.state.observations
            )
            if not profile_match:
                missing_evidence.append("selected fragment spatial profile")
                profile_arguments = (
                    {"fragment_id": fragment_id}
                    if fragment_id else {"fragment_smiles": fragment_smiles}
                )
                recommended_queries.append({
                    "tool": "get_fragment_spatial_profile",
                    "arguments": profile_arguments,
                })
        if not exact_geometry:
            missing_evidence.append("exact accepted candidate geometry")
            geometry_arguments = {
                "operation": operation,
                "fragment_smiles": transformation["fragment_smiles"],
            }
            if parent_attempt is not None:
                geometry_arguments["parent_attempt"] = parent_attempt
                geometry_arguments["replace_existing_substituent"] = True
            if transformation.get("fragment_id"):
                geometry_arguments["fragment_id"] = transformation["fragment_id"]
            if operation == "replace_fragment":
                geometry_arguments["replacement_site_id"] = transformation["replacement_site_id"]
            else:
                geometry_arguments["edit_atom_index"] = index
            recommended_queries.append({
                "tool": "validate_candidate_geometry",
                "arguments": geometry_arguments,
            })
        if missing_evidence:
            raise ReadyEvidenceError(transformation, missing_evidence, recommended_queries)
        self._emit("ready_gate_passed", {
            "operation": operation,
            "edit_atom_index": index,
            "replacement_site_id": transformation.get("replacement_site_id"),
            "cut_bond": transformation.get("cut_bond"),
            "fragment_id": transformation.get("fragment_id"),
            "fragment_smiles": transformation.get("fragment_smiles"),
        })

    def _optimization_settings(self) -> dict[str, Any]:
        configured = self.context.task.get("docking_optimization") or {}
        return {
            "primary_metric": configured.get("primary_metric", "minimizedAffinity"),
            "minimum_improvement": float(configured.get("minimum_improvement", 0.25)),
            "seed_stddev_penalty": float(configured.get("seed_stddev_penalty", 0.25)),
            "minimum_seed_win_fraction": float(
                configured.get("minimum_seed_win_fraction", 2 / 3)
            ),
            "hard_max_attempts": int(
                configured.get("hard_max_attempts", self.context.task.get("max_edit_attempts", 30))
            ),
        }

    def _record_candidate_history(
        self,
        report: dict[str, Any],
        transformation: dict[str, Any],
    ) -> None:
        validation = report.get("validation") or {}
        candidate = validation.get("candidate") or {}
        docking = report.get("docking") or {}
        comparison = docking.get("comparison") or {}
        primary_name = self._optimization_settings()["primary_metric"]
        self.state.candidate_history.append({
            "attempt": report.get("attempt"),
            "candidate_id": f"attempt-{report.get('attempt'):02d}",
            "parent_attempt": report.get("parent_attempt"),
            "generation": report.get("generation", 1),
            "record_type": "design_attempt",
            "transformation": transformation,
            "candidate_path": report.get("candidate_path"),
            "validation": {
                "status": validation.get("status"),
                "failure_class": validation.get("failure_class"),
                "canonical_smiles": candidate.get("canonical_smiles") or validation.get("canonical_smiles"),
                "property_delta": validation.get("property_delta"),
                "severe_clash_count": validation.get("severe_clash_count"),
                "formal_charge": candidate.get("formal_charge"),
                "heavy_atoms": candidate.get("heavy_atoms"),
                "molecular_weight": candidate.get("molecular_weight"),
            },
            "docking": {
                "status": docking.get("status"),
                "entered_docking": bool(
                    docking and not str(docking.get("status", "")).startswith("not_run")
                ),
                "completed": docking.get("status") == "complete",
                "primary_metric": primary_name,
                "primary_metric_summary": (comparison.get("metrics") or {}).get(primary_name),
                "pose_count_per_seed": docking.get("pose_count_per_seed"),
                "top_pose_properties": [
                    pose.get("properties", {}) for pose in (docking.get("poses") or [])[:1]
                ],
                "candidate_per_seed": docking.get("candidate_per_seed"),
                "pose_consensus": docking.get("pose_consensus"),
                "interaction_consensus": docking.get("interaction_consensus"),
            },
        })

    @staticmethod
    def _design_region(transformation: dict[str, Any]) -> str:
        if transformation.get("operation") == "replace_fragment":
            return f"replacement:{transformation.get('replacement_site_id')}"
        return f"atom:{transformation.get('edit_atom_index')}"

    def _validated_design_regions(self) -> list[str]:
        regions = set()
        for item in self.state.observations:
            if item.tool != "validate_candidate_geometry" or item.result.get("status") != "accepted":
                continue
            arguments = item.result.get("transformation") or item.arguments
            if (
                arguments.get("operation") == "replace_fragment"
                and not isinstance(arguments.get("replacement_site"), dict)
            ):
                continue
            regions.add(self._design_region(arguments))
        return sorted(region for region in regions if not region.endswith("None"))

    def _docked_design_regions(self) -> list[str]:
        regions = {
            self._design_region(item.get("transformation") or {})
            for item in self.state.docking_history
            if item.get("status") == "complete" and item.get("raw_quality_from_mean") is not None
        }
        return sorted(region for region in regions if not region.endswith("None"))

    @staticmethod
    def _modification_family(transformation: dict[str, Any]) -> str:
        if transformation.get("operation") == "replace_fragment":
            return "fragment_replacement"
        smiles = transformation.get("fragment_smiles")
        if not isinstance(smiles, str):
            return "other"
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return "other"
        dummy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
        if dummy:
            molecule = Chem.RWMol(molecule)
            molecule.RemoveAtom(dummy[0])
            molecule = molecule.GetMol()
        symbols = {atom.GetSymbol().upper() for atom in molecule.GetAtoms()}
        if symbols & {"F", "CL", "BR", "I"}:
            return "halogen"
        if any(
            atom.GetIsAromatic() and atom.GetSymbol().upper() in {"N", "O", "S"}
            for atom in molecule.GetAtoms()
        ):
            return "heteroaryl"
        if symbols & {"N", "O", "S"}:
            return "polar"
        if symbols <= {"C"}:
            return "alkyl"
        return "other"

    @classmethod
    def _local_modification_family(cls, transformation: dict[str, Any]) -> str:
        chemical_transformation = dict(transformation)
        chemical_transformation["operation"] = "replace_hydrogen"
        family = cls._modification_family(chemical_transformation)
        if transformation.get("operation") == "replace_fragment":
            return f"fragment_replacement:{family}"
        return family

    @staticmethod
    def _replace_hydrogen_site_supported(atom: Chem.Atom) -> bool:
        """Return whether the current single-bond editor supports replacing this H."""
        if atom.GetAtomicNum() <= 1 or atom.GetTotalNumHs() < 1:
            return False
        # Aromatic [nH] substitution requires explicit tautomer/protonation handling.
        return not (atom.GetSymbol() == "N" and atom.GetIsAromatic())

    def _global_search_coverage(self) -> dict[str, Any]:
        hydrogen_atoms = [
            atom for atom in self.context.ligand.GetAtoms()
            if atom.GetAtomicNum() > 1 and atom.GetTotalNumHs() > 0
        ]
        editable_atoms = sorted(
            atom.GetIdx()
            for atom in hydrogen_atoms
            if self._replace_hydrogen_site_supported(atom)
        )
        host_ineligible_atoms = [
            {
                "atom_index": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "reason": (
                    "The current replace_hydrogen editor does not support aromatic [nH] "
                    "substitution without explicit tautomer/protonation handling."
                ),
            }
            for atom in hydrogen_atoms
            if not self._replace_hydrogen_site_supported(atom)
        ]
        replacement_sites = [
            site["replacement_site_id"]
            for site in self.tools.list_fragment_replacement_sites(limit=100).get("sites", [])
        ]
        search_policy = self._search_policy()
        adaptive_search = search_policy["mode"] == "adaptive"
        atom_clearance: dict[int, float] = {}
        for index in editable_atoms:
            try:
                atom_clearance[index] = float(
                    self.tools.check_growth_space(index, 3.0)["minimum_clearance"]
                )
            except Exception:
                atom_clearance[index] = 0.0

        tracked_atom_indices = sorted({atom.GetIdx() for atom in hydrogen_atoms})
        attempted_by_atom: dict[int, list[dict[str, Any]]] = {
            index: [] for index in tracked_atom_indices
        }
        attempted_by_site: dict[str, list[dict[str, Any]]] = {site_id: [] for site_id in replacement_sites}

        # candidate_history is retained as a compatibility source for older runs;
        # new runs use exploration_attempts, which includes rejected edits.
        records = list(self.state.exploration_attempts)
        known_transformations = {
            json.dumps(item.get("transformation") or {}, sort_keys=True)
            for item in records
        }
        for item in self.state.candidate_history:
            transformation = item.get("transformation") or {}
            key = json.dumps(self._exploration_transformation(transformation), sort_keys=True)
            if key in known_transformations:
                continue
            records.append({
                "event": item.get("attempt"),
                "source": "legacy_candidate_history",
                "status": (item.get("validation") or {}).get("status") or "attempted",
                "family": self._modification_family(transformation),
                "transformation": self._exploration_transformation(transformation),
            })

        def coverage_record(record: dict[str, Any]) -> dict[str, Any]:
            transformation = record.get("transformation") or {}
            return {
                "event": record.get("event"),
                "attempt": record.get("attempt"),
                "source": record.get("source"),
                "status": record.get("status"),
                "family": record.get("family") or self._modification_family(transformation),
                "transformation": self._compact_transformation(transformation),
                "reason": record.get("reason"),
            }

        for item in records:
            transformation = item.get("transformation") or {}
            operation = transformation.get("operation", "replace_hydrogen")
            record = coverage_record(item)
            if operation == "replace_hydrogen":
                index = transformation.get("edit_atom_index")
                if index in attempted_by_atom:
                    attempted_by_atom[index].append(record)
            elif operation == "replace_fragment":
                site_id = transformation.get("replacement_site_id")
                if site_id in attempted_by_site:
                    attempted_by_site[site_id].append(record)

        host_closed_atoms = {item["atom_index"] for item in host_ineligible_atoms}
        closed_atoms = host_closed_atoms | {
            index for index in editable_atoms
            if self._is_unmodifiable("atom", index)
        }
        closed_sites = {
            site_id for site_id in replacement_sites
            if self._is_unmodifiable("replacement_site", site_id)
        }

        def family_closed(target_type: str, target_id: Any, family: str) -> bool:
            return self._is_unmodifiable(target_type, target_id, family)

        def seen_atom_families(index: int) -> set[str]:
            families = {
                "halogen" if record["family"] == "halogen" else "non_halogen"
                for record in attempted_by_atom[index]
            }
            for family in ("halogen", "non_halogen"):
                if family_closed("atom", index, family):
                    families.add(family)
            if index in closed_atoms:
                families.update({"halogen", "non_halogen"})
            return families

        best_halogen_hit_atoms = set()
        for item in self.state.docking_history:
            transformation = item.get("transformation") or {}
            if (
                transformation.get("operation") == "replace_hydrogen"
                and self._modification_family(transformation) == "halogen"
                and isinstance(item.get("raw_quality_from_mean"), (int, float))
                and float(item["raw_quality_from_mean"]) > 0
            ):
                best_halogen_hit_atoms.add(transformation.get("edit_atom_index"))

        all_families = {
            record["family"] or self._modification_family(record.get("transformation") or {})
            for record in records
        }
        for item in self.state.unmodifiable_targets:
            family = item.get("family")
            if item.get("scope") == "family" and family:
                all_families.add(family)
            elif item.get("scope") == "site":
                all_families.add(
                    "fragment_replacement" if item.get("target_type") == "replacement_site" else "non_halogen"
                )
        families = sorted(all_families)

        missing_atom_coverage = []
        for index in editable_atoms:
            required_families = {"halogen"} if atom_clearance[index] < 1.5 else {"halogen", "non_halogen"}
            missing_families = sorted(required_families - seen_atom_families(index))
            if missing_families:
                missing_atom_coverage.append({
                    "atom_index": index,
                    "clearance": atom_clearance[index],
                    "required_families": sorted(required_families),
                    "missing_families": missing_families,
                })

        def distinct_fragments(records_for_site: list[dict[str, Any]]) -> set[str]:
            values = set()
            for record in records_for_site:
                smiles = (record.get("transformation") or {}).get("fragment_smiles")
                if not isinstance(smiles, str):
                    continue
                molecule = Chem.MolFromSmiles(smiles)
                values.add(Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule else smiles)
            return values

        missing_replacement_coverage = []
        for site_id, site_records in attempted_by_site.items():
            distinct = distinct_fragments(site_records)
            if site_id not in closed_sites and len(distinct) < 2:
                missing_replacement_coverage.append({
                    "replacement_site_id": site_id,
                    "attempt_count": len(site_records),
                    "distinct_fragments": sorted(distinct),
                    "required_distinct_fragments": 2,
                })

        missing_atoms = [index for index in editable_atoms if index not in closed_atoms and not attempted_by_atom[index]]
        missing_sites = [site_id for site_id in replacement_sites if site_id not in closed_sites and not attempted_by_site[site_id]]
        halogen_hits_without_non_halogen = [
            index for index in sorted(best_halogen_hit_atoms)
            if not family_closed("atom", index, "non_halogen")
            and "non_halogen" not in seen_atom_families(index)
        ]
        missing_global_families = [
            family for family in ("halogen", "non_halogen", "fragment_replacement")
            if (
                family == "halogen" and "halogen" not in families
                or family == "non_halogen"
                and not any(item in families for item in ("alkyl", "polar", "heteroaryl", "other", "non_halogen"))
                or family == "fragment_replacement" and "fragment_replacement" not in families
            )
        ]
        missing_target_diversity = []
        if adaptive_search:
            open_targets = [
                {"target_type": "atom", "target_id": index}
                for index in editable_atoms if index not in closed_atoms
            ] + [
                {"target_type": "replacement_site", "target_id": site_id}
                for site_id in replacement_sites if site_id not in closed_sites
            ]
            complete = not open_targets
        else:
            open_targets = []
            complete = not (
                missing_atom_coverage or missing_replacement_coverage or missing_atoms
                or missing_sites or halogen_hits_without_non_halogen or missing_global_families
            )
        pending_obligations = []
        for index in missing_atoms:
            pending_obligations.append({
                "type": "atom",
                "atom_index": index,
                "required_action": "attempt a transformation or use MARK_UNMODIFIABLE",
            })
        for site_id in missing_sites:
            pending_obligations.append({
                "type": "replacement_site",
                "replacement_site_id": site_id,
                "required_action": "attempt a fragment or use MARK_UNMODIFIABLE",
            })
        for item in missing_atom_coverage:
            for family in item["missing_families"]:
                pending_obligations.append({
                    "type": "atom_family",
                    "atom_index": item["atom_index"],
                    "family": family,
                    "clearance": item["clearance"],
                })
        for item in missing_replacement_coverage:
            pending_obligations.append({
                "type": "replacement_fragments",
                "replacement_site_id": item["replacement_site_id"],
                "additional_distinct_fragments_required": (
                    item["required_distinct_fragments"] - len(item["distinct_fragments"])
                ),
            })
        for index in halogen_hits_without_non_halogen:
            if not any(
                item.get("type") == "atom_family"
                and item.get("atom_index") == index
                and item.get("family") == "non_halogen"
                for item in pending_obligations
            ):
                pending_obligations.append({
                    "type": "halogen_hit_followup",
                    "atom_index": index,
                    "family": "non_halogen",
                })
        for family in missing_global_families:
            pending_obligations.append({"type": "global_family", "family": family})
        if adaptive_search:
            pending_obligations = [
                {
                    "type": "target_review",
                    **item,
                    "required_action": (
                        "Study accumulated site, fragment, geometry, and docking evidence. Continue with "
                        "a chemically distinct evidence-backed transformation, or use MARK_UNMODIFIABLE "
                        "with scope site and a concrete evidence-based reason when no credible option remains."
                    ),
                }
                for item in open_targets
                if not any(
                    missing["target_type"] == item["target_type"]
                    and missing["target_id"] == item["target_id"]
                    for missing in missing_target_diversity
                )
            ]
        if search_policy["site_lock_enabled"] and self.state.site_strategy:
            self._refresh_site_search()
            strategy_sites = self._strategy_sites()
            strategy_keys = [
                self._target_key(item.get("target_type"), item.get("target_id"))
                for item in strategy_sites
            ]
            strategy_status = [
                self.state.site_search.get(key, {}) for key in strategy_keys
            ]
            strategy_ready = bool(strategy_sites) and len(strategy_sites) >= search_policy[
                "minimum_prioritized_sites"
            ]
            complete = strategy_ready and all(
                item.get("status") == "closed" for item in strategy_status
            )
            pending_obligations = []
            if not strategy_ready:
                pending_obligations.append({
                    "type": "site_strategy",
                    "required_action": (
                        "Call assess_edit_sites with at least "
                        f"{search_policy['minimum_prioritized_sites']} evidence-backed targets."
                    ),
                })
            elif self.state.active_target is not None:
                active_key = self._target_key(
                    self.state.active_target["target_type"],
                    self.state.active_target["target_id"],
                )
                pending_obligations.append({
                    "type": "active_target_local_search",
                    "active_target": self.state.active_target,
                    "site_search": self.state.site_search.get(active_key),
                    "required_action": (
                        "Continue the locked target while a chemically distinct evidence-backed option remains. "
                        "Patience is a review signal, not an automatic stopping condition. Close it only with "
                        "MARK_UNMODIFIABLE after the local evidence and candidate options are exhausted."
                    ),
                })
            open_targets = [
                {
                    "target_type": item.get("target_type"),
                    "target_id": item.get("target_id"),
                    "priority": item.get("priority"),
                    "site_type": item.get("site_type"),
                    "search_status": item.get("initial_search_status"),
                }
                for item in strategy_status
                if item.get("status") != "closed"
            ]
            missing_target_diversity = []
        return {
            "complete": complete,
            "editable_hydrogen_atoms": editable_atoms,
            "host_ineligible_hydrogen_atoms": host_ineligible_atoms,
            "atom_clearance": {str(index): clearance for index, clearance in atom_clearance.items()},
            "replacement_sites": replacement_sites,
            "exploration_attempts": self._llm_safe_value(self.state.exploration_attempts),
            "unmodifiable_targets": self._llm_safe_value(self.state.unmodifiable_targets),
            "attempted_atoms": {str(index): records for index, records in attempted_by_atom.items() if records},
            "attempted_replacement_sites": {site_id: records for site_id, records in attempted_by_site.items() if records},
            "closed_atoms": sorted(closed_atoms),
            "closed_replacement_sites": sorted(closed_sites),
            "modification_families_seen": families,
            "missing_edit_atoms": missing_atoms,
            "missing_atom_coverage": missing_atom_coverage,
            "missing_replacement_sites": missing_sites,
            "missing_replacement_coverage": missing_replacement_coverage,
            "halogen_hit_atoms_missing_non_halogen_followup": halogen_hits_without_non_halogen,
            "missing_global_families": missing_global_families,
            "missing_target_diversity": missing_target_diversity,
            "open_targets": open_targets,
            "pending_obligations": pending_obligations,
            "policy": {
                **search_policy,
                "meaning": (
                    "Adaptive mode treats the minimum transformation count as an exploration floor, not a "
                    "stopping target. Geometry rejection is learned evidence. After the floor is met, the LLM "
                    "must review site chemistry, fragment information, docking trend, pose consensus, and "
                    "interaction changes, then either continue with a credible distinct hypothesis or explicitly "
                    "close the target using MARK_UNMODIFIABLE scope site."
                    if adaptive_search else
                    "An explicit transformation attempt counts for coverage even when chemistry, valence, "
                    "conformer generation, or geometry rejects it. A host-validated MARK_UNMODIFIABLE declaration "
                    "also closes its declared site or family, but never counts as successful docking evidence."
                ),
                "fixed_region_attempts": False,
                "patience": False,
            },
        }

    def _stop_gate_rejection(self) -> dict[str, Any] | None:
        coverage = self._global_search_coverage()
        if coverage["complete"]:
            return None
        adaptive_search = coverage["policy"]["mode"] == "adaptive"
        return {
            "status": "rejected",
            "failure_class": "global_search_incomplete",
            "global_search": coverage,
            "instruction": (
                "Do not STOP yet. Follow pending_obligations as the authoritative work list. For each open "
                "target, inspect its accumulated chemical environment, geometry, fragment-library evidence, "
                "docking trend, pose consensus, and interaction changes. Continue with a chemically distinct "
                "evidence-backed transformation while a credible option remains. Once the diversity floor is met "
                "and the evidence no longer supports improvement, explicitly close that target with "
                "MARK_UNMODIFIABLE scope site and a concrete reason."
                if adaptive_search else
                "Do not STOP yet. Complete the pending obligations shown in global_search: attempt every "
                "editable hydrogen atom and every host replacement site, test the required modification families, "
                "and test a non-halogen follow-up at every atom where a halogen candidate beat the reference. "
                "Chemistry/geometry/valence/clash rejections count as explored evidence. Use MARK_UNMODIFIABLE "
                "with a precise target, scope, and reason when a site or family is not chemically supported; "
                "do not repeat an attempted transformation. The pending_obligations list is authoritative."
            ),
        }

    def _record_docking_result(
        self,
        attempt: int,
        transformation: dict[str, Any],
        candidate_path: Path,
        docking: dict[str, Any],
    ) -> dict[str, Any]:
        settings = self._optimization_settings()
        metric_name = settings["primary_metric"]
        metric = (docking.get("comparison") or {}).get("metrics", {}).get(metric_name)
        delta = None
        raw_quality = None
        quality = None
        direction = None
        seed_stddev = None
        seed_win_fraction = None
        stability_eligible = False
        if isinstance(metric, dict):
            summary = metric.get("delta_candidate_minus_reference") or {}
            value = summary.get("mean")
            if isinstance(value, (int, float)):
                delta = float(value)
                direction = metric.get("direction")
                raw_quality = -delta if direction == "lower_is_better" else delta
                stddev = summary.get("stddev", 0.0)
                seed_stddev = float(stddev) if isinstance(stddev, (int, float)) else 0.0
                win_fraction = metric.get("candidate_better_seed_fraction", 1.0)
                seed_win_fraction = (
                    float(win_fraction) if isinstance(win_fraction, (int, float)) else 0.0
                )
                stability_eligible = (
                    seed_win_fraction + 1e-9 >= settings["minimum_seed_win_fraction"]
                )
                if stability_eligible:
                    quality = raw_quality - settings["seed_stddev_penalty"] * seed_stddev

        previous_best = self.state.convergence.get("best_quality")
        minimum = settings["minimum_improvement"]
        is_new_best = quality is not None and (
            previous_best is None or quality > float(previous_best)
        )
        is_significant_improvement = quality is not None and (
            previous_best is None or quality > float(previous_best) + minimum
        )
        if is_new_best:
            best_quality = quality
            best_attempt = attempt
        else:
            best_quality = previous_best
            best_attempt = self.state.convergence.get("best_attempt")
        non_improving = (
            0
            if is_significant_improvement
            else int(self.state.convergence.get("non_improving_attempts", 0)) + 1
        )

        entry = {
            "attempt": attempt,
            "candidate_id": f"attempt-{attempt:02d}",
            "parent_attempt": transformation.get("parent_attempt"),
            "generation": transformation.get("generation", 1),
            "design_region": self._design_region(transformation),
            "transformation": transformation,
            "candidate_path": str(candidate_path),
            "status": docking.get("status"),
            "primary_metric": metric_name,
            "direction": direction,
            "delta_candidate_minus_reference": delta,
            "raw_quality_from_mean": raw_quality,
            "seed_stddev": seed_stddev,
            "seed_win_fraction": seed_win_fraction,
            "seed_stddev_penalty": settings["seed_stddev_penalty"],
            "minimum_seed_win_fraction": settings["minimum_seed_win_fraction"],
            "stability_eligible": stability_eligible,
            "quality": quality,
            "improvement_over_previous_best": (
                None if quality is None or previous_best is None else quality - float(previous_best)
            ),
            "is_new_best": is_new_best,
            "is_significant_improvement": is_significant_improvement,
            "best_quality_so_far": best_quality,
            "best_attempt_so_far": best_attempt,
            "comparison": docking.get("comparison"),
            "pose_consensus": docking.get("pose_consensus"),
            "interaction_consensus": docking.get("interaction_consensus"),
        }
        self.state.docking_history.append(entry)
        self._refresh_site_search()
        scored_count = sum(
            item.get("raw_quality_from_mean") is not None for item in self.state.docking_history
        )
        stability_eligible_count = sum(
            item.get("quality") is not None for item in self.state.docking_history
        )
        validated_regions = self._validated_design_regions()
        docked_regions = self._docked_design_regions()
        global_search = self._global_search_coverage()
        best_entry = next(
            (item for item in self.state.docking_history if item.get("attempt") == best_attempt),
            None,
        )
        self.state.convergence = {
            "status": "searching",
            "converged": False,
            "llm_controls_termination": True,
            "stop_authority": "llm_or_hard_safety_limit",
            "primary_metric": metric_name,
            "minimum_improvement": minimum,
            "scored_attempts": scored_count,
            "stability_eligible_attempts": stability_eligible_count,
            "seed_stddev_penalty": settings["seed_stddev_penalty"],
            "minimum_seed_win_fraction": settings["minimum_seed_win_fraction"],
            "best_attempt": best_attempt,
            "best_quality": best_quality,
            "best_delta_candidate_minus_reference": (
                best_entry.get("delta_candidate_minus_reference") if best_entry else None
            ),
            "best_reference_hit": bool(
                best_entry and best_entry.get("raw_quality_from_mean", 0) > 0
            ),
            "non_improving_attempts": non_improving,
            "validated_design_regions": validated_regions,
            "docked_design_regions": docked_regions,
            "design_region_count_is_descriptive_only": True,
            "global_search": global_search,
            "next_decision": (
                "LLM should continue with a new evidence-backed transformation. STOP is blocked until "
                "global_search.complete is true, unless hard_max_attempts is reached."
            ),
        }
        self._emit("docking_scored", {
            "attempt": attempt,
            "metric": metric_name,
            "direction": direction,
            "candidate_minus_reference": delta,
            "raw_quality_from_mean": raw_quality,
            "seed_stddev": seed_stddev,
            "seed_win_fraction": seed_win_fraction,
            "stability_eligible": stability_eligible,
            "quality": quality,
            "is_new_best": is_new_best,
            "best_attempt": best_attempt,
            "best_quality": best_quality,
            "active_parent_attempt": best_attempt,
            "non_improving_attempts": non_improving,
            "validated_design_regions": validated_regions,
            "docked_design_regions": docked_regions,
            "global_search": global_search,
            "converged": False,
        })
        return entry

    def _accepted_output(
        self,
        history: list[dict[str, Any]],
        reference_path: Path,
        stopping_reason: str,
    ) -> dict[str, Any]:
        best_attempt = self.state.convergence.get("best_attempt")
        if not isinstance(best_attempt, int):
            best_attempt = next(
                (item["attempt"] for item in reversed(history) if item.get("docking")),
                history[-1]["attempt"],
            )
        best = next(item for item in history if item["attempt"] == best_attempt)
        self.state.convergence["status"] = (
            "stopped_by_llm" if stopping_reason == "llm_stop" else stopping_reason
        )
        self.state.convergence["termination_reason"] = stopping_reason
        self.state.convergence["converged"] = False
        self.state.convergence["global_search"] = self._global_search_coverage()
        rbfe = {
            "stage": "rbfe",
            "status": "deferred",
            "message": "RBFE is intentionally deferred until docking pose selection and ligand mapping are validated.",
        }
        return {
            "status": "candidate_accepted",
            "stopping_reason": stopping_reason,
            "best_attempt": best_attempt,
            "candidate_path": best["candidate_path"],
            "reference_path": str(reference_path),
            "attempts": history,
            "docking": best.get("docking", {}),
            "docking_history": self.state.docking_history,
            "convergence": self.state.convergence,
            "rbfe": rbfe,
            "fep": rbfe,
        }

    def design(self, first_decision: dict[str, Any]) -> dict[str, Any]:
        decision = first_decision
        settings = self._optimization_settings()
        hard_max = settings["hard_max_attempts"]
        history: list[dict[str, Any]] = []
        seen_candidate_smiles: dict[str, int] = {}
        reference_path = self.run_dir / "reference-ligand.sdf"
        receptor_path = self.run_dir / "receptor-protein-only.pdb"

        self._design_phase = True
        self._refresh_site_search()
        self._emit("optimization_started", {
            **settings,
            "hard_max_attempts": hard_max,
            "site_strategy": self.state.site_strategy,
            "active_target": self.state.active_target,
        })
        for attempt in range(1, hard_max + 1):
            try:
                self._validate_design(decision)
            except ReadyDecisionError as error:
                if (
                    attempt == 1
                    and error.rejection.get("failure_class") in {
                        "site_lock_violation", "site_strategy_missing", "site_strategy_too_small",
                    }
                ):
                    decision = self._retry_ready_decision(decision, error.rejection)
                    if decision.get("action") == "STOP":
                        return {"status": "no_candidate_accepted", "stopping_reason": "llm_stop", "attempts": history}
                    self._validate_design(decision)
                else:
                    raise
            transformation = self._transformation(decision)
            parent_attempt = transformation.get("parent_attempt")
            parent_molecule = self._resolve_parent_candidate(parent_attempt)
            self._emit("parent_selected", {
                "attempt": attempt,
                "parent_attempt": parent_attempt or 0,
                "generation": transformation.get("generation", 1),
                "parent_metadata": self._parent_metadata_for(parent_attempt),
            })
            exploration_record = self._record_exploration_attempt(
                transformation,
                "submitted",
                "design",
                attempt=attempt,
            )
            self._emit("candidate_attempt_started", {
                "attempt": attempt,
                "operation": transformation.get("operation"),
                "edit_atom_index": transformation.get("edit_atom_index"),
                "cut_bond": transformation.get("cut_bond"),
                "fragment_id": transformation.get("fragment_id"),
                "fragment_smiles": transformation.get("fragment_smiles"),
            })
            try:
                result = apply_transformation(
                    parent_molecule,
                    transformation,
                    self.context.protein_atoms,
                    seed=17,
                )
                attempt_path = self.run_dir / f"edit-attempt-{attempt:02d}.sdf"
                write_sdf(result, attempt_path, name=f"edit-attempt-{attempt:02d}")
                report = {
                    "attempt": attempt,
                    "parent_attempt": parent_attempt,
                    "generation": transformation.get("generation", 1),
                    "decision": decision,
                    "transformation": transformation,
                    "validation": result.report,
                    "candidate_path": str(attempt_path),
                }
            except Exception as error:
                result = None
                report = {
                    "attempt": attempt,
                    "parent_attempt": parent_attempt,
                    "generation": transformation.get("generation", 1),
                    "decision": decision,
                    "transformation": transformation,
                    "validation": {
                        "status": "rejected",
                        "failure_stage": "deterministic_geometry_prescreen",
                        "failure_class": "candidate_construction_or_geometry",
                        "error": str(error),
                        "recommended_next_queries": [
                            "get_atom_environment",
                            "check_growth_space",
                            "validate_candidate_geometry",
                            "search_fragment_library",
                        ],
                    },
                    "candidate_path": None,
                }
            history.append(report)

            if result is None or result.report["status"] != "accepted":
                rejection = {
                    **report["validation"],
                    "failure_stage": "deterministic_geometry_prescreen",
                    "docking": {
                        "stage": "docking",
                        "status": "not_run_geometry_rejected",
                        "message": "Docking was not started because deterministic candidate validation failed.",
                    },
                }
                report["validation"] = rejection
                report["docking"] = rejection["docking"]
                self._update_exploration_attempt(
                    exploration_record,
                    "geometry_rejected",
                    reason=rejection.get("error"),
                    details={"failure_class": rejection.get("failure_class")},
                )
                self._emit("candidate_geometry_rejected", {
                    "attempt": attempt,
                    "failure_class": rejection.get("failure_class"),
                    "error": rejection.get("error"),
                    "severe_clash_count": rejection.get("severe_clash_count"),
                    "worst_clash_residue": rejection.get("worst_clash_residue"),
                    "worst_overlap": rejection.get("worst_overlap"),
                    "docking_status": "not_run_geometry_rejected",
                })
                self._record_candidate_history(report, transformation)
                self._refresh_site_search()
                self._write_json(f"edit-attempt-{attempt:02d}.json", report)
                if attempt == hard_max:
                    break
                decision = self._retry_ready_decision(decision, rejection)
                if decision.get("action") == "STOP":
                    if any(item.get("docking", {}).get("status") == "complete" for item in history):
                        return self._accepted_output(history, reference_path, "llm_stop")
                    return {"status": "no_candidate_accepted", "stopping_reason": "llm_stop", "attempts": history}
                continue

            candidate_smiles = result.report["candidate"]["canonical_smiles"]
            previous_attempt = seen_candidate_smiles.get(candidate_smiles)
            if previous_attempt is not None:
                duplicate = {
                    "status": "rejected",
                    "failure_stage": "candidate_identity_check",
                    "failure_class": "duplicate_candidate_structure",
                    "canonical_smiles": candidate_smiles,
                    "first_seen_attempt": previous_attempt,
                    "docking": {
                        "stage": "docking",
                        "status": "not_run_duplicate_candidate",
                    },
                }
                report["validation"] = duplicate
                report["docking"] = duplicate["docking"]
                self._update_exploration_attempt(
                    exploration_record,
                    "duplicate_structure",
                    reason="Canonical candidate structure was already attempted.",
                    details={"first_seen_attempt": previous_attempt},
                )
                self._record_candidate_history(report, transformation)
                self._refresh_site_search()
                self._write_json(f"edit-attempt-{attempt:02d}.json", report)
                if attempt == hard_max:
                    break
                decision = self._retry_ready_decision(decision, duplicate)
                if decision.get("action") == "STOP":
                    if any(item.get("docking", {}).get("status") == "complete" for item in history):
                        return self._accepted_output(history, reference_path, "llm_stop")
                    return {"status": "no_candidate_accepted", "stopping_reason": "llm_stop", "attempts": history}
                continue
            seen_candidate_smiles[candidate_smiles] = attempt
            self._update_exploration_attempt(
                exploration_record,
                "geometry_accepted",
                details={"canonical_smiles": candidate_smiles},
            )
            self._refresh_site_search()
            self._emit("candidate_geometry_accepted", {
                "canonical_smiles": candidate_smiles,
                "property_delta": result.report.get("property_delta"),
                "severe_clash_count": result.report.get("severe_clash_count"),
            })

            candidate_path = self.run_dir / f"candidate-{attempt:02d}.sdf"
            write_sdf(result, candidate_path, name=f"candidate-{attempt:02d}")
            report["candidate_path"] = str(candidate_path)
            if not reference_path.exists():
                write_sdf(
                    EditResult(self.context.ligand, {"status": "reference"}),
                    reference_path,
                    name="reference-ligand",
                )
            if not receptor_path.exists():
                self.context.write_receptor_pdb(receptor_path)

            self._emit("docking_started", {
                "attempt": attempt,
                "candidate_path": str(candidate_path),
                "reference_path": str(reference_path),
                "receptor_path": str(receptor_path),
            })
            if hasattr(self.docking_adapter, "run_with_reference_baseline"):
                docking = self.docking_adapter.run_with_reference_baseline(
                    candidate_path=candidate_path,
                    receptor_path=receptor_path,
                    reference_path=reference_path,
                    output_dir=self.run_dir / f"docking-attempt-{attempt:02d}",
                    reference_output_dir=self.run_dir / "docking-reference-baseline",
                    reference_result=self.reference_docking_result,
                )
                baseline = docking.get("reference_baseline")
                if isinstance(baseline, dict) and baseline.get("status") == "complete":
                    self.reference_docking_result = baseline
            else:
                docking = self.docking_adapter.run(
                    candidate_path=candidate_path,
                    receptor_path=receptor_path,
                    reference_path=reference_path,
                    output_dir=self.run_dir / f"docking-attempt-{attempt:02d}",
                )
            report["docking"] = docking
            self._update_exploration_attempt(
                exploration_record,
                "docked" if docking.get("status") == "complete" else "docking_failed",
                reason=docking.get("error") or docking.get("message"),
            )
            self._record_candidate_history(report, transformation)
            self._refresh_site_search()
            self._write_json(f"edit-attempt-{attempt:02d}.json", report)
            self._emit("docking_completed", {
                "attempt": attempt,
                "status": docking.get("status"),
                "seed_count": docking.get("seed_count"),
                "pose_count_per_seed": docking.get("pose_count_per_seed"),
                "failure_class": docking.get("failure_class"),
                "error": docking.get("error") or docking.get("message"),
            })

            if docking.get("status") != "complete":
                return self._accepted_output(history, reference_path, "docking_not_complete")

            trend_entry = self._record_docking_result(
                attempt, transformation, candidate_path, docking
            )
            if trend_entry.get("stability_eligible"):
                self.parent_candidates[attempt] = Chem.Mol(result.molecule)
                self.parent_metadata[attempt] = {
                    "attempt": attempt,
                    "generation": transformation.get("generation", 1),
                    "target_type": self._transformation_target(transformation)["target_type"],
                    "target_id": self._transformation_target(transformation)["target_id"],
                    "quality": trend_entry.get("quality"),
                    "canonical_smiles": candidate_smiles,
                    "candidate_path": str(candidate_path),
                }
                self._emit("parent_available", self.parent_metadata[attempt])
            self._write_json("docking-history.json", {
                "history": self.state.docking_history,
                "convergence": self.state.convergence,
            })
            if trend_entry.get("raw_quality_from_mean") is None:
                return self._accepted_output(
                    history,
                    reference_path,
                    "no_candidate_revision_without_comparable_metric",
                )
            if attempt == hard_max:
                return self._accepted_output(history, reference_path, "hard_safety_limit")

            docking_summary = self._docking_feedback_summary(docking)
            feedback = {
                "failure_class": "docking_evaluation",
                "docking": {
                    "stage": "docking",
                    "result": docking_summary,
                    "candidate_path": str(candidate_path),
                    "receptor_path": str(receptor_path),
                    "reference_path": str(reference_path),
                },
                "latest_docking": docking_summary,
                "latest_trend_entry": {
                    key: trend_entry.get(key)
                    for key in (
                        "attempt", "transformation", "primary_metric", "direction",
                        "delta_candidate_minus_reference", "raw_quality_from_mean",
                        "seed_stddev", "seed_win_fraction", "stability_eligible", "quality",
                        "is_new_best", "best_attempt_so_far", "best_quality_so_far",
                    )
                },
                "incumbent": {
                    "best_attempt": self.state.convergence.get("best_attempt"),
                    "best_quality": self.state.convergence.get("best_quality"),
                },
                "convergence": self.state.convergence,
                "recommended_next_queries": [
                    "search_fragment_library",
                    "get_ligand_fragment",
                    "get_atom_environment",
                    "check_growth_space",
                    "validate_candidate_geometry",
                ],
            }
            next_decision = self._retry_ready_decision(decision, feedback)
            if next_decision.get("action") == "STOP":
                return self._accepted_output(history, reference_path, "llm_stop")
            next_transformation = self._transformation(next_decision)
            if self._same_transformation(transformation, next_transformation):
                if trend_entry.get("raw_quality_from_mean") is None:
                    return self._accepted_output(
                        history, reference_path, "no_candidate_revision_without_comparable_metric"
                    )
                revision_feedback = {
                    **feedback,
                    "failure_class": "candidate_not_revised",
                    "instruction": (
                        "The proposed transformation is identical to the just-docked candidate. "
                        "Use the docking feedback to query useful new evidence or return READY with "
                        "a chemically distinct, fully validated transformation. You may return STOP "
                        "when your evidence-based search is complete; no fixed regional attempt "
                        "target controls termination."
                    ),
                }
                next_decision = self._retry_ready_decision(decision, revision_feedback)
                if next_decision.get("action") == "STOP":
                    return self._accepted_output(history, reference_path, "llm_stop")
                next_transformation = self._transformation(next_decision)
                if self._same_transformation(transformation, next_transformation):
                    raise RuntimeError(
                        "LLM repeated the same transformation twice without selecting a new hypothesis"
                    )
            decision = next_decision

        if any(item.get("docking", {}).get("status") == "complete" for item in history):
            return self._accepted_output(history, reference_path, "hard_safety_limit")
        return {
            "status": "no_candidate_accepted",
            "stopping_reason": "hard_safety_limit",
            "attempts": history,
        }

    def run(self) -> dict[str, Any]:
        self._emit("workflow_started", {
            "task": self.state.task,
            "run_dir": str(self.run_dir),
            "ligand_heavy_atoms": self.context.ligand.GetNumHeavyAtoms(),
            "protein_atoms": len(self.context.protein_atoms),
        })
        first_decision = self.collect_context()
        result = self.design(first_decision)
        final = {"state": self.state.compact_view(), "result": result}
        self._write_json("result.json", final)
        self._emit("workflow_completed", {
            "status": result.get("status"),
            "stopping_reason": result.get("stopping_reason"),
            "attempt_count": len(result.get("attempts", [])),
            "best_attempt": result.get("best_attempt"),
            "candidate_path": result.get("candidate_path"),
            "result_path": str(self.run_dir / "result.json"),
        })
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
        observations = payload["state"]["observations"]
        if not any(item["tool"] == "get_edit_site_candidates" for item in observations):
            return {
                "action": "QUERY",
                "question": "List host-supported edit targets with local pocket and directional facts.",
                "tool": "get_edit_site_candidates",
                "arguments": {},
            }
        if not any(item["tool"] == "assess_edit_sites" for item in observations):
            return {
                "action": "QUERY",
                "question": "Rank the supported edit sites and classify their local role.",
                "tool": "assess_edit_sites",
                "arguments": {
                    "sites": [{
                        "target_type": "atom",
                        "target_id": 10,
                        "priority": 1,
                        "site_type": "pocket_extension",
                        "rationale": "Use the scripted phenyl edit site as a small pocket-extension test.",
                    }],
                    "global_rationale": "The scripted client supplies one auditable prioritized target.",
                },
            }
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
