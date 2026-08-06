from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from molecular_agent.models import AgentState, REQUIRED_EVIDENCE
from molecular_agent.structure import ComplexContext
from molecular_agent.tools import ToolRegistry
from molecular_agent.workflow import ScriptedDemoClient, Workflow


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "input" / "task.json"


def test_input_pdb_sdf_identity():
    context = ComplexContext(TASK)
    assert len(context.ligand_pdb_atoms) == 24
    assert context.ligand.GetNumHeavyAtoms() == 24
    assert len(context.protein_atoms) > 8000


def test_evidence_gate_requires_all_categories():
    state = AgentState(task="increase activity", max_context_rounds=8)
    assert not state.ready
    state.evidence.update(REQUIRED_EVIDENCE - {"edit_site_geometry"})
    assert not state.ready
    state.evidence.add("edit_site_geometry")
    assert state.ready


def test_small_query_does_not_cover_required_evidence():
    tools = ToolRegistry(ComplexContext(TASK))
    _, evidence = tools.execute("get_pocket_residues", {"radius": 3.0})
    assert evidence == set()
    _, evidence = tools.execute("get_pocket_residues", {"radius": 5.0})
    assert evidence == {"pocket_environment"}


def test_ligand_fragment_returns_connected_atom_and_bond_subgraph():
    tools = ToolRegistry(ComplexContext(TASK))
    result, evidence = tools.execute(
        "get_ligand_fragment", {"atom_index": 0, "radius_bonds": 2}
    )
    assert evidence == {"fragment_properties"}
    assert result["atom_indices"]
    assert result["bond_indices"]
    assert result["properties"]["heavy_atoms"] == len(result["atom_indices"])


def test_scripted_workflow_produces_valid_local_candidate(tmp_path):
    result = Workflow(TASK, ScriptedDemoClient(), tmp_path).run()
    assert result["state"]["missing_site_evidence"] == []
    assert result["result"]["status"] == "candidate_accepted"
    candidate = Chem.SDMolSupplier(result["result"]["candidate_path"], removeHs=False)[0]
    assert candidate is not None
    assert Chem.GetFormalCharge(candidate) == 0
    assert candidate.GetNumHeavyAtoms() == 25
    assert result["result"]["docking"]["status"] == "not_configured"
    assert result["result"]["fep"]["status"] == "not_configured"


class WrongSiteClient(ScriptedDemoClient):
    def complete_json(self, payload):
        decision = super().complete_json(payload)
        if decision.get("action") == "READY":
            decision["edit_atom_index"] = 11
        return decision


def test_ready_cannot_switch_to_unqueried_edit_site(tmp_path):
    workflow = Workflow(TASK, WrongSiteClient(), tmp_path)
    with pytest.raises(RuntimeError, match="lacks site-specific"):
        workflow.collect_context()


class DuplicateClient:
    def complete_json(self, payload):
        return {
            "action": "QUERY",
            "question": "repeat",
            "tool": "get_ligand_info",
            "arguments": {},
            "expected_evidence": "identity",
        }


def test_duplicate_tool_call_is_blocked(tmp_path):
    workflow = Workflow(TASK, DuplicateClient(), tmp_path)
    with pytest.raises(RuntimeError, match="Duplicate tool call"):
        workflow.collect_context()


class RetryQueryClient:
    def __init__(self):
        self.calls = 0

    def complete_json(self, payload):
        self.calls += 1
        if payload["mode"] == "context_collection":
            if self.calls == 1:
                return {
                    "action": "QUERY",
                    "question": "site",
                    "tool": "get_atom_environment",
                    "arguments": {"atom_index": 0, "radius": 4.0},
                }
            if self.calls == 2:
                return {
                    "action": "QUERY",
                    "question": "space",
                    "tool": "check_growth_space",
                    "arguments": {"atom_index": 0, "distance": 1.5},
                }
            return {
                "action": "READY",
                "understanding": "site evidence",
                "edit_atom_index": 0,
                "edit_hypothesis": "small methyl edit",
                "fragment_smiles": "[*:1]C",
            }
        if payload["mode"] == "edit_retry":
            if self.calls == 4:
                return {
                    "action": "QUERY",
                    "question": "smaller fragment",
                    "tool": "get_fragment_properties",
                    "arguments": {"smiles": "[*:1]F"},
                }
            return {
                "action": "READY",
                "understanding": "retry with the same chemically valid small edit",
                "edit_atom_index": 0,
                "edit_hypothesis": "retry small fluorine edit",
                "fragment_smiles": "[*:1]F",
            }
        raise AssertionError(payload["mode"])


def test_edit_retry_can_query_before_ready(tmp_path):
    result = Workflow(TASK, RetryQueryClient(), tmp_path).run()
    assert result["result"]["status"] == "candidate_accepted"
    assert len(result["state"]["observations"]) == 3
    assert len(result["state"]["decisions"]) == 5
    assert (tmp_path / "decision-04.json").exists()
    assert (tmp_path / "decision-05.json").exists()
