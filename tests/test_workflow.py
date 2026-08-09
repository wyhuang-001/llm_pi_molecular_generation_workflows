from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from molecular_agent.adapters import NotConfiguredAdapter
from molecular_agent.models import AgentState, REQUIRED_EVIDENCE
from molecular_agent.structure import ComplexContext
from molecular_agent.tools import ToolRegistry
from molecular_agent.workflow import ScriptedDemoClient, Workflow
from scripts.openai_compatible_client import OpenAICompatibleChatClient


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "input" / "task.json"


def test_input_pdb_sdf_identity():
    context = ComplexContext(TASK)
    assert len(context.ligand_pdb_atoms) == 24
    assert context.ligand.GetNumHeavyAtoms() == 24
    assert len(context.protein_atoms) > 8000


def test_ligand_topology_comes_from_component_definition():
    context = ComplexContext(TASK)
    ligand = context.ligand
    assert context.ligand_source.endswith("input/raw/2A6.cif")
    assert Chem.MolToSmiles(ligand, isomericSmiles=True) == (
        "c1ccc(Nc2nc(OCC3CCCCC3)c3[nH]cnc3n2)cc1"
    )
    assert sum(atom.GetIsAromatic() for atom in ligand.GetAtoms()) == 15
    assert ligand.GetAtomWithIdx(1).GetTotalNumHs() == 1
    assert ligand.GetAtomWithIdx(1).GetIsAromatic()
    assert ligand.GetBondBetweenAtoms(1, 17).GetIsAromatic()
    assert ligand.GetBondBetweenAtoms(1, 18).GetIsAromatic()


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
        "get_ligand_fragment", {"atom_index": 1, "radius_bonds": 2}
    )
    assert evidence == {"fragment_properties"}
    assert result["atom_indices"]
    assert result["bond_indices"]
    assert result["properties"]["heavy_atoms"] == len(result["atom_indices"])


def test_ablation_tool_catalog_keeps_candidate_geometry_final_only():
    tools = ToolRegistry(ComplexContext(TASK))
    assert "validate_candidate_geometry" not in tools.catalog(include_candidate_geometry=False)
    assert "validate_candidate_geometry" in tools.catalog(include_candidate_geometry=True)


def test_candidate_geometry_evidence_records_exact_candidate_check():
    tools = ToolRegistry(ComplexContext(TASK))
    rejected, evidence = tools.execute(
        "validate_candidate_geometry", {"atom_index": 1, "fragment_smiles": "[*:1]C"}
    )
    assert rejected["status"] == "rejected"
    assert evidence == {"candidate_geometry"}
    accepted, evidence = tools.execute(
        "validate_candidate_geometry", {"atom_index": 9, "fragment_smiles": "[*:1]F"}
    )
    assert accepted["status"] == "accepted"
    assert evidence == {"candidate_geometry"}


def test_receptor_export_excludes_co_crystal_hetero_atoms(tmp_path):
    context = ComplexContext(TASK)
    path = context.write_receptor_pdb(tmp_path / "receptor.pdb")
    text = path.read_text(encoding="utf-8")
    assert "ATOM" in text
    assert "HETATM" not in text
    assert "2A6" not in text


def test_unconfigured_scoring_adapters_are_explicit(tmp_path):
    docking = NotConfiguredAdapter("docking").run()
    rbfe = NotConfiguredAdapter("rbfe").run()
    assert docking["status"] == "not_configured"
    assert rbfe["status"] == "not_configured"


def test_api_non_object_response_is_diagnosed(tmp_path, monkeypatch):
    fake_curl = tmp_path / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "output=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = \"-o\" ]; then output=$2; shift 2; else shift; fi\n"
        "done\n"
        "printf '%s' '{\"choices\":[{\"message\":{\"content\":\"[1,2]\"}}]}' > \"$output\"\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({
            "base_url": "https://example.invalid/v1",
            "model": "test",
            "api_key_env": "TEST_API_KEY",
        }),
        encoding="utf-8",
    )
    diagnostic = tmp_path / "api-error.json"
    client = OpenAICompatibleChatClient(config, "test")
    with pytest.raises(RuntimeError, match="got list"):
        client.complete_json({"mode": "test"}, diagnostic_path=diagnostic)
    report = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert report["failure"] == "assistant_json_type_list"
    assert report["assistant_content"] == "[1,2]"


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
    with pytest.raises(RuntimeError, match="lacks environment"):
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
                    "arguments": {"atom_index": 1, "radius": 4.0},
                }
            if self.calls == 2:
                return {
                    "action": "QUERY",
                    "question": "space",
                    "tool": "check_growth_space",
                    "arguments": {"atom_index": 1, "distance": 1.5},
                }
            if self.calls == 3:
                return {
                    "action": "QUERY",
                    "question": "candidate geometry",
                    "tool": "validate_candidate_geometry",
                    "arguments": {"atom_index": 1, "fragment_smiles": "[*:1]C"},
                }
            return {
                "action": "READY",
                "understanding": "site evidence",
                "edit_atom_index": 1,
                "edit_hypothesis": "small methyl edit",
                "fragment_smiles": "[*:1]C",
            }
        if payload["mode"] == "edit_retry":
            if self.calls == 5:
                return {
                    "action": "QUERY",
                    "question": "smaller fragment",
                    "tool": "get_fragment_properties",
                    "arguments": {"smiles": "[*:1]F"},
                }
            if self.calls == 6:
                return {
                    "action": "QUERY",
                    "question": "candidate geometry retry",
                    "tool": "validate_candidate_geometry",
                    "arguments": {"atom_index": 1, "fragment_smiles": "[*:1]F"},
                }
            return {
                "action": "READY",
                "understanding": "retry with the same chemically valid small edit",
                "edit_atom_index": 1,
                "edit_hypothesis": "retry small fluorine edit",
                "fragment_smiles": "[*:1]F",
            }
        raise AssertionError(payload["mode"])


def test_edit_retry_can_query_before_ready(tmp_path):
    result = Workflow(TASK, RetryQueryClient(), tmp_path).run()
    assert result["result"]["status"] == "no_candidate_accepted"
    assert len(result["state"]["observations"]) == 5
    assert len(result["state"]["decisions"]) == 9
    assert result["result"]["attempts"][1]["decision"]["fragment_smiles"] == "[*:1]F"
    assert result["state"]["decisions"][2]["tool"] == "validate_candidate_geometry"
    assert result["state"]["decisions"][4]["tool"] == "get_fragment_properties"
    assert result["state"]["decisions"][5]["tool"] == "validate_candidate_geometry"
    assert (tmp_path / "decision-04.json").exists()
    assert (tmp_path / "decision-09.json").exists()


class EnvironmentOnlyAblationClient:
    def complete_json(self, payload):
        if payload["mode"] == "collect_context":
            if not payload["state"]["tool_calls"]:
                return {
                    "action": "QUERY",
                    "question": "environment",
                    "tool": "get_atom_environment",
                    "arguments": {"atom_index": 10, "radius": 4.0},
                }
            return {
                "action": "READY",
                "understanding": "environment only",
                "edit_atom_index": 10,
                "edit_hypothesis": "small edit",
                "fragment_smiles": "[*:1]F",
            }
        raise AssertionError(payload["mode"])


class SevenQueryAblationClient:
    def complete_json(self, payload):
        if payload["mode"] != "collect_context":
            raise AssertionError(payload["mode"])
        if len(payload["state"]["tool_calls"]) < 7:
            return {
                "action": "QUERY",
                "question": "additional context",
                "tool": "get_ligand_info",
                "arguments": {},
            }
        return {
            "action": "READY",
            "understanding": "seven queries before final decision",
            "edit_atom_index": 10,
            "edit_hypothesis": "small edit",
            "fragment_smiles": "[*:1]F",
        }


def test_final_unbounded_run_is_not_forced_ready_at_budget_number(tmp_path):
    from scripts.compare_tool_budgets import run_budget

    config = tmp_path / "config.json"
    config.write_text(
        '{"base_url":"https://example.invalid/v1","model":"test","api_key_env":"TEST_KEY"}'
    )
    module = __import__("scripts.compare_tool_budgets", fromlist=["OpenAICompatibleChatClient"])
    original = module.OpenAICompatibleChatClient
    module.OpenAICompatibleChatClient = lambda *_args, **_kwargs: SevenQueryAblationClient()
    try:
        result = run_budget(
            TASK, config, tmp_path / "run", 6, "pocket", 6.0,
            require_site_evidence=True, unbounded=True,
        )
    finally:
        module.OpenAICompatibleChatClient = original
    assert result["tool_call_count"] == 7
    assert result["status"] == "site_evidence_gate_failed"


def test_final_unbounded_ablation_requires_site_evidence(tmp_path):
    from scripts.compare_tool_budgets import run_budget

    config = tmp_path / "config.json"
    config.write_text(
        '{"base_url":"https://example.invalid/v1","model":"test","api_key_env":"TEST_KEY"}'
    )
    original = __import__("scripts.compare_tool_budgets", fromlist=["OpenAICompatibleChatClient"]).OpenAICompatibleChatClient
    module = __import__("scripts.compare_tool_budgets", fromlist=["OpenAICompatibleChatClient"])
    module.OpenAICompatibleChatClient = lambda *_args, **_kwargs: EnvironmentOnlyAblationClient()
    try:
        result = run_budget(
            TASK, config, tmp_path / "run", 6, "pocket", 6.0,
            require_site_evidence=True, unbounded=True,
        )
    finally:
        module.OpenAICompatibleChatClient = original
    assert result["status"] == "site_evidence_gate_failed"
    assert result["state"]["unbounded_tool_calls"] is True
    assert result["result"]["ready_gate"]["missing"] == ["edit_site_geometry", "candidate_geometry"]
    assert not (tmp_path / "run" / "budget-06" / "candidate.sdf").exists()


def test_legacy_budget_does_not_require_site_evidence(tmp_path):
    from scripts.compare_tool_budgets import run_budget

    config = tmp_path / "config.json"
    config.write_text(
        '{"base_url":"https://example.invalid/v1","model":"test","api_key_env":"TEST_KEY"}'
    )
    original = __import__("scripts.compare_tool_budgets", fromlist=["OpenAICompatibleChatClient"]).OpenAICompatibleChatClient
    module = __import__("scripts.compare_tool_budgets", fromlist=["OpenAICompatibleChatClient"])
    module.OpenAICompatibleChatClient = lambda *_args, **_kwargs: EnvironmentOnlyAblationClient()
    try:
        result = run_budget(TASK, config, tmp_path / "run", 1, "pocket", 6.0)
    finally:
        module.OpenAICompatibleChatClient = original
    assert result["status"] != "site_evidence_gate_failed"
    assert result["state"]["site_evidence_gate_required"] is False
