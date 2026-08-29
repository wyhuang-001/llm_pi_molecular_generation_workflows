from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from molecular_agent.adapters import DockingAdapter, NotConfiguredAdapter
from molecular_agent.editing import apply_transformation
from molecular_agent.llm import ResponsesClient
from molecular_agent.models import AgentState, REQUIRED_EVIDENCE, ToolObservation
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


def test_llm_assessed_edit_site_strategy_is_host_validated():
    tools = ToolRegistry(ComplexContext(TASK))
    result, evidence = tools.execute(
        "assess_edit_sites",
        {
            "sites": [
                {
                    "target_type": "atom",
                    "target_id": 10,
                    "priority": 1,
                    "site_type": "pocket_extension",
                    "rationale": "The outward phenyl direction may reach unused pocket volume.",
                },
            ],
            "global_rationale": "Prioritize a geometrically accessible extension vector.",
        },
    )
    assert result["status"] == "complete"
    assert result["sites"][0]["priority"] == 1
    assert result["sites"][0]["site_type"] == "pocket_extension"
    assert evidence == {"site_strategy"}

    with pytest.raises(ValueError, match="Unknown or non-editable"):
        tools.assess_edit_sites([{
            "target_type": "atom",
            "target_id": 999,
            "priority": 1,
            "site_type": "uncertain",
            "rationale": "invalid host target",
        }])


def test_site_candidate_batch_is_operation_specific_and_geometry_screened():
    tools = ToolRegistry(ComplexContext(TASK))
    result, evidence = tools.execute(
        "generate_site_candidate_batch",
        {"target_type": "atom", "target_id": 10, "query": "fluoro", "limit": 3},
    )
    assert result["status"] == "complete"
    assert result["target_type"] == "atom"
    assert result["operation"] == "substitute"
    assert result["accepted_count"] >= 1
    assert all(item["transformation"]["edit_atom_index"] == 10 for item in result["candidates"])
    assert evidence == {"candidate_batch"}


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


def test_fragment_library_is_searchable_and_auditable():
    tools = ToolRegistry(ComplexContext(TASK))
    result, evidence = tools.execute(
        "search_fragment_library",
        {"query": "fluoro", "operation": "substitute", "limit": 5},
    )
    assert evidence == {"fragment_library"}
    assert result["match_mode"] == "chemical_substructure"
    assert result["fragments"][0]["fragment_id"] == "fluoro"
    assert result["fragments"][0]["smiles"] == "[*:1]F"
    assert result["library_path"].endswith("molecular_agent/data/fragments.json")


def test_fragment_library_rejects_natural_language_query_and_suggests_terms():
    tools = ToolRegistry(ComplexContext(TASK))
    rejected, evidence = tools.execute(
        "search_fragment_library",
        {
            "query": "small polar heterocycle hydrogen bond donor acceptor",
            "operation": "replace_fragment",
            "max_heavy_atoms": 10,
            "limit": 20,
        },
    )
    assert rejected["status"] == "rejected"
    assert rejected["failure_class"] == "unsupported_fragment_query"
    assert evidence == set()
    assert "heterocycle" in rejected["supported_chemical_queries"]
    assert rejected["recommended_queries"]

    accepted, evidence = tools.execute(
        "search_fragment_library",
        {"query": "heterocycle", "operation": "substitute", "limit": 5},
    )
    assert accepted["status"] == "complete"
    assert accepted["match_mode"] == "chemical_substructure"
    assert evidence == {"fragment_library"}


def test_fragment_library_strictly_filters_requested_operation(tmp_path):
    library_path = tmp_path / "fragments.json"
    library_path.write_text(json.dumps({
        "allowed_operations": ["substitute"],
        "fragments": [
            {"fragment_id": "sub-only", "name": "pyridyl", "smiles": "[*:1]c1ccncc1", "operation": "legacy-value"},
            {"fragment_id": "replace-ok", "name": "pyridyl replacement", "smiles": "[*:1]c1ccncc1", "allowed_operations": ["replace_fragment"]},
        ],
    }), encoding="utf-8")
    from molecular_agent.fragment_library import FragmentLibrary

    library = FragmentLibrary(library_path)
    substitute = library.search("pyridine", operation="substitute")
    replacement = library.search("pyridine", operation="replace_fragment")

    assert substitute["match_mode"] == "chemical_substructure"
    assert [item["fragment_id"] for item in substitute["fragments"]] == ["sub-only"]
    assert [item["fragment_id"] for item in replacement["fragments"]] == ["replace-ok"]
    assert all(
        "replace_fragment" in item.get("allowed_operations", [])
        for item in replacement["fragments"]
    )


def test_fragment_library_does_not_relabel_substituents_as_replacements():
    tools = ToolRegistry(ComplexContext(TASK))
    result, _evidence = tools.execute(
        "search_fragment_library",
        {"query": "phenyl", "operation": "replace_fragment", "limit": 5},
    )
    assert result["operation_compatible_records"] == 0
    assert result["fragments"] == []

    site = tools.list_fragment_replacement_sites(limit=1)["sites"][0]
    rejected, _evidence = tools.execute(
        "validate_candidate_geometry",
        {
            "operation": "replace_fragment",
            "replacement_site_id": site["replacement_site_id"],
            "fragment_id": "phenyl",
            "fragment_smiles": "[*:1]c1ccccc1",
        },
    )
    assert rejected["status"] == "rejected"
    assert "does not allow operation replace_fragment" in rejected["error"]


def test_polar_contacts_report_donor_acceptor_compatibility():
    tools = ToolRegistry(ComplexContext(TASK))
    result = tools.detect_basic_interactions(4.5)
    polar = [item for item in result["contacts"] if item["kind"] == "polar_contact_candidate"]

    assert polar
    assert all("ligand_roles" in item and "protein_roles" in item for item in polar)
    assert all("hydrogen_bond_role_compatible" in item for item in polar)
    incompatible = [item for item in polar if not item["hydrogen_bond_role_compatible"]]
    assert incompatible
    assert all(item["role_warning"] for item in incompatible)


def test_fragment_replacement_sites_preserve_the_larger_ring_rich_scaffold():
    tools = ToolRegistry(ComplexContext(TASK))
    result, evidence = tools.execute("list_fragment_replacement_sites", {"limit": 20})
    assert evidence == {"replacement_sites"}
    assert result["count"] > 0
    assert all(site["retained_heavy_atoms"] > site["removed_heavy_atoms"] for site in result["sites"])
    assert all(site["retained_ring_atoms"] >= site["removed_ring_atoms"] for site in result["sites"])
    assert all(site["removed_fraction"] <= 0.4 for site in result["sites"])
    assert all(site["replacement_site_id"].startswith("replacement-site-") for site in result["sites"])


def test_replacement_site_spatial_profile_returns_directional_facts():
    tools = ToolRegistry(ComplexContext(TASK))
    site = tools.list_fragment_replacement_sites(limit=1)["sites"][0]
    result, evidence = tools.execute(
        "get_replacement_site_spatial_profile",
        {"replacement_site_id": site["replacement_site_id"]},
    )
    assert result["status"] == "complete"
    assert evidence == {"replacement_site_spatial_profile"}
    assert result["replacement_site_id"] == site["replacement_site_id"]
    assert len(result["attachment_unit_vector"]) == 3
    assert result["direction_profiles"]
    assert all(
        "minimum_protein_atom_distance_along_probe" in profile
        for profile in result["direction_profiles"]
    )
    assert "limitation" in result


def test_fragment_spatial_profile_returns_attachment_centered_facts():
    tools = ToolRegistry(ComplexContext(TASK))
    fragment_smiles = "C1OCC1[*:1]"
    result, evidence = tools.execute(
        "get_fragment_spatial_profile",
        {"fragment_smiles": fragment_smiles},
    )
    assert result["status"] == "complete"
    assert evidence == {"fragment_spatial_profile"}
    assert result["fragment_smiles"] == fragment_smiles
    assert result["heavy_atoms"] == 4
    assert result["conformer_count"] >= 1
    assert result["max_attachment_distance"]["maximum"] > 0
    assert result["representative_conformer_atoms"]
    assert "limitation" in result


def test_fragment_replacement_site_id_resolves_cut_direction():
    tools = ToolRegistry(ComplexContext(TASK))
    site = tools.list_fragment_replacement_sites(limit=1)["sites"][0]
    result, evidence = tools.execute(
        "validate_candidate_geometry",
        {
            "operation": "replace_fragment",
            "replacement_site_id": site["replacement_site_id"],
            "fragment_smiles": "[*:1]F",
        },
    )
    assert evidence == {"candidate_geometry"}
    assert result["transformation"]["cut_bond"] == site["cut_bond"]
    assert result["transformation"]["edit_atom_index"] == site["retained_atom_index"]
    assert result["transformation"]["replacement_site"]["removed_atom_indices"] == site["removed_atom_indices"]


def test_fragment_replacement_rejects_direct_cut_bond_guessing():
    tools = ToolRegistry(ComplexContext(TASK))
    result, _evidence = tools.execute(
        "validate_candidate_geometry",
        {
            "operation": "replace_fragment",
            "cut_bond": [14, 15],
            "fragment_smiles": "[*:1]F",
        },
    )
    assert result["status"] == "rejected"
    assert "replacement_site_id" in result["error"]


def test_replace_fragment_does_not_require_hydrogen_on_retained_atom():
    context = ComplexContext(TASK)
    # Bond 14-15 connects the heteroaromatic scaffold to an O-cyclohexyl side chain.
    assert context.ligand.GetAtomWithIdx(14).GetTotalNumHs() == 0
    result = apply_transformation(
        context.ligand,
        {
            "operation": "replace_fragment",
            "edit_atom_index": 14,
            "cut_bond": [14, 15],
            "fragment_smiles": "[*:1]F",
        },
        context.protein_atoms,
    )
    assert result.report["operation"] == "replace_fragment"
    assert result.report["structure_change"]["cut_bond"] == [14, 15]
    assert 15 in result.report["structure_change"]["removed_atom_indices"]
    assert Chem.GetFormalCharge(result.molecule) == Chem.GetFormalCharge(context.ligand)


def test_llm_state_view_compacts_fragment_provenance_without_mutating_audit(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    result = {
        "status": "complete",
        "fragments": [{
            "fragment_id": "example",
            "smiles": "[*:1]F",
            "source_molecule_ids": [f"CHEMBL{i}" for i in range(100)],
        }],
    }
    workflow.state.observations.append(ToolObservation(
        tool="search_fragment_library",
        arguments={"query": "fluoro"},
        result=result,
        evidence={"fragment_library"},
    ))

    view = workflow._llm_state_view()

    assert "source_molecule_ids" not in view["observations"][0]["result"]["fragments"][0]
    assert len(workflow.state.observations[0].result["fragments"][0]["source_molecule_ids"]) == 100


def test_query_batch_executes_each_distinct_tool_call(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    action = workflow._handle_decision({
        "action": "QUERY_BATCH",
        "queries": [
            {"tool": "get_ligand_info", "arguments": {}},
            {"tool": "get_pocket_residues", "arguments": {"radius": 5.0}},
        ],
    })
    assert action == "QUERY_BATCH"
    assert [item.tool for item in workflow.state.observations] == [
        "get_ligand_info", "get_pocket_residues"
    ]


def test_query_batch_skips_repeated_item_and_executes_new_items(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    workflow._handle_decision({
        "action": "QUERY",
        "tool": "get_ligand_info",
        "arguments": {},
    })

    workflow._handle_decision({
        "action": "QUERY_BATCH",
        "queries": [
            {"tool": "get_ligand_info", "arguments": {}},
            {"tool": "get_pocket_residues", "arguments": {"radius": 5.0}},
        ],
    })

    assert [item.tool for item in workflow.state.observations] == [
        "get_ligand_info", "get_pocket_residues"
    ]
    assert workflow.state.tool_rejections[-1]["failure_class"] == "duplicate_tool_call"


def test_query_batch_skips_internal_duplicate_and_executes_once(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    workflow._handle_decision({
        "action": "QUERY_BATCH",
        "queries": [
            {"tool": "get_ligand_info", "arguments": {}},
            {"tool": "get_ligand_info", "arguments": {}},
        ],
    })
    assert [item.tool for item in workflow.state.observations] == ["get_ligand_info"]


def test_placeholder_fragment_id_is_treated_as_omitted(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    transformation = workflow._transformation({
        "operation": "replace_hydrogen",
        "edit_atom_index": 9,
        "fragment_id": "optional",
        "fragment_smiles": "[*:1]F",
    })
    assert "fragment_id" not in transformation
    assert transformation["fragment_smiles"] == "[*:1]F"


def test_transformation_identity_normalizes_cut_bond_order(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    workflow.state.candidate_history.append({
        "transformation": {
            "operation": "replace_fragment",
            "edit_atom_index": 14,
            "cut_bond": [15, 14],
            "fragment_smiles": "[*:1]F",
        }
    })
    assert workflow._transformation_was_attempted({
        "operation": "replace_fragment",
        "edit_atom_index": 14,
        "cut_bond": [14, 15],
        "fragment_smiles": "[*:1]F",
    })


def test_site_lock_blocks_jump_until_active_target_local_search_completes(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    workflow._execute_query({
        "action": "QUERY",
        "tool": "get_edit_site_candidates",
        "arguments": {},
    })
    workflow._execute_query({
        "action": "QUERY",
        "tool": "assess_edit_sites",
        "arguments": {
            "sites": [
                {
                    "target_type": "atom",
                    "target_id": 10,
                    "priority": 1,
                    "site_type": "pocket_extension",
                    "rationale": "Explore the first phenyl vector locally.",
                },
                {
                    "target_type": "atom",
                    "target_id": 9,
                    "priority": 2,
                    "site_type": "solvent_exposed",
                    "rationale": "Reserve the adjacent site for the next local phase.",
                },
            ],
        },
    })
    workflow._design_phase = True
    workflow._refresh_site_search()

    assert workflow.state.active_target["target_id"] == 10
    rejection = workflow._site_lock_rejection({
        "operation": "replace_hydrogen",
        "edit_atom_index": 9,
        "fragment_smiles": "[*:1]F",
    })
    assert rejection is not None
    assert rejection["failure_class"] == "site_lock_violation"
    assert rejection["active_target"]["target_id"] == 10
    with pytest.raises(RuntimeError, match="current active prioritized site"):
        workflow._execute_query({
            "action": "QUERY",
            "tool": "generate_site_candidate_batch",
            "arguments": {"target_type": "atom", "target_id": 9, "query": "fluoro"},
        })


def test_docking_trend_preserves_best_attempt_without_auto_convergence(tmp_path):
    task = json.loads(TASK.read_text(encoding="utf-8"))
    task["docking_optimization"] = {
        "primary_metric": "minimizedAffinity",
        "minimum_improvement": 0.25,
        "hard_max_attempts": 20,
    }
    task_path = tmp_path / "task.json"
    task["complex_path"] = str((ROOT / "input" / "complex.pdb").resolve())
    task_path.write_text(json.dumps(task), encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "2A6.cif").write_text(
        (ROOT / "input" / "raw" / "2A6.cif").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    workflow = Workflow(task_path, ScriptedDemoClient(), tmp_path / "run")
    qualities = [0.5, 1.0, 0.9, 1.1, 1.05]
    entries = []
    for attempt, quality in enumerate(qualities, start=1):
        entries.append(workflow._record_docking_result(
            attempt,
            {"operation": "replace_hydrogen", "edit_atom_index": 10, "fragment_smiles": f"f{attempt}"},
            tmp_path / f"candidate-{attempt}.sdf",
            {
                "status": "complete",
                "comparison": {
                    "metrics": {
                        "minimizedAffinity": {
                            "direction": "lower_is_better",
                            "delta_candidate_minus_reference": {"mean": -quality},
                        }
                    }
                },
            },
        ))
    assert workflow.state.convergence["converged"] is False
    assert workflow.state.convergence["llm_controls_termination"] is True
    assert workflow.state.convergence["best_attempt"] == 4
    assert [entry["best_quality_so_far"] for entry in entries] == [0.5, 1.0, 1.0, 1.1, 1.1]


def test_design_regions_are_descriptive_only(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    workflow._record_docking_result(
        1,
        {"operation": "replace_hydrogen", "edit_atom_index": 9, "fragment_smiles": "[*:1]F"},
        tmp_path / "candidate-1.sdf",
        {
            "status": "complete",
            "comparison": {"metrics": {"minimizedAffinity": {
                "direction": "lower_is_better",
                "delta_candidate_minus_reference": {"mean": -0.4, "stddev": 0.1},
                "candidate_better_seed_fraction": 1.0,
            }}},
        },
    )
    workflow._record_docking_result(
        2,
        {"operation": "replace_hydrogen", "edit_atom_index": 10, "fragment_smiles": "[*:1]Cl"},
        tmp_path / "candidate-2.sdf",
        {
            "status": "complete",
            "comparison": {"metrics": {"minimizedAffinity": {
                "direction": "lower_is_better",
                "delta_candidate_minus_reference": {"mean": 0.2, "stddev": 0.4},
                "candidate_better_seed_fraction": 1 / 3,
            }}},
        },
    )
    convergence = workflow.state.convergence
    assert convergence["validated_design_regions"] == []
    assert convergence["docked_design_regions"] == ["atom:10", "atom:9"]
    assert convergence["design_region_count_is_descriptive_only"] is True
    assert convergence["converged"] is False


def test_global_stop_gate_requires_non_halogen_followup_after_halogen_hit(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    workflow._record_docking_result(
        1,
        {"operation": "replace_hydrogen", "edit_atom_index": 9, "fragment_smiles": "[*:1]F"},
        tmp_path / "candidate-1.sdf",
        {
            "status": "complete",
            "comparison": {"metrics": {"minimizedAffinity": {
                "direction": "lower_is_better",
                "delta_candidate_minus_reference": {"mean": -0.8, "stddev": 0.1},
                "candidate_better_seed_fraction": 1.0,
            }}},
        },
    )
    rejection = workflow._stop_gate_rejection()
    assert rejection is not None
    assert rejection["failure_class"] == "global_search_incomplete"
    coverage = rejection["global_search"]
    assert 9 in coverage["halogen_hit_atoms_missing_non_halogen_followup"]
    assert coverage["complete"] is False


def test_global_search_counts_geometry_rejection_as_exploration(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    workflow._execute_query({
        "action": "QUERY",
        "tool": "validate_candidate_geometry",
        "arguments": {
            "operation": "replace_hydrogen",
            "edit_atom_index": 17,
            "fragment_smiles": "[*:1]F",
        },
    })
    coverage = workflow._global_search_coverage()
    assert coverage["attempted_atoms"]["17"][0]["status"] == "geometry_rejected"
    assert any(item["atom_index"] == 17 for item in coverage["host_ineligible_hydrogen_atoms"])
    assert 17 not in coverage["editable_hydrogen_atoms"]
    assert 17 not in coverage["missing_edit_atoms"]
    assert not any(item["atom_index"] == 17 for item in coverage["missing_atom_coverage"])


def test_unsupported_aromatic_nh_is_reported_but_not_a_pending_edit_site(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    coverage = workflow._global_search_coverage()
    assert any(item["atom_index"] == 17 for item in coverage["host_ineligible_hydrogen_atoms"])
    assert 17 not in coverage["editable_hydrogen_atoms"]
    assert not any(item["atom_index"] == 17 for item in coverage["missing_atom_coverage"])


def test_mark_unmodifiable_closes_site_coverage(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    for fragment in ("[*:1]F", "[*:1]C"):
        workflow._record_exploration_attempt(
            {
                "operation": "replace_hydrogen",
                "edit_atom_index": 9,
                "fragment_smiles": fragment,
            },
            "geometry_rejected",
            "validate_candidate_geometry",
        )
    workflow._handle_decision({
        "action": "MARK_UNMODIFIABLE",
        "target_type": "atom",
        "target_id": 9,
        "scope": "site",
        "reason": "The accumulated pocket evidence does not support a credible edit at this site.",
    })
    coverage = workflow._global_search_coverage()
    assert 9 in coverage["closed_atoms"]
    assert 9 not in coverage["missing_edit_atoms"]
    assert not any(item["atom_index"] == 9 for item in coverage["missing_atom_coverage"])
    assert workflow.state.exploration_attempts[-1]["status"] == "llm_unmodifiable"


def test_rejected_geometry_satisfies_its_family_for_editable_atom(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    workflow._execute_query({
        "action": "QUERY",
        "tool": "validate_candidate_geometry",
        "arguments": {
            "operation": "replace_hydrogen",
            "edit_atom_index": 1,
            "fragment_smiles": "[*:1]F",
        },
    })
    coverage = workflow._global_search_coverage()
    atom = next(item for item in coverage["missing_atom_coverage"] if item["atom_index"] == 1)
    assert "halogen" not in atom["missing_families"]
    assert workflow._transformation_was_attempted({
        "operation": "replace_hydrogen",
        "edit_atom_index": 1,
        "fragment_smiles": "[*:1]F",
    })


def test_candidate_history_distinguishes_exploration_from_docking(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    transformation = {
        "operation": "replace_hydrogen",
        "edit_atom_index": 1,
        "fragment_smiles": "[*:1]C",
    }
    report = {
        "attempt": 1,
        "transformation": transformation,
        "candidate_path": None,
        "validation": {"status": "rejected", "failure_class": "steric_clash"},
        "docking": {"status": "not_run_geometry_rejected"},
    }
    workflow._record_candidate_history(report, transformation)
    entry = workflow.state.candidate_history[-1]
    assert entry["record_type"] == "design_attempt"
    assert entry["docking"]["entered_docking"] is False
    assert entry["docking"]["completed"] is False


def test_pending_obligations_include_adaptive_target_diversity(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    coverage = workflow._global_search_coverage()
    assert any(item["type"] == "target_diversity" for item in coverage["pending_obligations"])
    assert not coverage["complete"]


def test_global_stop_gate_requires_all_sites_and_modification_families(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    workflow.context.task["search_policy"] = {"mode": "family_coverage"}
    coverage = workflow._global_search_coverage()
    for index in coverage["editable_hydrogen_atoms"]:
        workflow.state.candidate_history.append({
            "attempt": len(workflow.state.candidate_history) + 1,
            "transformation": {
                "operation": "replace_hydrogen",
                "edit_atom_index": index,
                "fragment_smiles": "[*:1]F",
            },
            "validation": {"status": "accepted"},
            "docking": {"status": "complete"},
        })
        if float(coverage["atom_clearance"][str(index)]) >= 1.5:
            workflow.state.candidate_history.append({
                "attempt": len(workflow.state.candidate_history) + 1,
                "transformation": {
                    "operation": "replace_hydrogen",
                    "edit_atom_index": index,
                    "fragment_smiles": "[*:1]C",
                },
                "validation": {"status": "accepted"},
                "docking": {"status": "complete"},
            })
    for site_id in coverage["replacement_sites"]:
        for fragment in ("[*:1]C", "[*:1]N"):
            workflow.state.candidate_history.append({
                "attempt": len(workflow.state.candidate_history) + 1,
                "transformation": {
                    "operation": "replace_fragment",
                    "replacement_site_id": site_id,
                    "fragment_smiles": fragment,
                },
                "validation": {"status": "rejected"},
                "docking": {"status": "not_run_geometry_rejected"},
            })
    assert workflow._global_search_coverage()["complete"] is True
    assert workflow._stop_gate_rejection() is None


def test_seed_stability_penalizes_noisy_candidate(tmp_path):
    workflow = Workflow(TASK, ScriptedDemoClient(), tmp_path)
    stable = workflow._record_docking_result(
        1,
        {"operation": "replace_hydrogen", "edit_atom_index": 9, "fragment_smiles": "stable"},
        tmp_path / "stable.sdf",
        {
            "status": "complete",
            "comparison": {"metrics": {"minimizedAffinity": {
                "direction": "lower_is_better",
                "delta_candidate_minus_reference": {"mean": -0.5, "stddev": 0.2},
                "candidate_better_seed_fraction": 1.0,
            }}},
        },
    )
    noisy = workflow._record_docking_result(
        2,
        {"operation": "replace_hydrogen", "edit_atom_index": 10, "fragment_smiles": "noisy"},
        tmp_path / "noisy.sdf",
        {
            "status": "complete",
            "comparison": {"metrics": {"minimizedAffinity": {
                "direction": "lower_is_better",
                "delta_candidate_minus_reference": {"mean": -0.6, "stddev": 1.0},
                "candidate_better_seed_fraction": 2 / 3,
            }}},
        },
    )
    assert stable["quality"] == pytest.approx(0.45)
    assert noisy["quality"] == pytest.approx(0.35)
    assert workflow.state.convergence["best_attempt"] == 1


def test_receptor_export_excludes_co_crystal_hetero_atoms(tmp_path):
    context = ComplexContext(TASK)
    path = context.write_receptor_pdb(tmp_path / "receptor.pdb")
    text = path.read_text(encoding="utf-8")
    assert "ATOM" in text
    assert "HETATM" not in text
    assert "2A6" not in text


def test_docking_adapter_reads_pose_scores_and_audits_reference(tmp_path):
    script = tmp_path / "fake_docking.py"
    script.write_text(
        "import sys\n"
        "from rdkit import Chem\n"
        "candidate, output_dir = sys.argv[1:]\n"
        "molecule = Chem.SDMolSupplier(candidate, removeHs=False)[0]\n"
        "writer = Chem.SDWriter(output_dir + '/docked.sdf')\n"
        "for score in ('-8.1', '-7.4'):\n"
        "    molecule.SetProp('minimizedAffinity', score)\n"
        "    writer.write(molecule)\n"
        "writer.close()\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.sdf"
    reference = tmp_path / "reference.sdf"
    candidate.write_text((ROOT / "input" / "ligand.sdf").read_text(encoding="utf-8"), encoding="utf-8")
    reference.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
    output_dir = tmp_path / "docking-output"
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n", encoding="utf-8")
    adapter = DockingAdapter(
        {
            "enabled": True,
            "command": [
                __import__("sys").executable,
                str(script),
                "{candidate}",
                "{output_dir}",
            ],
        },
        tmp_path,
    )
    result = adapter.run(
        candidate_path=candidate,
        receptor_path=receptor,
        reference_path=reference,
        output_dir=output_dir,
    )
    assert result["status"] == "complete"
    assert result["pose_count"] == 2
    assert [pose["properties"]["minimizedAffinity"] for pose in result["poses"]] == ["-8.1", "-7.4"]
    assert result["pose_selection"]["gnina_rank_1"]["rank"] == 1
    assert result["pose_selection"]["best_by_minimizedAffinity"]["rank"] == 1
    audit = json.loads((output_dir / "docking-command.json").read_text(encoding="utf-8"))
    assert audit["inputs"]["reference"] == str(reference.resolve())


def test_docking_reference_baseline_produces_relative_score_comparison(tmp_path):
    script = tmp_path / "fake_docking.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from rdkit import Chem\n"
        "candidate, output_dir = sys.argv[1:]\n"
        "molecule = Chem.SDMolSupplier(candidate, removeHs=False)[0]\n"
        "is_reference = Path(candidate).name == 'reference.sdf'\n"
        "molecule.SetProp('minimizedAffinity', '-8.0' if is_reference else '-8.6')\n"
        "molecule.SetProp('CNNscore', '0.5' if is_reference else '0.6')\n"
        "molecule.SetProp('CNNaffinity', '6.0' if is_reference else '6.2')\n"
        "writer = Chem.SDWriter(output_dir + '/docked.sdf')\n"
        "writer.write(molecule)\n"
        "writer.close()\n",
        encoding="utf-8",
    )
    source = (ROOT / "input" / "ligand.sdf").read_text(encoding="utf-8")
    candidate = tmp_path / "candidate.sdf"
    reference = tmp_path / "reference.sdf"
    candidate.write_text(source, encoding="utf-8")
    reference.write_text(source, encoding="utf-8")
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n",
        encoding="utf-8",
    )
    adapter = DockingAdapter(
        {
            "enabled": True,
            "command": [
                __import__("sys").executable,
                str(script),
                "{candidate}",
                "{output_dir}",
            ],
        },
        tmp_path,
    )
    result = adapter.run_with_reference_baseline(
        candidate_path=candidate,
        receptor_path=receptor,
        reference_path=reference,
        output_dir=tmp_path / "candidate-docking",
        reference_output_dir=tmp_path / "reference-docking",
    )
    assert result["status"] == "complete"
    assert result["reference_baseline"]["status"] == "complete"
    assert result["comparison"]["status"] == "complete"
    affinity = result["comparison"]["metrics"]["minimizedAffinity"]
    assert affinity["delta_candidate_minus_reference"]["mean"] == pytest.approx(-0.6)
    assert affinity["candidate_better_seed_fraction"] == pytest.approx(1.0)
    cnn_score = result["comparison"]["metrics"]["CNNscore"]
    assert cnn_score["delta_candidate_minus_reference"]["mean"] == pytest.approx(0.1)
    assert cnn_score["candidate_better_seed_fraction"] == pytest.approx(1.0)
    assert (tmp_path / "reference-docking" / "seed-00017" / "docking-result.json").exists()


def test_docking_multi_seed_aggregates_paired_comparisons(tmp_path):
    script = tmp_path / "fake_docking.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from rdkit import Chem\n"
        "candidate, output_dir, seed = sys.argv[1:]\n"
        "molecule = Chem.SDMolSupplier(candidate, removeHs=False)[0]\n"
        "is_reference = Path(candidate).name == 'reference.sdf'\n"
        "offset = {'17': 0.1, '29': 0.3, '43': -0.2}[seed]\n"
        "molecule.SetProp('minimizedAffinity', str(-8.0 + offset if is_reference else -8.5 + offset))\n"
        "writer = Chem.SDWriter(output_dir + '/docked.sdf')\n"
        "writer.write(molecule)\n"
        "writer.close()\n",
        encoding="utf-8",
    )
    source = (ROOT / "input" / "ligand.sdf").read_text(encoding="utf-8")
    candidate = tmp_path / "candidate.sdf"
    reference = tmp_path / "reference.sdf"
    candidate.write_text(source, encoding="utf-8")
    reference.write_text(source, encoding="utf-8")
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \n",
        encoding="utf-8",
    )
    adapter = DockingAdapter(
        {
            "enabled": True,
            "seeds": [17, 29, 43],
            "command": [
                __import__("sys").executable,
                str(script),
                "{candidate}",
                "{output_dir}",
                "{seed}",
            ],
        },
        tmp_path,
    )
    result = adapter.run_with_reference_baseline(
        candidate_path=candidate,
        receptor_path=receptor,
        reference_path=reference,
        output_dir=tmp_path / "candidate-docking",
        reference_output_dir=tmp_path / "reference-docking",
    )
    assert result["status"] == "complete"
    metric = result["comparison"]["metrics"]["minimizedAffinity"]
    assert metric["n"] == 3
    assert metric["delta_candidate_minus_reference"]["mean"] == pytest.approx(-0.5)
    assert metric["delta_candidate_minus_reference"]["stddev"] == pytest.approx(0.0)
    assert metric["candidate_better_seed_fraction"] == pytest.approx(1.0)
    assert (tmp_path / "candidate-docking" / "seed-00017" / "docking-result.json").exists()
    assert (tmp_path / "reference-docking" / "seed-00043" / "docking-result.json").exists()
    assert result["pose_consensus"]["stable"] is True
    assert result["pose_consensus"]["largest_consistent_cluster_fraction"] == 1.0
    assert result["interaction_consensus"]["status"] == "complete"
    assert "candidate_consensus_residues" in result["interaction_consensus"]


def test_docking_preflight_blocks_invalid_receptor_before_command(tmp_path):
    marker = tmp_path / "command-ran"
    script = tmp_path / "must-not-run.py"
    script.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.sdf"
    reference = tmp_path / "reference.sdf"
    source = (ROOT / "input" / "ligand.sdf").read_text(encoding="utf-8")
    candidate.write_text(source, encoding="utf-8")
    reference.write_text(source, encoding="utf-8")
    adapter = DockingAdapter(
        {
            "enabled": True,
            "command": [__import__("sys").executable, str(script)],
        },
        tmp_path,
    )
    result = adapter.run(
        candidate_path=candidate,
        receptor_path=tmp_path / "missing-receptor.pdb",
        reference_path=reference,
        output_dir=tmp_path / "docking",
    )
    assert result["status"] == "failed"
    assert result["failure_class"] == "docking_input_preflight"
    assert not marker.exists()


def test_unconfigured_scoring_adapters_are_explicit(tmp_path):
    docking = NotConfiguredAdapter("docking").run()
    rbfe = NotConfiguredAdapter("rbfe").run()
    assert docking["status"] == "not_configured"
    assert rbfe["status"] == "not_configured"


def test_responses_client_reads_configured_key_file(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    key_file = tmp_path / "api-key"
    key_file.write_text("secret-from-file\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "base_url": "https://example.invalid/v1",
        "model": "test",
        "api_key_env": "TEST_API_KEY",
        "api_key_file": str(key_file),
    }), encoding="utf-8")
    client = ResponsesClient(config)
    assert client.api_key == "secret-from-file"


def test_responses_client_extracts_json_after_model_narration():
    parsed = ResponsesClient._extract_json_object(
        'The next action is straightforward. {"action":"QUERY","tool":"get_ligand_info","arguments":{}}'
    )
    assert parsed == {
        "action": "QUERY",
        "tool": "get_ligand_info",
        "arguments": {},
    }


def test_responses_client_rejects_incomplete_json():
    with pytest.raises(json.JSONDecodeError):
        ResponsesClient._extract_json_object('Return QUERY now: {"action":"QUERY"')


def test_responses_client_ignores_null_output_content(tmp_path, monkeypatch):
    fake_curl = tmp_path / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "output=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = \"-o\" ]; then output=$2; shift 2; else shift; fi\n"
        "done\n"
        "printf '%s' '{\"output_text\":null,\"output\":[{\"content\":[{\"type\":\"reasoning\",\"text\":null},{\"type\":\"output_text\",\"text\":\"{\\\"action\\\":\\\"QUERY\\\",\\\"tool\\\":\\\"get_ligand_info\\\",\\\"arguments\\\":{}}\"}]}]}' > \"$output\"\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "base_url": "https://example.invalid/v1",
        "model": "test",
        "api_key_env": "TEST_API_KEY",
    }), encoding="utf-8")

    client = ResponsesClient(config)
    decision = client.complete_json({"mode": "context_collection", "state": {}})

    assert decision == {
        "action": "QUERY",
        "tool": "get_ligand_info",
        "arguments": {},
    }


def test_responses_client_repairs_incomplete_json_once(tmp_path, monkeypatch):
    fake_curl = tmp_path / "curl"
    counter = tmp_path / "counter"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "output=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = \"-o\" ]; then output=$2; shift 2; else shift; fi\n"
        "done\n"
        f"counter={str(counter)!r}\n"
        "n=0; [ -f \"$counter\" ] && n=$(cat \"$counter\"); n=$((n+1)); printf '%s' \"$n\" > \"$counter\"\n"
        "if [ \"$n\" -eq 1 ]; then\n"
        "  printf '%s' '{\"output_text\":\"analysis without complete JSON {\\\"action\\\":\\\"QUERY\\\"\"}' > \"$output\"\n"
        "else\n"
        "  printf '%s' '{\"output_text\":\"{\\\"action\\\":\\\"QUERY\\\",\\\"tool\\\":\\\"get_ligand_info\\\",\\\"arguments\\\":{}}\"}' > \"$output\"\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "base_url": "https://example.invalid/v1",
        "model": "test",
        "api_key_env": "TEST_API_KEY",
    }), encoding="utf-8")

    client = ResponsesClient(config)
    decision = client.complete_json({"mode": "context_collection", "state": {}})

    assert decision["action"] == "QUERY"
    assert decision["tool"] == "get_ligand_info"
    assert counter.read_text() == "2"


def test_responses_client_repairs_api_incomplete_message_without_reasoning_context(tmp_path, monkeypatch):
    fake_curl = tmp_path / "curl"
    counter = tmp_path / "counter"
    request_log = tmp_path / "requests.jsonl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "output=''\n"
        "request=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = \"-o\" ]; then output=$2; shift 2;\n"
        "  elif [ \"$1\" = \"--data-binary\" ]; then request=${2#@}; shift 2;\n"
        "  else shift; fi\n"
        "done\n"
        f"counter={str(counter)!r}\n"
        f"request_log={str(request_log)!r}\n"
        "n=0; [ -f \"$counter\" ] && n=$(cat \"$counter\"); n=$((n+1)); printf '%s' \"$n\" > \"$counter\"\n"
        "printf '%s\\n' \"$(cat \"$request\")\" >> \"$request_log\"\n"
        "if [ \"$n\" -eq 1 ]; then\n"
        "  printf '%s' '{\"status\":\"incomplete\",\"output\":[{\"type\":\"reasoning\",\"status\":\"incomplete\",\"content\":[{\"type\":\"output_text\",\"text\":\"long reasoning\"}]},{\"type\":\"message\",\"status\":\"incomplete\",\"content\":[{\"type\":\"output_text\",\"text\":\"\"}]}]}' > \"$output\"\n"
        "else\n"
        "  printf '%s' '{\"status\":\"completed\",\"output_text\":\"{\\\"action\\\":\\\"QUERY\\\",\\\"tool\\\":\\\"get_ligand_info\\\",\\\"arguments\\\":{}}\"}' > \"$output\"\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "base_url": "https://example.invalid/v1",
        "model": "test",
        "api_key_env": "TEST_API_KEY",
        "max_output_tokens": 16384,
        "reasoning_effort": "low",
        "repair_max_output_tokens": 4096,
        "repair_reasoning_effort": "low",
    }), encoding="utf-8")

    client = ResponsesClient(config)
    decision = client.complete_json({"mode": "context_collection", "state": {}})

    assert decision["action"] == "QUERY"
    requests = [json.loads(line) for line in request_log.read_text().splitlines()]
    assert requests[0]["max_output_tokens"] == 16384
    assert requests[0]["reasoning"] == {"effort": "low"}
    assert requests[1]["max_output_tokens"] == 4096
    assert requests[1]["reasoning"] == {"effort": "low"}
    repair_user_text = requests[1]["input"][1]["content"][0]["text"]
    assert "long reasoning" not in repair_user_text
    assert "incomplete_response" not in repair_user_text


def test_chat_client_accepts_json_code_fence(tmp_path, monkeypatch):
    fake_curl = tmp_path / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "output=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = \"-o\" ]; then output=$2; shift 2; else shift; fi\n"
        "done\n"
        "printf '%s' '{\"choices\":[{\"finish_reason\":\"stop\",\"message\":{\"content\":\"```json\\n{\\\"action\\\":\\\"READY\\\"}\\n```\"}}]}' > \"$output\"\n",
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
    client = OpenAICompatibleChatClient(config, "test")
    assert client.complete_json({"mode": "test"}) == {"action": "READY"}


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
    events = []
    result = Workflow(
        TASK,
        ScriptedDemoClient(),
        tmp_path,
        progress=lambda event, details: events.append((event, details)),
    ).run()
    assert result["state"]["missing_site_evidence"] == []
    assert result["result"]["status"] == "candidate_accepted"
    candidate = Chem.SDMolSupplier(result["result"]["candidate_path"], removeHs=False)[0]
    assert candidate is not None
    assert Chem.GetFormalCharge(candidate) == 0
    assert candidate.GetNumHeavyAtoms() == 25
    assert result["result"]["docking"]["status"] == "not_configured"
    assert result["result"]["fep"]["status"] == "deferred"
    event_names = [event for event, _details in events]
    assert "workflow_started" in event_names
    assert "tool_started" in event_names
    assert "candidate_geometry_accepted" in event_names
    assert "docking_completed" in event_names
    assert "workflow_completed" in event_names


class CompleteDockingAdapter:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return {
            "stage": "docking",
            "status": "complete",
            "pose_count": 1,
            "poses": [{"rank": 1, "properties": {"minimizedAffinity": "-8.0"}}],
        }


class NeverCalledRBFEAdapter:
    def run(self, **kwargs):
        raise AssertionError("RBFE must be deferred from the design loop")


class DockingRetryClient(ScriptedDemoClient):
    def complete_json(self, payload):
        if payload["mode"] == "edit_retry":
            assert payload["rejection"]["failure_class"] == "docking_evaluation"
            assert payload["rejection"]["docking"]["result"]["pose_count"] == 1
            context = payload["optimization_context"]
            assert context["candidate_history"]
            assert context["docking_history"]
            assert context["attempted_transformations"]
            assert context["candidate_history"][0]["validation"]["status"] == "accepted"
            assert context["docking_history"][0]["attempt"] == 1
            assert "comparison" in context["docking_history"][0]
            assert "reference_baseline" in context
            previous_site = payload["previous_design"]["edit_atom_index"]
            observations = payload["state"]["observations"]
            if previous_site == 10:
                if not any(item["tool"] == "get_atom_environment" and item["arguments"].get("atom_index") == 9 for item in observations):
                    return {
                        "action": "QUERY",
                        "question": "Inspect an alternative site after docking.",
                        "tool": "get_atom_environment",
                        "arguments": {"atom_index": 9, "radius": 5.0},
                    }
                if not any(item["tool"] == "check_growth_space" and item["arguments"].get("atom_index") == 9 for item in observations):
                    return {
                        "action": "QUERY",
                        "question": "Check growth space at the alternative site.",
                        "tool": "check_growth_space",
                        "arguments": {"atom_index": 9, "distance": 2.0},
                    }
                if not any(item["tool"] == "validate_candidate_geometry" and item["arguments"].get("atom_index") == 9 for item in observations):
                    return {
                        "action": "QUERY",
                        "question": "Validate the alternative fluorine candidate.",
                        "tool": "validate_candidate_geometry",
                        "arguments": {"atom_index": 9, "fragment_smiles": "[*:1]F"},
                    }
                return {
                    "action": "READY",
                    "understanding": "Docking motivated a tested alternative fluorination site.",
                    "edit_atom_index": 9,
                    "edit_hypothesis": "move fluorine to the alternative phenyl site",
                    "fragment_smiles": "[*:1]F",
                }
            observations = payload["state"]["observations"]
            if not any(
                item["tool"] == "get_atom_environment"
                and item["arguments"].get("atom_index") == 12
                for item in observations
            ):
                return {
                    "action": "QUERY",
                    "question": "Inspect a third candidate site after docking.",
                    "tool": "get_atom_environment",
                    "arguments": {"atom_index": 12, "radius": 5.0},
                }
            if not any(
                item["tool"] == "check_growth_space"
                and item["arguments"].get("atom_index") == 12
                for item in observations
            ):
                return {
                    "action": "QUERY",
                    "question": "Check growth space at the third candidate site.",
                    "tool": "check_growth_space",
                    "arguments": {"atom_index": 12, "distance": 2.0},
                }
            if not any(
                item["tool"] == "validate_candidate_geometry"
                and item["arguments"].get("atom_index") == 12
                for item in observations
            ):
                return {
                    "action": "QUERY",
                    "question": "Validate the third fluorine candidate.",
                    "tool": "validate_candidate_geometry",
                    "arguments": {"atom_index": 12, "fragment_smiles": "[*:1]F"},
                }
            return {
                "action": "READY",
                "understanding": "Docking feedback motivated a chemically distinct third site.",
                "edit_atom_index": 12,
                "edit_hypothesis": "move fluorine to the third tested phenyl site",
                "fragment_smiles": "[*:1]F",
            }
        return super().complete_json(payload)


class WrongSiteClient(ScriptedDemoClient):
    def complete_json(self, payload):
        decision = super().complete_json(payload)
        if decision.get("action") == "READY":
            decision["edit_atom_index"] = 11
        return decision


def test_docking_complete_is_feedback_and_rbfe_is_deferred(tmp_path):
    workflow = Workflow(TASK, DockingRetryClient(), tmp_path)
    docking = CompleteDockingAdapter()
    workflow.docking_adapter = docking
    workflow.rbfe_adapter = NeverCalledRBFEAdapter()
    result = workflow.run()
    assert result["result"]["status"] == "candidate_accepted"
    assert len(docking.calls) == 1
    assert len(result["result"]["attempts"]) == 1
    assert result["result"]["rbfe"]["status"] == "deferred"
    assert result["result"]["attempts"][0]["docking"]["status"] == "complete"
    assert (tmp_path / "docking-attempt-01").is_dir()


def test_ready_auto_completes_evidence_for_new_edit_site(tmp_path):
    workflow = Workflow(TASK, WrongSiteClient(), tmp_path)
    decision = workflow.collect_context()

    assert decision["action"] == "READY"
    assert decision["edit_atom_index"] == 11
    assert any(
        item.tool == "get_atom_environment"
        and item.arguments["atom_index"] == 11
        for item in workflow.state.observations
    )
    assert any(
        item.tool == "check_growth_space"
        and item.arguments["atom_index"] == 11
        for item in workflow.state.observations
    )
    assert any(
        item.tool == "validate_candidate_geometry"
        and item.arguments.get("edit_atom_index") == 11
        and item.result.get("status") == "accepted"
        for item in workflow.state.observations
    )


class MissingReplacementEnvironmentClient:
    def __init__(self):
        self.ready_attempts = 0

    def complete_json(self, payload):
        observations = payload["state"]["observations"]
        sites = next(
            (item["result"]["sites"] for item in observations if item["tool"] == "list_fragment_replacement_sites"),
            None,
        )
        if sites is None:
            return {
                "action": "QUERY",
                "question": "Enumerate replacement sites.",
                "tool": "list_fragment_replacement_sites",
                "arguments": {"limit": 50},
            }
        site = sites[-1]
        transformation = {
            "operation": "replace_fragment",
            "replacement_site_id": site["replacement_site_id"],
            "fragment_smiles": "[*:1]C1COC1",
        }
        if not any(item["tool"] == "validate_candidate_geometry" for item in observations):
            return {
                "action": "QUERY",
                "question": "Validate oxetane replacement.",
                "tool": "validate_candidate_geometry",
                "arguments": transformation,
            }
        rejection = next(
            (item for item in reversed(payload["state"]["tool_rejections"])
             if item.get("failure_class") == "ready_evidence_missing"),
            None,
        )
        if rejection and not any(
            item["tool"] == "get_atom_environment"
            and item["arguments"].get("atom_index") == site["retained_atom_index"]
            for item in observations
        ):
            query = rejection["recommended_queries"][0]
            return {
                "action": "QUERY",
                "question": "Complete the missing retained-atom evidence.",
                **query,
            }
        self.ready_attempts += 1
        return {
            "action": "READY",
            "understanding": "The host validated a directed oxetane side-chain replacement.",
            "edit_hypothesis": "Test a smaller polar side chain.",
            **transformation,
        }


def test_missing_ready_evidence_is_recoverable_for_fragment_replacement(tmp_path):
    client = MissingReplacementEnvironmentClient()
    workflow = Workflow(TASK, client, tmp_path)
    workflow.context.task["search_policy"] = {"mode": "family_coverage"}
    decision = workflow.collect_context()
    site = workflow.tools.resolve_replacement_site(decision["replacement_site_id"])

    assert client.ready_attempts == 1
    assert any(
        item.tool == "get_atom_environment"
        and item.arguments["atom_index"] == site["retained_atom_index"]
        for item in workflow.state.observations
    )
    assert any(
        item.get("failure_class") == "ready_evidence_missing"
        for item in workflow.state.tool_rejections
    )


class InvalidDecisionThenValidClient:
    def __init__(self):
        self.calls = 0

    def complete_json(self, payload):
        self.calls += 1
        if self.calls == 1:
            return {"cutoff": 4.0}
        return {
            "action": "QUERY",
            "question": "ligand identity",
            "tool": "get_ligand_info",
            "arguments": {},
        }


def test_invalid_llm_decision_is_repaired_without_execution(tmp_path):
    workflow = Workflow(TASK, InvalidDecisionThenValidClient(), tmp_path)
    decision = workflow._repair_decision(
        {"cutoff": 4.0}, workflow._query_payload(), "context_collection"
    )
    assert decision["action"] == "QUERY"
    assert len(workflow.state.observations) == 0
    assert list(tmp_path.glob("invalid-decision-*.json"))


def test_transformation_field_detector_does_not_require_completeness():
    contains = Workflow._contains_transformation_fields
    # The exact failure from runs/docking-loop-real-agent-20260814-192916.
    assert contains({
        "operation": "replace_fragment",
        "replacement_site_id": "replacement-site-001",
        "fragment_smiles": "[*:1]C#N",
    })
    # Partial transformation responses still need targeted READY repair.
    assert contains({"operation": "replace_fragment"})
    assert contains({
        "replacement_site_id": "replacement-site-001",
        "fragment_smiles": "[*:1]C#N",
    })
    # Action validity is a separate concern from transformation detection.
    assert contains({"action": "READY", "operation": "replace_fragment"})
    assert contains({"action": "INVALID", "operation": "replace_fragment"})
    # Unrelated invalid decisions and non-dict inputs are not transformations.
    assert not contains({"cutoff": 4.0})
    assert not contains(None)
    assert not contains("[*:1]C#N")


class PersistentBareTransformationClient:
    """Mirrors GLM-5.3: always returns the bare transformation, ignoring repairs."""

    def __init__(self):
        self.calls = 0
        self.repair_modes: list[str] = []

    def complete_json(self, payload):
        self.calls += 1
        self.repair_modes.append(payload.get("mode"))
        return {
            "operation": "replace_fragment",
            "replacement_site_id": "replacement-site-001",
            "fragment_smiles": "[*:1]C#N",
        }


def test_bare_transformation_only_normalizes_action_after_repair_fails(tmp_path):
    client = PersistentBareTransformationClient()
    workflow = Workflow(TASK, client, tmp_path)
    bare = {
        "operation": "replace_fragment",
        "replacement_site_id": "replacement-site-001",
        "fragment_smiles": "[*:1]C#N",
    }
    decision = workflow._repair_decision(bare, workflow._query_payload(), "edit_retry")

    # The host supplies only the missing workflow action. READY semantics remain
    # model-owned and will be rejected by normal validation if they are absent.
    assert decision == {**bare, "action": "READY"}
    assert "understanding" not in decision
    assert "edit_hypothesis" not in decision
    assert "knowledge_gaps" not in decision

    # Two targeted READY-schema repair attempts were made before normalization.
    assert client.calls == 2
    assert client.repair_modes == ["ready_schema_repair", "ready_schema_repair"]

    # Both repair attempts and the action normalization are auditable on disk.
    invalid_files = sorted(tmp_path.glob("invalid-decision-*.json"))
    assert len(invalid_files) == 2
    repair = json.loads(invalid_files[0].read_text())
    assert repair["repair_mode"] == "ready_schema_repair"
    assert repair["invalid_decision"] == bare
    normalized_files = list(tmp_path.glob("normalized-transformation-decision-*.json"))
    assert len(normalized_files) == 1
    audit = json.loads(normalized_files[0].read_text())
    assert audit["phase"] == "edit_retry"
    assert audit["original_decision"] == bare
    assert audit["normalized_decision"] == decision


class BareThenReadyClient:
    """Returns a bare transformation once, then a valid READY on repair."""

    def __init__(self):
        self.calls = 0

    def complete_json(self, payload):
        self.calls += 1
        if payload.get("mode") == "ready_schema_repair":
            return {
                "action": "READY",
                "operation": "replace_fragment",
                "replacement_site_id": "replacement-site-001",
                "fragment_smiles": "[*:1]C#N",
                "understanding": "The pocket tolerates a nitrile at this vector.",
                "edit_hypothesis": "Install a nitrile via host-enumerated site.",
                "knowledge_gaps": [],
            }
        return {
            "operation": "replace_fragment",
            "replacement_site_id": "replacement-site-001",
            "fragment_smiles": "[*:1]C#N",
        }


def test_bare_transformation_repair_recovers_when_model_wraps_ready(tmp_path):
    client = BareThenReadyClient()
    workflow = Workflow(TASK, client, tmp_path)
    bare = {
        "operation": "replace_fragment",
        "replacement_site_id": "replacement-site-001",
        "fragment_smiles": "[*:1]C#N",
    }
    decision = workflow._repair_decision(bare, workflow._query_payload(), "edit_retry")

    # The targeted READY-schema repair succeeded, so no normalization was needed.
    assert decision["action"] == "READY"
    assert decision["understanding"] == "The pocket tolerates a nitrile at this vector."
    assert client.calls == 1
    assert list(tmp_path.glob("invalid-decision-*.json"))
    assert not list(tmp_path.glob("normalized-transformation-decision-*.json"))


class DuplicateClient:
    def complete_json(self, payload):
        return {
            "action": "QUERY",
            "question": "repeat",
            "tool": "get_ligand_info",
            "arguments": {},
            "expected_evidence": "identity",
        }


def test_duplicate_tool_call_is_reused_without_fatal_failure(tmp_path):
    workflow = Workflow(TASK, DuplicateClient(), tmp_path)
    workflow.state.max_context_rounds = 2
    with pytest.raises(RuntimeError, match="no-information"):
        workflow.collect_context()
    assert workflow.state.tool_rejections
    assert workflow.state.tool_rejections[-1]["failure_class"] == "duplicate_tool_call"
    assert len(workflow.state.observations) == 1


class NeverCalledDockingAdapter:
    def run(self, **kwargs):
        raise AssertionError("Docking must not run after deterministic geometry rejection")


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
            assert payload["rejection"]["failure_stage"] == "deterministic_geometry_prescreen"
            assert payload["rejection"]["docking"]["status"] == "not_run_geometry_rejected"
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


def test_rejected_candidate_geometry_cannot_satisfy_ready_gate(tmp_path):
    workflow = Workflow(TASK, RetryQueryClient(), tmp_path)
    workflow.context.task["search_policy"] = {"mode": "family_coverage"}
    workflow.docking_adapter = NeverCalledDockingAdapter()
    with pytest.raises(RuntimeError, match="invalid READY decisions"):
        workflow.run()
    assert len(workflow.state.observations) == 3
    assert workflow.state.tool_rejections
    assert all(
        item.get("failure_class") == "ready_evidence_missing"
        for item in workflow.state.tool_rejections
    )
    assert all(
        "exact accepted candidate geometry" in item["missing_evidence"]
        for item in workflow.state.tool_rejections
    )


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


class GateFeedbackAblationClient:
    def complete_json(self, payload):
        if payload["state"]["decisions"] and "ready_gate_feedback" not in payload:
            raise AssertionError("gate feedback was not returned to the LLM")
        if not payload["state"]["decisions"]:
            return {
                "action": "READY",
                "understanding": "initial hypothesis without site evidence",
                "edit_atom_index": 9,
                "edit_hypothesis": "fluorinate the aniline ring",
                "fragment_smiles": "[*:1]F",
            }
        calls = payload["state"]["tool_calls"]
        if len(calls) == 0:
            return {
                "action": "QUERY",
                "question": "environment after gate feedback",
                "tool": "get_atom_environment",
                "arguments": {"atom_index": 9, "radius": 5.0},
            }
        if len(calls) == 1:
            return {
                "action": "QUERY",
                "question": "growth space after gate feedback",
                "tool": "check_growth_space",
                "arguments": {"atom_index": 9, "distance": 2.0},
            }
        if len(calls) == 2:
            return {
                "action": "QUERY",
                "question": "validate revised fluorine candidate",
                "tool": "validate_candidate_geometry",
                "arguments": {"atom_index": 9, "fragment_smiles": "[*:1]F"},
            }
        return {
            "action": "READY",
            "understanding": "site evidence and candidate geometry are verified",
            "edit_atom_index": 9,
            "edit_hypothesis": "fluorinate the aniline ring",
            "fragment_smiles": "[*:1]F",
        }


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


def test_final_unbounded_gate_feedback_allows_learning_and_revision(tmp_path):
    from scripts.compare_tool_budgets import run_budget

    config = tmp_path / "config.json"
    config.write_text(
        '{"base_url":"https://example.invalid/v1","model":"test","api_key_env":"TEST_KEY"}'
    )
    module = __import__("scripts.compare_tool_budgets", fromlist=["OpenAICompatibleChatClient"])
    original = module.OpenAICompatibleChatClient
    module.OpenAICompatibleChatClient = lambda *_args, **_kwargs: GateFeedbackAblationClient()
    try:
        result = run_budget(
            TASK, config, tmp_path / "run", 6, "pocket", 6.0,
            require_site_evidence=True, unbounded=True,
        )
    finally:
        module.OpenAICompatibleChatClient = original
    assert result["status"] == "candidate_geometry_accepted"
    assert result["tool_call_count"] == 3
    assert result["decision_count"] == 5
    assert result["result"]["ready_gate"]["status"] == "passed"
    assert result["result"]["rbfe"]["status"] == "deferred"


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
