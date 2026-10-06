"""Two-axis edit taxonomy: site_type x change_type, and the new deletion cell.

The host no longer exposes the historical one-dimensional operation names.  Every
edit is described by *where* it happens (``atom`` / ``bond`` / ``linker`` /
``ring``) and *what* happens there (``addition`` / ``deletion`` / ``replacement``).
These tests pin the vocabulary, the legacy projection used for frozen artifacts,
and the new ``bond:deletion`` capability.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molecular_agent.edit_taxonomy import (
    CHANGE_TYPES,
    SITE_ALLOWED_CHANGE_TYPES,
    SITE_TYPES,
    EditTaxonomyError,
    axes_for_operation,
    change_types_from_legacy_operations,
    infer_axes,
    normalize_transformation,
    operation_label,
    taxonomy_documentation,
)
from molecular_agent.editing import apply_transformation, resolve_edit_axes
from molecular_agent.structure import ComplexContext
from molecular_agent.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "input" / "task.json"


def test_operation_labels_cover_every_supported_cell() -> None:
    expected = {
        "atom:addition",
        "atom:replacement",
        "bond:deletion",
        "bond:replacement",
        "ring:addition",
        "ring:replacement",
    }
    produced = {
        operation_label(site_type, change_type)
        for site_type in SITE_TYPES
        for change_type in SITE_ALLOWED_CHANGE_TYPES[site_type]
    }
    assert expected <= produced
    assert "linker:addition" in produced and "linker:deletion" in produced


def test_legacy_operations_project_onto_the_axes() -> None:
    assert axes_for_operation("replace_hydrogen") == ("atom", "addition")
    assert axes_for_operation("replace_fragment") == ("bond", "replacement")
    assert axes_for_operation("element_swap") == ("atom", "replacement")
    assert axes_for_operation("ring_atom_swap") == ("ring", "replacement")
    assert axes_for_operation("bond:deletion") == ("bond", "deletion")
    assert change_types_from_legacy_operations(
        ["replace_fragment", "terminal_substituent_swap"]
    ) == ["replacement"]


def test_axes_that_do_not_exist_are_rejected() -> None:
    with pytest.raises(EditTaxonomyError):
        axes_for_operation("not_an_operation")
    with pytest.raises(EditTaxonomyError):
        normalize_transformation({"site_type": "bond", "change_type": "addition"})
    with pytest.raises(EditTaxonomyError):
        normalize_transformation({"site_type": "atom", "change_type": "deletion"})


def test_normalize_fills_the_canonical_fields() -> None:
    normalized = normalize_transformation(
        {"site_type": "bond", "change_type": "deletion", "bond_site_id": "bond-site-001"}
    )
    assert normalized["operation"] == "bond:deletion"
    # A legacy payload is still readable.
    legacy = normalize_transformation({"operation": "replace_fragment", "bond_site_id": "x"})
    assert (legacy["site_type"], legacy["change_type"]) == ("bond", "replacement")
    # A payload with only concrete fields is inferred, never silently misfiled.
    assert infer_axes({"edit_atom_index": 3, "fragment_smiles": "[*:1]C"}) == ("atom", "addition")
    assert infer_axes({"bond_site_id": "bond-site-001", "fragment_smiles": "[*:1]C"}) == ("bond", "replacement")
    assert infer_axes({"atom_index": 7, "element": "N"}) == ("atom", "replacement")


def test_taxonomy_documentation_is_machine_readable() -> None:
    document = taxonomy_documentation()
    assert document["schema"] == "site_type x change_type"
    assert set(document["site_types"]) == set(SITE_TYPES)
    assert set(document["change_types"]) == set(CHANGE_TYPES)


def test_bond_deletion_removes_the_side_and_keeps_the_scaffold() -> None:
    context = ComplexContext(TASK)
    tools = ToolRegistry(context)
    site = tools.get_edit_site_candidates()["bond_sites"][0]
    before = context.ligand.GetNumHeavyAtoms()

    result, evidence = tools.execute(
        "validate_candidate_geometry",
        {
            "site_type": "bond",
            "change_type": "deletion",
            "bond_site_id": site["bond_site_id"],
        },
    )

    assert result["status"] == "accepted"
    assert result["change_type"] == "deletion"
    assert result["operation"] == "bond:deletion"
    assert result["heavy_atom_delta"] == -site["removed_heavy_atoms"]
    assert result["candidate"]["heavy_atoms"] == before - site["removed_heavy_atoms"]
    # Deletion only removes atoms, so it can never report a new rigid clash.
    assert result["severe_clash_count"] == 0
    assert evidence == {"candidate_geometry"}
    # The formal charge is preserved.
    assert result["candidate"]["formal_charge"] == result["parent"]["formal_charge"]


def test_bond_deletion_reports_the_lost_reference_contacts_as_a_host_fact() -> None:
    tools = ToolRegistry(ComplexContext(TASK))
    sites = tools.get_edit_site_candidates()["bond_sites"]
    assert any(site["removed_reference_interactions"] for site in sites)
    site = next(site for site in sites if site["removed_reference_interactions"])
    assert all(
        {"kind", "protein_atom", "distance"} <= set(contact)
        for contact in site["removed_reference_interactions"]
    )


def test_deletion_is_rejected_at_an_atom_site() -> None:
    tools = ToolRegistry(ComplexContext(TASK))
    atom_site = tools.get_edit_site_candidates()["atom_sites"][0]
    result, _evidence = tools.execute(
        "validate_candidate_geometry",
        {
            "site_type": "atom",
            "change_type": "deletion",
            "edit_atom_index": atom_site["target_id"],
        },
    )
    assert result["status"] == "rejected"
    assert "change_type" in result["error"]


def test_linker_edits_are_declared_but_not_executed_yet() -> None:
    """The linker cell is part of the vocabulary; this stage does not execute it."""
    context = ComplexContext(TASK)
    with pytest.raises(ValueError):
        apply_transformation(
            context.ligand,
            {"site_type": "linker", "change_type": "deletion", "bond_site_id": "linker-001"},
            context.protein_atoms,
        )
    assert resolve_edit_axes({"site_type": "linker", "change_type": "addition"}) == (
        "linker",
        "addition",
    )


def test_bond_sites_report_the_removal_consequence_profile() -> None:
    """Deletion needs a two-sided ledger, not only the list of lost contacts."""
    tools = ToolRegistry(ComplexContext(TASK))
    sites = tools.get_edit_site_candidates()["bond_sites"]
    assert sites
    for site in sites:
        assert isinstance(site["removed_polar_roles"], dict)
        assert set(site["removed_polar_roles"]) <= {"donor", "acceptor"}
        assert isinstance(site["removed_rotatable_bonds"], int)
        assert site["removed_rotatable_bonds"] >= 0
        assert isinstance(site["core_overlap"], list)
        assert site["core_overlap"] == []

    # At least one site must show a polar capability and a flexibility change,
    # otherwise the gain side of the ledger would never be exercised.
    assert any(site["removed_polar_roles"] for site in sites)
    assert any(site["removed_rotatable_bonds"] > 0 for site in sites)


def test_design_dossier_surfaces_the_removal_profile() -> None:
    tools = ToolRegistry(ComplexContext(TASK))
    dossier = tools.get_design_dossier(panel_size=2)
    bond_sites = [site for site in dossier["sites"] if site.get("site_type") == "bond"]
    assert bond_sites
    for site in bond_sites:
        assert "removed_polar_roles" in site
        assert "removed_rotatable_bonds" in site
        assert "core_overlap" in site


def test_pose_core_overlap_is_computed_and_blocks_the_edit() -> None:
    """Removing a pose-comparison atom is a structural rejection, not a bad score."""
    tools = ToolRegistry(ComplexContext(TASK))
    site = tools.get_edit_site_candidates()["bond_sites"][0]
    internal = next(item for item in tools._bond_sites if item["bond_site_id"] == site["bond_site_id"])
    removed = internal["removed_atom_indices"]
    assert removed

    # The default task protects nothing, so the same removal is legal.
    assert tools._removal_profile(removed)["core_overlap"] == []

    # Protect one atom that sits on the removed side of this site.
    tools.context.task.setdefault("fragment_replacement", {})["protected_core_atom_indices"] = [
        removed[0]
    ]
    assert tools._removal_profile(removed)["core_overlap"] == [removed[0]]

    # The gate must reject both a deletion and a replacement that removes it.
    tools._bond_sites = [{**internal, "core_overlap": [removed[0]]}]
    for change_type, payload in (
        ("deletion", {}),
        ("replacement", {"fragment_smiles": "[*:1]C"}),
    ):
        result, _evidence = tools.execute(
            "validate_candidate_geometry",
            {
                "site_type": "bond",
                "change_type": change_type,
                "bond_site_id": site["bond_site_id"],
                **payload,
            },
        )
        assert result["status"] == "rejected"
        assert result["failure_class"] == "pose_core_removal"
        assert result["core_overlap"] == [removed[0]]
