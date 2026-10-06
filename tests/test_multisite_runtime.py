"""Runtime tests for the host-supplied edit-site table (multi-site mode).

These tests pin the interface contract that the v2 library was built for:

* the host loads ``edit_site_table_path`` and exposes exactly those sites;
* a protected site is still enumerated but rejects every operation;
* an unlisted site is rejected;
* an operation that the site does not list is rejected;
* every built candidate records its edit layer, so a one-atom change is never
  reported as a whole-fragment replacement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molecular_agent.fragment_library import FragmentLibrary, canonical_operation
from molecular_agent.structure import ComplexContext
from molecular_agent.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "4WKQ" / "task.v2.json"
CATALOG = ROOT / "4WKQ" / "design" / "fragments_v2.json"
SITES = ROOT / "4WKQ" / "design" / "sites-v2.json"

pytestmark = pytest.mark.skipif(
    not TASK.is_file() or not CATALOG.is_file() or not SITES.is_file(),
    reason="v2 task and library have not been built in this checkout",
)


@pytest.fixture(scope="module")
def tools() -> ToolRegistry:
    context = ComplexContext(TASK)
    library = FragmentLibrary(CATALOG)
    return ToolRegistry(context, library)


def test_operation_aliases_are_equivalent() -> None:
    assert canonical_operation("substitute") == canonical_operation("atom:addition")
    assert canonical_operation("bond:replacement") == "bond:replacement"


def test_site_table_is_loaded_and_exposed(tools: ToolRegistry) -> None:
    sites = tools.list_edit_sites()
    assert sites["source"] == "edit_site_table"
    table = json.loads(SITES.read_text(encoding="utf-8"))
    assert len(sites["atom_sites"]) == len(table["atom_sites"])
    assert len(sites["cut_sites"]) == len(table["cut_sites"])
    assert sites["protected_atom_indices"] == table["protected_atom_indices"]


def test_protected_core_rejects_every_operation(tools: ToolRegistry) -> None:
    result = tools.validate_candidate_geometry(
        target_type="atom", target_id=19, site_type="atom", change_type="addition", fragment_smiles="[*:1]O"
    )
    assert result["status"] == "rejected"
    assert result["failure_class"] == "protected_edit_site"


def test_unlisted_site_is_rejected(tools: ToolRegistry) -> None:
    result = tools.validate_candidate_geometry(
        target_type="bond",
        target_id="cut-999",
        site_type="bond", change_type="replacement",
        fragment_smiles="[*:1]C",
    )
    assert result["status"] == "rejected"
    assert result["failure_class"] == "unlisted_edit_site"


def test_operation_not_allowed_at_site_is_rejected(tools: ToolRegistry) -> None:
    # Element swap is only listed for the morpholine oxygen and the halogen atoms.
    result = tools.validate_candidate_geometry(
        target_type="atom", target_id=7, site_type="atom", change_type="replacement", element="N"
    )
    assert result["status"] == "rejected"
    assert result["failure_class"] == "change_type_not_allowed_at_site"


def test_one_atom_substituent_is_reported_as_minimal_edit(tools: ToolRegistry) -> None:
    result = tools.validate_candidate_geometry(
        target_type="atom", target_id=8, site_type="atom", change_type="addition", fragment_smiles="[*:1]O"
    )
    assert result["status"] == "accepted"
    layer = result["edit_layers"]
    assert layer["tier"] == "T1"
    assert layer["compile_as"] == "minimal_edit"
    assert layer["heavy_atom_changes"] == 1
    assert layer["ring_change_kinds"] == []


def test_ring_skeleton_element_replacement_is_reported_as_whole_ring(tools: ToolRegistry) -> None:
    # Morpholine oxygen -> CH2: exactly one heavy atom changes, but the ring
    # skeleton changed, so the host must not narrate it as a minimal edit.
    result = tools.validate_candidate_geometry(
        target_type="atom", target_id=2, site_type="ring", change_type="replacement", element="C"
    )
    assert result["status"] == "accepted"
    layer = result["edit_layers"]
    assert layer["tier"] == "T2"
    assert layer["compile_as"] == "whole_fragment"
    assert layer["heavy_atom_changes"] == 1
    assert "skeleton_substitution" in layer["ring_change_kinds"]


def test_short_cut_replacement_is_reported_correctly(tools: ToolRegistry) -> None:
    result = tools.validate_candidate_geometry(
        target_type="bond",
        target_id="cut-010",
        site_type="bond", change_type="replacement",
        fragment_smiles="[*:1]Br",
    )
    assert result["status"] == "accepted"
    layer = result["edit_layers"]
    assert layer["tier"] == "T1"
    assert layer["substitutions"] == 1


def test_catalog_records_carry_no_site_assignment(tools: ToolRegistry) -> None:
    for record in tools.fragment_library.records:
        assert "target_id" not in record
        assert "site_id" not in record
        assert "cut_bond" not in record


def test_symmetry_equivalent_atoms_are_marked(tools: ToolRegistry) -> None:
    table = json.loads(SITES.read_text(encoding="utf-8"))
    by_index = {record["atom_index"]: record for record in table["atom_sites"]}
    # The morpholine ring has two interchangeable carbon pairs under its own symmetry.
    assert sorted(by_index[0]["symmetry_equivalent_atom_indices"]) == [0, 4]
    assert sorted(by_index[1]["symmetry_equivalent_atom_indices"]) == [1, 3]
    # The representative is the most usable member of the class, lowest index on ties.
    assert by_index[0]["symmetry_representative"] is True
    assert by_index[4]["equivalent_to"] == "atom-000"
    assert by_index[1]["symmetry_representative"] is True
    assert by_index[3]["equivalent_to"] == "atom-001"
    assert "symmetry_convention" in table


def test_symmetry_redundant_target_is_canonicalised_not_rejected(tools: ToolRegistry) -> None:
    result = tools.validate_candidate_geometry(
        target_type="atom", target_id=3, site_type="atom", change_type="addition", fragment_smiles="[*:1]N"
    )
    assert result["status"] == "accepted"
    info = result["transformation"]["site_canonicalization"]
    assert info["requested_target_id"] == 3
    assert info["canonical_target_id"] == 1
    assert info["reason"] == "symmetry_equivalent_site"


def test_symmetry_equivalent_targets_give_the_same_molecule(tools: ToolRegistry) -> None:
    left = tools.validate_candidate_geometry(
        target_type="atom", target_id=1, site_type="atom", change_type="addition", fragment_smiles="[*:1]N"
    )
    right = tools.validate_candidate_geometry(
        target_type="atom", target_id=3, site_type="atom", change_type="addition", fragment_smiles="[*:1]N"
    )
    assert left["canonical_smiles"] == right["canonical_smiles"]


def test_exhausted_retry_budget_stops_gracefully(tmp_path) -> None:
    """Exhausting the retry loop must not raise away already-docked candidates."""
    from molecular_agent.workflow import Workflow

    class AlwaysInvalidClient:
        """An operation the chosen site does not allow: the host rejects every decision."""

        def __init__(self):
            self.calls = 0

        def complete_json(self, payload):
            self.calls += 1
            return {
                "action": "READY",
                "understanding": "invalid on purpose",
                "edit_hypothesis": "invalid on purpose",
                "site_type": "atom", "change_type": "replacement",
                "target_type": "atom",
                "target_id": 7,
                "element": "N",
            }

    client = AlwaysInvalidClient()
    workflow = Workflow(TASK, client, tmp_path / "run")
    decision = workflow._retry_direct_decision(
        {"action": "READY"}, {"failure_class": "steric_clash", "error": "test"}
    )
    assert decision["action"] == "STOP"
    assert decision["stop_reason"] == "repeated_invalid_edit"
    assert client.calls >= 1
