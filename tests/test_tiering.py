"""Tests for the two-layer (T1/T2) edit classification.

The rules under test are the frozen contract:

* T1 — one or two heavy-atom changes with no ring-framework change.
* T2 — three or more heavy-atom changes, or any ring-framework change.
* The ring rule takes precedence: a single-atom ring change is still T2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from molecular_agent.tiering import T1, T2, classify_change, classify_fragment_smiles, ring_sizes

REFERENCE = "OCCCN1CCOCC1"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "smiles, expected_tier, expected_reason_fragment",
    [
        # ring-side-chain decoration: not a skeleton change -> T1
        ("CC1COCCN1CCCO[*:1]", T1, "1 heavy-atom"),
        ("O[C@@H](CO[*:1])CN1CCOCC1", T1, "1 heavy-atom"),
        # chain length +/-1 or +/-2 -> T1
        ("O[C@@H](CCN1CCOCC1)CO[*:1]", T1, "2 heavy-atom"),
        ("C(CCN1CCOCC1)CO[*:1]", T1, "2 heavy-atom"),
        ("C1CN(CCO[*:1])CCO1", T1, "2 heavy-atom"),
        # ring skeleton: one atom differs, still T2
        ("C1CCN(CCCO[*:1])CC1", T2, "ring-skeleton"),
        # ring size: one atom differs, still T2
        ("C1CCN(CCCO[*:1])C1", T2, "ring-size"),
        # ring skeleton substitution plus a methyl -> T2
        ("CN1CCN(CCCO[*:1])CC1", T2, "ring-skeleton"),
        # ring framework replaced by an open chain -> T2
        ("COCCN(CCOC)CCO[*:1]", T2, "ring"),
        # three heavy-atom changes without a ring change -> T2
        ("O[C@@H](CCO[*:1])CN1CCOCC1", T2, "heavy-atom changes"),
    ],
)
def test_frozen_tier_examples(smiles: str, expected_tier: str, expected_reason_fragment: str) -> None:
    result = classify_fragment_smiles(REFERENCE, smiles)
    assert result["tier"] == expected_tier
    assert expected_reason_fragment in result["reason"]


def test_ring_rule_takes_precedence_over_atom_count() -> None:
    # Morpholine -> piperidine: exactly one heavy atom differs (O -> CH2) but the
    # ring skeleton changed, so the fragment is a whole-ring replacement.
    result = classify_fragment_smiles(REFERENCE, "C1CCN(CCCO[*:1])CC1")
    assert result["cost"] == 1
    assert result["tier"] == T2
    assert result["ring_skeleton_change"] is True
    assert "skeleton_substitution" in result["ring_change_kinds"]


def test_element_substitution_counts_as_one_change() -> None:
    result = classify_fragment_smiles(REFERENCE, "C1CCN(CCCO[*:1])CC1")
    assert result["substitutions"] == 1
    assert result["added"] == 1
    assert result["lost"] == 1
    assert result["cost"] == 1


def test_reference_against_itself_is_a_no_op() -> None:
    result = classify_fragment_smiles(REFERENCE, REFERENCE)
    assert result["cost"] == 0
    assert result["tier"] == T1
    assert result["added"] == 0 and result["lost"] == 0


def test_ring_sizes_reports_every_ring_membership() -> None:
    morpholine = Chem.MolFromSmiles("N1CCOCC1")
    sizes = ring_sizes(morpholine)
    assert set(sizes) == {0, 1, 2, 3, 4, 5}
    assert all(value == {6} for value in sizes.values())


def test_classify_change_rejects_empty_molecules() -> None:
    with pytest.raises(ValueError):
        classify_change(Chem.MolFromSmiles(""), Chem.MolFromSmiles("CC"))


def test_catalog_is_site_free() -> None:
    """The designer-facing catalog must not pre-decide the site or the cut."""
    catalog_path = ROOT / "4WKQ/design/fragments_v2.json"
    if not catalog_path.is_file():
        pytest.skip("v2 catalog has not been built in this checkout")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    for forbidden in ("cut_bond", "reference_ligand_smiles", "reference_side_chain_smiles", "retained_atom_indices"):
        assert forbidden not in catalog
    assert catalog["fragments"], "catalog must not be empty"
    for entry in catalog["fragments"]:
        assert "site" not in entry
        assert "tier" not in entry
        assert "compile_as" not in entry
        assert entry["size_class"] in {"minimal", "small", "medium", "large"}
        assert entry["allowed_operations"]
        assert entry["smiles"].count("[*:1]") == 1
        assert entry["formal_charge"] == 0
        assert "origin" not in entry


def test_site_table_has_no_fragment_assignment() -> None:
    sites_path = ROOT / "4WKQ/design/sites-v2.json"
    if not sites_path.is_file():
        pytest.skip("v2 site table has not been built in this checkout")
    sites = json.loads(sites_path.read_text(encoding="utf-8"))
    for record in sites["atom_sites"] + sites["cut_sites"]:
        assert "fragment_id" not in record
        assert "fragments" not in record
        assert "library_record" not in record


def test_site_table_covers_every_ligand_atom() -> None:
    sites_path = ROOT / "4WKQ/design/sites-v2.json"
    if not sites_path.is_file():
        pytest.skip("v2 site table has not been built in this checkout")
    sites = json.loads(sites_path.read_text(encoding="utf-8"))
    coverage = sites["atom_coverage"]
    assert coverage["assigned_atoms"] == coverage["ligand_heavy_atoms"]
    assert coverage["unassigned_atoms"] == []
    protected = set(sites["protected_atom_indices"])
    assert protected, "the quinazoline core must be protected"
    for record in sites["atom_sites"]:
        if record["atom_index"] in protected:
            assert record["allowed_operations"] == []
            assert record["protection"] == "protected"
    for record in sites["cut_sites"]:
        assert protected <= set(record["retained_atom_indices"]), "a cut must keep the protected core"
        assert record["cut_bond"][0] != record["cut_bond"][1]
