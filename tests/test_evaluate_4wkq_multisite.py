"""Product-keyed scoring for the cut-point-free pool.

The frozen-cut evaluator keys on ``fragment_id``; that is invalid once one compound
can be reached through several (site, fragment) pairs.  These tests pin the new key:
the **product molecule**.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_4wkq_benchmark import (
    enumerate_pool_products,
    freeze_labels_multisite,
    score_product_attempts,
)

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "4WKQ" / "benchmark-private"
ACTIVITIES = PRIVATE / "activity-source-v1" / "activities.json"
TASK = ROOT / "4WKQ" / "task.benchmark-multisite.json"
REACHABILITY = PRIVATE / "reachability-multisite.json"
PUBLIC = ROOT / "4WKQ" / "design"

pytestmark = pytest.mark.skipif(
    not (TASK.is_file() and REACHABILITY.is_file() and ACTIVITIES.is_file()),
    reason="multisite pool / activity source not present in this checkout",
)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("evaluator-multisite") / "v1"
    report = freeze_labels_multisite(ACTIVITIES, TASK, REACHABILITY, PUBLIC, out)
    labels = json.loads((out / "labels.json").read_text())
    return {"report": report, "labels": labels}


def test_freeze_covers_the_literature_set(frozen: dict) -> None:
    report = frozen["report"]
    assert report["mode"] == "multisite"
    assert report["measured_compounds"] == 26
    assert report["known_high_compounds"] > 0
    by_structure = frozen["labels"]["by_structure"]
    assert len(by_structure) == 26
    assert all(item["experimental"]["ic50_nM"] > 0 for item in by_structure.values())


def test_hits_key_on_the_product_molecule(frozen: dict) -> None:
    by_structure = frozen["labels"]["by_structure"]
    measured = next(iter(by_structure))
    unknown = "CCOc1ccccc1"
    rows = score_product_attempts(
        [
            {"attempt": 1, "product_smiles": measured, "quality": 1.0, "pose_passed": True},
            {"attempt": 2, "product_smiles": unknown, "quality": 0.5, "pose_passed": True},
        ],
        by_structure,
        10,
    )
    assert rows[0]["in_measured_set"] is True
    assert rows[0]["compound_id"] == by_structure[measured]["compound_id"]
    assert rows[0]["known_high"] == by_structure[measured]["known_high"]
    assert rows[1]["in_measured_set"] is False
    assert rows[1]["experimental"] is None
    assert rows[1]["known_high"] is False


def test_the_same_molecule_proposed_twice_is_a_duplicate(frozen: dict) -> None:
    """Two different (site, fragment) pairs reaching one compound are one candidate."""
    by_structure = frozen["labels"]["by_structure"]
    target = next(iter(by_structure))
    with pytest.raises(ValueError, match="Repeated candidate molecule"):
        score_product_attempts(
            [
                {"attempt": 1, "product_smiles": target,
                 "site_id": "cut-002", "change_type": "replacement", "fragment_id": "MS-001"},
                {"attempt": 2, "product_smiles": target,
                 "site_id": "cut-001", "change_type": "replacement", "fragment_id": "MS-004"},
            ],
            by_structure,
            10,
        )


def test_budget_is_enforced(frozen: dict) -> None:
    by_structure = frozen["labels"]["by_structure"]
    attempts = [{"attempt": index, "product_smiles": f"C{'C' * index}"} for index in range(1, 4)]
    with pytest.raises(ValueError, match="budget exceeded"):
        score_product_attempts(attempts, by_structure, 2)


def test_pool_product_space_is_enumerable_and_excludes_the_native_ligand() -> None:
    products = enumerate_pool_products(TASK, PUBLIC)
    assert len(products) > 100
    native = json.loads(REACHABILITY.read_text())["native_smiles"]
    assert native not in products
