"""Offline experience library: structure, store, match and rank trajectory evidence."""
from __future__ import annotations

from molecular_agent.experience import (
    ExperienceLibrary,
    build_evidence_record,
    structure_experience,
)


def _record(attempt, site_id, operation, fragment_id, delta, pose, outcome, failure_class=None):
    return build_evidence_record(
        attempt=attempt,
        task="optimize",
        ligand_smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        site={"target_type": "atom", "target_id": site_id, "region": "morpholine"},
        operation=operation,
        fragment={"fragment_id": fragment_id, "fragment_smiles": "C[*:1]"},
        pocket={"residues": ["ASP:A:800"]},
        evidence={"delta_candidate_minus_reference": delta, "quality": 0.3, "seed_win_fraction": 1.0,
                  "seed_stddev": 0.05, "pose_native_like": pose},
        outcome=outcome,
        failure={"failure_class": failure_class} if failure_class else None,
    )


def test_structure_experience_aggregates_sites_fragments_operations():
    records = [
        _record(1, 1, "atom:addition", "BG-a", -0.6, True, "new_best"),
        _record(2, 1, "atom:addition", "BG-b", -0.4, True, "improved"),
        _record(3, 2, "ring:replacement", "N", 0.1, False, "pose_failed", "pose_retention"),
    ]
    exp = structure_experience(records)
    assert exp["record_count"] == 3
    assert exp["schema"] == "simple-molecular-agent.experience.v1"

    sites = {f"{s['target_type']}:{s['target_id']}": s for s in exp["site_experience"]}
    assert sites["atom:1"]["attempts"] == 2
    assert sites["atom:1"]["best_delta"] == -0.6
    assert sites["atom:1"]["best_fragment"] == "BG-a"
    assert sites["atom:1"]["pose_pass_rate"] == 1.0
    assert sites["atom:2"]["attempts"] == 1

    frags = {f["fragment"]: f for f in exp["fragment_experience"]}
    assert frags["BG-a"]["best_delta"] == -0.6
    assert frags["BG-a"]["sites"] == ["atom:1"]

    ops = {o["operation"]: o for o in exp["operation_experience"]}
    assert ops["atom:addition"]["attempts"] == 2
    assert ops["atom:addition"]["new_best_count"] == 1

    failures = {f["failure_class"]: f for f in exp["failure_experience"]}
    assert failures["pose_retention"]["attempts"] == 1
    assert failures["pose_retention"]["sites"] == ["atom:2"]


def test_library_store_load_and_ranked_match(tmp_path):
    lib = ExperienceLibrary(tmp_path)
    ligand = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    exp = structure_experience([
        _record(1, 1, "atom:addition", "BG-a", -0.6, True, "new_best"),
        _record(2, 1, "atom:addition", "BG-b", -0.4, True, "improved"),
        _record(3, 7, "atom:addition", "BG-c", -0.2, True, "improved"),
        _record(4, 2, "ring:replacement", "N", 0.1, False, "pose_failed", "pose_retention"),
    ])
    lib.store(ligand, exp)
    loaded = lib.load(ligand)
    assert loaded is not None
    assert loaded["record_count"] == 4

    query = {
        "sites": [{"target_type": "atom", "target_id": 1}],
        "fragments": [{"fragment_id": "BG-a", "fragment_smiles": "C[*:1]"}],
        "operations": ["atom:addition"],
        "pocket_residues": ["ASP:A:800"],
        "failure_classes": {"*"},
    }
    ranked = lib.match(ligand, query)
    kinds = {item["kind"] for item in ranked}
    # site 1, fragment BG-a, and the operation all match and must rank
    assert "site" in kinds and "fragment" in kinds and "operation" in kinds
    # site 1 has the best delta and must be the top site entry
    site_item = next(item for item in ranked if item["kind"] == "site")
    assert "best_delta=-0.6" in site_item["text"]
    text = lib.context_text(ranked)
    assert "site atom:1" in text and "fragment BG-a" in text


def test_library_miss_returns_empty(tmp_path):
    lib = ExperienceLibrary(tmp_path)
    assert lib.match("CCO", {"sites": [], "fragments": [], "operations": [],
                             "pocket_residues": [], "failure_classes": {"*"}}) == []
