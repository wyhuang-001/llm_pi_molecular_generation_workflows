"""Multisite closed pool: cut-point-free, product-keyed, multi-change-type.

The frozen single-cut pool is checked in ``test_closed_pool.py`` and must keep
working unchanged.  These tests cover the added ``multisite`` mode:

* the whole host site table is exposed, not one frozen cut;
* a hit is keyed by the **product molecule**, so the same compound is the same
  candidate no matter which (site, fragment) pair reached it;
* ``deletion`` needs no fragment, ``atom:addition`` and ``bond:replacement`` do;
* the runtime module still never touches the private evaluator file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from molecular_agent.closed_pool import ClosedPool
from molecular_agent.fragment_library import FragmentLibrary
from molecular_agent.structure import ComplexContext
from molecular_agent.tools import ToolRegistry
from molecular_agent.workflow import Workflow

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "4WKQ" / "task.benchmark-multisite.json"
CATALOG = ROOT / "4WKQ" / "design" / "fragments-multisite.json"
PRIVATE = ROOT / "4WKQ" / "benchmark-private" / "reachability-multisite.json"
MANIFEST = ROOT / "4WKQ" / "design" / "pool-manifest-multisite.json"

pytestmark = pytest.mark.skipif(
    not TASK.is_file() or not CATALOG.is_file() or not MANIFEST.is_file(),
    reason="multisite pool has not been built in this checkout",
)


@pytest.fixture(scope="module")
def pool() -> ClosedPool:
    context = ComplexContext(TASK)
    library = FragmentLibrary((context.input_dir / context.task["fragment_library_path"]).resolve())
    return ClosedPool(context, ToolRegistry(context, library))


def test_multisite_pool_exposes_the_whole_site_table(pool: ClosedPool) -> None:
    assert pool.mode == "multisite"
    assert len(pool.bond_sites) > 1, "a cut-point-free pool must expose more than one cut"
    assert len(pool.atom_sites) > 1
    contract = pool.dossier(pool.tools)["selection_contract"]
    assert contract["mode"] == "closed_pool_multisite"
    assert contract["full_catalog_visible"] is True


def test_manifest_declares_the_cut_space_and_verification_rule() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["mode"] == "multisite_reachability"
    assert manifest["counts"]["bond_sites"] > 1
    assert "reconstruction" in manifest["rules"]["verification"]
    assert manifest["rules"]["background"]["activity_used"] is False


def test_replacement_deletion_and_addition_all_build_a_product(pool: ClosedPool) -> None:
    bond = pool.bond_sites[0]
    fragment = next(r for r in pool.by_id.values() if "replacement" in r["allowed_change_types"])
    replacement = pool.normalize({
        "action": "READY", "site_type": "bond", "change_type": "replacement",
        "bond_site_id": bond["bond_site_id"], "fragment_id": fragment["fragment_id"],
    })
    deletion = pool.normalize({
        "action": "READY", "site_type": "bond", "change_type": "deletion",
        "bond_site_id": bond["bond_site_id"],
    })
    atom = next(s for s in pool.atom_sites if "addition" in s["allowed_change_types"])
    additive = next(r for r in pool.by_id.values() if "addition" in r["allowed_change_types"])
    addition = pool.normalize({
        "action": "READY", "site_type": "atom", "change_type": "addition",
        "edit_atom_index": atom["target_id"], "fragment_id": additive["fragment_id"],
    })
    products = {pool.product_smiles(item) for item in (replacement, deletion, addition)}
    assert len(products) == 3
    assert all(product for product in products)
    assert pool.native_smiles not in products


def test_one_compound_is_one_candidate_however_it_is_reached(pool: ClosedPool) -> None:
    """The whole point of a cut-point-free pool: hits key on the product molecule."""
    if not PRIVATE.is_file():
        pytest.skip("private reachability map is not present")
    detail = json.loads(PRIVATE.read_text(encoding="utf-8"))["compound_detail"]
    multi = next(
        (item for item in detail.values()
         if item.get("reachable") and len(item.get("decompositions") or []) > 1),
        None,
    )
    assert multi is not None, "the literature series must have multi-decomposition compounds"
    products = set()
    reachable = 0
    for item in multi["decompositions"]:
        decision = {
            "action": "READY",
            "site_type": item["site_type"],
            "change_type": item["change_type"],
        }
        if item["site_type"] in {"bond", "linker"}:
            decision["bond_site_id"] = item["site_id"]
        else:
            decision["edit_atom_index"] = item["site_id"]
        if item["fragment_smiles"]:
            # the public catalog only ships the minimal representation per compound,
            # so the other legal decompositions are not selectable by design
            if item["fragment_smiles"] not in pool.by_smiles:
                continue
            decision["fragment_smiles"] = item["fragment_smiles"]
        normalized = pool.normalize(decision)
        products.add(pool.product_smiles(normalized))
        reachable += 1
    assert reachable >= 1
    assert len(products) == 1, "every available decomposition must give the same product"
    assert pool.native_smiles not in products


def test_change_type_and_site_must_be_allowed(pool: ClosedPool) -> None:
    bond = pool.bond_sites[0]
    with pytest.raises(ValueError, match="only allows|does not allow change_type"):
        pool.normalize({
            "action": "READY", "site_type": "bond", "change_type": "addition",
            "bond_site_id": bond["bond_site_id"], "fragment_smiles": "[*:1]C",
        })
    with pytest.raises(ValueError, match="exposed site list"):
        pool.normalize({
            "action": "READY", "site_type": "bond", "change_type": "deletion",
            "bond_site_id": "cut-999",
        })
    replacement_only = next(
        r for r in pool.by_id.values() if r["allowed_change_types"] == ["replacement"]
    )
    atom = next(s for s in pool.atom_sites if "addition" in s["allowed_change_types"])
    with pytest.raises(ValueError, match="does not allow change_type"):
        pool.normalize({
            "action": "READY", "site_type": "atom", "change_type": "addition",
            "edit_atom_index": atom["target_id"],
            "fragment_id": replacement_only["fragment_id"],
        })


def test_runtime_module_never_references_the_private_evaluator_file() -> None:
    source = (ROOT / "molecular_agent" / "closed_pool.py").read_text(encoding="utf-8")
    assert "benchmark-private" not in source
    assert "reachability" not in source


def test_end_to_end_multisite_run_with_mock_docking(tmp_path):
    """Full loop: dossier -> decision -> host build -> docking -> history -> evaluator.

    Real graph construction, real geometry and real product-keyed bookkeeping; the
    docking adapter is mocked, so no GNINA, PLIP, network or paid LLM is used.
    """
    from types import SimpleNamespace

    from scripts.evaluate_4wkq_benchmark import evaluate_multisite, freeze_labels_multisite
    from scripts.run_4wkq_baseline import PublicBaselineClient

    activities = ROOT / "4WKQ" / "benchmark-private" / "activity-source-v1" / "activities.json"
    evaluator = tmp_path / "evaluator"
    freeze_labels_multisite(activities, TASK, PRIVATE, ROOT / "4WKQ" / "design", evaluator)

    context = ComplexContext(TASK)
    client = PublicBaselineClient("random", 17, ligand=context.ligand)
    run = tmp_path / "run"
    workflow = Workflow(TASK, client, run)
    # A deterministic pose-passing mock: no GNINA, but the full quality/ranking path
    # and the product-keyed feedback are exercised.
    def fake_docking(**kwargs):
        candidate = kwargs.get("candidate_path")
        seed = abs(hash(Path(str(candidate)).name)) % 1000
        return {
            "status": "complete", "seed_count": 3, "paired_seed_fraction": 1.0,
            "pose_retention": {"status": "passed"},
            "comparison": {"status": "complete", "metrics": {"minimizedAffinity": {
                "direction": "lower_is_better",
                "delta_candidate_minus_reference": {"mean": -(seed / 1000.0), "stddev": 0.05},
                "candidate_better_seed_count": 3, "candidate_better_seed_fraction": 1.0}}},
            "pose_consensus": {"stable": True}, "interaction_consensus": {},
            "candidate_per_seed": {}, "reference_baseline": {},
        }

    workflow.docking_adapter = SimpleNamespace(
        prepare_reference=lambda *args: {"status": "complete"},
        run_with_reference_baseline=fake_docking,
    )
    result = workflow.run()["result"]

    attempts = result["attempts"]
    # The scripted baseline stops when the workflow stops asking, so assert the
    # invariants rather than an exact count.
    assert 10 <= len(attempts) <= workflow.closed_pool.budget
    products = {
        item["validation"]["candidate"]["canonical_smiles"] for item in attempts
    }
    assert len(products) == len(attempts), "every proposal must be a distinct molecule"
    kinds = {item["transformation"]["change_type"] for item in attempts}
    assert kinds <= {"addition", "deletion", "replacement"}

    # the LLM-facing payload must describe the multisite contract, not a sole C6 site
    payload = workflow._direct_payload("after the run")
    text = json.dumps(payload)
    assert "sole C6 site" not in text
    assert payload["state"]["learning_summary"]["scope"] == "public_structures_and_observed_docking_only"
    assert payload["state"]["learning_summary"]["tested_count"] == len(attempts)

    # the evaluator scores by product molecule and never sees a fragment key
    report = evaluate_multisite(run, evaluator, TASK, ROOT / "4WKQ" / "design")
    assert report["mode"] == "multisite"
    assert report["unique_proposals"] == len(attempts)
    assert report["hit_key"] == "product_molecule_canonical_smiles"
    assert report["proposals"]["unknown_n"] > 0
    assert report["rows"] and any(row["quality"] is not None for row in report["rows"])
    assert report["final_top_k"]["1"]["n"] == 1
    assert report["pool_product_space"] > 100


def test_element_swap_keeps_the_pose_gate_evaluable(tmp_path):
    """A swapped atom is a different chemical atom, so it must lose parent provenance.

    Keeping the provenance made the pose-retention mapping raise "Retained atom
    chemistry changed" for every element swap, so the whole edit class advertised by
    the site table was structurally unevaluable.
    """
    import json as _json

    from molecular_agent.editing import EditResult, apply_transformation, write_sdf
    from molecular_agent.pose_retention import core_mappings, read_one

    context = ComplexContext(TASK)
    native = Chem.RemoveHs(Chem.Mol(context.ligand))
    result = apply_transformation(
        native,
        {"site_type": "ring", "change_type": "replacement", "edit_atom_index": 2, "element": "C"},
        context.protein_atoms,
    )
    assert not result.molecule.GetAtomWithIdx(2).HasProp("_reference_atom_index")
    pose = tmp_path / "swap.sdf"
    write_sdf(EditResult(result.molecule, result.report), pose, name="swap")
    source = read_one(pose)
    provenance = _json.loads(source.GetProp("reference_atom_indices"))
    assert provenance.count(None) == 1, "only the edited atom loses its reference identity"
    # the pose gate can now evaluate this candidate instead of erroring out
    mappings = core_mappings(native, source, source, [10, 11, 12, 15, 16])
    assert len(mappings) >= 1

    # if the edited atom were part of the comparison core, the gate must refuse it
    try:
        core_mappings(native, source, source, [2])
    except ValueError as error:
        assert "core" in str(error).lower()
    else:
        raise AssertionError("an edit inside the comparison core must be rejected")


def test_pose_evaluation_failure_does_not_end_the_run(tmp_path):
    """``evaluation_failed`` is a candidate-level outcome, not a run terminator."""
    from types import SimpleNamespace

    from scripts.run_4wkq_baseline import PublicBaselineClient

    run = tmp_path / "run"
    client = PublicBaselineClient("random", 17, ligand=ComplexContext(TASK).ligand)
    workflow = Workflow(TASK, client, run)
    calls = {"count": 0}

    def docking(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "evaluation_failed", "failure_class": "pose_evaluation",
                    "error": "No majority of paired, core/anchor-gated Evaluation Poses",
                    "pose_retention": {"status": "failed"},
                    "comparison": {"metrics": {}}, "paired_seed_fraction": 0,
                    "candidate_per_seed": {}, "reference_baseline": {}}
        seed = abs(hash(Path(str(kwargs.get("candidate_path"))).name)) % 1000
        return {"status": "complete", "seed_count": 3, "paired_seed_fraction": 1.0,
                "pose_retention": {"status": "passed"},
                "comparison": {"status": "complete", "metrics": {"minimizedAffinity": {
                    "direction": "lower_is_better",
                    "delta_candidate_minus_reference": {"mean": -(seed / 1000.0), "stddev": 0.02},
                    "candidate_better_seed_count": 3, "candidate_better_seed_fraction": 1.0}}},
                "pose_consensus": {"stable": True}, "interaction_consensus": {},
                "candidate_per_seed": {}, "reference_baseline": {}}

    workflow.docking_adapter = SimpleNamespace(
        prepare_reference=lambda *args: {"status": "complete"},
        run_with_reference_baseline=docking,
    )
    result = workflow.run()["result"]
    assert result["stopping_reason"] != "docking_not_complete"
    assert len(result["attempts"]) > 1, "the search must continue after a pose-gate failure"


def test_mark_unmodifiable_closes_site_in_direct_mode(tmp_path):
    """Direct mode supports MARK_UNMODIFIABLE, including ring/halogen atoms without H."""
    class Dummy:
        def complete_json(self, payload):
            return {"action": "STOP", "stop_reason": "no_promising_edit"}

    workflow = Workflow(TASK, Dummy(), tmp_path / "run")
    # morpholine ring O (no hydrogen) and meta-Cl are host sites now
    assert workflow._record_unmodifiable({
        "action": "MARK_UNMODIFIABLE", "target_type": "atom", "target_id": 2,
        "completion_reason": "no_promising_edit",
    }) is True
    assert workflow._record_unmodifiable({
        "action": "MARK_UNMODIFIABLE", "target_type": "bond", "target_id": "cut-001",
        "completion_reason": "site_complete",
    }) is True
    workflow._prepare_initial_context()
    dossier = workflow._direct_dossier()
    site_ids = {(s.get("target_type"), s.get("target_id")) for s in dossier.get("sites", [])}
    assert ("atom", 2) not in site_ids
    assert ("bond", "cut-001") not in site_ids
