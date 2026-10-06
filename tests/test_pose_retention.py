import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from molecular_agent.adapters import DockingAdapter
from molecular_agent.editing import EditResult, _retained_fragment, write_sdf
from molecular_agent.plip_adapter import PLIPAdapter
from molecular_agent.pose_retention import (PoseRetention, anchor_check, clique_family, core_mappings,
                                            direct_rmsd, geometry_anchor_check, heavy, read_one)
from molecular_agent.structure import ComplexContext
from molecular_agent.workflow import Workflow

ROOT = Path(__file__).resolve().parents[1]


def molecule(smiles="CCO"):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m, randomSeed=17) == 0
    return m


def shifted(m, dx, score=-8):
    m = Chem.Mol(m)
    conf = m.GetConformer()
    for i, pos in enumerate(conf.GetPositions()):
        conf.SetAtomPosition(i, tuple(pos + np.array([dx, 0, 0])))
    m.SetProp("minimizedAffinity", str(score))
    return m


def sdf(path, molecules):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as writer:
        for m in molecules:
            writer.write(m)
    return path


class FakeDocking(DockingAdapter):
    def __init__(self, path, **protocol):
        super().__init__({"enabled": True, "seeds": [17, 29, 43], "plip": {"enabled": False},
                          "pose_retention": {"enabled": True, "core_atom_indices": [0, 1, 2],
                                             "minimum_passing_seeds": 2, **protocol}}, path)
        self.calls = []
        self.shifts = {}
        self.reference_shifts = {}

    def _run_with_retries(self, **values):
        self.calls.append(values)
        source = read_one(values["candidate_path"])
        is_reference = values["candidate_path"] == values["reference_path"]
        shift = (self.reference_shifts if is_reference else self.shifts).get(values["seed"], 0)
        path = Path(values["output_dir"]) / "docked.sdf"
        # Rank 1 is a wrong mode with an enticing score; rank 3 is best among eligible.
        sdf(path, [shifted(source, 6+shift, -99), shifted(source, .2+shift, -8),
                   shifted(source, .4+shift, -9 if is_reference else -10)])
        return {"status": "complete", "pose_path": str(path.resolve())}


@pytest.fixture
def setup(tmp_path):
    native = sdf(tmp_path / "native.sdf", [molecule()])
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text("ATOM\nEND\n")  # No anchor/preflight needed in the mocked command runner.
    candidate = sdf(tmp_path / "candidate.sdf", [read_one(native)])
    adapter = FakeDocking(tmp_path)
    return native, receptor, candidate, adapter


def test_direct_rmsd_is_not_fitted_and_supports_atom_permutation():
    native = molecule("c1ccccc1")
    permuted = Chem.RenumberAtoms(native, list(reversed(range(native.GetNumAtoms()))))
    options = core_mappings(native, native, permuted, list(range(6)))
    xyz = heavy(native).GetConformer().GetPositions()
    assert min(direct_rmsd(o["core_xyz"], xyz)[0] for o in options) < 1e-9
    displaced = shifted(native, 5)
    options = core_mappings(native, native, displaced, list(range(6)))
    assert min(direct_rmsd(o["core_xyz"], xyz)[0] for o in options) == pytest.approx(5)


def test_charge_and_stereochemistry_identity_rejected():
    native = molecule("CCN")
    different = molecule("CC[NH3+]")
    with pytest.raises(ValueError, match="chemical identity"):
        core_mappings(native, native, different, [0, 1, 2])
    original = molecule("F[C@H](Cl)Br")
    opposite = molecule("F[C@@H](Cl)Br")
    with pytest.raises(ValueError, match="stereochemistry"):
        core_mappings(original, original, opposite, [0, 1, 2, 3])


def test_provenance_survives_fragment_deletion_and_sdf(tmp_path):
    parent = heavy(molecule("OCCC"))
    for atom in parent.GetAtoms():
        atom.SetIntProp("_reference_atom_index", atom.GetIdx())
    scaffold, _, mapping = _retained_fragment(parent, (1, 2))
    assert mapping == {0: 0, 1: 1}
    # Supply coordinates (the editor normally generates these after attachment).
    scaffold.AddConformer(Chem.Conformer(scaffold.GetNumAtoms()))
    for i in range(scaffold.GetNumAtoms()):
        scaffold.GetConformer().SetAtomPosition(i, parent.GetConformer().GetAtomPosition(i))
    Chem.SanitizeMol(scaffold)
    path = tmp_path / "edited.sdf"
    write_sdf(EditResult(scaffold, {}), path)
    source = read_one(path)
    options = core_mappings(parent, source, source, [0, 1])
    assert options[0]["retained_original_heavy_fraction"] == .5
    with pytest.raises(ValueError, match="core was deleted"):
        core_mappings(parent, source, source, [0, 1, 2])
    source.ClearProp("reference_atom_indices")
    with pytest.raises(ValueError, match="No chemically valid"):
        core_mappings(parent, source, source, [0, 1])


def test_complete_link_family_does_not_chain():
    groups = {seed: [{"seed": seed, "rank": 1, "core_xyz": [[x, 0, 0]], "crystal_core_rmsd": x}]
              for seed, x in [(17, 0), (29, 1.5), (43, 3)]}
    family = clique_family(groups, 2)
    assert len(family) == 2
    assert {r["seed"] for r in family} == {17, 29}


def test_gate_then_score_and_frozen_cache(setup, tmp_path):
    native, receptor, candidate, adapter = setup
    protocol = PoseRetention(adapter)
    baseline = protocol.prepare(native, receptor, tmp_path / "baseline")
    assert baseline["status"] == "complete"
    assert baseline["valid_seeds"] == [17, 29, 43]
    assert baseline["geometry_reference"]["rank"] == 2  # Geometry != score baseline.
    assert all(r["original_pose_rank"] == 3 for r in baseline["reference_by_seed"].values())
    assert protocol.prepare(native, receptor, tmp_path / "baseline")["status"] == "complete"
    assert len(adapter.calls) == 3
    result = adapter.run_with_reference_baseline(candidate, receptor, native, tmp_path / "candidate",
                                                 tmp_path / "baseline")
    assert result["status"] == "complete"
    assert len(adapter.calls) == 6  # Reference never redocked for a candidate.
    metric = result["comparison"]["metrics"]["minimizedAffinity"]
    assert metric["delta_candidate_minus_reference"]["mean"] == -1
    for row in result["per_seed"]:
        assert row["candidate"]["original_pose_rank"] == 3
        assert float(read_one(Path(row["candidate"]["pose_path"])).GetProp("minimizedAffinity")) == -10
    receptor.write_text("ATOM changed\n")
    assert protocol.prepare(native, receptor, tmp_path / "baseline")["failure_class"] == "frozen_reference_mismatch"


def test_frozen_artifact_tampering_is_rejected(setup, tmp_path):
    native, receptor, _, adapter = setup
    protocol = PoseRetention(adapter)
    baseline = protocol.prepare(native, receptor, tmp_path / "baseline")
    Path(baseline["geometry_reference_path"]).write_text("tampered")
    assert protocol.prepare(native, receptor, tmp_path / "baseline")["status"] == "calibration_failed"
    assert len(adapter.calls) == 3


def test_calibration_fails_no_majority_no_best_effort(setup, tmp_path):
    native, receptor, _, adapter = setup
    adapter.reference_shifts = {29: 4, 43: 5}
    baseline = PoseRetention(adapter).prepare(native, receptor, tmp_path / "baseline")
    assert baseline["status"] == "calibration_failed"
    assert "geometry_reference" not in baseline


def test_majority_two_of_three_retains_failed_seed_denominator(setup, tmp_path):
    native, receptor, candidate, adapter = setup
    protocol = PoseRetention(adapter)
    baseline = protocol.prepare(native, receptor, tmp_path / "baseline")
    adapter.shifts = {43: 6}
    result = protocol.run_candidate(candidate, native, receptor, tmp_path / "candidate", baseline)
    assert result["status"] == "complete"
    assert result["paired_seeds"] == [17, 29]
    assert result["outlier_or_failed_seeds"] == [43]
    metric = result["comparison"]["metrics"]["minimizedAffinity"]
    assert metric["candidate_better_seed_fraction"] == 2/3
    assert metric["candidate_better_fraction_of_paired_seeds"] == 1
    assert result["per_seed"][2]["status"] == "no_eligible_pose"


def test_no_eligible_pose_never_falls_back_to_rank_one(setup, tmp_path):
    native, receptor, candidate, adapter = setup
    protocol = PoseRetention(adapter)
    baseline = protocol.prepare(native, receptor, tmp_path / "baseline")
    adapter.shifts = {17: 6, 29: 6, 43: 6}
    result = protocol.run_candidate(candidate, native, receptor, tmp_path / "candidate", baseline)
    assert result["status"] == "no_eligible_pose"
    assert "evaluation_pose_path" not in result
    assert not result["evaluation_pose_paths"]
    assert not result["comparison"]["metrics"]


def test_plip_hard_anchors_require_atom_direction_geometry_not_residue():
    anchor = {"ligand_atom_index": 3, "ligand_role": "donor", "protein_residue": "LEU:A:83",
              "protein_atom_name": "O", "max_distance": 3.5, "min_donor_angle": 100}
    report = {"status": "complete", "preparation": {
        "atom_mapping": [{"pose_atom_index": 5, "complex_serial": 9000}],
        "receptor_atom_mapping": [{"complex_serial": 670, "residue": "LEU:A:83", "atom_name": "O"}]},
        "records": [{"type": "hydrogen_bond", "residue": "LEU:A:83", "fields": {
            "donoridx": "9000", "acceptoridx": "670", "protisdon": "False",
            "dist_d-a": "2.9", "don_angle": "145"}}]}
    assert anchor_check(report, [anchor], {3: 5})["status"] == "passed"
    assert anchor_check(report, [anchor], {3: 6})["status"] == "failed"
    report["records"][0]["fields"]["protisdon"] = "True"
    assert anchor_check(report, [anchor], {3: 5})["status"] == "failed"
    assert anchor_check({"status": "failed"}, [anchor], {3: 5})["status"] == "unavailable"


def test_real_crystal_geometry_anchor_and_tpo_preparation(tmp_path):
    context = ComplexContext(ROOT / "input/task.single_edit.json")
    native = Chem.AddHs(context.ligand, addCoords=True)
    receptor = context.write_receptor_pdb(tmp_path / "receptor.pdb")
    anchors = context.task["pose_retention"]["anchors"]
    assert geometry_anchor_check(native, receptor, anchors, {21: 21})["status"] == "passed"
    assert geometry_anchor_check(shifted(native, 5), receptor, anchors, {21: 21})["status"] == "failed"
    assert geometry_anchor_check(Chem.RemoveHs(native), receptor, anchors, {21: 21})["status"] == "unavailable"
    assert DockingAdapter._receptor_preflight(receptor)[0]
    manifest = json.loads(receptor.with_suffix(".preparation.json").read_text())
    assert manifest["retained_hetero_atom_count"] == 22
    poses = sdf(tmp_path / "poses.sdf", [shifted(native, 5), native])
    pdb = tmp_path / "complex.pdb"
    provenance = PLIPAdapter.build_complex(receptor, poses, pdb, pose_rank=2)
    assert provenance["pose_rank"] == 2
    atoms = [r for r in pdb.read_text().splitlines() if r.startswith(("ATOM", "HETATM"))]
    assert [int(r[6:11]) for r in atoms] == list(range(1, len(atoms)+1))
    assert not any(r.startswith("TER") for r in pdb.read_text().splitlines())
    ligand = [r for r in atoms if r[17:20] == "LIG"]
    xyz = [float(ligand[21][p:p+8]) for p in (30, 38, 46)]
    assert xyz == pytest.approx(native.GetConformer().GetAtomPosition(21), abs=.0006)


def test_calibration_failure_precedes_any_llm_or_browser(tmp_path):
    class Client:
        def complete_json(self, payload):
            pytest.fail("LLM must not be called before successful calibration")
    workflow = Workflow(ROOT / "input/task.single_edit.json", Client(), tmp_path)
    workflow._prepare_initial_context = lambda: pytest.fail("Browser must not run before calibration")
    result = workflow.run()["result"]
    assert result["status"] == "reference_calibration_failed"
    assert result["attempts"] == []


def test_rejected_candidate_cannot_be_accepted_by_history_fallback(tmp_path):
    workflow = Workflow(ROOT / "input/task.single_edit.json", object(), tmp_path)
    history = [{"attempt": 1, "candidate_path": "wrong.sdf", "docking": {"status": "no_eligible_pose"}}]
    assert workflow._accepted_output(history, tmp_path / "reference.sdf", "llm_stop")["status"] == "no_candidate_accepted"


def test_missing_seed_quality_penalty_and_failed_gate_cannot_promote(setup, tmp_path):
    native, receptor, candidate, adapter = setup
    protocol = PoseRetention(adapter)
    baseline = protocol.prepare(native, receptor, tmp_path / "baseline")
    adapter.shifts = {43: 6}
    result = protocol.run_candidate(candidate, native, receptor, tmp_path / "candidate", baseline)
    workflow = Workflow(ROOT / "input/task.single_edit.json", object(), tmp_path / "workflow")
    transformation = {"site_type": "atom", "change_type": "addition", "edit_atom_index": 10, "fragment_smiles": "[*:1]C"}
    entry = workflow._record_docking_result(1, transformation, candidate, result)
    assert entry["quality"] == pytest.approx(1-1/3)
    assert entry["seed_coverage_penalty"] == pytest.approx(1/3)
    result["status"] = "no_eligible_pose"
    result["pose_retention"]["status"] = "failed"
    entry = workflow._record_docking_result(2, transformation, candidate, result)
    assert entry["quality"] is None
    assert not entry["stability_eligible"]
    assert workflow.state.convergence["best_attempt"] == 1


def test_no_eligible_pose_is_returned_to_designer_not_accepted(tmp_path):
    class Client:
        calls = []
        def complete_json(self, payload):
            self.calls.append(payload)
            return {"action": "STOP", "stop_reason": "unable_to_repair"}
    class RejectedDocking:
        def run_with_reference_baseline(self, **values):
            return {"status": "no_eligible_pose", "seed_count": 3,
                    "pose_retention": {"status": "failed"}, "comparison": {"metrics": {}},
                    "paired_seed_fraction": 0, "candidate_per_seed": {}, "reference_baseline": {}}
    client = Client()
    workflow = Workflow(ROOT / "input/task.single_edit.json", client, tmp_path)
    workflow.context.task["external_research"] = {"enabled": False, "required": False}
    workflow._prepare_initial_context()
    workflow.docking_adapter = RejectedDocking()
    result = workflow.design({"action": "READY", "site_type": "atom", "change_type": "addition",
                              "understanding": "Test original ligand context", "edit_hypothesis": "Try para methyl",
                              "edit_atom_index": 10, "fragment_smiles": "[*:1]C"})
    assert result["status"] == "no_candidate_accepted"
    assert len(client.calls) == 1
    feedback = client.calls[0]
    assert "no_eligible_pose" in json.dumps(feedback)
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["docking"]["status"] == "no_eligible_pose"


def test_plip_unavailable_is_evaluation_failure_not_anchor_absence(setup, tmp_path, monkeypatch):
    native, receptor, _, adapter = setup
    adapter.config["plip"]["enabled"] = True
    adapter.config["pose_retention"]["anchors"] = [{
        "type": "hydrogen_bond", "method": "plip", "ligand_atom_index": 2, "ligand_role": "donor",
        "protein_residue": "ALA:A:1", "protein_atom_name": "O", "max_distance": 3.5, "min_donor_angle": 100}]
    protocol = PoseRetention(adapter)
    monkeypatch.setattr(protocol.plip, "run", lambda *args, **kwargs: {"status": "failed", "error": "tool crashed"})
    raw = adapter._run_with_retries(candidate_path=native, reference_path=native,
                                   seed=17, output_dir=tmp_path / "raw")
    result = protocol.evaluate(native, native, receptor, raw, tmp_path / "evaluation", 17)
    assert result["status"] == "evaluation_failed"
    assert result["poses"][1]["reason"] == "anchor_evaluation_unavailable"
    assert protocol.select(result) is None


def test_bad_numeric_protocol_rejected(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        PoseRetention(FakeDocking(tmp_path, candidate_reference_rmsd=float("nan")))
    with pytest.raises(ValueError, match="nonnegative"):
        PoseRetention(FakeDocking(tmp_path, missing_seed_penalty=-1))
