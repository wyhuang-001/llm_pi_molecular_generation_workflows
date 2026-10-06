"""Calibrated, immutable docking reference and post-docking pose selection.

No ligand alignment, positional constraints, candidate-specific MCS, or rank-1 fallback.
Geometry, interaction anchors and score baselines are distinct audited artifacts.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Lipinski

from .plip_adapter import PLIPAdapter, compare_plip
from .editing import VDW
from .structure import parse_pdb

PROTOCOL_VERSION = 1
MAX_MATCHES = 4096


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_one(path: Path) -> Chem.Mol:
    molecules = list(Chem.SDMolSupplier(str(path), removeHs=False))
    if len(molecules) != 1 or molecules[0] is None or not molecules[0].GetNumConformers():
        raise ValueError(f"Expected exactly one readable molecule with coordinates: {path}")
    return molecules[0]


def heavy(molecule: Chem.Mol) -> Chem.Mol:
    molecule = Chem.Mol(molecule)
    for atom in molecule.GetAtoms():
        atom.SetIntProp("_input_index", atom.GetIdx())
        atom.SetAtomMapNum(0)
    return Chem.RemoveHs(molecule)


def matches(target: Chem.Mol, query: Chem.Mol) -> tuple:
    result = target.GetSubstructMatches(query, uniquify=False, useChirality=True,
                                       maxMatches=MAX_MATCHES + 1)
    if len(result) > MAX_MATCHES:
        raise ValueError("Symmetry enumeration limit exceeded; refusing a partial minimum")
    if not result:
        raise ValueError("No chemically valid atom mapping")
    return result


def direct_rmsd(a: list | np.ndarray, b: list | np.ndarray) -> tuple[float, float]:
    """Same receptor frame: deliberately no Kabsch fit/translation."""
    displacement = np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=1)
    if not np.isfinite(displacement).all() or not len(displacement):
        raise ValueError("Invalid coordinates/core")
    return float(np.sqrt(np.mean(displacement ** 2))), float(displacement.max())


def core_mappings(native: Chem.Mol, source: Chem.Mol, pose: Chem.Mol, core: list[int]) -> list[dict]:
    """Full chemical isomorphisms source->pose plus Host edit provenance.

    Legacy offline additions may use a full ORIGINAL graph substructure match; deletions
    require provenance. Never derive or shrink a core with an MCS per candidate.
    """
    n, s, p = heavy(native), heavy(source), heavy(pose)
    if Chem.MolToSmiles(s, isomericSmiles=True) != Chem.MolToSmiles(p, isomericSmiles=True):
        raise ValueError("Docked pose chemical identity/stereochemistry differs from input")
    if any(i < 0 or i >= n.GetNumAtoms() for i in core):
        raise ValueError("Core atom outside original heavy-atom graph")
    source_maps = []
    if source.HasProp("reference_atom_indices"):
        ids = json.loads(source.GetProp("reference_atom_indices"))
        if len(ids) != source.GetNumAtoms():
            raise ValueError("Invalid reference atom provenance length")
        mapping = {}
        for atom in s.GetAtoms():
            ref = ids[atom.GetIntProp("_input_index")]
            if ref is None:
                continue
            if type(ref) is not int or ref < 0 or ref >= n.GetNumAtoms() or ref in mapping:
                raise ValueError("Invalid/duplicate original atom identity")
            original = n.GetAtomWithIdx(ref)
            signature = lambda a: (a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(), a.GetIsAromatic())
            if signature(atom) != signature(original):
                raise ValueError("Retained atom chemistry changed")
            mapping[ref] = atom.GetIdx()
        if not set(core).issubset(mapping):
            raise ValueError("Fixed comparison core was deleted; no smaller MCS is permitted")
        for i, j in itertools.combinations(mapping, 2):
            nb, sb = n.GetBondBetweenAtoms(i, j), s.GetBondBetweenAtoms(mapping[i], mapping[j])
            signature = lambda b: (str(b.GetBondType()), b.GetIsAromatic()) if b else None
            if signature(nb) != signature(sb):
                raise ValueError("Retained core/scaffold bond identity changed")
        source_maps.append(mapping)
    else:
        # Strict legacy/offline compatibility; matches the whole original, never just a convenient ring.
        source_maps = [dict(enumerate(m)) for m in matches(s, n)]
    pose_matches = matches(p, s)
    coordinates = pose.GetConformer().GetPositions()
    options, seen = [], set()
    for source_map in source_maps:
        for match in pose_matches:
            ref_to_pose = {ref: p.GetAtomWithIdx(match[idx]).GetIntProp("_input_index")
                           for ref, idx in source_map.items()}
            ordered = tuple(ref_to_pose[i] for i in core)
            if ordered in seen:
                continue
            seen.add(ordered)
            xyz = coordinates[list(ordered)]
            if not np.isfinite(xyz).all():
                raise ValueError("Nonfinite pose coordinates")
            options.append({"core_xyz": xyz.tolist(), "reference_to_pose": ref_to_pose,
                            "retained_original_heavy_fraction": len(ref_to_pose) / n.GetNumAtoms()})
    return options


def heavy_steric_check(pose: Chem.Mol, receptor: Path, maximum_overlap: float) -> dict:
    """Final-pose receptor overlap, all ligand/receptor heavy atoms, no fitted coordinates.

    Explicit protein H are intentionally excluded: a favorable D-H...A bond must not
    be mislabeled as a heavy-atom collision. This is a geometric gate, not an energy.
    """
    protein = [a for a in parse_pdb(receptor) if a.element not in {"H", "D", "T"}]
    ligand = [a for a in pose.GetAtoms() if a.GetAtomicNum() != 1]
    if not protein or not ligand:
        raise ValueError("No heavy atoms for final-pose steric evaluation")
    if any(a.element not in VDW for a in protein) or any(a.GetSymbol().upper() not in VDW for a in ligand):
        raise ValueError("Unknown element radius in final-pose steric evaluation")
    pxyz = np.array([a.xyz for a in protein])
    lxyz = pose.GetConformer().GetPositions()[[a.GetIdx() for a in ligand]]
    if not np.isfinite(pxyz).all() or not np.isfinite(lxyz).all():
        raise ValueError("Nonfinite final-pose steric coordinates")
    distances = np.linalg.norm(lxyz[:, None, :] - pxyz[None, :, :], axis=2)
    overlaps = np.array([VDW[a.GetSymbol().upper()] for a in ligand])[:, None] + np.array([VDW[a.element] for a in protein])[None, :] - distances
    i, j = np.unravel_index(np.argmax(overlaps), overlaps.shape)
    worst = protein[j]
    value = float(overlaps[i, j])
    return {"status": "passed" if value <= maximum_overlap else "failed", "maximum_allowed_overlap": maximum_overlap,
            "max_heavy_atom_overlap": value, "overlapping_pair_count": int(np.sum(overlaps > maximum_overlap)),
            "worst_ligand_pose_atom_index": ligand[i].GetIdx(),
            "worst_protein_atom": f"{worst.residue_name}:{worst.chain}:{worst.residue_number}:{worst.name}"}


def anchor_check(plip: dict, anchors: list[dict], mapping: dict[int, int]) -> dict:
    """Atom identity + donor/acceptor direction + PLIP-assigned H-bond geometry.

    Non-anchor PLIP interactions stay soft feedback. Missing PLIP is UNKNOWN, not zero.
    """
    if not anchors:
        return {"status": "passed", "anchors": []}
    if plip.get("status") != "complete":
        return {"status": "unavailable", "reason": "PLIP evaluation unavailable"}
    preparation = plip.get("preparation", {})
    ligand_serials = {row["pose_atom_index"]: row["complex_serial"]
                      for row in preparation.get("atom_mapping", [])}
    protein = {row["complex_serial"]: row for row in preparation.get("receptor_atom_mapping", [])}
    results = []
    for anchor in anchors:
        ref = anchor["ligand_atom_index"]
        ligand = ligand_serials.get(mapping.get(ref))
        ligand_donor = anchor["ligand_role"] == "donor"
        matched = []
        for record in plip.get("records", []):
            if record["type"] != "hydrogen_bond":
                continue
            fields = record["fields"]
            if fields.get("protisdon", "").lower() != ("false" if ligand_donor else "true"):
                continue
            try:
                lig_index = int(fields["donoridx" if ligand_donor else "acceptoridx"])
                prot_index = int(fields["acceptoridx" if ligand_donor else "donoridx"])
                distance, angle = float(fields["dist_d-a"]), float(fields["don_angle"])
            except (KeyError, ValueError, TypeError):
                continue
            prot = protein.get(prot_index, {})
            if (lig_index == ligand and prot.get("residue") == anchor["protein_residue"]
                    and prot.get("atom_name") == anchor["protein_atom_name"]
                    and prot.get("insertion_code", "") == anchor.get("insertion_code", "")
                    and 0 < distance <= anchor["max_distance"]
                    and anchor["min_donor_angle"] <= angle <= 180):
                matched.append({"donor_acceptor_distance": distance, "donor_angle": angle,
                                "ligand_complex_serial": lig_index, "protein_complex_serial": prot_index})
        results.append({"anchor": anchor, "passed": bool(matched), "matches": matched})
    return {"status": "passed" if all(row["passed"] for row in results) else "failed", "anchors": results}


def geometry_anchor_check(pose: Chem.Mol, receptor: Path, anchors: list[dict], mapping: dict[int, int]) -> dict:
    """Conservative explicit-H donor -> protein backbone carbonyl-O anchors.

    PLIP can suppress an otherwise valid pair in favor of a competing acceptor; use
    independently specified atom geometry, not that residue-level selection, for these
    reviewed anchors. No inferred receptor hydrogens/protonation and no H optimization.
    """
    if not anchors:
        return {"status": "passed", "anchors": []}
    lines = [line for line in receptor.read_text().splitlines() if line[:6].strip() in {"ATOM", "HETATM"}]
    donor_ids = {match[0] for match in Lipinski._HDonors(pose)}
    xyz = pose.GetConformer().GetPositions()
    results = []
    for anchor in anchors:
        target = [line for line in lines if
                  f"{line[17:20].strip()}:{line[21:22].strip()}:{int(line[22:26])}" == anchor["protein_residue"]
                  and line[26:27].strip() == anchor.get("insertion_code", "")]
        oxygen = [line for line in target if line[12:16].strip() == "O"]
        carbon = [line for line in target if line[12:16].strip() == "C"]
        coordinates = lambda line: np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if (len(oxygen) != 1 or len(carbon) != 1 or
                oxygen[0][76:78].strip().upper() != "O" or carbon[0][76:78].strip().upper() != "C" or
                not 1.05 <= np.linalg.norm(coordinates(oxygen[0])-coordinates(carbon[0])) <= 1.65):
            return {"status": "unavailable", "reason": "Protein backbone carbonyl identity/geometry is ambiguous"}
        index = mapping.get(anchor["ligand_atom_index"])
        if index is None or index not in donor_ids:
            results.append({"anchor": anchor, "passed": False, "reason": "Ligand atom is not a chemical H-bond donor"})
            continue
        hydrogens = [a.GetIdx() for a in pose.GetAtomWithIdx(index).GetNeighbors() if a.GetAtomicNum() == 1]
        if not hydrogens:
            return {"status": "unavailable", "reason": "Explicit donor H missing; do not invent an H-bond angle"}
        acceptor = coordinates(oxygen[0])
        distance = float(np.linalg.norm(xyz[index]-acceptor))
        details = []
        for h in hydrogens:
            u, v = xyz[index]-xyz[h], acceptor-xyz[h]
            norm = float(np.linalg.norm(u)*np.linalg.norm(v))
            if norm <= 0 or not math.isfinite(norm):
                continue
            angle = float(np.degrees(np.arccos(np.clip(np.dot(u, v)/norm, -1, 1))))
            h_distance = float(np.linalg.norm(v))
            details.append({"donor_acceptor_distance": distance, "hydrogen_acceptor_distance": h_distance,
                            "donor_angle": angle, "hydrogen_pose_atom_index": h,
                            "passed": 0 < distance <= anchor["max_distance"] and
                            0 < h_distance <= anchor["max_hydrogen_distance"] and angle >= anchor["min_donor_angle"]})
        if not details:
            return {"status": "unavailable", "reason": "Donor H coordinates are degenerate/nonfinite"}
        results.append({"anchor": anchor, "passed": any(row["passed"] for row in details), "geometry": details})
    return {"status": "passed" if all(row["passed"] for row in results) else "failed", "anchors": results}


def clique_family(groups: dict[int, list[dict]], threshold: float, budget: int = 200000) -> list[dict]:
    """Largest complete-link family, at most one pose/seed; no single-link chaining.

    Ties use crystal recovery, not GNINA score. Search is bounded and fails explicitly
    rather than silently using a greedy result if an unusually large protocol is supplied.
    """
    best: list[dict] = []
    best_cost = math.inf
    visits = 0
    ordered = list(groups.values())

    def visit(level: int, chosen: list[dict]) -> None:
        nonlocal best, best_cost, visits
        visits += 1
        if visits > budget:
            raise ValueError("Reference family search budget exceeded")
        if len(chosen) + len(ordered) - level < len(best):
            return
        if level == len(ordered):
            cost = sum(row["crystal_core_rmsd"] for row in chosen)
            if len(chosen) > len(best) or (len(chosen) == len(best) and cost < best_cost):
                best, best_cost = list(chosen), cost
            return
        for row in ordered[level]:
            if all(direct_rmsd(row["core_xyz"], other["core_xyz"])[0] <= threshold for other in chosen):
                visit(level+1, chosen+[row])
        visit(level+1, chosen)

    visit(0, [])
    return best


class PoseRetention:
    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.config = dict(adapter.config.get("pose_retention") or {})
        self.core = self.config.get("core_atom_indices", [])
        if (not isinstance(self.core, list) or not self.core or
                any(type(i) is not int or i < 0 for i in self.core) or len(set(self.core)) != len(self.core)):
            raise ValueError("pose_retention requires fixed, unique original core_atom_indices")
        self.seeds = adapter._seeds()
        self.minimum = self.config.get("minimum_passing_seeds", 2)
        if type(self.minimum) is not int:
            raise ValueError("minimum_passing_seeds must be an integer, not a rounded fraction")
        if not 1 <= self.minimum <= len(self.seeds) or self.minimum <= len(self.seeds) / 2:
            raise ValueError("minimum_passing_seeds must be a strict majority of configured seeds")
        self.top_n = self.config.get("top_n", 20)
        if type(self.top_n) is not int or self.top_n < 1:
            raise ValueError("top_n must be positive")
        self.native_limit = self._positive("reference_native_rmsd", 2.0)
        self.candidate_limit = self._positive("candidate_reference_rmsd", 2.0)
        self.max_displacement = self._positive("max_core_displacement", 3.0)
        self.family_limit = self._positive("family_rmsd", 2.0)
        self.steric_limit = (self._positive("maximum_heavy_atom_overlap", .55)
                             if "maximum_heavy_atom_overlap" in self.config else None)
        penalty = float(self.config.get("missing_seed_penalty", 1.0))
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError("missing_seed_penalty must be finite and nonnegative")
        self.metric = self.config.get("primary_metric", "minimizedAffinity")
        if self.metric not in adapter.SCORE_METRICS:
            raise ValueError("Unsupported pose selection score")
        self.lower = adapter.SCORE_METRICS[self.metric] == "lower_is_better"
        self.anchors = self.config.get("anchors", [])
        for anchor in self.anchors:
            if (anchor.get("type") != "hydrogen_bond" or anchor.get("ligand_role") not in {"donor", "acceptor"}
                    or anchor.get("ligand_atom_index") not in self.core
                    or not anchor.get("protein_residue") or not anchor.get("protein_atom_name")
                    or not 0 < anchor.get("max_distance", 0) < 10
                    or not 0 < anchor.get("min_donor_angle", 0) <= 180):
                raise ValueError("Anchors need a fixed core atom, protein atom, donor role and explicit geometry")
            method = anchor.get("method", "plip")
            if method not in {"plip", "explicit_atom_geometry"}:
                raise ValueError("Unsupported hard anchor evaluation method")
            if method == "explicit_atom_geometry" and (anchor["ligand_role"] != "donor" or
                    anchor["protein_atom_name"] != "O" or
                    anchor.get("protein_role") != "backbone_carbonyl_acceptor" or
                    not 0 < anchor.get("max_hydrogen_distance", 0) < 5):
                raise ValueError("Explicit geometry currently supports only ligand donors -> reviewed backbone carbonyl O")
        self.plip_anchors = [a for a in self.anchors if a.get("method", "plip") == "plip"]
        self.geometry_anchors = [a for a in self.anchors if a.get("method") == "explicit_atom_geometry"]
        if self.plip_anchors and not (adapter.config.get("plip") or {}).get("enabled"):
            raise ValueError("PLIP hard anchors require enabled PLIP")
        self.plip = PLIPAdapter(adapter.config.get("plip") or {})

    def check_anchors(self, pose: Chem.Mol, receptor: Path, plip: dict, mapping: dict) -> dict:
        checks = [anchor_check(plip, self.plip_anchors, mapping),
                  geometry_anchor_check(pose, receptor, self.geometry_anchors, mapping)]
        status = "unavailable" if any(c["status"] == "unavailable" for c in checks) else (
            "passed" if all(c["status"] == "passed" for c in checks) else "failed")
        return {"status": status, "checks": checks}

    def _positive(self, key: str, default: float) -> float:
        value = float(self.config.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        return value

    def signature(self, native: Path, receptor: Path) -> dict:
        executables = {}
        command = self.adapter.config.get("command") or []
        for name in ([command[0]] if command else []) + [self.plip.config.get("executable", "plip")]:
            resolved = shutil.which(str(name))
            if resolved:
                path = Path(resolved).resolve()
                stat = path.stat()
                executables[str(name)] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        try:
            plip_version = version("plip")
        except PackageNotFoundError:
            plip_version = "unavailable"
        modules = ["pose_retention.py", "plip_adapter.py", "adapters.py", "structure.py", "editing.py"]
        return {"protocol_version": PROTOCOL_VERSION, "rdkit_version": rdBase.rdkitVersion,
                "plip_version": plip_version, "executables": executables,
                "implementation_sha256": {name: digest(Path(__file__).with_name(name)) for name in modules},
                "native_sha256": digest(native), "receptor_sha256": digest(receptor),
                "docking_and_plip_protocol": self.adapter.config}

    def evaluate(self, native_path: Path, source_path: Path, receptor: Path, raw: dict,
                 out: Path, seed: int, reference: dict | None = None) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        native, source = read_one(native_path), read_one(source_path)
        native_xyz = heavy(native).GetConformer().GetPositions()[self.core]
        records = []
        supplier = Chem.SDMolSupplier(raw["pose_path"], removeHs=False)
        for rank in range(1, min(len(supplier), self.top_n) + 1):
            pose = supplier[rank-1]
            row: dict = {"seed": seed, "rank": rank, "status": "ineligible"}
            records.append(row)
            try:
                if pose is None or not pose.GetNumConformers():
                    raise ValueError("Unreadable pose")
                properties = {}
                for key in self.adapter.SCORE_METRICS:
                    if pose.HasProp(key):
                        try:
                            value = float(pose.GetProp(key))
                        except ValueError:
                            continue
                        if math.isfinite(value):
                            properties[key] = pose.GetProp(key)
                score = float(properties.get(self.metric, "nan"))
                if not math.isfinite(score):
                    raise ValueError("Missing/nonfinite primary score")
                row["properties"], row["score"] = properties, score
                options = core_mappings(native, source, pose, self.core)
                target_xyz = reference["core_xyz"] if reference else native_xyz
                for option in options:
                    option["reference_core_rmsd"], option["max_core_displacement"] = direct_rmsd(option["core_xyz"], target_xyz)
                    option["crystal_core_rmsd"], _ = direct_rmsd(option["core_xyz"], native_xyz)
                options.sort(key=lambda item: item["reference_core_rmsd"])
                row.update(options[0])
                limit = self.candidate_limit if reference else self.native_limit
                geometric = [item for item in options if item["reference_core_rmsd"] <= limit
                             and item["max_core_displacement"] <= self.max_displacement]
                if not geometric:
                    row["reason"] = "core_geometry_gate"
                    continue
                if self.steric_limit is not None:
                    row["steric_check"] = heavy_steric_check(pose, receptor, self.steric_limit)
                    if row["steric_check"]["status"] != "passed":
                        row["reason"] = "final_pose_heavy_atom_clash"
                        continue
                plip = self.plip.run(receptor, Path(raw["pose_path"]), out / f"pose-{rank:03d}", pose_rank=rank)
                row["plip"] = plip
                if self.plip_anchors and plip.get("status") != "complete":
                    row.update(status="evaluation_failed", reason="anchor_evaluation_unavailable")
                    continue
                for option in geometric:
                    check = self.check_anchors(pose, receptor, plip, option["reference_to_pose"])
                    row["anchor_check"] = check
                    if check["status"] == "unavailable":
                        row.update(status="evaluation_failed", reason="anchor_evaluation_unavailable")
                        break
                    if check["status"] == "passed":
                        row.update(option, status="eligible")
                        break
                if row["status"] == "ineligible":
                    row["reason"] = "interaction_anchor_gate"
            except (ValueError, RuntimeError, KeyError, IndexError, TypeError) as exc:
                row.update(status="evaluation_failed", reason="identity_mapping_or_score", error=str(exc))
        # A failed evaluation could conceal the best eligible score: do not silently pick another.
        failed = any(row["status"] == "evaluation_failed" for row in records)
        eligible = [row for row in records if row["status"] == "eligible"]
        result = {"status": "evaluation_failed" if failed or not records else "complete",
                  "seed": seed, "total_output_poses": len(supplier), "evaluated_pose_count": len(records),
                  "eligible_pose_count": len(eligible), "poses": records}
        save_json(out / "pose-evaluation.json", result)
        return result

    def select(self, evaluation: dict) -> dict | None:
        if evaluation["status"] != "complete":
            return None
        eligible = [row for row in evaluation["poses"] if row["status"] == "eligible"]
        return min(eligible, key=lambda row: (row["score"] if self.lower else -row["score"], row["rank"])) if eligible else None

    @staticmethod
    def export_pose(raw: dict, selected: dict, destination: Path) -> dict:
        molecule = Chem.SDMolSupplier(raw["pose_path"], removeHs=False)[selected["rank"]-1]
        with Chem.SDWriter(str(destination)) as writer:
            writer.write(molecule)
        result = dict(raw)
        result.update(pose_path=str(destination.resolve()), source_pose_path=raw["pose_path"],
                      original_pose_rank=selected["rank"], evaluation_pose=selected,
                      poses=[{"rank": 1, "original_rank": selected["rank"], "properties": selected["properties"]}],
                      pose_count=1, plip=selected.get("plip", {}),
                      top_pose={"rank": selected["rank"], "properties": selected["properties"]},
                      pose_selection={"policy": "geometry_and_anchor_gate_then_primary_score",
                                      "original_rank": selected["rank"], "evaluation_pose_path": str(destination.resolve())})
        return result

    def prepare(self, native: Path, receptor: Path, out: Path) -> dict:
        """Run once before design; cache is tied to exact prepared inputs and protocol."""
        out = out.resolve()
        out.mkdir(parents=True, exist_ok=True)
        manifest = out / "calibration.json"
        signature = self.signature(native, receptor)
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved.get("signature") != signature:
                return {"status": "calibration_failed", "failure_class": "frozen_reference_mismatch",
                        "error": "Prepared inputs/protocol changed; use a new run directory"}
            for filename, expected in saved.get("artifact_sha256", {}).items():
                path = Path(filename)
                if not path.exists() or digest(path) != expected:
                    return {"status": "calibration_failed", "failure_class": "frozen_reference_mismatch",
                            "error": f"Frozen artifact missing/changed: {filename}"}
            return saved
        if any(out.glob("seed-*/docked.sdf")):
            return {"status": "calibration_failed", "error": "Unversioned/interrupted reference outputs exist; use a new run directory"}
        raw_runs: dict = {}
        result: dict = {"status": "calibration_failed", "signature": signature,
                        "core_atom_indices": self.core, "seeds": self.seeds, "per_seed": [],
                        "minimum_passing_seeds": self.minimum,
                        "coordinate_policy": "direct receptor-frame RMSD; no ligand alignment or positional restraint"}
        try:
            native_mol = read_one(native)
            native_xyz = heavy(native_mol).GetConformer().GetPositions()[self.core].tolist()
            if self.steric_limit is not None:
                result["crystal_steric_check"] = heavy_steric_check(native_mol, receptor, self.steric_limit)
                if result["crystal_steric_check"]["status"] != "passed":
                    raise ValueError("Prepared crystal fails the final-pose heavy-atom overlap gate")
            native_plip = self.plip.run(receptor, native, out / "crystal-plip")
            native_map = {i: heavy(native_mol).GetAtomWithIdx(i).GetIntProp("_input_index") for i in self.core}
            check = self.check_anchors(native_mol, receptor, native_plip, native_map)
            result["crystal_anchor_check"] = check
            result["crystal_plip"] = native_plip
            if check["status"] != "passed":
                raise ValueError("Configured hard anchors are not supported by prepared crystal (or PLIP unavailable)")
            raw_runs, evaluations, groups = {}, {}, {}
            for seed in self.seeds:
                directory = out / f"seed-{seed:05d}"
                raw = self.adapter._run_with_retries(candidate_path=native, receptor_path=receptor,
                    reference_path=native, output_dir=directory, seed=seed)
                raw_runs[seed] = raw
                if raw.get("status") != "complete":
                    result["per_seed"].append({"seed": seed, "status": "docking_failed", "docking": raw})
                    continue
                evaluation = self.evaluate(native, native, receptor, raw, directory / "native-gate", seed)
                evaluations[seed] = evaluation
                result["per_seed"].append(evaluation)
                if evaluation["status"] == "complete":
                    groups[seed] = [row for row in evaluation["poses"] if row["status"] == "eligible"]
            family = clique_family(groups, self.family_limit)
            if len(family) < self.minimum:
                raise ValueError("No absolute native-like, anchor-supported reference family with majority seed support")
            representative = min(family, key=lambda row: (
                sum(direct_rmsd(row["core_xyz"], other["core_xyz"])[0] for other in family),
                row["crystal_core_rmsd"]))
            geometry = self.export_pose(raw_runs[representative["seed"]], representative, out / "geometry-reference.sdf")
            frozen_family = []
            for row in family:
                exported = self.export_pose(raw_runs[row["seed"]], row,
                                            out / f"geometry-family-seed-{row['seed']:05d}.sdf")
                frozen_family.append({"seed": row["seed"], "rank": row["rank"],
                                      "pose_path": exported["pose_path"], "core_xyz": row["core_xyz"],
                                      "crystal_core_rmsd": row["crystal_core_rmsd"]})
            result.update(geometry_reference=representative, geometry_reference_path=geometry["pose_path"],
                          reference_family=frozen_family, crystal_core_xyz=native_xyz)
            # Independent score baseline: best score per seed among native-like poses in this fixed mode.
            selected, selected_rows = {}, {}
            for seed, evaluation in evaluations.items():
                eligible = []
                for row in evaluation["poses"]:
                    if row["status"] != "eligible":
                        continue
                    rmsd, maximum = direct_rmsd(row["core_xyz"], representative["core_xyz"])
                    if rmsd <= self.candidate_limit and maximum <= self.max_displacement:
                        eligible.append(row)
                choice = self.select({"status": evaluation["status"], "poses": eligible})
                if choice:
                    selected_rows[seed] = choice
            score_family = clique_family({seed: [row] for seed, row in selected_rows.items()}, self.family_limit)
            valid = {row["seed"] for row in score_family}
            if len(valid) < self.minimum:
                raise ValueError("Reference score Evaluation Poses do not have majority family support")
            for seed in sorted(valid):
                selected[seed] = self.export_pose(raw_runs[seed], selected_rows[seed], out / f"evaluation-seed-{seed:05d}.sdf")
            result.update(status="complete", reference_by_seed={str(seed): value for seed, value in selected.items()},
                          valid_seeds=sorted(valid), invalid_seeds=sorted(set(self.seeds)-valid),
                          passing_seed_count=len(valid), passing_seed_fraction=len(valid)/len(self.seeds),
                          score_baseline_policy="best primary score among gated poses in the frozen mode; paired per seed",
                          anchors=self.anchors)
        except (ValueError, RuntimeError, OSError, KeyError, IndexError) as exc:
            result.update(failure_class="reference_calibration", error=str(exc))
        # Freeze all evidence, not just the representative. Cache cannot consume modified old poses.
        result["artifact_sha256"] = {str(path): digest(path) for path in out.rglob("*")
                                     if path.is_file() and path != manifest and not path.name.endswith(".tmp")}
        for raw in raw_runs.values():
            if raw.get("pose_path") and Path(raw["pose_path"]).is_file():
                result["artifact_sha256"][str(Path(raw["pose_path"]).resolve())] = digest(Path(raw["pose_path"]))
        save_json(manifest, result)
        return result

    @staticmethod
    def _pose_evidence(
        per_seed: list[dict[str, Any]], family_seeds: set[int], paired: set[int],
        minimum_family_size: int,
    ) -> dict[str, Any]:
        """Build the compact, multi-objective evidence sent back to the planner.

        The three axes are deliberately kept separate: RMSD measures geometry, pose status
        measures eligibility/consensus, and PLIP comparison describes interaction changes.
        None of these fields is an experimental affinity claim.
        """
        rmsd_rows = []
        pose_rows = []
        interaction_rows = []
        for item in per_seed:
            seed = item.get("seed")
            evaluation = item.get("evaluation_pose") or {}
            pose_rows.append({
                "seed": seed,
                "status": item.get("status"),
                "selected_rank": evaluation.get("rank"),
                "family_member": bool(item.get("family_member")),
                "paired_baseline_available": bool(item.get("paired_baseline_available")),
                "native_like": item.get("status") == "eligible" and bool(item.get("family_member")),
            })
            if item.get("status") == "eligible":
                rmsd_rows.append({
                    "seed": seed,
                    "reference_core_rmsd": evaluation.get("reference_core_rmsd"),
                    "crystal_core_rmsd": evaluation.get("crystal_core_rmsd"),
                    "max_core_displacement": evaluation.get("max_core_displacement"),
                })
            comparison = item.get("plip_comparison") or {}
            if comparison.get("status") == "complete":
                interaction_rows.append({
                    "seed": seed,
                    "retained": comparison.get("retained", []),
                    "gained": comparison.get("gained", []),
                    "lost": comparison.get("lost", []),
                })

        def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
            return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]

        def consensus(rows: list[dict[str, Any]], key: str) -> list[str]:
            counts: dict[str, int] = {}
            for row in rows:
                for value in row.get(key, []):
                    counts[str(value)] = counts.get(str(value), 0) + 1
            threshold = max(1, len(rows) // 2 + 1)
            return sorted(value for value, count in counts.items() if count >= threshold)

        reference_rmsd = numeric(rmsd_rows, "reference_core_rmsd")
        crystal_rmsd = numeric(rmsd_rows, "crystal_core_rmsd")
        max_displacement = numeric(rmsd_rows, "max_core_displacement")
        native_like = len(paired) >= minimum_family_size and bool(family_seeds)
        return {
            "status": "complete" if rmsd_rows else "unavailable",
            "native_like": native_like,
            "rmsd": {
                "coordinate_frame": "shared_receptor_frame_without_post_alignment",
                "core_atom_rmsd_to_frozen_reference_per_seed": rmsd_rows,
                "mean_core_rmsd_to_frozen_reference": (sum(reference_rmsd) / len(reference_rmsd) if reference_rmsd else None),
                "max_core_rmsd_to_frozen_reference": max(reference_rmsd) if reference_rmsd else None,
                "mean_core_rmsd_to_crystal": (sum(crystal_rmsd) / len(crystal_rmsd) if crystal_rmsd else None),
                "max_core_displacement": max(max_displacement) if max_displacement else None,
            },
            "pose": {
                "classification": "native_like" if native_like else "no_majority_native_like_pose",
                "family_seeds": sorted(family_seeds),
                "paired_seeds": sorted(paired),
                "family_size": len(family_seeds),
                "paired_size": len(paired),
                "minimum_family_size": minimum_family_size,
                "per_seed": pose_rows,
            },
            "interactions": {
                "status": "complete" if interaction_rows else "unavailable",
                "per_seed": interaction_rows,
                "retained_consensus": consensus(interaction_rows, "retained"),
                "gained_consensus": consensus(interaction_rows, "gained"),
                "lost_consensus": consensus(interaction_rows, "lost"),
            },
            "interpretation": (
                "RMSD, native-like pose eligibility, and interaction changes are separate audited evidence. "
                "They support ranking and structural interpretation, not experimental affinity or activity."
            ),
        }

    def run_candidate(self, candidate: Path, native: Path, receptor: Path, out: Path, calibration: dict) -> dict:
        out = out.resolve()
        out.mkdir(parents=True, exist_ok=True)
        if calibration.get("status") != "complete":
            return {"status": "calibration_failed", "reference_baseline": calibration,
                    "error": calibration.get("error", "Reference calibration failed")}
        per_seed, selected = [], {}
        for seed in self.seeds:
            directory = out / f"seed-{seed:05d}"
            raw = self.adapter._run_with_retries(candidate_path=candidate, receptor_path=receptor,
                reference_path=native, output_dir=directory, seed=seed)
            item = {"seed": seed, "status": "evaluation_failed", "raw_docking": raw}
            per_seed.append(item)
            if raw.get("status") != "complete":
                item["reason"] = "docking_failed"
                continue
            try:
                evaluation = self.evaluate(native, candidate, receptor, raw, directory / "pose-gate", seed,
                                           calibration["geometry_reference"])
            except (ValueError, OSError, RuntimeError) as exc:
                item.update(reason="pose_evaluation_unavailable", error=str(exc))
                continue
            item["pose_evaluation"] = evaluation
            choice = self.select(evaluation)
            if not choice:
                item.update(status="evaluation_failed" if evaluation["status"] != "complete" else "no_eligible_pose")
                continue
            selected[seed] = choice
            item.update(status="eligible", evaluation_pose=choice,
                        candidate=self.export_pose(raw, choice, directory / "evaluation-pose.sdf"))
        # Do not drop outlier seeds and then pretend every seed passed.
        family = clique_family({seed: [row] for seed, row in selected.items()}, self.family_limit)
        family_seeds = {row["seed"] for row in family}
        paired = family_seeds & set(calibration["valid_seeds"])
        comparisons = []
        for item in per_seed:
            seed = item["seed"]
            item["family_member"] = seed in family_seeds
            item["paired_baseline_available"] = seed in calibration["valid_seeds"]
            if seed in selected and seed not in family_seeds:
                item.update(status="pose_family_outlier")
            if seed in paired:
                reference = calibration["reference_by_seed"][str(seed)]
                item["reference"] = reference
                comparison = self.adapter.compare_results(item["candidate"], reference)
                comparison.update(seed=seed, candidate_original_rank=selected[seed]["rank"],
                                  reference_original_rank=reference["original_pose_rank"])
                item["comparison"] = comparison
                item["plip_comparison"] = compare_plip(item["candidate"].get("plip", {}), reference.get("plip", {}))
                comparisons.append(comparison)
        comparison = self.adapter.aggregate_comparisons(comparisons, sorted(paired))
        comparison.update(total_configured_seed_count=len(self.seeds), paired_seed_count=len(paired),
                          missing_or_outlier_seeds=sorted(set(self.seeds)-paired),
                          pose_selection_policy="same exported Evaluation Pose for score, PLIP and consensus")
        for metric in comparison.get("metrics", {}).values():
            metric["candidate_better_fraction_of_paired_seeds"] = metric["candidate_better_seed_fraction"]
            metric["candidate_better_seed_fraction"] = metric["candidate_better_seed_count"] / len(self.seeds)
        complete = len(paired) >= self.minimum and comparison.get("status") == "complete"
        evaluation_failed = any(item["status"] == "evaluation_failed" for item in per_seed)
        status = "complete" if complete else ("evaluation_failed" if evaluation_failed else "no_eligible_pose")
        result = {"stage": "docking", "status": status, "seed_count": len(self.seeds), "seeds": self.seeds,
                  "passing_seed_count": len(family_seeds), "paired_seed_count": len(paired),
                  "passing_seed_fraction": len(family_seeds)/len(self.seeds),
                  "paired_seed_fraction": len(paired)/len(self.seeds),
                  "geometry_passing_seeds": sorted(selected), "family_seeds": sorted(family_seeds),
                  "paired_seeds": sorted(paired), "outlier_or_failed_seeds": sorted(set(self.seeds)-paired),
                  "minimum_passing_seeds": self.minimum, "per_seed": per_seed, "comparison": comparison,
                  "pose_retention": {"status": "passed" if complete else "failed", "core_atom_indices": self.core,
                                     "frozen_geometry_reference": calibration["geometry_reference_path"],
                                     "rmsd_threshold": self.candidate_limit, "family_rmsd_threshold": self.family_limit},
                  "reference_baseline": calibration,
                  "pose_consensus": {"status": "complete", "stable": complete, "method": "fixed-core complete-link",
                                     "seed_count": len(self.seeds), "family_seed_count": len(family_seeds)}}
        result["pose_evidence"] = self._pose_evidence(
            per_seed, family_seeds, paired, self.minimum
        )
        result["candidate_per_seed"] = {
            str(item["seed"]): {"status": item["status"],
                "top_pose": (item.get("candidate", {}).get("poses") or [None])[0],
                "pose_path": item.get("candidate", {}).get("pose_path"),
                "pose_selection": {key: item.get("evaluation_pose", {}).get(key) for key in
                    ("rank", "reference_core_rmsd", "crystal_core_rmsd", "max_core_displacement", "anchor_check", "steric_check")},
                "family_member": item["family_member"],
                "audit_path": str(out / f"seed-{item['seed']:05d}" / "pose-gate" / "pose-evaluation.json")}
            for item in per_seed}
        result["plip_comparison"] = {str(item["seed"]): item.get("plip_comparison", {"status": "unavailable"})
                                     for item in per_seed}
        result["evaluation_pose_paths"] = {str(item["seed"]): item["candidate"]["pose_path"]
                                           for item in per_seed if item["seed"] in paired}
        if complete:
            # Deterministic representative output; all paired seed poses remain available.
            result["evaluation_pose_path"] = result["evaluation_pose_paths"][str(min(paired))]
        if not complete:
            result.update(failure_class="pose_retention" if not evaluation_failed else "pose_evaluation",
                          error="No majority of paired, core/anchor-gated Evaluation Poses in one family; no rank-1 fallback")
        save_json(out / "docking-result.json", result)
        return result
