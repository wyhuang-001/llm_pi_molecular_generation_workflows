#!/usr/bin/env python
"""Build a cut-point-free 4WKQ library that still covers the literature series.

The frozen C6 pool required every candidate to re-specify the whole ether-linked
side chain, so 78 fragments (all O-linked, 8-12 heavy atoms) covered 26 literature
compounds.  This builder instead:

1. enumerates **every** legal edit site of the ligand (all directed non-ring cuts
   and all hydrogen-bearing atom sites), not one frozen cut;
2. for each literature compound, finds every site that rebuilds it in one edit;
3. **verifies each candidate by exact graph reconstruction** against the parent,
   so a decomposition is only kept when the assembled product equals the target;
4. keeps the minimal-fragment representation per compound, and records the
   operation in the two-axis taxonomy plus the T1/T2 tier of the overall change;
5. fills the rest of the closed pool with a diversity-selected background from the
   general library, matched per size class and attachment element.

Outputs (public vs private is a hard boundary):

* ``4WKQ/design/fragments-multisite.json``          public catalog, no provenance
* ``4WKQ/design/pool-manifest-multisite.json``      public counts, rules and hashes
* ``4WKQ/benchmark-private/reachability-multisite.json`` evaluator-only site/fragment
  to compound mapping, tiers and literature membership

Nothing here is read by the public workflow: the designer sees the catalog and the
site table, never this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molecular_agent.closed_pool import canonical, sha256  # noqa: E402
from molecular_agent.editing import (  # noqa: E402
    _attach_fragment_graph,
    _retained_fragment,
)
from molecular_agent.fragment_library import FragmentLibrary  # noqa: E402
from molecular_agent.structure import ComplexContext  # noqa: E402
from molecular_agent.tiering import DRUG_LIKE_ELEMENTS, RARE_ELEMENTS, classify_change  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "4WKQ/task.benchmark.json"
FROZEN_REACHABILITY = "4WKQ/benchmark-private/reachability.json"
FROZEN_SITE_TABLE = "4WKQ/design/sites-v2.json"
GENERAL_LIBRARY = "molecular_agent/data/fragment_library_v2.json"

SCHEMA_VERSION = 1
BACKGROUND_SEED = 410627
BACKGROUND_PER_CLASS = 60
SIMILARITY_STOP = 0.90
#: Literature-justified aniline substituents, always added to the multisite catalog.
#: erlotinib uses 3-ethynyl, the discovery series uses m-methyl, and halogens are the
#: routine aniline scan; each is a single-port fragment usable for addition/replacement.
ANILINE_SUBSTITUENTS = [
    ("C#C[*:1]", "ethynyl"),
    ("C[*:1]", "methyl"),
    ("F[*:1]", "fluoro"),
    ("Cl[*:1]", "chloro"),
    ("Br[*:1]", "bromo"),
    ("I[*:1]", "iodo"),
]
REACTIVE = [
    Chem.MolFromSmarts(smarts)
    for smarts in ("[O]-[O]", "[N]-[N]", "[N]-[O]", "[C](=O)[F,Cl,Br,I]")
]
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# --------------------------------------------------------------------------- #
# site space
# --------------------------------------------------------------------------- #
def bond_sites(tools) -> list[dict]:
    return [dict(site) for site in tools.list_bond_sites(limit=200)["sites"]]


def atom_sites(tools) -> list[dict]:
    return [dict(site) for site in tools.get_edit_site_candidates()["atom_sites"]]


# --------------------------------------------------------------------------- #
# decomposition of one target relative to the parent
# --------------------------------------------------------------------------- #
def _fragment_at(target: Chem.Mol, anchor_image: int, keep: set[int]) -> str | None:
    """Attachment SMILES of ``target`` beyond ``anchor_image``."""
    rw = Chem.RWMol(Chem.Mol(target))
    drop = sorted(keep - {anchor_image}, reverse=True)
    for index in drop:
        rw.RemoveAtom(index)
    shift = sum(1 for index in drop if index < anchor_image)
    local = anchor_image - shift
    atom = rw.GetAtomWithIdx(local)
    atom.SetAtomicNum(0)
    atom.SetAtomMapNum(1)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.SetIsAromatic(False)
    try:
        molecule = rw.GetMol()
        Chem.SanitizeMol(molecule)
    except Exception:
        return None
    if len(Chem.GetMolFrags(molecule)) != 1 or Chem.GetFormalCharge(molecule) != 0:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def decompose_at_bond_site(
    parent: Chem.Mol, target: Chem.Mol, site: dict, target_smiles: str
) -> dict | None:
    """Return the decomposition of ``target`` at one directed cut, verified by assembly."""
    cut = (int(site["cut_bond"][0]), int(site["cut_bond"][1]))
    if parent.GetBondBetweenAtoms(*cut) is None:
        return None
    try:
        # ``_retained_fragment`` returns the retained scaffold with the dummy already
        # removed, which is exactly the parent graph ``_attach_fragment_graph`` expects.
        scaffold_mol, retained_anchor, old_to_new = _retained_fragment(parent, cut)
    except Exception:
        return None
    retained_parent = sorted(old_to_new)
    anchor_local = retained_parent.index(cut[0])

    core_query = Chem.RWMol(Chem.Mol(parent))
    for index in sorted(set(range(parent.GetNumAtoms())) - set(retained_parent), reverse=True):
        core_query.RemoveAtom(index)
    core_query = core_query.GetMol()
    try:
        Chem.SanitizeMol(core_query)
    except Exception:
        return None

    match = target.GetSubstructMatch(core_query)
    if not match or len(match) != core_query.GetNumAtoms():
        return None
    anchor_image = match[anchor_local]
    keep_images = set(match)

    if len(keep_images) == target.GetNumAtoms():
        # a bond:deletion leaves the retained core with a hydrogen at the anchor
        try:
            product = Chem.MolToSmiles(
                _attach_fragment_graph(scaffold_mol, retained_anchor, "[*:1][H]"),
                isomericSmiles=True,
            )
        except Exception:
            return None
        if product != target_smiles:
            return None
        return {
            "site_type": "bond",
            "change_type": "deletion",
            "fragment_smiles": None,
            "site_id": site["bond_site_id"],
            "anchor": cut[0],
            "cut_bond": list(cut),
        }

    fragment = _fragment_at(target, anchor_image, keep_images)
    if fragment is None:
        return None
    try:
        product = Chem.MolToSmiles(
            _attach_fragment_graph(scaffold_mol, retained_anchor, fragment),
            isomericSmiles=True,
        )
    except Exception:
        return None
    if product != target_smiles:
        return None
    return {
        "site_type": "bond",
        "change_type": "replacement",
        "fragment_smiles": fragment,
        "site_id": site["bond_site_id"],
        "anchor": cut[0],
        "cut_bond": list(cut),
    }


def decompose_at_atom_site(
    parent: Chem.Mol, target: Chem.Mol, site: dict, target_smiles: str
) -> dict | None:
    """Return an atom:addition / atom:ring replacement decomposition, verified by assembly."""
    anchor = site["target_id"]
    change_types = site.get("allowed_change_types") or []
    match = target.GetSubstructMatch(parent)
    if match and len(match) == parent.GetNumAtoms():
        anchor_image = match[anchor]
        keep_images = set(match)
        fragment = _fragment_at(target, anchor_image, keep_images)
        if fragment and "addition" in change_types:
            try:
                product = Chem.MolToSmiles(
                    _attach_fragment_graph(parent, anchor, fragment), isomericSmiles=True
                )
            except Exception:
                product = None
            if product == target_smiles:
                return {
                    "site_type": site.get("site_type", "atom"),
                    "change_type": "addition",
                    "fragment_smiles": fragment,
                    "site_id": site.get("site_id", f"atom-{anchor:03d}"),
                    "anchor": anchor,
                    "cut_bond": None,
                }
    # single element swap: everything else identical, one atom differs
    if parent.GetNumAtoms() == target.GetNumAtoms() and "replacement" in change_types:
        differing = [
            index for index in range(parent.GetNumAtoms())
            if parent.GetAtomWithIdx(index).GetAtomicNum()
            != target.GetAtomWithIdx(index).GetAtomicNum()
        ]
        if len(differing) == 1 and differing[0] == anchor:
            element = target.GetAtomWithIdx(anchor).GetSymbol()
            rw = Chem.RWMol(Chem.Mol(parent))
            rw.GetAtomWithIdx(anchor).SetAtomicNum(target.GetAtomWithIdx(anchor).GetAtomicNum())
            rw.GetAtomWithIdx(anchor).SetNoImplicit(False)
            rw.GetAtomWithIdx(anchor).SetNumExplicitHs(0)
            rw.GetAtomWithIdx(anchor).SetIsAromatic(False)
            try:
                product = Chem.MolToSmiles(rw.GetMol(), isomericSmiles=True)
            except Exception:
                product = None
            if product == target_smiles:
                return {
                    "site_type": site.get("site_type", "atom"),
                    "change_type": "replacement",
                    "fragment_smiles": None,
                    "element": element,
                    "site_id": site.get("site_id", f"atom-{anchor:03d}"),
                    "anchor": anchor,
                    "cut_bond": None,
                }
    return None


def decompositions(parent: Chem.Mol, target: Chem.Mol, cuts: list[dict], atoms: list[dict]):
    target_smiles = Chem.MolToSmiles(target, isomericSmiles=True)
    found = []
    for site in cuts:
        item = decompose_at_bond_site(parent, target, site, target_smiles)
        if item:
            found.append(item)
    for site in atoms:
        item = decompose_at_atom_site(parent, target, site, target_smiles)
        if item:
            found.append(item)
    return found


# --------------------------------------------------------------------------- #
# catalog records
# --------------------------------------------------------------------------- #
def attachable(smiles: str) -> bool:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return False
    if Chem.GetFormalCharge(molecule) != 0:
        return False
    if not 1 <= molecule.GetNumHeavyAtoms() <= 12:
        return False
    if any(
        atom.GetAtomicNum() not in DRUG_LIKE_ELEMENTS | RARE_ELEMENTS | {0} or atom.GetIsotope()
        for atom in molecule.GetAtoms()
    ):
        return False
    dummies = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 1 or dummies[0].GetDegree() != 1 or dummies[0].GetAtomMapNum() != 1:
        return False
    if any(atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 0):
        return False
    return not any(molecule.HasSubstructMatch(pattern) for pattern in REACTIVE)


def record_for(fragment_id: str, smiles: str, change_types: list[str], tier: str,
               sites: list[str]) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    dummy = next(atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0)
    anchor = dummy.GetNeighbors()[0]
    properties = FragmentLibrary._properties(smiles)
    return {
        "fragment_id": fragment_id,
        "smiles": smiles,
        "canonical_smiles": properties["canonical_smiles"],
        "heavy_atoms": properties["heavy_atoms"],
        "size_class": properties["size_class"],
        "attachment_atom_element": anchor.GetSymbol(),
        "attachment_atom_is_aromatic": bool(anchor.GetIsAromatic()),
        "chemical_tags": properties["chemical_tags"],
        "formal_charge": properties["formal_charge"],
        "properties": {
            key: properties[key]
            for key in (
                "molecular_weight", "logp", "hbd", "hba", "tpsa",
                "rotatable_bonds", "ring_count", "aromatic_ring_count",
            )
        },
        "allowed_change_types": sorted(change_types),
        # legacy spelling so the standard library/panel machinery keeps working
        "allowed_operations": sorted({
            "replace_hydrogen" if value == "addition" else "replace_fragment"
            for value in change_types
        }),
        "min_edit_tier": tier,
        "usable_at_site_types": sorted({site.split(":")[0] for site in sites}),
    }


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(task_path: Path, reachability_path: Path, general_path: Path,
          background_per_class: int, minimal_only: bool = True,
          maximum_unique_candidates: int = 30,
          maximum_design_requests: int = 150) -> tuple[dict, dict, dict]:
    context = ComplexContext(task_path)
    parent = Chem.RemoveHs(Chem.Mol(context.ligand))
    from molecular_agent.tools import ToolRegistry

    tools = ToolRegistry(context)
    cuts, atoms = bond_sites(tools), atom_sites(tools)

    records = json.loads(reachability_path.read_text(encoding="utf-8"))["records"]
    targets = [r for r in records if not r.get("is_reference")]

    fragments: dict[str, dict] = {}        # the catalog set (minimal by default)
    all_fragments: dict[str, dict] = {}    # every legal decomposition's fragment
    private_entries: dict[str, dict] = {}
    coverage = Counter()
    no_fragment_compounds = []

    for record in targets:
        compound = record["source_compound_id"]
        target = Chem.RemoveHs(Chem.MolFromSmiles(record["canonical_smiles"]))
        target_smiles = Chem.MolToSmiles(target, isomericSmiles=True)
        tier = classify_change(parent, target)["tier"]
        found = decompositions(parent, target, cuts, atoms)
        if not found:
            coverage["uncovered"] += 1
            private_entries[compound] = {
                "structure": record["canonical_smiles"],
                "tier": tier,
                "reachable": False,
                "decompositions": [],
            }
            continue
        coverage["covered"] += 1
        # canonical, minimal representation: smallest fragment, otherwise no-fragment edit
        with_fragment = [item for item in found if item["fragment_smiles"]]
        best = min(
            with_fragment,
            key=lambda item: Chem.MolFromSmiles(item["fragment_smiles"]).GetNumHeavyAtoms(),
            default=None,
        )
        if best is None:
            no_fragment_compounds.append(compound)
        for item in found:
            if not item["fragment_smiles"]:
                continue
            smiles = item["fragment_smiles"]
            for target_map in (all_fragments,):
                entry = target_map.setdefault(smiles, {
                    "change_types": set(), "tiers": set(), "sites": set(), "compounds": set(),
                })
                entry["change_types"].add(item["change_type"])
                entry["tiers"].add(tier)
                entry["sites"].add(f"{item['site_type']}:{item['site_id']}")
                entry["compounds"].add(compound)
        if best is not None:
            smiles = best["fragment_smiles"]
            entry = fragments.setdefault(smiles, {
                "change_types": set(), "tiers": set(), "sites": set(), "compounds": set(),
            })
            entry["change_types"].add(best["change_type"])
            entry["tiers"].add(tier)
            entry["sites"].add(f"{best['site_type']}:{best['site_id']}")
            entry["compounds"].add(compound)
        private_entries[compound] = {
            "structure": record["canonical_smiles"],
            "tier": tier,
            "reachable": True,
            "minimal_representation": best,
            "n_decompositions": len(found),
            "decompositions": found,
        }

    catalog_source = all_fragments if not minimal_only else fragments

    # background from the general library, matched per size class
    general = json.loads(general_path.read_text(encoding="utf-8"))["fragments"]
    taken = {canonical(smiles) for smiles in catalog_source}
    by_class: dict[str, list[dict]] = defaultdict(list)
    for record in general:
        smiles = record["smiles"]
        if record["canonical_smiles"] in taken or not attachable(smiles):
            continue
        by_class[record["size_class"]].append(record)

    rng = random.Random(BACKGROUND_SEED)
    background: list[tuple[str, str]] = []
    for size_class in ("minimal", "small", "medium", "large"):
        candidates = sorted(by_class.get(size_class, []), key=lambda item: item["canonical_smiles"])
        if not candidates:
            continue
        fingerprints = [FP.GetFingerprint(Chem.MolFromSmiles(item["smiles"])) for item in candidates]
        seeds = [
            FP.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in sorted(catalog_source)
        ]
        worst = [0.0] * len(candidates)
        for seed in seeds:
            similarities = DataStructs.BulkTanimotoSimilarity(seed, fingerprints)
            worst = [max(current, value) for current, value in zip(worst, similarities)]
        chosen: set[int] = set()
        while len(chosen) < background_per_class and len(chosen) < len(candidates):
            position = min(
                (index for index in range(len(candidates)) if index not in chosen),
                key=lambda index: (worst[index], index),
            )
            if worst[position] >= SIMILARITY_STOP and chosen:
                break
            chosen.add(position)
            background.append((candidates[position]["smiles"], size_class))
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[position], fingerprints)
            worst = [max(current, value) for current, value in zip(worst, similarities)]

    catalog_records = []
    for number, smiles in enumerate(sorted(catalog_source), start=1):
        entry = catalog_source[smiles]
        catalog_records.append(record_for(
            f"MS-{number:03d}", smiles, sorted(entry["change_types"]),
            min(entry["tiers"]), sorted(entry["sites"]),
        ))
    for number, (smiles, size_class) in enumerate(background, start=1):
        catalog_records.append(record_for(
            f"BG-{size_class}-{number:03d}", smiles, ["addition", "replacement"], "T1", [],
        ))
    # Literature-justified aniline substituents (gefitinib -> erlotinib ethynyl, m-methyl,
    # halogen scans). Always included so the aniline region is explorable, even though the
    # diversity-first background would not guarantee them.
    seen_smiles = {canonical(record["smiles"]) for record in catalog_records}
    for number, (smiles, name) in enumerate(ANILINE_SUBSTITUENTS, start=1):
        if canonical(smiles) in seen_smiles:
            continue
        catalog_records.append(record_for(
            f"AS-{name}", smiles, ["addition", "replacement"], "T1", [],
        ))

    size_counts = Counter(record["size_class"] for record in catalog_records)
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Cut-point-free attachable-fragment catalog. Records carry chemistry and the "
            "change types they may be used for, but no site assignment, no cut bond and no "
            "reference-ligand position. The designer matches a record to a host-listed site "
            "and the host validates and compiles the edit."
        ),
        "attachment_convention": (
            "[*:1] marks the atom the host bonds to the selected site atom. The attachment "
            "atom element is reported per record and is not fixed."
        ),
        "size_classes": {
            "minimal": {"min_heavy_atoms": 1, "max_heavy_atoms": 1},
            "small": {"min_heavy_atoms": 2, "max_heavy_atoms": 4},
            "medium": {"min_heavy_atoms": 5, "max_heavy_atoms": 8},
            "large": {"min_heavy_atoms": 9, "max_heavy_atoms": 12},
        },
        "counts": {
            "total": len(catalog_records),
            "by_size_class": dict(sorted(size_counts.items())),
            "literature_derived": len(catalog_source),
            "background": len(background),
        },
        "fragments": catalog_records,
    }
    private = {
        "role": "evaluator_only_do_not_send_to_designer",
        "native_smiles": Chem.MolToSmiles(parent, isomericSmiles=True),
        "literature_membership": "see reachability.json (source of these compounds)",
        "coverage": dict(coverage),
        "no_fragment_compounds": no_fragment_compounds,
        # the evaluator keys hits on the product molecule, never on a fragment id
        "product_index": {
            canonical(record["structure"]): compound
            for compound, record in private_entries.items() if record["reachable"]
        },
        "catalog_derivation": "minimal representation per compound" if minimal_only
                              else "every legal decomposition",
        "fragment_to_compounds": {
            smiles: sorted(entry["compounds"]) for smiles, entry in catalog_source.items()
        },
        "all_decomposition_fragments": {
            smiles: sorted(entry["compounds"]) for smiles, entry in all_fragments.items()
        },
        "compound_detail": private_entries,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": "multisite_reachability",
        "task": str(task_path.relative_to(ROOT)),
        "inputs": {
            "task_sha256": sha256(task_path),
            "reachability_sha256": sha256(reachability_path),
            "general_library_sha256": sha256(general_path),
        },
        "rules": {
            "cut_space": "every directed non-ring single bond that keeps the protected core",
            "site_space": "the task edit-site table (atom and cut sites)",
            "verification": "exact graph reconstruction against the parent; a decomposition is "
                            "kept only when the assembled product equals the target SMILES",
            "minimal_representation": "smallest attachment fragment per compound",
            "background": {
                "source": str(general_path.relative_to(ROOT)),
                "per_size_class_cap": background_per_class,
                "seed": BACKGROUND_SEED,
                "similarity_stop": SIMILARITY_STOP,
                "activity_used": False,
            },
        },
        "candidate_count": len(catalog_records),
        "maximum_unique_candidates": maximum_unique_candidates,
        "maximum_design_requests": maximum_design_requests,
        "counts": {
            "catalog_total": len(catalog_records),
            "literature_compounds": len(targets),
            "literature_covered": coverage.get("covered", 0),
            "literature_uncovered": coverage.get("uncovered", 0),
            "literature_derived_fragments": len(catalog_source),
            "all_decomposition_fragments": len(all_fragments),
            "compounds_needing_no_fragment": len(no_fragment_compounds),
            "bond_sites": len(cuts),
            "atom_sites": len(atoms),
        },
    }
    return catalog, private, manifest


def _halogen_cut_sites(task_path: Path) -> list[dict]:
    """Add a directed cut site for every terminal halogen on an aromatic carbon.

    The frozen site table protects the whole aniline including Cl/F, so the
    aniline-halogen bonds are absent from ``cut_sites``. These extra sites make the
    Cl/F substituents replaceable (single -> multi) via the ordinary bond machinery
    while keeping the aniline ring itself protected.
    """
    context = ComplexContext(task_path)
    molecule = Chem.RemoveHs(Chem.Mol(context.ligand))
    total = molecule.GetNumHeavyAtoms()
    ring_atoms = {index for ring in molecule.GetRingInfo().AtomRings() for index in ring}
    sites: list[dict] = []
    halogen_numbers = {9: "F", 17: "Cl", 35: "Br", 53: "I"}
    for atom in molecule.GetAtoms():
        number = atom.GetAtomicNum()
        if number not in halogen_numbers or atom.IsInRing() or atom.GetDegree() != 1:
            continue
        neighbor = next(iter(atom.GetNeighbors()))
        if neighbor.GetAtomicNum() != 6 or not neighbor.GetIsAromatic():
            continue
        retained_index, removed_index = neighbor.GetIdx(), atom.GetIdx()
        retained = [index for index in range(total) if index != removed_index]
        removed = [removed_index]
        pos = molecule.GetConformer().GetAtomPosition
        retained_point, removed_point = pos(retained_index), pos(removed_index)
        sites.append({
            "site_id": f"cut-{halogen_numbers[number]}",
            "target_type": "replacement_site",
            "region": "aniline-halogen",
            "label": f"remove the {halogen_numbers[number].lower()} substituent",
            "cut_bond": [retained_index, removed_index],
            "retained_atom_index": retained_index,
            "removed_side_atom_index": removed_index,
            "retained_atom_indices": retained,
            "removed_atom_indices": removed,
            "retained_heavy_atoms": total - 1,
            "removed_heavy_atoms": 1,
            "removed_fraction": round(1 / total, 3),
            "retained_scaffold_smiles": Chem.MolFragmentToSmiles(
                molecule, atomsToUse=retained, isomericSmiles=True
            ),
            "removed_fragment_smiles": Chem.MolFragmentToSmiles(
                molecule, atomsToUse=removed, isomericSmiles=True
            ),
            "attachment_vector": [
                round(removed_point.x - retained_point.x, 3),
                round(removed_point.y - retained_point.y, 3),
                round(removed_point.z - retained_point.z, 3),
            ],
            "allowed_operations": ["deletion", "replacement"],
            "allowed_change_types": ["deletion", "replacement"],
            "protection": "open",
        })
    return sites


def derive_site_table(task_path: Path, source: Path) -> dict:
    """Copy the frozen site table and widen the change types it permits.

    The frozen table predates ``bond:deletion`` and lists only replacement-style
    operations, so it is copied rather than edited: the original file and its hash
    stay valid for the frozen benchmark.
    """
    table = json.loads(source.read_text(encoding="utf-8"))
    legacy_to_change = {"element_swap": "replacement"}
    for record in table.get("atom_sites", []):
        change_types = {"addition"}
        for operation in record.get("allowed_operations") or []:
            if operation in legacy_to_change:
                change_types.add(legacy_to_change[operation])
        record["allowed_change_types"] = sorted(change_types)
    for record in table.get("cut_sites", []):
        # a directed non-ring cut can always be executed as a deletion or a replacement
        record["allowed_change_types"] = ["deletion", "replacement"]
    table["cut_sites"] = list(table.get("cut_sites", [])) + _halogen_cut_sites(task_path)
    table["schema_version"] = int(table.get("schema_version", 1))
    table["derived_from"] = str(source.relative_to(ROOT))
    table["derived_note"] = (
        "Multisite variant of the frozen site table. Sites, protection and geometry are "
        "unchanged; cut sites additionally advertise deletion, and terminal aniline "
        "halogens (Cl/F) gain directed cut sites so they can be replaced with fragments."
    )
    return table


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--reachability", default=FROZEN_REACHABILITY)
    parser.add_argument("--general-library", default=GENERAL_LIBRARY)
    parser.add_argument("--background-per-class", type=int, default=BACKGROUND_PER_CLASS)
    parser.add_argument("--maximum-unique-candidates", type=int, default=30)
    parser.add_argument("--maximum-design-requests", type=int, default=150)
    parser.add_argument("--include-all-decompositions", action="store_true",
                        help="catalog every legal decomposition instead of the minimal one per compound")
    parser.add_argument("--catalog", default="4WKQ/design/fragments-multisite.json")
    parser.add_argument("--site-table", default="4WKQ/design/sites-multisite.json")
    parser.add_argument("--manifest", default="4WKQ/design/pool-manifest-multisite.json")
    parser.add_argument("--private", default="4WKQ/benchmark-private/reachability-multisite.json")
    args = parser.parse_args()

    catalog, private, manifest = build(
        (ROOT / args.task).resolve(),
        (ROOT / args.reachability).resolve(),
        (ROOT / args.general_library).resolve(),
        args.background_per_class,
        minimal_only=not args.include_all_decompositions,
        maximum_unique_candidates=args.maximum_unique_candidates,
        maximum_design_requests=args.maximum_design_requests,
    )
    write_json(ROOT / args.catalog, catalog)
    write_json(ROOT / args.private, private)

    task = json.loads((ROOT / args.task).read_text(encoding="utf-8"))
    source_table = task.get("edit_site_table_path")
    if source_table:
        # always derive from the frozen original, even when the task now points at
        # the derived table, so re-running never changes the provenance
        frozen_source = ROOT / FROZEN_SITE_TABLE
        origin = frozen_source if frozen_source.is_file() else (ROOT / args.task).parent / source_table
        site_table = derive_site_table((ROOT / args.task).resolve(), origin)
        write_json(ROOT / args.site_table, site_table)

    manifest["public_files"] = {
        "catalog": args.catalog,
        "catalog_sha256": sha256(ROOT / args.catalog),
    }
    if source_table:
        manifest["public_files"]["site_table"] = args.site_table
        manifest["public_files"]["site_table_sha256"] = sha256(ROOT / args.site_table)
        manifest["public_files"]["site_table_derived_from"] = (
            FROZEN_SITE_TABLE if (ROOT / FROZEN_SITE_TABLE).is_file() else source_table
        )
    manifest["private_files"] = {
        "reachability": args.private,
        "reachability_sha256": sha256(ROOT / args.private),
    }
    write_json(ROOT / args.manifest, manifest)

    counts = manifest["counts"]
    print(json.dumps(counts, ensure_ascii=False, indent=1))
    print(f"\ncatalog  : {args.catalog}")
    print(f"manifest : {args.manifest}")
    print(f"private  : {args.private}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
