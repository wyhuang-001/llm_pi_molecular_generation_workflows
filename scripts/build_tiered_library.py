"""Build the two-layer (T1/T2) fragment libraries.

Outputs
-------
* ``molecular_agent/data/edit_rules_v2.json``      T1 minimal-edit rule table
* ``molecular_agent/data/fragment_library_v2.json`` general library with family
  tagging, plus the base-representative T2 catalog view
* ``4WKQ/design/fragments_v2.json``                rebuilt 4WKQ catalog (public)
* ``4WKQ/design/sites-v2.json``                    edit-site table (public)
* ``4WKQ/design/pool-manifest-v2.json``            counts, hashes, frozen rules
* ``4WKQ/benchmark-private/tier-map-v2.json``      tierung of the private pool

Existing v1 files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molecular_agent.tiering import (  # noqa: E402
    DRUG_LIKE_ELEMENTS,
    RARE_ELEMENTS,
    T1,
    T2,
    classify_change,
    classify_fragment_smiles,
    ring_sizes,
)

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SIDE_CHAIN = "OCCCN1CCOCC1"
REFERENCE_LIGAND = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
TIER_THRESHOLD_HEAVY_ATOMS = 3
MAX_ADDED_T2_FRAGMENTS = 60
BACKGROUND_PER_SIZE_CLASS = 90

T1_GROUPS_ONE_ATOM = [
    ("methyl", "[*:1]C"),
    ("hydroxyl", "[*:1]O"),
    ("amino", "[*:1]N"),
    ("thiol", "[*:1]S"),
    ("fluoro", "[*:1]F"),
    ("chloro", "[*:1]Cl"),
    ("bromo", "[*:1]Br"),
    ("iodo", "[*:1]I"),
    ("methylene", "[*:1]=C"),
    ("boryl", "[*:1]B"),
    ("silyl", "[*:1][Si]"),
    ("phosphanyl", "[*:1]P"),
]

T1_GROUPS_TWO_ATOMS = [
    ("methoxy", "[*:1]OC"),
    ("methylamino", "[*:1]NC"),
    ("cyano", "[*:1]C#N"),
    ("ethynyl", "[*:1]C#C"),
    ("vinyl", "[*:1]C=C"),
    ("formyl", "[*:1]C=O"),
    ("hydroxymethyl", "[*:1]CO"),
    ("aminomethyl", "[*:1]CN"),
    ("chloromethyl", "[*:1]CCl"),
    ("methylthio", "[*:1]SC"),
    ("nitroso", "[*:1]N=O"),
]

T1_ELEMENT_SWAPS = [
    ("carbon_to_nitrogen", 6, 7),
    ("nitrogen_to_carbon", 7, 6),
    ("carbon_to_oxygen", 6, 8),
    ("oxygen_to_carbon", 8, 6),
    ("nitrogen_to_oxygen", 7, 8),
    ("oxygen_to_nitrogen", 8, 7),
    ("carbon_to_sulfur", 6, 16),
    ("sulfur_to_carbon", 16, 6),
    ("oxygen_to_sulfur", 8, 16),
    ("sulfur_to_oxygen", 16, 8),
]

REGION_DEFINITIONS = [
    {
        "region": "C6-linker",
        "atoms": [6, 7, 8],
        "protection": "open",
        "atom_operations": ["replace_hydrogen"],
        "cut_operations": ["replace_fragment", "terminal_substituent_swap", "ring_atom_swap", "ring_size_edit"],
        "overlap_group": "C6-side-chain",
    },
    {
        "region": "C6-morpholine-substituent",
        "atoms": [0, 1, 3, 4],
        "protection": "open",
        "atom_operations": ["replace_hydrogen"],
        "cut_operations": [],
        "overlap_group": "C6-side-chain",
    },
    {
        "region": "C6-morpholine-skeleton",
        "atoms": [2, 5],
        "protection": "open",
        "atom_operations": [],
        "atom_element_swap_atoms": [2],
        "cut_operations": ["replace_fragment", "ring_atom_swap", "ring_size_edit"],
        "overlap_group": "C6-side-chain",
    },
    {
        "region": "C6-aryl-ether",
        "atoms": [9],
        "protection": "open",
        "atom_operations": [],
        "cut_operations": ["replace_fragment", "terminal_substituent_swap"],
        "overlap_group": "C6-side-chain",
    },
    {
        "region": "C7-methoxy",
        "atoms": [13, 14],
        "protection": "open",
        "atom_operations": ["replace_hydrogen"],
        "cut_operations": ["replace_fragment", "terminal_substituent_swap"],
        "overlap_group": "C7-methoxy",
    },
    {
        "region": "aniline-halogen",
        "atoms": [26, 28],
        "protection": "open",
        "atom_operations": [],
        "atom_element_swap_atoms": [26, 28],
        "cut_operations": ["replace_fragment", "terminal_substituent_swap"],
        "overlap_group": "aniline-ring",
    },
    {
        "region": "aniline-ring",
        "atoms": [23, 24, 25, 27, 29, 30],
        "protection": "open",
        "atom_operations": ["replace_hydrogen"],
        "cut_operations": [],
        "overlap_group": "aniline-ring",
    },
    {
        "region": "aniline-amine",
        "atoms": [22],
        "protection": "caution",
        "protection_reason": "aniline NH is a hinge-adjacent hydrogen-bond donor",
        "atom_operations": ["replace_hydrogen", "terminal_substituent_swap"],
        "cut_operations": ["replace_fragment"],
        "overlap_group": "aniline-ring",
    },
    {
        "region": "quinazoline-core",
        "atoms": [10, 11, 12, 15, 16, 17, 18, 19, 20, 21],
        "protection": "protected",
        "protection_reason": "hinge-binding core including the verified N3 anchor (atom 18)",
        "atom_operations": [],
        "cut_operations": [],
        "overlap_group": "quinazoline-core",
    },
]

# Directed non-ring single-bond cuts.  ``retained`` must keep the protected
# quinazoline core, so every cut removes a substituent rather than the scaffold.
CUT_DEFINITIONS = [
    {"cut_bond": [6, 5], "region": "C6-morpholine-skeleton", "label": "remove the morpholine ring"},
    {"cut_bond": [7, 6], "region": "C6-linker", "label": "remove the morpholine and one linker carbon"},
    {"cut_bond": [8, 7], "region": "C6-linker", "label": "remove the morpholine and two linker carbons"},
    {"cut_bond": [9, 8], "region": "C6-linker", "label": "remove the whole C6 side chain beyond the ether oxygen"},
    {"cut_bond": [10, 9], "region": "C6-aryl-ether", "label": "remove the whole C6 side chain including the ether oxygen"},
    {"cut_bond": [11, 13], "region": "C7-methoxy", "label": "remove the C7 methoxy group"},
    {"cut_bond": [13, 14], "region": "C7-methoxy", "label": "remove the C7 methyl only"},
    {"cut_bond": [21, 22], "region": "aniline-amine", "label": "remove the whole aniline arm"},
    {"cut_bond": [22, 23], "region": "aniline-amine", "label": "remove the aniline ring, keeping the NH"},
    {"cut_bond": [25, 26], "region": "aniline-halogen", "label": "remove the chlorine"},
    {"cut_bond": [27, 28], "region": "aniline-halogen", "label": "remove the fluorine"},
]


def region_of_atom(atom_index: int) -> str:
    for definition in REGION_DEFINITIONS:
        if atom_index in definition["atoms"]:
            return definition["region"]
    return "unassigned"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule)


def properties(smiles: str) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    return {
        "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "molecular_weight": round(Descriptors.MolWt(molecule), 2),
        "logp": round(CrippenLogP(molecule), 2),
        "hbd": Lipinski.NumHDonors(molecule),
        "hba": Lipinski.NumHAcceptors(molecule),
        "tpsa": round(rdMolDescriptors.CalcTPSA(molecule), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
        "formal_charge": Chem.GetFormalCharge(molecule),
    }


def CrippenLogP(molecule: Chem.Mol) -> float:
    from rdkit.Chem import Crippen

    return Crippen.MolLogP(molecule)


# --------------------------------------------------------------------------
# T1 rule table
# --------------------------------------------------------------------------


def build_edit_rules() -> dict:
    def group(record: tuple[str, str], family: str) -> dict:
        name, smiles = record
        molecule = Chem.MolFromSmiles(smiles)
        elements = sorted({atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 0})
        return {
            "name": name,
            "smiles": smiles,
            "attachment_count": 1,
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
            "elements": elements,
            "rare_elements": sorted({element for element in elements if _atomic_number(element) in RARE_ELEMENTS}),
            "allowed_operations": ["replace_hydrogen"],
            "rule_family": family,
        }

    return {
        "schema_version": 2,
        "layer": "T1_small_edits",
        "description": (
            "Small edits are compiled as a minimal edit on the parent. They must not be "
            "executed as a whole-fragment replacement even when the catalog also lists an "
            "equivalent assembled structure."
        ),
        "tier_rule": {
            "t1": "one or two heavy-atom changes, no ring-size change and no ring-skeleton substitution",
            "t2": "three or more heavy-atom changes, or any ring framework change",
            "ring_rule_precedence": (
                "A ring-size change or a ring-skeleton atom substitution is T2 even when only one "
                "heavy atom differs."
            ),
        },
        "element_whitelist": sorted(_symbol(number) for number in DRUG_LIKE_ELEMENTS),
        "rare_elements": sorted(_symbol(number) for number in RARE_ELEMENTS),
        "substituent_attach": {
            "one_atom_groups": [group(item, "substituent_attach_1") for item in T1_GROUPS_ONE_ATOM],
            "two_atom_groups": [group(item, "substituent_attach_2") for item in T1_GROUPS_TWO_ATOMS],
        },
        "element_swap": {
            "scope": ["chain_atom", "ring_substituent_atom"],
            "forbidden_scope": ["ring_skeleton_atom"],
            "swaps": [
                {
                    "name": name,
                    "from": _symbol(source),
                    "to": _symbol(target),
                    "tier": T1,
                    "note": "a ring-skeleton swap is T2 and must be routed to whole-ring replacement",
                }
                for name, source, target in T1_ELEMENT_SWAPS
            ],
        },
        "chain_length_edit": {
            "deltas": [-2, -1, 1, 2],
            "tier": T1,
            "note": "applies to a linker between two retained attachment points",
        },
        "substituent_delete": {
            "heavy_atoms_max": 2,
            "tier": T1,
            "allowed_operations": ["delete_substituent"],
        },
        "terminal_substituent_swap": {
            "heavy_atoms_max": 2,
            "tier": T1,
            "allowed_operations": ["terminal_substituent_swap"],
            "note": (
                "Exchanges a terminal substituent for another small group, for example methyl for "
                "ethyl, or chlorine for bromine. It never touches ring-skeleton atoms."
            ),
        },
    }


def _symbol(atomic_number: int) -> str:
    return Chem.GetPeriodicTable().GetElementSymbol(atomic_number)


def _atomic_number(symbol: str) -> int:
    return Chem.GetPeriodicTable().GetAtomicNumber(symbol)


# --------------------------------------------------------------------------
# general library
# --------------------------------------------------------------------------


def one_atom_edit_neighbours(smiles: str, library: set[str]) -> set[str]:
    """Library members reachable from ``smiles`` by deleting or swapping one atom."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return set()
    found: set[str] = set()

    def normalise(candidate: Chem.Mol) -> str | None:
        try:
            candidate = Chem.Mol(candidate)
            Chem.SanitizeMol(candidate)
            if Chem.GetFormalCharge(candidate) != 0:
                return None
            return Chem.MolToSmiles(candidate)
        except Exception:
            return None

    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0 or atom.GetDegree() == 0:
            continue
        if atom.GetNeighbors()[0].GetAtomicNum() == 0:
            continue
        if atom.GetDegree() == 1 or atom.GetIsAromatic():
            edited = Chem.RWMol(molecule)
            edited.RemoveAtom(atom.GetIdx())
            value = normalise(edited.GetMol())
            if value and value in library and value != smiles:
                found.add(value)
        swaps = {6: [7, 8], 7: [6, 8], 8: [6, 7], 9: [6, 7, 8], 16: [6, 7, 8], 17: [6, 7, 8]}
        for target in swaps.get(atom.GetAtomicNum(), []):
            edited = Chem.RWMol(molecule)
            replacement = edited.GetAtomWithIdx(atom.GetIdx())
            replacement.SetAtomicNum(target)
            replacement.SetNoImplicit(False)
            replacement.SetNumExplicitHs(0)
            value = normalise(edited.GetMol())
            if value and value in library and value != smiles:
                found.add(value)
    return found


def build_general_library(source_path: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    fragments = source["fragments"]
    smiles_set = {record["canonical_smiles"] for record in fragments}
    by_smiles = {record["canonical_smiles"]: record for record in fragments}

    parents: dict[str, str] = {}

    def find(item: str) -> str:
        parents.setdefault(item, item)
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_left] = root_right

    edges = 0
    neighbours: dict[str, set[str]] = {}
    for smiles in smiles_set:
        found = one_atom_edit_neighbours(smiles, smiles_set)
        neighbours[smiles] = found
        for neighbour in found:
            union(smiles, neighbour)
            edges += 1

    families: dict[str, list[str]] = {}
    for smiles in smiles_set:
        families.setdefault(find(smiles), []).append(smiles)

    def role_of(smiles: str) -> str:
        """A fragment that is one atom larger than a direct neighbour is an extension.

        Reachability is judged on direct one-atom edits, not on a whole connected
        component, so a chain of one-atom steps does not label distinct chemistry
        as a mere variant.
        """
        heavy = by_smiles[smiles]["heavy_atoms"]
        return "extension" if any(
            by_smiles[neighbour]["heavy_atoms"] < heavy for neighbour in neighbours[smiles]
        ) else "base"

    records = []
    for root, members in families.items():
        for value in sorted(members):
            record = by_smiles[value]
            role = role_of(value)
            records.append({
                "fragment_id": record["fragment_id"],
                "smiles": record["smiles"],
                "canonical_smiles": record["canonical_smiles"],
                "operation": record.get("operation", "replace_fragment"),
                "heavy_atoms": record["heavy_atoms"],
                "size_class": record["size_class"],
                "chemical_tags": record.get("chemical_tags", []),
                "formal_charge": record["formal_charge"],
                "properties": {
                    key: record[key]
                    for key in ("molecular_weight", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "ring_count")
                    if key in record
                },
                "family_id": "fam-" + hashlib.sha1(root.encode()).hexdigest()[:12],
                "family_role": role,
                "family_size": len(members),
                "family_base_smiles": None if role == "base" else min(
                    (neighbour for neighbour in neighbours[value]
                     if by_smiles[neighbour]["heavy_atoms"] < record["heavy_atoms"]),
                    key=lambda item: (by_smiles[item]["heavy_atoms"], item),
                ),
                "heavy_atom_delta_from_base": 2 if role == "extension" else 0,
                "compile_as": "whole_fragment" if role == "base" else "minimal_edit_preferred",
            })

    added_minimal = add_missing_single_atom_entries(records, by_smiles)

    return {
        "schema_version": 2,
        "description": (
            "General fragment library with one-atom-edit family tagging. Base entries are real "
            "fragments; one-atom variants are reachable as T1 minimal edits and should not be "
            "executed as whole-fragment replacements."
        ),
        "source_library": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256(source_path),
        "tier_threshold_heavy_atoms": TIER_THRESHOLD_HEAVY_ATOMS,
        "family_graph": {
            "edge_definition": "single heavy-atom deletion or single element swap between library members",
            "edges": edges,
            "families": len(families),
            "family_size_distribution": dict(
                sorted(Counter(len(members) for members in families.values()).items(), reverse=True)
            ),
        },
        "counts": {
            "entries": len(records),
            "base_entries": sum(1 for record in records if record["family_role"] == "base"),
            "extensions": sum(1 for record in records if record["family_role"] == "extension"),
            "newly_added_single_atom_entries": added_minimal,
        },
        "fragments": records,
        "base_catalog": [record["canonical_smiles"] for record in records if record["family_role"] == "base"],
    }


def add_missing_single_atom_entries(records: list[dict], by_smiles: dict) -> int:
    """Append curated one-heavy-atom entries that the source library lacks."""
    present = {record["canonical_smiles"] for record in records}
    added = 0
    for name, smiles in T1_GROUPS_ONE_ATOM:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumHeavyAtoms() != 1:
            continue
        value = Chem.MolToSmiles(molecule)
        if value in present:
            continue
        present.add(value)
        elements = sorted({atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 0})
        records.append({
            "fragment_id": f"t1-single-{name}",
            "smiles": smiles,
            "canonical_smiles": value,
            "operation": "replace_hydrogen",
            "heavy_atoms": 1,
            "size_class": "minimal",
            "chemical_tags": ["single_atom_group"],
            "formal_charge": Chem.GetFormalCharge(molecule),
            "properties": properties(smiles),
            "family_id": "t1-single-atom",
            "family_role": "base",
            "family_size": 1,
            "family_base_smiles": None,
            "heavy_atom_delta_from_base": 0,
            "compile_as": "minimal_edit",
            "rare_elements": sorted(element for element in elements if _atomic_number(element) in RARE_ELEMENTS),
        })
        added += 1
    return added


# --------------------------------------------------------------------------
# 4WKQ fragment catalog
#
# The catalog lists *fragments only*: each record is one neutral molecule with a
# single mapped attachment atom (``[*:1]``).  It carries no site id, no cut
# bond, no retained scaffold and no "which position of the reference ligand"
# information.  Choosing the site and matching a fragment to it is the
# designer's job; the host validates and compiles the edit.
# --------------------------------------------------------------------------

REACTIVE_PATTERNS = [
    Chem.MolFromSmarts(pattern)
    for pattern in ("[O]-[O]", "[N]-[N]", "[N]-[O]", "[C](=O)[F,Cl,Br,I]")
]


def generic_attachable(smiles: str) -> bool:
    """True when the fragment is a neutral single piece with one attachment atom.

    The attachment atom may be any drug-like element: removing the dummy leaves a free
    valence, so ``[*:1]F`` means "bond fluorine to the site" and is a valid record.
    """
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
    return not any(molecule.HasSubstructMatch(pattern) for pattern in REACTIVE_PATTERNS)


def fragment_record(
    fragment_id: str,
    smiles: str,
    family_by_canonical: dict[str, dict],
) -> dict:
    """Build one designer-facing fragment record in the general-library format."""
    from molecular_agent.fragment_library import FragmentLibrary

    molecule = Chem.MolFromSmiles(smiles)
    dummy = next(atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0)
    anchor = dummy.GetNeighbors()[0]
    properties = FragmentLibrary._properties(smiles)
    family = family_by_canonical.get(properties["canonical_smiles"], {})
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
                "molecular_weight",
                "logp",
                "hbd",
                "hba",
                "tpsa",
                "rotatable_bonds",
                "ring_count",
                "aromatic_ring_count",
            )
        },
        "allowed_operations": ["replace_hydrogen", "replace_fragment"],
        "family_id": family.get("family_id"),
        "family_role": family.get("family_role"),
        "family_base_smiles": family.get("family_base_smiles"),
    }


def build_4wkq_fragment_catalog(
    pool_path: Path,
    general_library: dict,
    per_class_background_cap: int = 90,
) -> tuple[dict, dict]:
    """Assemble the task fragment catalog in the general-library format.

    Returns the public catalog and a private build report that records
    provenance counts and the reference-relative tier of the v1 pool.  The
    public catalog never states which fragment came from which source.
    """
    from rdkit.Chem import DataStructs, rdFingerprintGenerator

    pool = json.loads(pool_path.read_text(encoding="utf-8"))["fragments"]
    family_by_canonical = {
        record["canonical_smiles"]: record for record in general_library["fragments"]
    }

    entries: list[dict] = []
    seen: set[str] = set()
    provenance = {"one_atom_groups": 0, "two_atom_groups": 0, "benchmark_pool": 0, "background": 0}

    def add(smiles: str, fragment_id: str, bucket: str) -> bool:
        if not generic_attachable(smiles):
            return False
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        if canonical in seen:
            return False
        seen.add(canonical)
        entries.append(fragment_record(fragment_id, smiles, family_by_canonical))
        provenance[bucket] += 1
        return True

    for name, smiles in T1_GROUPS_ONE_ATOM:
        add(smiles, f"G-1-{name}", "one_atom_groups")
    for name, smiles in T1_GROUPS_TWO_ATOMS:
        add(smiles, f"G-2-{name}", "two_atom_groups")
    for record in pool:
        add(record["smiles"], record["fragment_id"], "benchmark_pool")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    by_class: dict[str, list[dict]] = {}
    for record in general_library["fragments"]:
        if record["canonical_smiles"] in {
            Chem.MolToSmiles(Chem.MolFromSmiles(entry["smiles"]), isomericSmiles=True) for entry in entries
        }:
            continue
        if not generic_attachable(record["smiles"]):
            continue
        by_class.setdefault(record["size_class"], []).append(record)

    for size_class in ("minimal", "small", "medium", "large"):
        candidates = sorted(by_class.get(size_class, []), key=lambda item: item["canonical_smiles"])
        if not candidates:
            continue
        fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(record["smiles"])) for record in candidates]
        seed_fingerprints = [
            generator.GetFingerprint(Chem.MolFromSmiles(entry["smiles"])) for entry in entries
        ]
        # Greedy max-min diversity.  The first pick already avoids whatever the
        # mandatory catalog covers, and each later pick avoids everything chosen
        # so far.  Only near-identity stops the loop.
        worst_case = [0.0] * len(candidates)
        for seed in seed_fingerprints:
            similarities = DataStructs.BulkTanimotoSimilarity(seed, fingerprints)
            worst_case = [max(current, value) for current, value in zip(worst_case, similarities)]
        chosen: set[int] = set()
        while len(chosen) < per_class_background_cap and len(chosen) < len(candidates):
            position = min(
                (index for index in range(len(candidates)) if index not in chosen),
                key=lambda item: (worst_case[item], item),
            )
            if worst_case[position] >= 0.9 and chosen:
                break
            chosen.add(position)
            record = candidates[position]
            add(record["smiles"], f"BG-{size_class}-{len(chosen):03d}", "background")
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[position], fingerprints)
            worst_case = [max(current, value) for current, value in zip(worst_case, similarities)]

    size_counts = Counter(entry["size_class"] for entry in entries)
    catalog = {
        "schema_version": 2,
        "description": (
            "Attachable-fragment catalog. Each record is one neutral fragment with a single "
            "mapped attachment atom. Records carry no site assignment, no cut bond and no "
            "reference-ligand position: the designer matches a fragment to one of the supplied "
            "edit sites and the host validates and compiles the edit."
        ),
        "attachment_convention": (
            "[*:1] marks the atom bond that the host forms to the selected site. A record means "
            "\"bond the site atom to the attachment atom of this fragment\"."
        ),
        "size_classes": {
            "minimal": {"min_heavy_atoms": 1, "max_heavy_atoms": 1},
            "small": {"min_heavy_atoms": 2, "max_heavy_atoms": 4},
            "medium": {"min_heavy_atoms": 5, "max_heavy_atoms": 8},
            "large": {"min_heavy_atoms": 9, "max_heavy_atoms": 12},
        },
        "redundancy_policy": (
            "The catalog may list fragments that differ from each other by one atom, for example "
            "methyl and methoxy, because the layer is a property of the site plus operation plus "
            "fragment. Redundancy is resolved at design time: if the product differs from its "
            "parent by only one or two heavy atoms without a ring change, the host compiles a "
            "minimal edit instead of a whole-fragment replacement. The general library's "
            "family_role marks one-atom extensions for bookkeeping only."
        ),
        "tier_rule": {
            "t1": "1-2 heavy-atom changes at the chosen site without a ring-framework change -> minimal edit",
            "t2": ">=3 heavy-atom changes or any ring-framework change -> whole-fragment replacement",
            "ring_rule_precedence": (
                "A ring-size change or a ring-skeleton atom substitution is T2 even when only one "
                "heavy atom differs."
            ),
            "note": "The tier is a property of the chosen site plus operation plus fragment, so it is computed at design time, not stored per fragment.",
        },
        "counts": {
            "total": len(entries),
            "by_size_class": dict(sorted(size_counts.items())),
        },
        "fragments": entries,
    }
    report = {
        "provenance": provenance,
        "by_size_class": dict(sorted(size_counts.items())),
        "v1_pool_tier": {
            record["fragment_id"]: classify_fragment_smiles(REFERENCE_SIDE_CHAIN, record["smiles"])["tier"]
            for record in pool
        },
        "v1_pool_detail": {
            record["fragment_id"]: {
                "smiles": record["smiles"],
                **{
                    key: value
                    for key, value in classify_fragment_smiles(REFERENCE_SIDE_CHAIN, record["smiles"]).items()
                    if key in {"tier", "cost", "added", "lost", "substitutions", "ring_change_kinds", "reason"}
                },
            }
            for record in pool
        },
    }
    return catalog, report


# --------------------------------------------------------------------------
# edit sites
# --------------------------------------------------------------------------


def build_site_table(task_path: Path) -> dict:
    """Build the edit-site table from the host's authoritative complex parsing.

    Uses ``ComplexContext``, so the ligand graph, atom order and hydrogen counts
    match the task's ``cut_bond`` / ``core_atom_indices`` exactly.  Requires the
    ``molecular-agent-docking`` environment (gemmi + RDKit).
    """
    from molecular_agent.structure import ComplexContext

    context = ComplexContext(task_path)
    ligand = Chem.RemoveHs(Chem.Mol(context.ligand))
    compounds = np.array(
        [context.ligand.GetConformer().GetAtomPosition(index) for index in range(context.ligand.GetNumAtoms())]
    )
    heavy_indices = [index for index, atom in enumerate(context.ligand.GetAtoms()) if atom.GetAtomicNum() > 1]
    coordinates = compounds[heavy_indices]
    if ligand.GetNumAtoms() != len(coordinates):
        raise ValueError("Ligand heavy-atom count does not match the host conformer")

    protein_points = [
        (f"{atom.residue_name}{atom.residue_number}", atom.name, np.asarray(atom.xyz, dtype=float))
        for atom in context.protein_atoms
        if atom.element != "H"
    ]
    protein_coordinates = np.array([item[2] for item in protein_points])
    protein_labels = [f"{item[0]}:{item[1]}" for item in protein_points]

    ligand_lines = [
        atom for atom in context.ligand_pdb_atoms if atom.element != "H"
    ]
    rings = ring_sizes(ligand)

    per_atom = []
    for atom in ligand.GetAtoms():
        index = atom.GetIdx()
        position = coordinates[index]
        heavy_neighbours = [neighbour for neighbour in atom.GetNeighbors() if neighbour.GetAtomicNum() > 1]
        clearance = None
        nearest = None
        if heavy_neighbours and atom.GetTotalNumHs() >= 1:
            neighbour_positions = [coordinates[neighbour.GetIdx()] for neighbour in heavy_neighbours]
            direction = position - np.mean(neighbour_positions, axis=0)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                direction = direction / norm
                probes = []
                for distance in (2.5, 3.5):
                    distances = np.linalg.norm(protein_coordinates - (position + direction * distance), axis=1)
                    best = int(np.argmin(distances))
                    probes.append({
                        "probe_distance": distance,
                        "nearest_protein_atom": protein_labels[best],
                        "clearance": round(float(distances[best]), 2),
                    })
                clearance = min(item["clearance"] for item in probes)
                nearest = probes[0]["nearest_protein_atom"]
        distances = np.linalg.norm(protein_coordinates - position, axis=1)
        order = np.argsort(distances)[:4]
        per_atom.append({
            "atom_index": index,
            "element": atom.GetSymbol(),
            "pdb_atom_name": ligand_lines[index].name if index < len(ligand_lines) else "",
            "region": region_of_atom(index),
            "hydrogen_count": atom.GetTotalNumHs(),
            "is_ring_atom": index in rings,
            "is_aromatic": atom.GetIsAromatic(),
            "ring_sizes": sorted(rings.get(index, set())),
            "nearest_protein_residues": [
                {"atom": protein_labels[int(position_index)], "distance": round(float(distances[position_index]), 2)}
                for position_index in order
            ],
            "outward_probe_clearance": clearance,
            "outward_probe_nearest_atom": nearest,
            "probe_verdict": _probe_verdict(clearance),
        })

    by_index = {record["atom_index"]: record for record in per_atom}
    assigned: dict[int, str] = {}
    for definition in REGION_DEFINITIONS:
        for index in definition["atoms"]:
            if index in assigned:
                raise ValueError(
                    f"Atom {index} is claimed by both {assigned[index]} and {definition['region']}"
                )
            assigned[index] = definition["region"]
    missing = sorted(set(by_index) - set(assigned))
    if missing:
        raise ValueError(f"Ligand atoms without a region assignment: {missing}")

    protected_atoms: list[int] = []
    # Symmetry-equivalent atoms produce the identical product when the same fragment is
    # attached, so they must be presented as one edit position.  Without this the
    # designer can pick a different-but-equivalent site id and the host rejects the
    # result as a duplicate structure, which wastes a full decision round.
    symmetry_ranks = list(Chem.CanonicalRankAtoms(ligand, breakTies=False))
    symmetry_classes: dict[int, list[int]] = {}
    for index, rank in enumerate(symmetry_ranks):
        symmetry_classes.setdefault(rank, []).append(index)
    verdict_order = {"open": 0, "tight": 1, "not_applicable": 2, "blocked": 3}
    representatives: dict[int, int] = {}
    for rank, members in symmetry_classes.items():
        representative = min(
            members,
            key=lambda index: (
                verdict_order.get(by_index[index]["probe_verdict"], 4), index,
            ),
        )
        for member in members:
            representatives[member] = representative

    atom_sites = []
    for definition in REGION_DEFINITIONS:
        protection = definition["protection"]
        if protection == "protected":
            protected_atoms.extend(definition["atoms"])
        for index in definition["atoms"]:
            record = by_index[index]
            if protection == "protected" or record["hydrogen_count"] < 1:
                operations: list[str] = []
            else:
                operations = list(definition["atom_operations"])
            if index in definition.get("atom_element_swap_atoms", []):
                operations = operations + ["element_swap"]
            representative = representatives[index]
            class_members = sorted(symmetry_classes[symmetry_ranks[index]])
            atom_sites.append({
                "site_id": f"atom-{index:03d}",
                "target_type": "atom",
                "region": definition["region"],
                "atom_index": index,
                "element": record["element"],
                "pdb_atom_name": record["pdb_atom_name"],
                "hydrogen_count": record["hydrogen_count"],
                "is_ring_atom": record["is_ring_atom"],
                "is_aromatic": record["is_aromatic"],
                "probe_clearance": record["outward_probe_clearance"],
                "probe_verdict": record["probe_verdict"],
                "outward_probe_nearest_atom": record["outward_probe_nearest_atom"],
                "nearest_protein_residues": record["nearest_protein_residues"],
                "allowed_operations": operations,
                "protection": protection,
                "protection_reason": definition.get("protection_reason"),
                "overlap_group": definition["overlap_group"],
                "symmetry_class": symmetry_ranks[index],
                "symmetry_equivalent_atom_indices": class_members,
                "symmetry_representative": index == representative,
                "canonical_atom_index": representative,
                "equivalent_to": None if index == representative else f"atom-{representative:03d}",
            })

    cut_sites = []
    for number, definition in enumerate(CUT_DEFINITIONS, start=1):
        retained, removed = definition["cut_bond"]
        region_definition = next(
            item for item in REGION_DEFINITIONS if item["region"] == definition["region"]
        )
        graph = Chem.RWMol(ligand)
        graph.RemoveBond(retained, removed)
        components = Chem.GetMolFrags(graph.GetMol(), asMols=False, sanitizeFrags=False)
        if len(components) != 2:
            raise ValueError(f"Cut {definition['cut_bond']} does not split the ligand in two")
        component_sets = [set(component) for component in components]
        retained_component = next(item for item in component_sets if retained in item)
        removed_component = next(item for item in component_sets if retained not in item)
        if not set(protected_atoms) <= retained_component:
            raise ValueError(f"Cut {definition['cut_bond']} would remove protected core atoms")
        retained_point = coordinates[retained]
        removed_point = coordinates[removed]
        protection = region_definition["protection"]
        operations = [] if protection == "protected" else list(region_definition["cut_operations"])
        cut_sites.append({
            "site_id": f"cut-{number:03d}",
            "target_type": "bond",
            "region": definition["region"],
            "label": definition["label"],
            "cut_bond": [retained, removed],
            "retained_atom_index": retained,
            "removed_side_atom_index": removed,
            "retained_atom_indices": sorted(retained_component),
            "removed_atom_indices": sorted(removed_component),
            "retained_heavy_atoms": len(retained_component),
            "removed_heavy_atoms": len(removed_component),
            "removed_fraction": round(len(removed_component) / ligand.GetNumAtoms(), 3),
            "removed_fragment_smiles": Chem.MolFragmentToSmiles(
                ligand, atomsToUse=sorted(removed_component), isomericSmiles=True
            ),
            "retained_scaffold_smiles": Chem.MolFragmentToSmiles(
                ligand, atomsToUse=sorted(retained_component), isomericSmiles=True
            ),
            "attachment_vector": [
                round(float(removed_point[0] - retained_point[0]), 3),
                round(float(removed_point[1] - retained_point[1]), 3),
                round(float(removed_point[2] - retained_point[2]), 3),
            ],
            "removed_side_clearance": by_index[removed]["outward_probe_clearance"],
            "allowed_operations": operations,
            "protection": protection,
            "overlap_group": region_definition["overlap_group"],
        })

    return {
        "schema_version": 2,
        "source_task": str(task_path.relative_to(ROOT)),
        "source_task_sha256": sha256(task_path),
        "source_complex": context.complex_path.name,
        "source_complex_sha256": sha256(context.complex_path),
        "ligand_selector": context.ligand_selector,
        "atom_index_basis": "task-canonical heavy-atom order (index 9 = OAV, 10 = CBA, 18 = N3)",
        "probe_method": (
            "rigid outward-vector probe at 2.5 and 3.5 A from the heavy-atom neighbour centroid; "
            "receptor heavy atoms only; reports geometry, not a free-energy verdict"
        ),
        "verdict_rule": {
            "open": "minimum clearance > 3.2 A",
            "tight": "2.6 A < minimum clearance <= 3.2 A",
            "blocked": "minimum clearance <= 2.6 A",
        },
        "protection_semantics": (
            "A protected site is still enumerated and auditable; the host rejects every operation on it. "
            "Atomic clearance is evidence, not a free-energy verdict."
        ),
        "input_contract": (
            "The host supplies this site list and the fragment catalog. The designer chooses which "
            "fragment goes to which site. This file never names a fragment."
        ),
        "symmetry_convention": (
            "Atoms with the same symmetry_class are interchangeable: attaching the same fragment to any "
            "of them yields the identical molecule. canonical_atom_index names the representative the "
            "host uses, and equivalent_to points a redundant atom at it. A designer may name any member; "
            "the host canonicalises the choice instead of failing it."
        ),
        "protected_atom_indices": sorted(protected_atoms),
        "atom_coverage": {
            "ligand_heavy_atoms": ligand.GetNumAtoms(),
            "assigned_atoms": len(assigned),
            "unassigned_atoms": missing,
        },
        "atom_sites": atom_sites,
        "cut_sites": cut_sites,
    }


def _probe_verdict(clearance: float | None) -> str:
    if clearance is None:
        return "not_applicable"
    if clearance > 3.2:
        return "open"
    if clearance > 2.6:
        return "tight"
    return "blocked"


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-library", default="molecular_agent/data/fragments_unified.json")
    parser.add_argument("--pool", default="4WKQ/design/fragments.json")
    parser.add_argument("--task", default="4WKQ/task.benchmark.json")
    parser.add_argument("--private-dir", default="4WKQ/benchmark-private")
    args = parser.parse_args()

    general_source = (ROOT / args.general_library).resolve()
    pool_path = (ROOT / args.pool).resolve()
    task = json.loads((ROOT / args.task).read_text(encoding="utf-8"))
    complex_candidate = (ROOT / task["complex_path"])
    if not complex_candidate.is_file():
        complex_candidate = (ROOT / args.task).parent / task["complex_path"]
    complex_path = complex_candidate.resolve()
    if not complex_path.is_file():
        raise FileNotFoundError(f"Complex not found: {complex_path}")

    print("== building T1 rule table ==")
    rules = build_edit_rules()
    rules_path = ROOT / "molecular_agent/data/edit_rules_v2.json"
    write_json(rules_path, rules)

    print("== tagging general library families ==")
    general = build_general_library(general_source)
    general_path = ROOT / "molecular_agent/data/fragment_library_v2.json"
    write_json(general_path, general)

    print("== building the 4WKQ fragment catalog ==")
    catalog, catalog_report = build_4wkq_fragment_catalog(pool_path, general)
    catalog_path = ROOT / "4WKQ/design/fragments_v2.json"
    write_json(catalog_path, catalog)

    print("== building edit-site table ==")
    sites = build_site_table((ROOT / args.task).resolve())
    sites_path = ROOT / "4WKQ/design/sites-v2.json"
    write_json(sites_path, sites)

    private_path = ROOT / args.private_dir / "tier-map-v2.json"
    write_json(private_path, {
        "role": "evaluator_only_do_not_send_to_designer",
        "note": (
            "Reference-relative tier of the v1 pool. The tier is only meaningful together with "
            "the frozen C6 cut, so it is bookkeeping for the evaluator and never part of the "
            "designer-facing catalog. Literature membership stays in reachability.json."
        ),
        "pool_v1_size": len(catalog_report["v1_pool_tier"]),
        "tier_counts": dict(Counter(catalog_report["v1_pool_tier"].values())),
        "entries": catalog_report["v1_pool_detail"],
    })

    background = catalog_report["provenance"]["background"]
    manifest = {
        "schema_version": 2,
        "tier_threshold_heavy_atoms": TIER_THRESHOLD_HEAVY_ATOMS,
        "design_contract": {
            "host_supplies": ["edit sites", "fragment catalog", "deterministic validation and compilation"],
            "designer_decides": ["which site", "which operation", "which fragment"],
            "catalog_carries": ["fragment identity", "size class", "chemistry", "allowed operations"],
            "catalog_does_not_carry": [
                "site assignment",
                "cut bond",
                "retained scaffold",
                "reference-ligand position",
                "provenance or literature membership",
                "per-fragment tier",
            ],
        },
        "public_files": {
            "catalog": str(catalog_path.relative_to(ROOT)),
            "catalog_sha256": sha256(catalog_path),
            "sites": str(sites_path.relative_to(ROOT)),
            "sites_sha256": sha256(sites_path),
            "edit_rules": str(rules_path.relative_to(ROOT)),
            "edit_rules_sha256": sha256(rules_path),
            "general_library": str(general_path.relative_to(ROOT)),
            "general_library_sha256": sha256(general_path),
        },
        "input_files": {
            "pool_v1": {"path": str(pool_path.relative_to(ROOT)), "sha256": sha256(pool_path)},
            "general_library_v1": {"path": str(general_source.relative_to(ROOT)), "sha256": sha256(general_source)},
            "complex": {"path": str(complex_path.relative_to(ROOT)), "sha256": sha256(complex_path)},
        },
        "counts": {
            "catalog_total": catalog["counts"]["total"],
            "catalog_by_size_class": catalog["counts"]["by_size_class"],
            "catalog_background_selected": background,
            "catalog_one_atom_groups": catalog_report["provenance"]["one_atom_groups"],
            "catalog_two_atom_groups": catalog_report["provenance"]["two_atom_groups"],
            "v1_pool_retained": catalog_report["provenance"]["benchmark_pool"],
            "general_library_entries": general["counts"]["entries"],
            "general_library_base_entries": general["counts"]["base_entries"],
            "general_library_extensions": general["counts"]["extensions"],
            "atom_site_count": len(sites["atom_sites"]),
            "cut_site_count": len(sites["cut_sites"]),
            "protected_atom_count": len(sites["protected_atom_indices"]),
        },
    }
    manifest_path = ROOT / "4WKQ/design/pool-manifest-v2.json"
    write_json(manifest_path, manifest)

    print("\n== summary ==")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
