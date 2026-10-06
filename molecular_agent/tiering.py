"""Two-layer edit classification for scaffold-preserving molecular optimization.

The library distinguishes two edit layers:

* ``T1`` — small edits: any single-atom substitution, two-heavy-atom changes,
  ring-side-chain (non-skeleton) decoration, and chain-length changes of one or
  two atoms.  Compiled as a *minimal edit* on the parent rather than as a whole
  fragment replacement.
* ``T2`` — large edits: three or more heavy-atom changes, **or** any change to
  the ring skeleton (ring-size change, ring-atom element substitution, ring
  formed/opened), even when only one heavy atom differs.

The ring rules take precedence over the atom-count rule: a one-atom ring
skeleton change is still T2 because it replaces the whole ring.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import rdFMCS

T1 = "T1"
T2 = "T2"

# Elements considered acceptable in drug-like fragments.  ``rare`` elements are
# kept but flagged so a caller can exclude them without editing the classifier.
DRUG_LIKE_ELEMENTS = {6, 7, 8, 9, 15, 16, 17, 35, 53}
RARE_ELEMENTS = {5, 14, 15, 34}  # B, Si, P, Se
EXCLUDED_ELEMENTS = {33, 51, 52, 80, 81, 82, 83, 84}  # As, Sb, Te, Hg, Tl, Pb, Bi, Po


def ring_sizes(molecule: Chem.Mol) -> dict[int, set[int]]:
    """Return ``atom_index -> {ring sizes containing that atom}``."""
    sizes: dict[int, set[int]] = {}
    for ring in molecule.GetRingInfo().AtomRings():
        for index in ring:
            sizes.setdefault(index, set()).add(len(ring))
    return sizes


def _matched_environment(
    index: int,
    molecule: Chem.Mol,
    matched: set[int],
    translation: dict[int, int] | None = None,
) -> set[int]:
    """Matched neighbours of ``index``, optionally translated to another frame."""
    neighbours = [
        neighbour.GetIdx()
        for neighbour in molecule.GetAtomWithIdx(index).GetNeighbors()
        if neighbour.GetIdx() in matched
    ]
    if translation is None:
        return set(neighbours)
    return {translation[index] for index in neighbours}


def classify_change(reference: Chem.Mol, fragment: Chem.Mol, timeout: int = 5) -> dict[str, Any]:
    """Classify a fragment relative to a reference side chain.

    Both molecules must be heavy-atom molecules without explicit hydrogens.  The
    returned ``cost`` is a graph edit distance in which an element substitution
    counts as one change (not one deletion plus one addition).
    """
    if reference.GetNumAtoms() == 0 or fragment.GetNumAtoms() == 0:
        raise ValueError("classify_change requires non-empty molecules")

    result = rdFMCS.FindMCS(
        [fragment, reference],
        timeout=timeout,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        matchValences=False,
        matchChiralTag=False,
    )
    if result.numAtoms == 0:
        return {
            "tier": T2,
            "cost": fragment.GetNumHeavyAtoms() + reference.GetNumHeavyAtoms(),
            "added": fragment.GetNumHeavyAtoms(),
            "lost": reference.GetNumHeavyAtoms(),
            "substitutions": 0,
            "ring_size_change": True,
            "ring_skeleton_change": False,
            "ring_change_kinds": ["no_common_substructure"],
            "reason": "no common substructure with the reference",
        }

    query = Chem.MolFromSmarts(result.smartsString)
    fragment_match = list(fragment.GetSubstructMatch(query))
    reference_match = list(reference.GetSubstructMatch(query))
    fragment_matched = set(fragment_match)
    reference_matched = set(reference_match)
    # Both environments are compared in the reference atom frame.
    fragment_to_reference = {fragment_match[position]: reference_match[position] for position in range(len(fragment_match))}

    added = sorted(set(range(fragment.GetNumAtoms())) - fragment_matched)
    lost = sorted(set(range(reference.GetNumAtoms())) - reference_matched)

    fragment_rings = ring_sizes(fragment)
    reference_rings = ring_sizes(reference)

    # Pair an added atom with a lost atom that occupies the same matched
    # attachment point: that is an element/valence substitution.
    substitutions = 0
    substitution_pairs: list[tuple[int, int]] = []
    used_lost: set[int] = set()
    for added_index in added:
        environment = _matched_environment(
            added_index, fragment, fragment_matched, fragment_to_reference
        )
        if not environment:
            continue
        for lost_index in lost:
            if lost_index in used_lost:
                continue
            lost_environment = _matched_environment(lost_index, reference, reference_matched)
            if environment == lost_environment:
                substitutions += 1
                substitution_pairs.append((added_index, lost_index))
                used_lost.add(lost_index)
                break

    paired_added = {pair[0] for pair in substitution_pairs}
    paired_lost = {pair[1] for pair in substitution_pairs}

    ring_kinds: set[str] = set()

    # Matched atoms whose ring-size sets disagree: the ring itself changed size.
    for fragment_index, reference_index in zip(fragment_match, reference_match):
        if fragment_rings.get(fragment_index, set()) != reference_rings.get(reference_index, set()):
            ring_kinds.add("size_change")
            break

    # Added/lost ring atoms: without a same-position substitution partner the
    # ring framework itself was built or destroyed.
    for added_index in added:
        if added_index not in fragment_rings:
            continue
        partner = next((pair[1] for pair in substitution_pairs if pair[0] == added_index), None)
        if partner is None:
            ring_kinds.add("ring_formed")
        elif fragment_rings[added_index] != reference_rings.get(partner, set()):
            ring_kinds.add("size_change")
        else:
            ring_kinds.add("skeleton_substitution")

    for lost_index in lost:
        if lost_index not in reference_rings:
            continue
        partner = next((pair[0] for pair in substitution_pairs if pair[1] == lost_index), None)
        if partner is None:
            ring_kinds.add("ring_removed")
        elif fragment_rings.get(partner, set()) != reference_rings[lost_index]:
            ring_kinds.add("size_change")

    # Matched ring atoms whose element changed (possible when a caller relaxes
    # the MCS atom comparison to CompareAny instead of CompareElements).
    for fragment_index, reference_index in zip(fragment_match, reference_match):
        if fragment_index in fragment_rings and reference_index in reference_rings:
            if fragment.GetAtomWithIdx(fragment_index).GetAtomicNum() != reference.GetAtomWithIdx(reference_index).GetAtomicNum():
                ring_kinds.add("skeleton_substitution")

    cost = len(added) + len(lost) - substitutions
    ring_size_change = "size_change" in ring_kinds
    ring_skeleton_change = "skeleton_substitution" in ring_kinds
    ring_change = bool(ring_kinds)
    tier = T2 if (ring_change or cost >= 3) else T1

    return {
        "tier": tier,
        "cost": cost,
        "added": len(added),
        "lost": len(lost),
        "substitutions": substitutions,
        "substitution_pairs": substitution_pairs,
        "added_indices": added,
        "lost_indices": lost,
        "ring_size_change": ring_size_change,
        "ring_skeleton_change": ring_skeleton_change,
        "ring_change_kinds": sorted(ring_kinds),
        "reason": _reason(cost, sorted(ring_kinds)),
    }


def _reason(cost: int, ring_kinds: list[str]) -> str:
    if "size_change" in ring_kinds:
        return "ring-size change: whole-ring replacement"
    if "skeleton_substitution" in ring_kinds:
        return "ring-skeleton atom substitution: whole-ring replacement"
    if "ring_removed" in ring_kinds:
        return "ring framework removed: large fragment"
    if "ring_formed" in ring_kinds:
        return "ring framework formed: large fragment"
    if cost >= 3:
        return f"{cost} heavy-atom changes: large fragment"
    return f"{cost} heavy-atom change(s) without ring-skeleton change: minimal edit"


def strip_attachment_dummy(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES and delete every attachment dummy atom.

    Deleting the dummy after parsing (instead of editing the SMILES text) keeps
    records such as ``[*:1]=C`` valid.  The bond order to the deleted dummy is
    not part of the tier decision, which only counts heavy-atom and ring changes.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    indices = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if not indices:
        return molecule
    editable = Chem.RWMol(molecule)
    for index in sorted(indices, reverse=True):
        editable.RemoveAtom(index)
    stripped = editable.GetMol()
    try:
        Chem.SanitizeMol(stripped)
    except Exception:
        return None
    return stripped


def classify_fragment_smiles(reference_smiles: str, fragment_smiles: str) -> dict[str, Any]:
    """Convenience wrapper that removes any ``[*:1]`` attachment dummy first."""
    reference = strip_attachment_dummy(reference_smiles)
    fragment = strip_attachment_dummy(fragment_smiles)
    if reference is None or fragment is None:
        raise ValueError(
            f"Invalid SMILES supplied to classify_fragment_smiles: {reference_smiles!r}, {fragment_smiles!r}"
        )
    if reference.GetNumAtoms() == 0 or fragment.GetNumAtoms() == 0:
        return {
            "tier": T2,
            "cost": fragment.GetNumHeavyAtoms() + reference.GetNumHeavyAtoms(),
            "added": fragment.GetNumHeavyAtoms(),
            "lost": reference.GetNumHeavyAtoms(),
            "substitutions": 0,
            "ring_size_change": False,
            "ring_skeleton_change": False,
            "ring_change_kinds": ["no_common_substructure"],
            "reason": "an attachment-only fragment has no comparable skeleton",
        }
    return classify_change(reference, fragment)
