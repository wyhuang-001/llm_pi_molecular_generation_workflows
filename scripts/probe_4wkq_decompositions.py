#!/usr/bin/env python
"""Probe: how a cut-point-free 4WKQ library would decompose the literature series.

For every literature compound it aligns the compound to gefitinib with one MCS,
then reads off what changed:

* removed = native atoms outside the common substructure
* added   = target atoms outside the common substructure
* boundary = bonds crossing between the retained part and the removed / added side

The number of boundary bonds then names the operation in the two-axis taxonomy:

* removed empty, added present, 1 boundary bond            -> atom:addition
* added empty, removed present, 1 boundary bond            -> bond:deletion
* added empty, removed present, 2 boundary bonds           -> linker:deletion
* both present, 1 boundary bond                            -> bond:replacement
* a single swapped atom                                    -> atom:/ring:replacement

Read-only: writes nothing, is not part of the workflow.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molecular_agent.structure import ComplexContext  # noqa: E402
from molecular_agent.tiering import classify_change  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "4WKQ" / "task.benchmark.json"
REACHABILITY = ROOT / "4WKQ" / "benchmark-private" / "reachability.json"


def mcs_alignment(native: Chem.Mol, target: Chem.Mol, timeout: int = 10):
    result = rdFMCS.FindMCS(
        [native, target],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        matchValences=False,
        timeout=timeout,
    )
    if not result.smartsString or result.numAtoms == 0:
        return None
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None:
        return None
    native_match = native.GetSubstructMatch(query)
    target_match = target.GetSubstructMatch(query)
    if not native_match or not target_match:
        return None
    return native_match, target_match


def boundary_bonds(molecule: Chem.Mol, side: set[int]) -> list[tuple[int, int]]:
    edges = []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (begin in side) != (end in side):
            edges.append((begin, end))
    return edges


def side_smiles(molecule: Chem.Mol, side: set[int], anchor: int) -> str | None:
    """Attachment SMILES of ``side`` with ``anchor`` turned into ``[*:1]``."""
    if not side:
        return None
    rw = Chem.RWMol(Chem.Mol(molecule))
    for index in sorted(set(range(molecule.GetNumAtoms())) - side, reverse=True):
        rw.RemoveAtom(index)
    shift = sum(
        1 for index in range(molecule.GetNumAtoms())
        if index not in side and index < anchor
    )
    local_anchor = anchor - shift
    atom = rw.GetAtomWithIdx(local_anchor)
    atom.SetAtomicNum(0)
    atom.SetAtomMapNum(1)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(0)
    atom.SetIsAromatic(False)
    try:
        molecule_out = rw.GetMol()
        Chem.SanitizeMol(molecule_out)
    except Exception:
        return None
    if len(Chem.GetMolFrags(molecule_out)) != 1:
        return None
    if Chem.GetFormalCharge(molecule_out) != 0:
        return None
    return Chem.MolToSmiles(molecule_out, isomericSmiles=True)


def classify_edit(native: Chem.Mol, target: Chem.Mol):
    alignment = mcs_alignment(native, target)
    if alignment is None:
        return [], None
    native_match, target_match = alignment
    removed = set(range(native.GetNumAtoms())) - set(native_match)
    added = set(range(target.GetNumAtoms())) - set(target_match)
    removed_edges = boundary_bonds(native, removed) if removed else []
    added_edges = boundary_bonds(target, added) if added else []

    candidates: list[dict] = []

    # a single swapped element with everything else retained
    if len(removed) == 1 and len(added) == 1:
        old = native.GetAtomWithIdx(next(iter(removed))).GetSymbol()
        new = target.GetAtomWithIdx(next(iter(added))).GetSymbol()
        if len(removed_edges) == len(added_edges):
            candidates.append({
                "change_type": "replacement",
                "site_type": "atom",
                "site": next(iter(removed)),
                "fragment_smiles": None,
                "element": new,
                "detail": f"{old}->{new}",
            })

    # addition: nothing removed, one new side attached at one native atom
    if not removed and added and len(added_edges) == 1:
        begin, end = added_edges[0]
        anchor = begin if begin not in added else end
        smiles = side_smiles(target, added | {anchor}, anchor)
        if smiles:
            candidates.append({
                "change_type": "addition",
                "site_type": "atom",
                "site": anchor,
                "fragment_smiles": smiles,
                "element": None,
                "detail": "grow at a hydrogen site",
            })

    if removed:
        if len(removed_edges) == 1:
            begin, end = removed_edges[0]
            anchor = begin if begin not in removed else end
            if added and len(added_edges) == 1:
                # the fragment lives on the target, so its anchor must be the
                # target-side atom of the target boundary bond, not the native one
                t_begin, t_end = added_edges[0]
                target_anchor = t_begin if t_begin not in added else t_end
                smiles = side_smiles(target, added | {target_anchor}, target_anchor)
                if smiles:
                    candidates.append({
                        "change_type": "replacement",
                        "site_type": "bond",
                        "site": anchor,
                        "fragment_smiles": smiles,
                        "element": None,
                        "detail": "swap the whole removed side",
                    })
            elif not added:
                candidates.append({
                    "change_type": "deletion",
                    "site_type": "bond",
                    "site": anchor,
                    "fragment_smiles": None,
                    "element": None,
                    "detail": "delete the removed side",
                })
        elif len(removed_edges) == 2:
            candidates.append({
                "change_type": "deletion",
                "site_type": "linker",
                "site": tuple(sorted(removed_edges[0])),
                "fragment_smiles": None,
                "element": None,
                "detail": f"remove {len(removed)} linker atom(s)",
            })

    return candidates, {
        "removed": len(removed),
        "added": len(added),
        "removed_edges": len(removed_edges),
        "added_edges": len(added_edges),
    }


def main() -> int:
    context = ComplexContext(TASK)
    native = Chem.RemoveHs(Chem.Mol(context.ligand))
    records = json.loads(REACHABILITY.read_text(encoding="utf-8"))["records"]
    targets = [r for r in records if not r.get("is_reference")]

    print(f"native: {Chem.MolToSmiles(native, isomericSmiles=True)}  ({native.GetNumAtoms()} heavy atoms)")
    print(f"literature compounds: {len(targets)}\n")

    print(f"{'compound':16} {'tier':4} {'rem/added/edges':>16}  operations")
    tiers = Counter()
    op_counts = Counter()
    fragments: dict[str, set[str]] = defaultdict(set)
    sites: dict[tuple, set[str]] = defaultdict(set)
    attachment = Counter()
    minimal_fragments: dict[str, set[str]] = defaultdict(set)

    detail_rows = []
    for record in targets:
        cid = record["source_compound_id"]
        target = Chem.RemoveHs(Chem.MolFromSmiles(record["canonical_smiles"]))
        tier = classify_change(native, target)
        tiers[tier["tier"]] += 1
        candidates, stats = classify_edit(native, target)
        ops = []
        for item in candidates:
            label = f"{item['site_type']}:{item['change_type']}"
            ops.append(label)
            op_counts[label] += 1
            sites[(item["site_type"], item["site"])].add(cid)
            if item["fragment_smiles"]:
                fragments[item["fragment_smiles"]].add(cid)
                anchor_atom = next(
                    atom for atom in Chem.MolFromSmiles(item["fragment_smiles"]).GetAtoms()
                    if atom.GetAtomicNum() == 0
                ).GetNeighbors()[0]
                attachment[anchor_atom.GetSymbol()] += 1
        if not any(item["fragment_smiles"] for item in candidates):
            # the maximal-MCS alignment is not always a single-edit one; the frozen
            # C6 cut is known to reproduce these compounds, so fall back to it
            for fallback in record.get("attachment_fragments") or []:
                candidates.append({
                    "change_type": "replacement",
                    "site_type": "bond",
                    "site": "C6",
                    "fragment_smiles": fallback,
                    "element": None,
                    "detail": "frozen C6 cut fallback",
                })
        if not ops and candidates:
            ops.append("C6 fallback")
        shape = f"{stats['removed']}/{stats['added']}/{stats['removed_edges']}+{stats['added_edges']}" if stats else "-"
        print(f"{cid:16} {tier['tier']:4} {shape:>16}  {', '.join(sorted(set(ops)))}")
        smallest = min(
            (item for item in candidates if item["fragment_smiles"]),
            key=lambda item: Chem.MolFromSmiles(item["fragment_smiles"]).GetNumHeavyAtoms(),
            default=None,
        )
        if smallest:
            minimal_fragments[smallest["fragment_smiles"]].add(cid)
        detail_rows.append((cid, tier["tier"], candidates))

    print(f"\ntier of gefitinib -> compound: {dict(tiers)}")
    print(f"operation mix: {dict(op_counts)}")
    print(f"distinct sites that can reach a literature compound: {len(sites)}")
    print(f"distinct attachment fragments required: {len(fragments)}")
    print(f"attachment element of required fragments: {dict(attachment)}")
    sizes = Counter(
        Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() for smiles in fragments
    )
    print(f"fragment heavy atoms: {dict(sorted(sizes.items()))}")
    print(f"fragments shared by >1 compound: {sum(1 for v in fragments.values() if len(v) > 1)}")

    print("\n-- smallest representation per compound --")
    for smiles, cids in sorted(
        minimal_fragments.items(),
        key=lambda item: Chem.MolFromSmiles(item[0]).GetNumHeavyAtoms(),
    ):
        atoms = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
        print(f"  {atoms:>2} atoms  {smiles:38} -> {', '.join(sorted(cids))}")

    print("\n-- no-fragment (minimal-edit) compounds --")
    for cid, tier, candidates in detail_rows:
        if any(item["fragment_smiles"] is None for item in candidates):
            kinds = sorted({f"{i['site_type']}:{i['change_type']} {i['detail']}"
                            for i in candidates if item["fragment_smiles"] is None})
            print(f"  {cid:16} {tier}  {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
