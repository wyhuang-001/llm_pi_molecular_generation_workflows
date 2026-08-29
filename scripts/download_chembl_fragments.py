#!/usr/bin/env python3
"""Build an auditable one-attachment ChEMBL BRICS fragment library.

The REST API is useful for small samples. For large runs this script downloads
ChEMBL's official chemreps snapshot once and processes it as a gzip stream.
A SQLite checkpoint makes the BRICS derivation restartable without retaining
all parent molecules in memory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors


API_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
STATUS_URL = "https://www.ebi.ac.uk/chembl/api/data/status.json"
FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/"
ATTRIBUTION_URL = f"{FTP_BASE}/REQUIRED.ATTRIBUTION"
CHEMREPS_NAME = "chembl_37_chemreps.txt.gz"
CHEMREPS_URL = f"{FTP_BASE}/{CHEMREPS_NAME}"
CHEMREPS_SHA256 = "ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7"


def fetch_json(url: str, retries: int = 5) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "simple-molecular-agent/0.1",
                },
            )
            with urlopen(request, timeout=120) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object from {url}")
            return value
        except Exception as error:  # network services can reset long jobs
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_error}") from last_error


def normalize_attachment(fragment_smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(fragment_smiles)
    if molecule is None:
        return None
    dummies = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    dummy = dummies[0]
    dummy.SetIsotope(0)
    dummy.SetAtomMapNum(1)
    if len(dummy.GetNeighbors()) != 1:
        return None
    bond = molecule.GetBondBetweenAtoms(dummy.GetIdx(), dummy.GetNeighbors()[0].GetIdx())
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def derive_record(smiles: str, max_fragment_heavy_atoms: int) -> list[str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return []
    fragments = []
    for raw_fragment in BRICS.BRICSDecompose(molecule, minFragmentSize=2):
        normalized = normalize_attachment(raw_fragment)
        fragment = Chem.MolFromSmiles(normalized) if normalized else None
        if fragment is None or fragment.GetNumHeavyAtoms() > max_fragment_heavy_atoms:
            continue
        fragments.append(normalized)
    return sorted(set(fragments))


def derive_fragments(
    molecules: list[dict[str, Any]], max_fragment_heavy_atoms: int
) -> dict[str, dict[str, Any]]:
    parents: dict[str, set[str]] = defaultdict(set)
    for record in molecules:
        molecule_id = record.get("molecule_chembl_id")
        smiles = (record.get("molecule_structures") or {}).get("canonical_smiles")
        if not isinstance(molecule_id, str) or not isinstance(smiles, str):
            continue
        try:
            fragments = derive_record(smiles, max_fragment_heavy_atoms)
        except Exception:
            continue
        for fragment in fragments:
            parents[fragment].add(molecule_id)
    return _fragment_records(parents)


def _fragment_records(parents: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    fragments = {}
    for index, smiles in enumerate(sorted(parents), start=1):
        fragments[smiles] = {
            "fragment_id": f"chembl-brics-{index:06d}",
            "name": f"ChEMBL BRICS fragment {index}",
            "smiles": smiles,
            "operation": "substitute",
            "source_molecule_ids": sorted(parents[smiles])[:50],
            "source_molecule_count": len(parents[smiles]),
        }
    return fragments


def download_molecules(limit: int, max_parent_mw: float, page_size: int) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    urls = []
    offset = 0
    while len(records) < limit:
        query = urlencode({
            "molecule_properties__mw_freebase__lte": max_parent_mw,
            "limit": min(page_size, limit - len(records)),
            "offset": offset,
        })
        url = f"{API_URL}?{query}"
        payload = fetch_json(url)
        batch = payload.get("molecules") or []
        if not batch:
            break
        records.extend(batch)
        urls.append(url)
        offset += len(batch)
    return records[:limit], urls


def download_resumable(url: str, path: Path, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"Using existing source file or resuming: {path}", flush=True)
    command = [
        "curl", "--fail", "--location", "--retry", "8", "--retry-all-errors",
        "--retry-delay", "3", "--connect-timeout", "30", "--max-time", "0",
        "--continue-at", "-", "--output", str(path), url,
    ]
    subprocess.run(command, check=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    print(f"Verified {path} SHA256={actual}", flush=True)


def open_checkpoint(path: Path, settings: dict[str, Any]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS fragments (smiles TEXT PRIMARY KEY, source_ids TEXT NOT NULL, source_count INTEGER NOT NULL)"
    )
    existing = dict(connection.execute("SELECT key, value FROM metadata"))
    for key, value in settings.items():
        if key in existing and existing[key] != str(value):
            raise RuntimeError(
                f"Checkpoint setting {key}={existing[key]} conflicts with requested {value}"
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, str(value)),
        )
    connection.commit()
    return connection


def metadata(connection: sqlite3.Connection, key: str, default: int = 0) -> int:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return int(row[0]) if row else default


def set_metadata(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in values.items()],
    )
    connection.commit()


def checkpoint_fragments(connection: sqlite3.Connection, parents: dict[str, set[str]]) -> None:
    for smiles, ids in parents.items():
        row = connection.execute(
            "SELECT source_ids, source_count FROM fragments WHERE smiles = ?", (smiles,)
        ).fetchone()
        previous_ids = set(json.loads(row[0])) if row else set()
        previous_count = int(row[1]) if row else 0
        merged_ids = previous_ids | ids
        connection.execute(
            "INSERT OR REPLACE INTO fragments(smiles, source_ids, source_count) VALUES (?, ?, ?)",
            (
                smiles,
                json.dumps(sorted(merged_ids)[:50]),
                previous_count + len(ids),
            ),
        )


def commit_checkpoint(
    connection: sqlite3.Connection,
    parents: dict[str, set[str]],
    values: dict[str, Any],
) -> None:
    """Atomically commit fragments and the source position they represent."""
    with connection:
        checkpoint_fragments(connection, parents)
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            [(key, str(value)) for key, value in values.items()],
        )


def build_from_chemreps(
    source_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    max_parent_mw: float,
    max_fragment_heavy_atoms: int,
    max_molecules: int,
    checkpoint_every: int,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    settings = {
        "source": source_path.resolve(),
        "max_parent_mw": max_parent_mw,
        "max_fragment_heavy_atoms": max_fragment_heavy_atoms,
        "max_molecules": max_molecules,
    }
    connection = open_checkpoint(checkpoint_path, settings)
    if metadata(connection, "complete") and output_path.exists():
        fragment_count = connection.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
        result = {
            "output": str(output_path),
            "molecules": metadata(connection, "selected_molecules"),
            "fragments": fragment_count,
            "checkpoint": str(checkpoint_path),
            "resumed_complete": True,
        }
        connection.close()
        return result
    processed_lines = metadata(connection, "processed_lines")
    selected = metadata(connection, "selected_molecules")
    invalid = metadata(connection, "invalid_molecules")
    decomposition_errors = metadata(connection, "decomposition_errors")
    total_lines = metadata(connection, "total_lines")
    parents: dict[str, set[str]] = {}
    with gzip.open(source_path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
        columns = header.rstrip("\n").split("\t")
        try:
            chembl_index = columns.index("chembl_id")
            smiles_index = columns.index("canonical_smiles")
        except ValueError as error:
            raise RuntimeError(f"Unexpected chemreps header: {header!r}") from error
        for line_number, line in enumerate(handle, start=1):
            total_lines = line_number
            if line_number <= processed_lines:
                continue
            values = line.rstrip("\n").split("\t")
            if len(values) <= max(chembl_index, smiles_index):
                invalid += 1
                continue
            molecule_id, smiles = values[chembl_index], values[smiles_index]
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                invalid += 1
                continue
            if Descriptors.MolWt(molecule) > max_parent_mw:
                continue
            selected += 1
            try:
                fragments = derive_record(smiles, max_fragment_heavy_atoms)
            except Exception as error:
                decomposition_errors += 1
                if decomposition_errors <= 20:
                    print(
                        f"Skipping BRICS failure molecule={molecule_id} error={error!r}",
                        flush=True,
                    )
                fragments = []
            for fragment in fragments:
                parents.setdefault(fragment, set()).add(molecule_id)
            if selected % checkpoint_every == 0:
                commit_checkpoint(connection, parents, {
                    "processed_lines": line_number,
                    "selected_molecules": selected,
                    "invalid_molecules": invalid,
                    "decomposition_errors": decomposition_errors,
                    "total_lines": total_lines,
                })
                parents.clear()
                print(
                    f"processed_lines={line_number} selected={selected} fragments="
                    f"{connection.execute('SELECT COUNT(*) FROM fragments').fetchone()[0]}",
                    flush=True,
                )
            if max_molecules and selected >= max_molecules:
                break
    commit_checkpoint(connection, parents, {
        "processed_lines": total_lines,
        "selected_molecules": selected,
        "invalid_molecules": invalid,
        "decomposition_errors": decomposition_errors,
        "total_lines": total_lines,
    })
    rows = connection.execute(
        "SELECT smiles, source_ids, source_count FROM fragments ORDER BY smiles"
    ).fetchall()
    records = []
    for index, (smiles, source_ids, source_count) in enumerate(rows, start=1):
        records.append({
            "fragment_id": f"chembl-brics-{index:06d}",
            "name": f"ChEMBL BRICS fragment {index}",
            "smiles": smiles,
            "operation": "substitute",
            "source_molecule_ids": json.loads(source_ids),
            "source_molecule_count": source_count,
        })
    payload = {
        "schema_version": 1,
        "allowed_operations": ["substitute", "replace_fragment"],
        "source": {
            **source_metadata,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "derivation": "One-attachment fragments generated with RDKit BRICSDecompose; attachment normalized to [*:1].",
            "downloaded_molecule_count": selected,
            "invalid_molecule_count": invalid,
            "brics_decomposition_error_count": decomposition_errors,
            "processed_source_lines": total_lines,
        },
        "filters": {
            "max_parent_mw": max_parent_mw,
            "max_fragment_heavy_atoms": max_fragment_heavy_atoms,
            "one_attachment_point": True,
        },
        "fragments": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    set_metadata(connection, {"complete": 1})
    connection.close()
    return {"output": str(output_path), "molecules": selected, "fragments": len(records), "checkpoint": str(checkpoint_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "rest", "ftp"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("molecular_agent/data/chembl_fragments.json"))
    parser.add_argument("--molecules", type=int, default=1000, help="REST limit; use 0 with --source ftp for all matching molecules")
    parser.add_argument("--max-parent-mw", type=float, default=350.0)
    parser.add_argument("--max-fragment-heavy-atoms", type=int, default=12)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    args = parser.parse_args()
    if args.molecules < 0 or args.page_size < 1 or args.checkpoint_every < 1:
        raise SystemExit("molecules must be non-negative; page-size and checkpoint-every must be positive")
    source = "ftp" if args.source == "ftp" or (args.source == "auto" and args.molecules == 0) else "rest"
    status = fetch_json(STATUS_URL)
    if source == "rest":
        if args.molecules < 1:
            raise SystemExit("REST mode requires --molecules >= 1")
        molecules, request_urls = download_molecules(args.molecules, args.max_parent_mw, args.page_size)
        fragments = derive_fragments(molecules, args.max_fragment_heavy_atoms)
        payload = {
            "schema_version": 1,
            "allowed_operations": ["substitute", "replace_fragment"],
            "source": {
                "name": "ChEMBL", "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "api": API_URL, "status_api": STATUS_URL,
                "release": status.get("chembl_db_version"), "release_date": status.get("chembl_release_date"),
                "request_urls": request_urls, "license": "CC BY-SA 3.0", "license_url": LICENSE_URL,
                "required_attribution": ATTRIBUTION_URL,
                "citation": "Mendez et al., Nucleic Acids Research 2019, DOI:10.1093/nar/gky1075",
                "derivation": "One-attachment fragments generated with RDKit BRICSDecompose; attachment normalized to [*:1].",
                "downloaded_molecule_count": len(molecules),
            },
            "filters": {"max_parent_mw": args.max_parent_mw, "max_fragment_heavy_atoms": args.max_fragment_heavy_atoms, "one_attachment_point": True},
            "fragments": list(fragments.values()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "molecules": len(molecules), "fragments": len(fragments)}))
        return

    source_path = args.cache_dir / CHEMREPS_NAME
    download_resumable(CHEMREPS_URL, source_path, CHEMREPS_SHA256)
    checkpoint = args.checkpoint or args.output.with_suffix(".checkpoint.sqlite")
    result = build_from_chemreps(
        source_path, args.output, checkpoint, args.max_parent_mw,
        args.max_fragment_heavy_atoms, args.molecules, args.checkpoint_every,
        {
            "name": "ChEMBL", "api": API_URL, "status_api": STATUS_URL,
            "ftp": CHEMREPS_URL, "release": status.get("chembl_db_version"),
            "release_date": status.get("chembl_release_date"), "license": "CC BY-SA 3.0",
            "license_url": LICENSE_URL, "required_attribution": ATTRIBUTION_URL,
            "citation": "Mendez et al., Nucleic Acids Research 2019, DOI:10.1093/nar/gky1075",
        },
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
