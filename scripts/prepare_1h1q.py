from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import gemmi
from rdkit import Chem


URLS = {
    "complex": "https://files.rcsb.org/download/1H1Q.pdb",
    "component": "https://files.rcsb.org/ligands/download/2A6.cif",
    "ligand_graph": "https://files.rcsb.org/ligands/download/2A6_ideal.sdf",
}


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "simple-molecular-agent/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("input"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    paths = {
        "complex": output / "complex.pdb",
        "component": raw / "2A6.cif",
        "ligand_graph": raw / "2A6_ideal.sdf",
    }
    for key, url in URLS.items():
        download(url, paths[key])

    block = gemmi.cif.read_file(str(paths["component"])).sole_block()
    table = block.find(
        "_chem_comp_atom.",
        ["atom_id", "model_Cartn_x", "model_Cartn_y", "model_Cartn_z"],
    )
    rows = list(table)
    coordinates = {row[0]: tuple(float(row[index]) for index in (1, 2, 3)) for row in rows}

    molecule = Chem.SDMolSupplier(str(paths["ligand_graph"]), removeHs=False)[0]
    if molecule is None:
        raise RuntimeError("Could not read downloaded 2A6 ligand graph")
    atom_names = [row[0] for row in rows]
    if len(atom_names) != molecule.GetNumAtoms():
        raise RuntimeError("RCSB CIF/SDF atom count mismatch")
    conformer = molecule.GetConformer()
    conformer.Set3D(True)
    for index, atom_name in enumerate(atom_names):
        x, y, z = coordinates[atom_name]
        conformer.SetAtomPosition(index, (x, y, z))
        molecule.GetAtomWithIdx(index).SetProp("atom_name", atom_name)
    molecule.SetProp("_Name", "NU6094_2A6_1H1Q_chain_A")
    molecule.SetProp("pose_source", "experimental coordinates from PDB 1H1Q ligand 2A6 chain A")
    writer = Chem.SDWriter(str(output / "ligand.sdf"))
    writer.write(molecule)
    writer.close()

    sources = {
        "system": "CDK2/cyclin A with NU6094",
        "pdb_id": "1H1Q",
        "ligand_code": "2A6",
        "resolution_angstrom": 2.5,
        "urls": URLS,
        "files": {str(path.relative_to(output)): sha256(path) for path in paths.values()},
        "reference": {
            "doi": "10.1038/nsb842",
            "pubmed": "12244298",
            "note": "NU6094 IC50 is about 970 nM for CDK2/cyclin A in ChEMBL assay CHEMBL662528. The stronger para-sulfamoyl analog NU6102 is held out from the agent input.",
        },
    }
    sources["files"]["ligand.sdf"] = sha256(output / "ligand.sdf")
    (output / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
