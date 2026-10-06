"""Read-only PLIP analysis of an explicitly selected pose, independent of the LLM."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from typing import Any

from rdkit import Chem


class PLIPAdapter:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @staticmethod
    def build_complex(receptor: Path, poses: Path, destination: Path, pose_rank: int = 1) -> dict[str, Any]:
        supplier = Chem.SDMolSupplier(str(poses), removeHs=False)
        molecule = supplier[pose_rank-1] if 1 <= pose_rank <= len(supplier) else None
        if molecule is None or not molecule.GetNumConformers():
            raise ValueError(f"GNINA rank-{pose_rank} pose is unreadable; do not silently select another rank")
        # Receptor is already prepared/allowlisted by Host. Do not drop TPO here.
        protein = [line for line in receptor.read_text().splitlines()
                   if line[:6].strip() in {"ATOM", "HETATM", "CONECT"}]
        if not protein:
            raise ValueError("Receptor has no ATOM records")
        receptor_atoms = [line for line in protein if line[:6].strip() in {"ATOM", "HETATM"}]
        if any(line[17:20].strip() == "LIG" and line[21:22] == "Z" and int(line[22:26]) == 1
               for line in receptor_atoms):
            raise ValueError("Prepared receptor collides with reserved ligand ID LIG:Z:1")
        source_serials = [int(line[6:11]) for line in receptor_atoms]
        if len(set(source_serials)) != len(source_serials):
            raise ValueError("Receptor has duplicate PDB atom serials")
        # PLIP/OpenBabel report 1-based atom-order indices, not arbitrary input PDB serials.
        # Make both coincide, and retain source serials for auditing.
        serial_map = {serial: i+1 for i, serial in enumerate(source_serials)}
        receptor_atoms = [line[:6] + f"{i+1:5d}" + line[11:] for i, line in enumerate(receptor_atoms)]
        connectivity = []
        for line in protein:
            if line.startswith("CONECT"):
                ids = [int(line[i:i+5]) for i in range(6, len(line), 5) if line[i:i+5].strip()]
                if ids and all(i in serial_map for i in ids):
                    connectivity.append("CONECT" + "".join(f"{serial_map[i]:5d}" for i in ids))
        protein = receptor_atoms + connectivity
        offset = len(receptor_atoms)
        if offset + molecule.GetNumAtoms() > 99999:
            raise ValueError("Complex exceeds PDB serial range")
        for atom in molecule.GetAtoms():
            info = Chem.AtomPDBResidueInfo()
            info.SetName(f"{atom.GetSymbol()}{atom.GetIdx()+1}".rjust(4)[:4])
            info.SetResidueName("LIG")
            info.SetChainId("Z")
            info.SetResidueNumber(1)
            info.SetIsHeteroAtom(True)
            atom.SetMonomerInfo(info)
        ligand_lines = []
        mapping = []
        for line in Chem.MolToPDBBlock(molecule).splitlines():
            if line.startswith("HETATM"):
                serial = int(line[6:11])
                ligand_lines.append(line[:6] + f"{offset+serial:5d}" + line[11:])
                mapping.append({"pose_atom_index": serial-1, "complex_serial": offset+serial})
            elif line.startswith("CONECT"):
                indices = [int(line[i:i+5]) for i in range(6, len(line), 5) if line[i:i+5].strip()]
                ligand_lines.append("CONECT" + "".join(f"{offset+i:5d}" for i in indices))
        # PLIP's PDBParser increments its original-ID map at TER even with --nofix.
        # Omit TER so reported IDs, PDB serials and OpenBabel atom order coincide.
        # The separate chain/residue identifiers already delimit the ligand.
        destination.write_text("\n".join(protein + ligand_lines + ["END", ""]))
        receptor_mapping = [{"complex_serial": int(line[6:11]), "source_serial": source_serials[i],
                             "atom_name": line[12:16].strip(),
                             "residue": f"{line[17:20].strip()}:{line[21:22].strip()}:{int(line[22:26])}",
                             "insertion_code": line[26:27].strip()} for i, line in enumerate(receptor_atoms)]
        return {"pose_rank": pose_rank, "ligand_id": "LIG:Z:1", "atom_mapping": mapping,
                "receptor_atom_mapping": receptor_mapping,
                "coordinate_precision": "PDB 0.001 angstrom; no optimization",
                "hydrogen_policy": "preserve supplied atoms; PLIP --nohydro --nofix",
                "explicit_hydrogen_count": sum(a.GetAtomicNum() == 1 for a in molecule.GetAtoms()),
                "warning": "No hydrogens added. Missing input hydrogens may reduce H-bond detection; PDB bond perception may differ from SDF."}

    @staticmethod
    def parse_report(path: Path) -> dict[str, Any]:
        root = ET.parse(path).getroot()
        records = []
        found = False
        for site in root.findall(".//bindingsite"):
            ids = site.find("identifiers")
            if ids is None or ids.findtext("hetid") != "LIG" or ids.findtext("chain") != "Z" or ids.findtext("position") != "1":
                continue
            found = True
            interactions = site.find("interactions")
            if interactions is None:
                continue
            for group in interactions:
                for interaction in group:
                    fields = {node.tag: node.text for node in interaction if not list(node)}
                    records.append({"type": interaction.tag,
                                    "residue": f"{fields.get('restype')}:{fields.get('reschain')}:{fields.get('resnr')}",
                                    "fields": fields})
        if not found:
            raise ValueError("PLIP report does not contain selected ligand LIG:Z:1")
        counts: dict[str, int] = {}
        for record in records:
            counts[record["type"]] = counts.get(record["type"], 0) + 1
        return {"records": records, "counts": counts,
                "interaction_keys": sorted({r['type'] + '|' + r['residue'] for r in records})}

    def run(self, receptor: Path, poses: Path, output_dir: Path, pose_rank: int = 1) -> dict[str, Any]:
        if not self.config.get("enabled", False):
            return {"status": "disabled"}
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {"status": "failed", "pose_rank": pose_rank}
        try:
            complex_path = output_dir / "evaluated-complex.pdb"
            result["preparation"] = self.build_complex(receptor, poses, complex_path, pose_rank)
            report = output_dir / "report.xml"
            report.unlink(missing_ok=True)
            command = [str(self.config.get("executable", "plip")), "-f", str(complex_path),
                       "-o", str(output_dir), "-x", "--name", "report", "--nohydro", "--nofix",
                       "--maxthreads", "1"]
            result.update(command=command, complex_path=str(complex_path), report_path=str(report))
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=float(self.config.get("timeout_seconds", 900)))
            (output_dir / "stdout.txt").write_text(completed.stdout)
            (output_dir / "stderr.txt").write_text(completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"PLIP exit {completed.returncode}; see stderr.txt")
            result.update(self.parse_report(report), status="complete")
            # Full serial provenance is an artifact, not thousands of repeated context rows.
            preparation = result["preparation"]
            mapping_path = output_dir / "atom-mapping.json"
            mapping_path.write_text(json.dumps({key: preparation[key] for key in
                                               ("atom_mapping", "receptor_atom_mapping")}, indent=2))
            involved = set()
            for record in result["records"]:
                for key, value in record["fields"].items():
                    if key.endswith("idx") and str(value).isdigit():
                        involved.add(int(value))
            preparation["receptor_atom_mapping"] = [row for row in preparation["receptor_atom_mapping"]
                                                    if row["complex_serial"] in involved]
            preparation["atom_mapping_path"] = str(mapping_path)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, ET.ParseError) as error:
            result["error"] = str(error)
        (output_dir / "interaction-result.json").write_text(json.dumps(result, indent=2))
        return result


def compare_plip(candidate: dict, reference: dict) -> dict:
    if candidate.get("status") != "complete" or reference.get("status") != "complete":
        return {"status": "unavailable", "candidate_status": candidate.get("status"),
                "reference_status": reference.get("status")}
    c, r = set(candidate["interaction_keys"]), set(reference["interaction_keys"])
    return {"status": "complete", "candidate_pose_rank": candidate.get("pose_rank", 1),
            "reference_pose_rank": reference.get("pose_rank", 1), "candidate_counts": candidate["counts"],
            "reference_counts": reference["counts"], "retained": sorted(c & r),
            "gained": sorted(c-r), "lost": sorted(r-c),
            "candidate_report": candidate["report_path"], "reference_report": reference["report_path"],
            "limitation": "Type/residue comparison, not atom-level equivalence or affinity. No added hydrogens."}
