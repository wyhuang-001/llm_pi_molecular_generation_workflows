"""Read-only offline audit of an old run's ALL top-N poses; no GNINA/LLM calls.

Use the OLD receptor with OLD poses/scores, never reinterpret them as corrected-TPO
results. Output is diagnostic only; a new prepared-receptor calibration is still needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem
from molecular_agent.adapters import DockingAdapter
from molecular_agent.pose_retention import PoseRetention, read_one, save_json


class OfflineAdapter(DockingAdapter):
    def __init__(self, config: dict, output: Path, source: Path, native: Path):
        super().__init__(config, output)
        self.source, self.native = source, native

    def _run_with_retries(self, **values):
        candidate, seed = Path(values["candidate_path"]), values["seed"]
        folder = "docking-reference-baseline" if candidate == self.native else f"docking-attempt-{int(candidate.stem.split('-')[-1]):02d}"
        path = self.source / folder / f"seed-{seed:05d}" / "docked.sdf"
        if not path.is_file():
            return {"status": "failed", "error": f"Missing legacy output: {path}"}
        return {"status": "complete", "pose_path": str(path.resolve()), "diagnostic_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task", type=Path, default=Path("input/task.single_edit.json"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidates", nargs="*", type=int, default=[28, 23, 19])
    args = parser.parse_args()
    source, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory must be new; old evidence is never overwritten")
    output.mkdir(parents=True)
    native = output / "prepared-crystal.sdf"
    with Chem.SDWriter(str(native)) as writer:
        writer.write(Chem.AddHs(read_one(source / "reference-ligand.sdf"), addCoords=True))
    receptor = output / "legacy-receptor.pdb"
    shutil.copyfile(source / "receptor-protein-only.pdb", receptor)
    protocol = json.loads(args.task.read_text())["pose_retention"]
    seeds = sorted(int(path.name.split('-')[-1]) for path in (source / "docking-reference-baseline").glob("seed-*"))
    adapter = OfflineAdapter({"enabled": True, "seeds": seeds, "pose_retention": protocol,
                              "plip": {"enabled": True}, "diagnostic_only": True}, output, source, native)
    evaluator = PoseRetention(adapter)
    calibration = evaluator.prepare(native, receptor, output / "reference-calibration")
    summary = {"diagnostic_only": True, "source_run": str(source),
               "warning": "Old receptor/old scores (TPO omitted). Not a production baseline for corrected preparation.",
               "calibration_status": calibration["status"], "error": calibration.get("error"),
               "reference_valid_seeds": calibration.get("valid_seeds"), "reference": [], "candidates": {}}
    for evaluation in calibration.get("per_seed", []):
        summary["reference"].append({"seed": evaluation["seed"], "status": evaluation["status"],
                                     "eligible_ranks": [row["rank"] for row in evaluation.get("poses", []) if row["status"] == "eligible"],
                                     "best_crystal_core_rmsd": min((row.get("crystal_core_rmsd", float('inf'))
                                                                     for row in evaluation.get("poses", [])), default=None)})
    if calibration["status"] == "complete":
        summary["geometry_reference"] = {k: calibration["geometry_reference"][k] for k in ("seed", "rank", "crystal_core_rmsd")}
        for attempt in args.candidates:
            result = evaluator.run_candidate(source / f"candidate-{attempt:02d}.sdf", native, receptor,
                                             output / f"candidate-{attempt:02d}", calibration)
            summary["candidates"][str(attempt)] = {k: result.get(k) for k in
                ("status", "paired_seeds", "outlier_or_failed_seeds", "comparison", "candidate_per_seed")}
    save_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "candidates"}, indent=2))
    for attempt, result in summary["candidates"].items():
        print("candidate", attempt, result["status"], "paired seeds", result["paired_seeds"],
              "mean delta", result["comparison"].get("metrics", {}).get("minimizedAffinity", {}).get("delta_candidate_minus_reference"))
    print("Audit:", output / "summary.json")


if __name__ == "__main__":
    main()
