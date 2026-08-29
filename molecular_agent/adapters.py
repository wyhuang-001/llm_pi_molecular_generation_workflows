from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable


class NotConfiguredAdapter:
    def __init__(self, stage: str, reason: str | None = None):
        self.stage = stage
        self.reason = reason

    def run(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": "not_configured",
            "message": self.reason
            or f"{self.stage} adapter is not configured or its dependencies are unavailable.",
        }


class CommandAdapter:
    """Run an explicitly configured external scientific command and audit its execution."""

    def __init__(
        self,
        stage: str,
        config: dict[str, Any],
        run_dir: Path,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.stage = stage
        self.config = config
        self.run_dir = run_dir
        self.progress = progress

    def _not_configured(self, reason: str) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": "not_configured",
            "message": reason,
        }

    def run(self, **values: Any) -> dict[str, Any]:
        if self.config.get("enabled") is not True:
            return self._not_configured(
                "Adapter is disabled. Set enabled=true only after installing and validating dependencies."
            )
        command_template = self.config.get("command")
        if not isinstance(command_template, list) or not command_template:
            return self._not_configured("No command list configured.")
        command = [str(item).format(**values) for item in command_template]
        executable = shutil.which(command[0])
        if executable is None and not Path(command[0]).is_file():
            return self._not_configured(f"Executable not found: {command[0]}")
        output_dir = Path(values.get("output_dir", self.run_dir)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        audit = {
            "stage": self.stage,
            "status": "running",
            "command": command,
            "command_display": shlex.join(command),
            "cwd": str(output_dir),
            "inputs": {key: str(value) for key, value in values.items() if key != "output_dir"},
        }
        (output_dir / f"{self.stage}-command.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self.progress:
            self.progress(f"{self.stage}_command_started", {
                "command": audit["command_display"],
                "output_dir": str(output_dir),
            })
        try:
            result = subprocess.run(
                command,
                cwd=output_dir,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                check=False,
                timeout=float(self.config.get("timeout_seconds", 86400)),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            audit.update({"status": "failed", "error": str(exc)})
            self._write_audit(output_dir, audit)
            return audit
        audit.update({
            "status": "complete" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout[-10000:],
            "stderr": result.stderr[-10000:],
        })
        self._write_audit(output_dir, audit)
        if self.progress:
            self.progress(f"{self.stage}_command_completed", {
                "status": audit["status"],
                "returncode": result.returncode,
                "output_dir": str(output_dir),
            })
        return audit

    @staticmethod
    def _write_audit(output_dir: Path, audit: dict[str, Any]) -> None:
        (output_dir / f"{audit['stage']}-result.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class DockingAdapter(CommandAdapter):
    SCORE_METRICS = {
        "minimizedAffinity": "lower_is_better",
        "CNNscore": "higher_is_better",
        "CNNaffinity": "higher_is_better",
        "CNN_VS": "higher_is_better",
    }

    def __init__(
        self,
        config: dict[str, Any],
        run_dir: Path,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        super().__init__("docking", config, run_dir, progress=progress)
        self._reference_results: dict[int, dict[str, Any]] = {}

    def _seeds(self) -> list[int]:
        configured = self.config.get("seeds", [self.config.get("seed", 17)])
        if not isinstance(configured, list) or not configured:
            raise ValueError("docking.seeds must be a non-empty list of integers")
        seeds = [int(seed) for seed in configured]
        if len(set(seeds)) != len(seeds):
            raise ValueError("docking.seeds must not contain duplicates")
        return seeds

    @staticmethod
    def _readable_3d_sdf(path: Path, label: str) -> tuple[bool, str]:
        if not path.is_file():
            return False, f"{label} SDF does not exist: {path}"
        try:
            from rdkit import Chem

            molecules = list(Chem.SDMolSupplier(str(path), removeHs=False))
        except Exception as exc:
            return False, f"{label} SDF could not be read: {exc}"
        readable = [molecule for molecule in molecules if molecule is not None]
        if len(readable) != 1:
            return False, f"{label} SDF must contain exactly one readable molecule; found {len(readable)}"
        if readable[0].GetNumConformers() < 1:
            return False, f"{label} SDF has no 3D conformer"
        conformer = readable[0].GetConformer()
        if not conformer.Is3D():
            return False, f"{label} SDF conformer is not marked as 3D"
        return True, "ok"

    @classmethod
    def compare_results(
        cls,
        candidate_result: dict[str, Any],
        reference_result: dict[str, Any],
        pose_rank: int = 1,
    ) -> dict[str, Any]:
        """Compare the same GNINA score fields at a selected pose rank."""
        candidate_poses = candidate_result.get("poses") or []
        reference_poses = reference_result.get("poses") or []
        candidate_pose = next(
            (pose for pose in candidate_poses if pose.get("rank") == pose_rank), None
        )
        reference_pose = next(
            (pose for pose in reference_poses if pose.get("rank") == pose_rank), None
        )
        comparison: dict[str, Any] = {
            "status": "complete",
            "pose_rank": pose_rank,
            "candidate_pose": candidate_pose,
            "reference_pose": reference_pose,
            "metrics": {},
            "meaning": (
                "Deltas are candidate minus reference from independent docking runs "
                "with the same receptor, autobox and configured GNINA protocol. "
                "They are not experimental affinity or activity predictions."
            ),
        }
        if candidate_pose is None or reference_pose is None:
            comparison.update(
                {
                    "status": "failed",
                    "failure_class": "reference_score_comparison",
                    "error": f"Pose rank {pose_rank} is missing from candidate or reference docking output",
                }
            )
            return comparison

        candidate_properties = candidate_pose.get("properties") or {}
        reference_properties = reference_pose.get("properties") or {}
        for metric, direction in cls.SCORE_METRICS.items():
            candidate_value = candidate_properties.get(metric)
            reference_value = reference_properties.get(metric)
            if candidate_value is None or reference_value is None:
                continue
            try:
                candidate_number = float(candidate_value)
                reference_number = float(reference_value)
            except (TypeError, ValueError):
                continue
            delta = candidate_number - reference_number
            comparison["metrics"][metric] = {
                "candidate": candidate_number,
                "reference": reference_number,
                "delta_candidate_minus_reference": delta,
                "direction": direction,
                "candidate_better_by_metric": (
                    delta < 0 if direction == "lower_is_better" else delta > 0
                ),
            }
        if not comparison["metrics"]:
            comparison.update(
                {
                    "status": "failed",
                    "failure_class": "reference_score_comparison",
                    "error": "No comparable numeric GNINA score fields were found",
                }
            )
        return comparison

    @staticmethod
    def _receptor_preflight(path: Path) -> tuple[bool, str]:
        if not path.is_file():
            return False, f"receptor PDB does not exist: {path}"
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return False, f"receptor PDB could not be read: {exc}"
        atom_lines = [line for line in lines if line[:6].strip() == "ATOM"]
        hetero_lines = [line for line in lines if line[:6].strip() == "HETATM"]
        if not atom_lines:
            return False, "receptor PDB contains no ATOM records"
        if hetero_lines:
            return False, "receptor PDB contains HETATM records; use protein-only receptor output"
        return True, "ok"

    def _preflight(
        self,
        candidate_path: Path,
        receptor_path: Path,
        reference_path: Path | None,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        checks = {}
        checks["candidate_sdf"] = self._readable_3d_sdf(candidate_path, "candidate")
        checks["reference_sdf"] = (
            self._readable_3d_sdf(reference_path, "reference")
            if reference_path
            else (False, "reference ligand is required for autobox docking")
        )
        checks["receptor_pdb"] = self._receptor_preflight(receptor_path)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            probe = output_dir / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            checks["output_directory"] = (True, "ok")
        except OSError as exc:
            checks["output_directory"] = (False, f"output directory is not writable: {exc}")
        failures = {name: message for name, (ok, message) in checks.items() if not ok}
        if failures:
            return {
                "stage": "docking",
                "status": "failed",
                "failure_class": "docking_input_preflight",
                "checks": {name: {"status": "passed" if ok else "failed", "message": message} for name, (ok, message) in checks.items()},
                "error": "Docking input preflight failed",
                "failures": failures,
            }
        return None

    def run(
        self,
        candidate_path: Path,
        receptor_path: Path,
        reference_path: Path | None = None,
        output_dir: Path | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        output_dir = (output_dir or (self.run_dir / "docking")).resolve()
        if self.config.get("enabled") is True:
            preflight = self._preflight(candidate_path, receptor_path, reference_path, output_dir)
            if preflight is not None:
                self._write_audit(output_dir, preflight)
                return preflight
        run_seed = int(seed if seed is not None else self.config.get("seed", 17))
        result = super().run(
            candidate=str(candidate_path.resolve()),
            receptor=str(receptor_path.resolve()),
            reference=str(reference_path.resolve()) if reference_path else "",
            output_dir=str(output_dir),
            seed=run_seed,
        )
        output_path = output_dir / str(self.config.get("output_filename", "docked.sdf"))
        if result.get("status") == "complete" and output_path.is_file():
            try:
                from rdkit import Chem

                supplier = Chem.SDMolSupplier(str(output_path), removeHs=False)
                poses = []
                for rank, molecule in enumerate(supplier, start=1):
                    if molecule is None:
                        continue
                    properties = {
                        key: molecule.GetProp(key)
                        for key in molecule.GetPropNames()
                    }
                    poses.append({"rank": rank, "properties": properties})
                result["pose_path"] = str(output_path)
                result["pose_count"] = len(poses)
                result["poses"] = poses[:20]
                result["pose_selection"] = self._pose_selection_summary(poses)
                if not poses:
                    result["status"] = "failed"
                    result["error"] = f"Docking output contains no readable poses: {output_path}"
            except Exception as exc:
                result["status"] = "failed"
                result["postprocess_error"] = str(exc)
        elif result.get("status") == "complete":
            result["status"] = "failed"
            result["error"] = f"Docking completed without output pose file: {output_path}"
        if result.get("status") != "not_configured":
            self._write_audit(output_dir, result)
        return result

    @classmethod
    def _pose_selection_summary(cls, poses: list[dict[str, Any]]) -> dict[str, Any]:
        selections: dict[str, Any] = {
            "gnina_rank_1": poses[0] if poses else None,
            "meaning": (
                "GNINA rank 1 follows pose/CNN ordering and need not have the most favorable "
                "minimizedAffinity. Metric-specific alternatives are reported for audit."
            ),
        }
        for metric, direction in cls.SCORE_METRICS.items():
            numeric = []
            for pose in poses:
                try:
                    value = float((pose.get("properties") or {}).get(metric))
                except (TypeError, ValueError):
                    continue
                numeric.append((value, pose))
            if not numeric:
                continue
            selected = (
                min(numeric, key=lambda item: item[0])
                if direction == "lower_is_better"
                else max(numeric, key=lambda item: item[0])
            )
            selections[f"best_by_{metric}"] = {
                "rank": selected[1].get("rank"),
                "value": selected[0],
                "properties": selected[1].get("properties") or {},
            }
        return selections

    @staticmethod
    def _top_pose_consensus(
        results: dict[int, dict[str, Any]], rmsd_threshold: float = 2.0
    ) -> dict[str, Any]:
        import numpy as np
        from rdkit import Chem

        coordinates: dict[int, np.ndarray] = {}
        for seed, result in results.items():
            path = result.get("pose_path")
            if not path:
                continue
            molecule = next(
                (item for item in Chem.SDMolSupplier(str(path), removeHs=False) if item is not None),
                None,
            )
            if molecule is None or molecule.GetNumConformers() < 1:
                continue
            conformer = molecule.GetConformer()
            heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
            coordinates[seed] = np.array([
                list(conformer.GetAtomPosition(index)) for index in heavy
            ])
        pairs = []
        adjacency = {seed: {seed} for seed in coordinates}
        seeds = sorted(coordinates)
        for position, left in enumerate(seeds):
            for right in seeds[position + 1:]:
                left_xyz, right_xyz = coordinates[left], coordinates[right]
                if left_xyz.shape != right_xyz.shape:
                    continue
                rmsd = float(np.sqrt(np.mean(np.sum((left_xyz - right_xyz) ** 2, axis=1))))
                pairs.append({"seed_pair": [left, right], "heavy_atom_rmsd": round(rmsd, 3)})
                if rmsd <= rmsd_threshold:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
        visited = set()
        component_sizes = []
        for seed in seeds:
            if seed in visited:
                continue
            stack = [seed]
            component = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(adjacency[current] - component)
            visited.update(component)
            component_sizes.append(len(component))
        largest_cluster = max(component_sizes, default=0)
        values = [item["heavy_atom_rmsd"] for item in pairs]
        all_pairs_consistent = bool(seeds) and len(pairs) == len(seeds) * (len(seeds) - 1) // 2 and all(
            value <= rmsd_threshold for value in values
        )
        return {
            "pose_rank": 1,
            "coordinate_frame": "shared_receptor_frame_without_post_alignment",
            "seed_count": len(seeds),
            "rmsd_threshold": rmsd_threshold,
            "pairwise_heavy_atom_rmsd": pairs,
            "mean_pairwise_rmsd": round(float(np.mean(values)), 3) if values else None,
            "max_pairwise_rmsd": max(values) if values else None,
            "largest_consistent_cluster_size": largest_cluster,
            "largest_consistent_cluster_fraction": (
                largest_cluster / len(seeds) if seeds else None
            ),
            "all_pairs_consistent": all_pairs_consistent,
            "stable": all_pairs_consistent,
            "limitation": (
                "This checks whether rank-1 poses occupy a similar geometry across seeds; it does "
                "not prove that the pose is experimentally correct."
            ),
        }

    @staticmethod
    def _contact_consensus(
        candidate_results: dict[int, dict[str, Any]],
        reference_results: dict[int, dict[str, Any]],
        receptor_path: Path,
        cutoff: float = 4.0,
    ) -> dict[str, Any]:
        import numpy as np
        from rdkit import Chem

        protein_rows = []
        for line in receptor_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line[:6].strip() != "ATOM":
                continue
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except ValueError:
                continue
            residue = f"{line[17:20].strip()}:{line[21:22].strip()}:{line[22:26].strip()}"
            protein_rows.append((residue, xyz))
        if not protein_rows:
            return {"status": "unavailable", "error": "No readable receptor ATOM coordinates"}
        protein_xyz = np.array([row[1] for row in protein_rows])
        protein_residues = [row[0] for row in protein_rows]

        def contacts(results: dict[int, dict[str, Any]]) -> dict[int, list[str]]:
            per_seed = {}
            for seed, result in results.items():
                path = result.get("pose_path")
                if not path:
                    continue
                molecule = next(
                    (item for item in Chem.SDMolSupplier(str(path), removeHs=False) if item is not None),
                    None,
                )
                if molecule is None:
                    continue
                conformer = molecule.GetConformer()
                ligand_xyz = np.array([
                    list(conformer.GetAtomPosition(atom.GetIdx()))
                    for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
                ])
                close = np.any(
                    np.sum((ligand_xyz[:, None, :] - protein_xyz[None, :, :]) ** 2, axis=2)
                    <= cutoff ** 2,
                    axis=0,
                )
                per_seed[seed] = sorted({
                    protein_residues[index] for index, is_close in enumerate(close) if is_close
                })
            return per_seed

        candidate = contacts(candidate_results)
        reference = contacts(reference_results)
        minimum_count = max(1, len(candidate) // 2 + 1)

        def consensus(per_seed: dict[int, list[str]]) -> list[str]:
            counts: dict[str, int] = {}
            for residues in per_seed.values():
                for residue in residues:
                    counts[residue] = counts.get(residue, 0) + 1
            return sorted(residue for residue, count in counts.items() if count >= minimum_count)

        candidate_consensus = consensus(candidate)
        reference_consensus = consensus(reference)
        return {
            "status": "complete",
            "cutoff": cutoff,
            "minimum_seed_count_for_consensus": minimum_count,
            "candidate_per_seed": candidate,
            "reference_per_seed": reference,
            "candidate_consensus_residues": candidate_consensus,
            "reference_consensus_residues": reference_consensus,
            "gained_consensus_residues": sorted(set(candidate_consensus) - set(reference_consensus)),
            "lost_consensus_residues": sorted(set(reference_consensus) - set(candidate_consensus)),
            "limitation": (
                "Residue contacts are distance-only and do not assign hydrogen-bond directionality "
                "or energetic contribution."
            ),
        }

    @classmethod
    def aggregate_comparisons(
        cls, comparisons: list[dict[str, Any]], seeds: list[int], pose_rank: int = 1
    ) -> dict[str, Any]:
        complete = [item for item in comparisons if item.get("status") == "complete"]
        result: dict[str, Any] = {
            "status": "complete" if len(complete) == len(seeds) else "failed",
            "pose_rank": pose_rank,
            "seed_count": len(seeds),
            "completed_seed_count": len(complete),
            "seeds": seeds,
            "per_seed": comparisons,
            "metrics": {},
            "meaning": (
                "Each seed compares candidate minus reference from paired independent GNINA runs. "
                "Summary statistics describe sampling stability; they are not experimental affinity "
                "or activity predictions."
            ),
        }
        if len(complete) != len(seeds):
            result.update(
                {
                    "failure_class": "reference_score_comparison",
                    "error": "At least one seed did not produce a complete candidate/reference comparison.",
                }
            )
            return result

        for metric, direction in cls.SCORE_METRICS.items():
            values = [
                item["metrics"][metric]["delta_candidate_minus_reference"]
                for item in complete
                if metric in item.get("metrics", {})
            ]
            if not values:
                continue
            better_count = sum(
                item["metrics"][metric]["candidate_better_by_metric"]
                for item in complete
                if metric in item.get("metrics", {})
            )
            result["metrics"][metric] = {
                "direction": direction,
                "n": len(values),
                "delta_candidate_minus_reference": {
                    "mean": mean(values),
                    "stddev": stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                },
                "candidate_better_seed_count": better_count,
                "candidate_better_seed_fraction": better_count / len(values),
            }
        if not result["metrics"]:
            result.update(
                {
                    "status": "failed",
                    "failure_class": "reference_score_comparison",
                    "error": "No comparable numeric GNINA score fields were found across seeds.",
                }
            )
        return result

    def run_with_reference_baseline(
        self,
        candidate_path: Path,
        receptor_path: Path,
        reference_path: Path,
        output_dir: Path,
        reference_output_dir: Path,
        reference_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run paired reference/candidate docking for every configured seed."""
        if self.config.get("enabled") is not True:
            return self.run(
                candidate_path=candidate_path,
                receptor_path=receptor_path,
                reference_path=reference_path,
                output_dir=output_dir,
            )

        seeds = self._seeds()
        reference_results: dict[int, dict[str, Any]] = {}
        candidate_results: dict[int, dict[str, Any]] = {}
        comparisons = []
        for seed in seeds:
            reference_dir = reference_output_dir / f"seed-{seed:05d}"
            candidate_dir = output_dir / f"seed-{seed:05d}"
            baseline = self._reference_results.get(seed)
            if baseline is None:
                baseline = self.run(
                    candidate_path=reference_path,
                    receptor_path=receptor_path,
                    reference_path=reference_path,
                    output_dir=reference_dir,
                    seed=seed,
                )
                if baseline.get("status") == "complete":
                    self._reference_results[seed] = baseline
            reference_results[seed] = baseline
            if baseline.get("status") != "complete":
                comparisons.append(
                    {
                        "seed": seed,
                        "status": "failed",
                        "failure_class": "reference_docking_baseline",
                        "error": "Reference docking baseline failed.",
                    }
                )
                continue
            candidate = self.run(
                candidate_path=candidate_path,
                receptor_path=receptor_path,
                reference_path=reference_path,
                output_dir=candidate_dir,
                seed=seed,
            )
            candidate_results[seed] = candidate
            comparison = self.compare_results(candidate, baseline, pose_rank=1)
            comparison["seed"] = seed
            comparisons.append(comparison)

        aggregate = self.aggregate_comparisons(comparisons, seeds, pose_rank=1)
        failed_reference = [seed for seed, item in reference_results.items() if item.get("status") != "complete"]
        failed_candidate = [seed for seed, item in candidate_results.items() if item.get("status") != "complete"]
        first_candidate = next((candidate_results[seed] for seed in seeds if seed in candidate_results), {})
        result = dict(first_candidate)
        result["status"] = "complete" if aggregate["status"] == "complete" else "failed"
        result["seed_count"] = len(seeds)
        result["seeds"] = seeds
        result["pose_count"] = next(
            (int(candidate_results[seed].get("pose_count", 0)) for seed in seeds if seed in candidate_results),
            0,
        )
        result["pose_count_per_seed"] = {
            str(seed): int(candidate_results[seed].get("pose_count", 0))
            for seed in candidate_results
        }
        result["total_pose_count"] = sum(result["pose_count_per_seed"].values())
        result["reference_baseline"] = {
            "stage": "docking_reference_baseline",
            "status": "complete" if not failed_reference else "failed",
            "seed_count": len(seeds) - len(failed_reference),
            "seeds": seeds,
            "failed_seeds": failed_reference,
            "per_seed": {
                str(seed): {
                    "status": reference_results[seed].get("status"),
                    "pose_path": reference_results[seed].get("pose_path"),
                    "pose_count": reference_results[seed].get("pose_count"),
                    "top_pose": (reference_results[seed].get("poses") or [None])[0],
                    "pose_selection": reference_results[seed].get("pose_selection"),
                    "audit_path": str((reference_output_dir / f"seed-{seed:05d}" / "docking-result.json").resolve()),
                }
                for seed in seeds
            },
        }
        result["candidate_per_seed"] = {
            str(seed): {
                "status": candidate_results[seed].get("status"),
                "pose_path": candidate_results[seed].get("pose_path"),
                "pose_count": candidate_results[seed].get("pose_count"),
                "top_pose": (candidate_results[seed].get("poses") or [None])[0],
                "pose_selection": candidate_results[seed].get("pose_selection"),
                "audit_path": str((output_dir / f"seed-{seed:05d}" / "docking-result.json").resolve()),
            }
            for seed in candidate_results
        }
        result["comparison"] = aggregate
        result["pose_consensus"] = self._top_pose_consensus(
            candidate_results,
            rmsd_threshold=float(self.config.get("pose_consensus_rmsd_threshold", 2.0)),
        )
        result["interaction_consensus"] = self._contact_consensus(
            candidate_results,
            reference_results,
            receptor_path,
            cutoff=float(self.config.get("interaction_contact_cutoff", 4.0)),
        )
        if failed_candidate:
            result["failure_class"] = "candidate_docking"
        elif failed_reference:
            result["failure_class"] = "reference_docking_baseline"
        self._write_audit(output_dir, result)
        return result


class RBFEAdapter(CommandAdapter):
    def __init__(
        self,
        config: dict[str, Any],
        run_dir: Path,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        super().__init__("rbfe", config, run_dir, progress=progress)

    def run(
        self,
        candidate_path: Path,
        receptor_path: Path,
        reference_path: Path,
        **_: Any,
    ) -> dict[str, Any]:
        if self.config.get("enabled") is not True:
            return self._not_configured(
                "RBFE is disabled. Set enabled=true only after installing AsyncFEP dependencies and configuring GPUs."
            )
        target = str(self.config.get("target", "current_target"))
        output_dir = self.run_dir / "rbfe"
        output_dir.mkdir(parents=True, exist_ok=True)
        targets_path = output_dir / "reference-target.yaml"
        target_data = {
            "targets": {
                target: {
                    "receptor": str(receptor_path.resolve()),
                    "reference": str(reference_path.resolve()),
                    "reference_name": "co_crystal_reference",
                    "ff_scheme": self.config.get("ff_scheme", "ff14sb-gaff2-tip3p"),
                }
            }
        }
        targets_path.write_text(
            json.dumps(target_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        command = self.config.get("command")
        if command is None:
            script = self.config.get("script")
            if not script:
                return self._not_configured("RBFE requires command or AsyncFEP script configuration.")
            python = str(self.config.get("python", "python"))
            command = [python, str(script), "--stage", str(self.config.get("stage", "all"))]
            command += ["--target", "{target}", "--targets", "{targets}"]
            command += ["--output", "{output_dir}", "--ligands", "{candidate}"]
            gpus = self.config.get("gpus", [])
            if gpus:
                command += ["--gpus", *[str(gpu) for gpu in gpus]]
        configured = dict(self.config)
        configured["command"] = command
        runner = CommandAdapter("rbfe", configured, self.run_dir)
        return runner.run(
            candidate=str(candidate_path.resolve()),
            reference=str(reference_path.resolve()),
            receptor=str(receptor_path.resolve()),
            output_dir=str(output_dir),
            target=target,
            targets=str(targets_path.resolve()),
        )


def configured_adapters(
    config_path: Path,
    run_dir: Path,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[CommandAdapter, CommandAdapter]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    docking_config = config.get("docking") or {}
    rbfe_config = config.get("rbfe") or {}
    docking = (
        DockingAdapter(docking_config, run_dir, progress=progress)
        if docking_config
        else NotConfiguredAdapter("docking")
    )
    rbfe = (
        RBFEAdapter(rbfe_config, run_dir, progress=progress)
        if rbfe_config
        else NotConfiguredAdapter("rbfe")
    )
    return docking, rbfe
