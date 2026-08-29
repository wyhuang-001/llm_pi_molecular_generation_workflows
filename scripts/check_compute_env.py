#!/usr/bin/env python3
"""Validate the shared molecular design, docking, and AsyncFEP environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportCheck:
    module: str
    label: str
    distribution: str | None = None


IMPORTS = (
    ImportCheck("numpy", "NumPy"),
    ImportCheck("scipy", "SciPy"),
    ImportCheck("rdkit", "RDKit"),
    ImportCheck("gemmi", "Gemmi"),
    ImportCheck("pydantic", "Pydantic"),
    ImportCheck("openbabel", "Open Babel"),
    ImportCheck("vina", "AutoDock Vina"),
    ImportCheck("meeko", "Meeko"),
    ImportCheck("prody", "ProDy"),
    ImportCheck("MDAnalysis", "MDAnalysis"),
    ImportCheck("prolif", "ProLIF"),
    ImportCheck("spyrmsd", "spyrmsd"),
    ImportCheck("freesasa", "FreeSASA", "freesasa-python"),
    ImportCheck("openmm", "OpenMM"),
    ImportCheck("yaml", "PyYAML", "pyyaml"),
    ImportCheck("matplotlib", "Matplotlib"),
    ImportCheck("pandas", "pandas"),
    ImportCheck("pyarrow", "PyArrow"),
    ImportCheck("parmed", "ParmEd"),
    ImportCheck("pymbar", "pymbar"),
    ImportCheck("jax", "JAX"),
    ImportCheck("jaxlib", "JAXlib"),
    ImportCheck("duckdb", "DuckDB"),
    ImportCheck("click", "Click"),
    ImportCheck("pynvml", "nvidia-ml-py", "nvidia-ml-py"),
    ImportCheck("psutil", "psutil"),
    ImportCheck("openff.toolkit", "OpenFF Toolkit", "openff-toolkit"),
    ImportCheck("openmmforcefields", "OpenMMForceFields"),
    ImportCheck("pdbfixer", "PDBFixer"),
    ImportCheck(
        "molecular_agent", "simple-molecular-agent", "simple-molecular-agent"
    ),
)

ASYNCFEP_IMPORTS = (
    ImportCheck("prepare", "bloom-prepare", "bloom-prepare"),
)

COMMANDS = (
    ("curl", "HTTP client"),
    ("obabel", "Open Babel CLI"),
    ("vina", "AutoDock Vina CLI"),
    ("mk_prepare_ligand.py", "Meeko ligand preparation"),
    ("mk_prepare_receptor.py", "Meeko receptor preparation"),
    ("antechamber", "AmberTools antechamber"),
    ("parmchk2", "AmberTools parmchk2"),
    ("tleap", "AmberTools tleap"),
    ("pdb4amber", "AmberTools pdb4amber"),
    ("reduce", "AmberTools Reduce"),
    ("xtb", "xTB"),
    ("molecular-agent", "molecular-agent CLI"),
)

ASYNCFEP_COMMANDS = (("bloom", "AsyncFEP/Bloom CLI"),)


class Reporter:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def pass_(self, label: str, detail: str = "") -> None:
        suffix = f" - {detail}" if detail else ""
        print(f"PASS  {label}{suffix}")

    def fail(self, label: str, detail: str) -> None:
        self.failures.append(f"{label}: {detail}")
        print(f"FAIL  {label} - {detail}")

    def warn(self, label: str, detail: str) -> None:
        self.warnings.append(f"{label}: {detail}")
        print(f"WARN  {label} - {detail}")


def package_version(check: ImportCheck, module: object) -> str:
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    distribution = check.distribution or check.module.split(".", 1)[0]
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "version unknown"


def check_python(reporter: Reporter) -> None:
    version = sys.version_info
    detail = f"{version.major}.{version.minor}.{version.micro} ({sys.executable})"
    if version[:2] == (3, 11):
        reporter.pass_("Python", detail)
    else:
        reporter.fail("Python", f"requires 3.11, found {detail}")


def check_imports(
    reporter: Reporter, checks: tuple[ImportCheck, ...]
) -> dict[str, object]:
    loaded: dict[str, object] = {}
    for check in checks:
        try:
            module = importlib.import_module(check.module)
        except Exception as exc:  # import-time binary failures matter here
            reporter.fail(check.label, f"cannot import {check.module}: {exc}")
            continue
        loaded[check.module] = module
        reporter.pass_(check.label, package_version(check, module))
    return loaded


def check_numpy_abi(reporter: Reporter, loaded: dict[str, object]) -> None:
    numpy_module = loaded.get("numpy")
    if numpy_module is None:
        return
    version_text = str(getattr(numpy_module, "__version__", ""))
    try:
        major, minor = (int(value) for value in version_text.split(".", 2)[:2])
    except (TypeError, ValueError):
        reporter.fail("NumPy compatibility", f"cannot parse version {version_text!r}")
        return
    if (major, minor) == (1, 26):
        reporter.pass_("NumPy compatibility", "1.26.x shared ABI")
    else:
        reporter.fail("NumPy compatibility", f"requires 1.26.x, found {version_text}")


def check_commands(reporter: Reporter, commands: tuple[tuple[str, str], ...]) -> None:
    for command, label in commands:
        path = shutil.which(command)
        if path:
            reporter.pass_(label, path)
        else:
            reporter.fail(label, f"command not found: {command}")


def check_gnina(reporter: Reporter, required: bool) -> None:
    path = shutil.which("gnina")
    if path:
        reporter.pass_("GNINA", path)
    elif required:
        reporter.fail("GNINA", "command not found and --require-gnina was set")
    else:
        reporter.warn(
            "GNINA",
            "optional command not found; Vina is installed but is not a drop-in replacement",
        )


def check_openmm_platforms(
    reporter: Reporter, loaded: dict[str, object], require_gpu: bool
) -> None:
    if "openmm" not in loaded:
        return
    try:
        from openmm import Platform

        names = [
            Platform.getPlatform(index).getName()
            for index in range(Platform.getNumPlatforms())
        ]
    except Exception as exc:
        reporter.fail("OpenMM platforms", str(exc))
        return
    if not names:
        reporter.fail("OpenMM platforms", "no platforms registered")
        return
    reporter.pass_("OpenMM platforms", ", ".join(names))
    if "CUDA" in names:
        reporter.pass_("OpenMM CUDA", "CUDA platform registered")
    elif require_gpu:
        reporter.fail("OpenMM CUDA", "CUDA platform missing and --require-gpu was set")
    else:
        reporter.warn("OpenMM CUDA", "CUDA platform not visible; acceptable on a CPU/login node")


def check_asyncfep(reporter: Reporter, root: Path | None) -> None:
    if root is None:
        reporter.warn("AsyncFEP source", "not checked; pass --asyncfep-root PATH")
        return
    root = root.expanduser().resolve()
    required = (
        root / "pyproject.toml",
        root / "main.py",
        root / "tools" / "reference_rbfe_pipeline.py",
        root / "bloom_prepare" / "pyproject.toml",
        root / "bloom_prepare" / "prepare" / "csfep" / "prepare_rbfe_topologies.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        reporter.fail("AsyncFEP source", "missing: " + ", ".join(missing))
    else:
        reporter.pass_("AsyncFEP source", str(root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asyncfep-root",
        type=Path,
        help="Path to the AsyncFEP checkout installed into this environment",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless OpenMM exposes its CUDA platform",
    )
    parser.add_argument(
        "--require-gnina",
        action="store_true",
        help="Fail unless the GNINA executable is available",
    )
    parser.add_argument(
        "--skip-asyncfep",
        action="store_true",
        help="Skip bloom-core, bloom-prepare, Bloom CLI, and source-tree checks",
    )
    args = parser.parse_args()
    if args.skip_asyncfep and args.asyncfep_root is not None:
        parser.error("--skip-asyncfep cannot be combined with --asyncfep-root")
    return args


def main() -> int:
    args = parse_args()
    reporter = Reporter()
    print("Compute environment check")
    print("=" * 72)
    check_python(reporter)
    import_checks = IMPORTS
    command_checks = COMMANDS
    if args.skip_asyncfep:
        reporter.warn(
            "AsyncFEP checks",
            "bloom-core, bloom-prepare, Bloom CLI, and source checks were skipped",
        )
    else:
        import_checks += ASYNCFEP_IMPORTS
        command_checks += ASYNCFEP_COMMANDS
    loaded = check_imports(reporter, import_checks)
    check_numpy_abi(reporter, loaded)
    check_commands(reporter, command_checks)
    check_gnina(reporter, args.require_gnina)
    check_openmm_platforms(reporter, loaded, args.require_gpu)
    if not args.skip_asyncfep:
        check_asyncfep(reporter, args.asyncfep_root)
    print("=" * 72)
    if reporter.failures:
        print(
            f"RESULT: FAIL ({len(reporter.failures)} failure(s), "
            f"{len(reporter.warnings)} warning(s))"
        )
        return 1
    print(f"RESULT: PASS ({len(reporter.warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
