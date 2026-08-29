#!/usr/bin/env python3
"""Check only the Python and command dependencies needed for docking."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


IMPORTS = (
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("rdkit", "RDKit"),
    ("gemmi", "Gemmi"),
    ("pydantic", "Pydantic"),
    ("openbabel", "Open Babel"),
    ("vina", "AutoDock Vina"),
    ("meeko", "Meeko"),
    ("prody", "ProDy"),
    ("MDAnalysis", "MDAnalysis"),
    ("prolif", "ProLIF"),
    ("spyrmsd", "spyrmsd"),
    ("freesasa", "FreeSASA"),
    ("molecular_agent", "simple-molecular-agent"),
)

COMMANDS = (
    ("curl", "curl"),
    ("gnina", "GNINA"),
    ("vina", "AutoDock Vina CLI"),
    ("obabel", "Open Babel CLI"),
    ("mk_prepare_ligand.py", "Meeko ligand preparation"),
    ("mk_prepare_receptor.py", "Meeko receptor preparation"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gnina", action="store_true")
    args = parser.parse_args()
    failures = []
    warnings = []
    print("Docking-only environment check")
    print("=" * 72)
    if sys.version_info[:2] != (3, 11):
        failures.append(f"Python 3.11 required, found {sys.version.split()[0]}")
        print(f"FAIL  Python - {failures[-1]}")
    else:
        print(f"PASS  Python - {sys.version.split()[0]}")
    for module_name, label in IMPORTS:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "version unknown")
            print(f"PASS  {label} - {version}")
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            print(f"FAIL  {label} - {exc}")
    for command, label in COMMANDS:
        path = shutil.which(command)
        if path:
            print(f"PASS  {label} - {path}")
        elif command == "gnina" and not args.require_gnina:
            warnings.append(f"{label}: command not found: {command}")
            print(f"WARN  {label} - command not found: {command}")
        else:
            failures.append(f"{label}: command not found: {command}")
            print(f"FAIL  {label} - command not found: {command}")
    gnina = shutil.which("gnina")
    if gnina:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        result = subprocess.run(
            [gnina, "--version"], capture_output=True, text=True, env=env, timeout=60
        )
        detail = (result.stdout or result.stderr).strip().splitlines()
        if result.returncode == 0:
            print(f"PASS  GNINA runtime - {detail[0] if detail else 'started successfully'}")
        else:
            message = detail[-1] if detail else f"exit code {result.returncode}"
            failures.append(f"GNINA runtime: {message}")
            print(f"FAIL  GNINA runtime - {message}")
    print("=" * 72)
    if failures:
        print(f"RESULT: FAIL ({len(failures)} failure(s), {len(warnings)} warning(s))")
        return 1
    print(f"RESULT: PASS ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
