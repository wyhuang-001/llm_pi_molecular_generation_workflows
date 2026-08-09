from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


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

    def __init__(self, stage: str, config: dict[str, Any], run_dir: Path):
        self.stage = stage
        self.config = config
        self.run_dir = run_dir

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
        return audit

    @staticmethod
    def _write_audit(output_dir: Path, audit: dict[str, Any]) -> None:
        (output_dir / f"{audit['stage']}-result.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class DockingAdapter(CommandAdapter):
    def __init__(self, config: dict[str, Any], run_dir: Path):
        super().__init__("docking", config, run_dir)

    def run(self, candidate_path: Path, receptor_path: Path, **_: Any) -> dict[str, Any]:
        output_dir = self.run_dir / "docking"
        return super().run(
            candidate=str(candidate_path.resolve()),
            receptor=str(receptor_path.resolve()),
            output_dir=str(output_dir),
        )


class RBFEAdapter(CommandAdapter):
    def __init__(self, config: dict[str, Any], run_dir: Path):
        super().__init__("rbfe", config, run_dir)

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


def configured_adapters(config_path: Path, run_dir: Path) -> tuple[CommandAdapter, CommandAdapter]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    docking_config = config.get("docking") or {}
    rbfe_config = config.get("rbfe") or {}
    docking = (
        DockingAdapter(docking_config, run_dir)
        if docking_config
        else NotConfiguredAdapter("docking")
    )
    rbfe = (
        RBFEAdapter(rbfe_config, run_dir)
        if rbfe_config
        else NotConfiguredAdapter("rbfe")
    )
    return docking, rbfe
