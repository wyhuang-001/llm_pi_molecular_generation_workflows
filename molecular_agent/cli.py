from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

from .llm import ResponsesClient
from .structure import ComplexContext
from .workflow import ScriptedDemoClient, Workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal LLM-guided molecular design workflow")
    parser.add_argument("--task", type=Path, default=Path("input/task.json"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/latest"))
    parser.add_argument("--check-input", action="store_true")
    parser.add_argument("--scripted-demo", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress live workflow events")
    parser.add_argument("--full-json", action="store_true", help="Print the complete result JSON")
    return parser


def _progress_printer(event: str, details: dict[str, Any]) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    payload = json.dumps(details, ensure_ascii=False, separators=(",", ":"), default=str)
    print(f"[{timestamp}] {event}: {payload}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    progress = None if args.quiet else _progress_printer
    if args.check_input:
        context = ComplexContext(args.task)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "task": context.task["task"],
                    "protein_atoms": len(context.protein_atoms),
                    "ligand_pdb_atoms": len(context.ligand_pdb_atoms),
                    "ligand_heavy_atoms": context.ligand.GetNumHeavyAtoms(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    try:
        client = (
            ScriptedDemoClient()
            if args.scripted_demo
            else ResponsesClient(
                args.config,
                diagnostic_dir=args.run_dir / "llm",
                progress=progress,
            )
        )
        result = Workflow(
            args.task,
            client,
            args.run_dir,
            config_path=args.config,
            progress=progress,
        ).run()
    except Exception as error:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (args.run_dir / "workflow-error.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if progress:
            progress("workflow_failed", failure)
        else:
            print(f"Workflow failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if args.full_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        final = result["result"]
        print(json.dumps({
            "status": final.get("status"),
            "stopping_reason": final.get("stopping_reason"),
            "attempt_count": len(final.get("attempts", [])),
            "best_attempt": final.get("best_attempt"),
            "candidate_path": final.get("candidate_path"),
            "docking_status": (final.get("docking") or {}).get("status"),
            "convergence": final.get("convergence"),
            "result_path": str((args.run_dir / "result.json").resolve()),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
