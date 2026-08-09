from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    client = ScriptedDemoClient() if args.scripted_demo else ResponsesClient(args.config)
    result = Workflow(args.task, client, args.run_dir, config_path=args.config).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
