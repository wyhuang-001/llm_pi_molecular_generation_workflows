from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molecular_agent.adapters import configured_adapters
from molecular_agent.editing import EditResult, apply_substituent, write_sdf
from scripts.openai_compatible_client import OpenAICompatibleChatClient
from molecular_agent.structure import ComplexContext


ABLATION_PROMPT = """You are running an ablation experiment for structure-based molecular design.
The user provided a task and protein-ligand complex coordinates. The coordinate text is the only
initial structural input. Do not claim interactions, chemistry, affinity, docking, or free energy
unless supported by the coordinate text or a tool result.

At each step return exactly one JSON object:
QUERY: {"action":"QUERY","question":"...","tool":"...","arguments":{},"expected_evidence":"..."}
READY: {"action":"READY","understanding":"...","edit_atom_index":0,"edit_hypothesis":"...","fragment_smiles":"[*:1]...","knowledge_gaps":[]}
PROPOSE_TOOL: {"action":"PROPOSE_TOOL","name":"...","purpose":"...","input_schema":{},"implementation_plan":"...","why_existing_tools_are_insufficient":"..."}

Choose tools based on the current information gap. The host will enforce the per-run tool budget.
When asked to return READY, include all required fields and propose one small connected substituent
with exactly one mapped dummy atom [*:1]. Preserve the scaffold, formal charge, and stereochemistry
where possible. A design is only a hypothesis; deterministic RDKit and rigid-protein checks decide
whether a candidate can be written. Never invent an affinity improvement.

The host provides ligand_atom_map as fixed input metadata. When choosing edit_atom_index,
use the zero-based rdkit_index from that map. Never use PDB serial numbers, protein atom
serials, or residue numbers as edit_atom_index.
For the final unbounded verification run, query get_atom_environment, check_growth_space,
and validate_candidate_geometry for the same atom and proposed fragment before READY.
"""


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ligand_atom_map(context: ComplexContext) -> list[dict[str, Any]]:
    conformer = context.ligand.GetConformer()
    available = set(range(len(context.ligand_pdb_atoms)))
    rows = []
    for rdkit_atom in context.ligand.GetAtoms():
        point = conformer.GetAtomPosition(rdkit_atom.GetIdx())
        candidates = []
        for pdb_index in available:
            pdb_atom = context.ligand_pdb_atoms[pdb_index]
            if pdb_atom.element != rdkit_atom.GetSymbol().upper():
                continue
            squared_distance = (
                (pdb_atom.xyz[0] - point.x) ** 2
                + (pdb_atom.xyz[1] - point.y) ** 2
                + (pdb_atom.xyz[2] - point.z) ** 2
            )
            candidates.append((squared_distance, pdb_index, pdb_atom))
        if not candidates:
            raise ValueError(f"No PDB coordinate match for RDKit ligand atom {rdkit_atom.GetIdx()}")
        squared_distance, pdb_index, pdb_atom = min(candidates)
        if squared_distance > 1e-6:
            raise ValueError(
                f"PDB/RDKit coordinate mismatch for ligand atom {rdkit_atom.GetIdx()}: "
                f"{squared_distance ** 0.5:.4f} A"
            )
        available.remove(pdb_index)
        rows.append({
            "rdkit_index": rdkit_atom.GetIdx(),
            "pdb_serial": pdb_atom.serial,
            "pdb_atom_name": pdb_atom.name,
            "element": rdkit_atom.GetSymbol(),
            "aromatic": rdkit_atom.GetIsAromatic(),
            "replaceable_hydrogens": rdkit_atom.GetTotalNumHs(),
            "xyz": [round(point.x, 3), round(point.y, 3), round(point.z, 3)],
            "bonded_rdkit_indices": sorted(neighbor.GetIdx() for neighbor in rdkit_atom.GetNeighbors()),
        })
    if available:
        raise ValueError("Some ligand PDB atoms were not mapped to the reconstructed ligand")
    return rows


def coordinate_text(context: ComplexContext, scope: str, pocket_radius: float) -> str:
    lines = context.complex_path.read_text(encoding="utf-8").splitlines()
    if scope == "full":
        return "\n".join(
            line for line in lines if line[:6].strip() in {"ATOM", "HETATM", "CONECT"}
        )

    ligand_xyz = []
    conformer = context.ligand.GetConformer()
    for atom in context.ligand.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        ligand_xyz.append((point.x, point.y, point.z))
    selected_serials = {atom.serial for atom in context.ligand_pdb_atoms}
    pocket_residues = set()
    for atom in context.protein_atoms:
        if any(
            (atom.xyz[0] - x) ** 2 + (atom.xyz[1] - y) ** 2 + (atom.xyz[2] - z) ** 2
            <= pocket_radius**2
            for x, y, z in ligand_xyz
        ):
            pocket_residues.add((atom.chain, atom.residue_name, atom.residue_number))
    selected_serials.update(
        atom.serial
        for atom in context.protein_atoms
        if (atom.chain, atom.residue_name, atom.residue_number) in pocket_residues
    )
    selected = []
    for line in lines:
        record = line[:6].strip()
        if record in {"ATOM", "HETATM"} and int(line[6:11]) in selected_serials:
            selected.append(line)
        elif record == "CONECT":
            values = [
                int(line[index : index + 5])
                for index in range(6, len(line), 5)
                if line[index : index + 5].strip()
            ]
            if values and values[0] in selected_serials:
                selected.append(line)
    return "\n".join(selected)


def base_payload(
    context: ComplexContext,
    coordinates: str,
    catalog: dict[str, Any],
    budget: int,
    state: dict[str, Any],
    mode: str = "collect_context",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "mode": mode,
        "task": context.task["task"],
        "tool_budget": budget,
        "tools_used": len(state["tool_calls"]),
        "state": state,
        "complex_coordinates": coordinates,
        "tool_catalog": catalog,
        "instruction": (
            "Use at most the remaining tool budget. Query a tool if useful. Return READY when you "
            "can make the best evidence-calibrated local edit."
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def _complete_json(client: Any, payload: dict[str, Any], diagnostic_path: Path) -> dict[str, Any]:
    try:
        return client.complete_json(payload, diagnostic_path=diagnostic_path)
    except TypeError as error:
        if "diagnostic_path" not in str(error):
            raise
        return client.complete_json(payload)


class SiteEvidenceGateError(ValueError):
    def __init__(self, decision: dict[str, Any], report: dict[str, Any]):
        self.decision = decision
        self.report = report
        super().__init__(json.dumps(report, ensure_ascii=False))


def validate_ready(
    context: ComplexContext,
    decision: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    require_site_evidence: bool = True,
) -> dict[str, Any]:
    required = ("understanding", "edit_atom_index", "edit_hypothesis", "fragment_smiles")
    missing = [key for key in required if not decision.get(key)]
    if missing:
        raise ValueError(f"READY is missing fields: {missing}")
    index = decision["edit_atom_index"]
    if not isinstance(index, int) or not 0 <= index < context.ligand.GetNumAtoms():
        raise ValueError(f"Invalid edit_atom_index: {index!r}")
    environment_verified = any(
        call["tool"] == "get_atom_environment"
        and call["arguments"].get("atom_index") == index
        and "edit_site_environment" in call.get("evidence", [])
        for call in tool_calls
    )
    growth_verified = any(
        call["tool"] == "check_growth_space"
        and call["arguments"].get("atom_index") == index
        and "edit_site_geometry" in call.get("evidence", [])
        for call in tool_calls
    )
    candidate_verified = any(
        call["tool"] == "validate_candidate_geometry"
        and call["arguments"].get("atom_index") == index
        and call["arguments"].get("fragment_smiles") == decision.get("fragment_smiles")
        and "candidate_geometry" in call.get("evidence", [])
        and call.get("result", {}).get("status") == "accepted"
        for call in tool_calls
    )
    report = {
        "status": (
            "passed"
            if (
                not require_site_evidence
                or environment_verified and growth_verified and candidate_verified
            )
            else "failed"
        ),
        "required": require_site_evidence,
        "edit_atom_index": index,
        "edit_site_environment_verified": environment_verified,
        "edit_site_geometry_verified": growth_verified,
        "candidate_geometry_verified": candidate_verified,
        "missing": [
            name for name, verified in (
                ("edit_site_environment", environment_verified),
                ("edit_site_geometry", growth_verified),
                ("candidate_geometry", candidate_verified),
            ) if require_site_evidence and not verified
        ],
    }
    if report["status"] != "passed":
        raise SiteEvidenceGateError(decision, report)
    return report


def run_budget(
    task_path: Path,
    config_path: Path,
    output_root: Path,
    budget: int,
    coordinate_scope: str,
    pocket_radius: float,
    require_site_evidence: bool = False,
    unbounded: bool = False,
) -> dict[str, Any]:
    context = ComplexContext(task_path)
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    client = OpenAICompatibleChatClient(config_path, system_prompt=ABLATION_PROMPT)
    run_dir = output_root / f"budget-{budget:02d}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    coordinates = coordinate_text(context, coordinate_scope, pocket_radius)
    atom_map = ligand_atom_map(context)
    state: dict[str, Any] = {
        "budget": budget,
        "unbounded_tool_calls": unbounded,
        "model": config_data.get("model"),
        "coordinate_scope": coordinate_scope,
        "coordinate_chars": len(coordinates),
        "ligand_atom_map_count": len(atom_map),
        "site_evidence_gate_required": require_site_evidence,
        "tool_calls": [],
        "decisions": [],
    }
    write_json(run_dir / "input.json", {
        "task": context.task,
        "coordinate_scope": coordinate_scope,
        "coordinate_chars": len(coordinates),
        "complex_path": str(context.complex_path),
        "ligand_atom_map": atom_map,
        "coordinates": coordinates,
    })
    from molecular_agent.tools import ToolRegistry

    tools = ToolRegistry(context)
    catalog = tools.catalog(include_candidate_geometry=unbounded)
    final_decision: dict[str, Any] | None = None
    ready_gate: dict[str, Any] | None = None
    status = "incomplete"
    error: str | None = None
    gate_retry_tool_count: int | None = None
    gate_retry_decision: dict[str, Any] | None = None

    try:
        while True:
            remaining = None if unbounded else budget - len(state["tool_calls"])
            force_ready = remaining is not None and remaining <= 0
            instruction = (
                "There is no tool-call limit for this final verification run. Query any registered "
                "tools needed to verify the proposed edit site, then return READY."
                if unbounded
                else (
                    "Tool budget is exhausted. Return READY now; do not return QUERY or PROPOSE_TOOL."
                    if force_ready
                    else "You may QUERY one registered tool or return READY."
                )
            )
            feedback = {}
            if gate_retry_decision is not None:
                feedback = {
                    "previous_ready": gate_retry_decision,
                    "ready_gate_feedback": ready_gate,
                    "instruction": (
                        "The previous READY was rejected by deterministic evidence or candidate "
                        "geometry. Use the feedback to identify the missing knowledge, query new "
                        "tools, and return a revised READY with a new or newly validated edit. "
                        "Do not repeat READY without new evidence."
                    ),
                }
            payload = base_payload(
                context, coordinates, catalog, budget, state,
                extra={
                    "instruction": instruction,
                    "tool_budget": None if unbounded else budget,
                    "tool_call_limit": None if unbounded else budget,
                    "ligand_atom_map": atom_map,
                    **feedback,
                    "metadata_contract": (
                        "The ligand_atom_map is fixed host metadata, not a tool result and not part "
                        "of the tool budget. Use rdkit_index for edit_atom_index."
                    ),
                },
            )
            decision = _complete_json(
                client,
                payload,
                run_dir / f"api-error-{len(state['decisions']) + 1:02d}.json",
            )
            state["decisions"].append(decision)
            write_json(run_dir / f"decision-{len(state['decisions']):02d}.json", decision)
            action = decision.get("action")
            if action == "READY":
                try:
                    ready_gate = validate_ready(
                        context, decision, state["tool_calls"], require_site_evidence
                    )
                except SiteEvidenceGateError as exc:
                    if not unbounded:
                        raise
                    ready_gate = exc.report
                    current_tool_count = len(state["tool_calls"])
                    if (
                        gate_retry_tool_count is not None
                        and current_tool_count == gate_retry_tool_count
                    ):
                        raise
                    gate_retry_tool_count = current_tool_count
                    gate_retry_decision = decision
                    continue
                final_decision = decision
                break
            if action == "PROPOSE_TOOL":
                write_json(run_dir / "tool-proposal.json", decision)
                status = "tool_proposal_requires_host_review"
                break
            if action != "QUERY":
                raise ValueError(f"Invalid action: {action!r}")
            if force_ready:
                raise ValueError("LLM requested a tool after the per-run budget was exhausted")
            tool_name = decision.get("tool")
            arguments = decision.get("arguments", {})
            if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                raise ValueError("QUERY requires a tool name and object arguments")
            if tool_name not in catalog:
                raise ValueError(f"Tool is not available in this ablation group: {tool_name}")
            result, evidence = tools.execute(tool_name, arguments)
            call = {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "evidence": sorted(evidence),
            }
            state["tool_calls"].append(call)
            write_json(run_dir / f"tool-call-{len(state['tool_calls']):02d}.json", call)

        if final_decision is not None:
            try:
                edit_result = apply_substituent(
                    context.ligand,
                    final_decision["edit_atom_index"],
                    final_decision["fragment_smiles"],
                    context.protein_atoms,
                    seed=17,
                )
                candidate_path = run_dir / "candidate.sdf"
                write_sdf(edit_result, candidate_path, name=f"budget-{budget:02d}")
                status = (
                    "candidate_geometry_accepted"
                    if edit_result.report["status"] == "accepted"
                    else "candidate_geometry_rejected"
                )
                docking = {"stage": "docking", "status": "not_run_geometry_rejected"}
                rbfe = {"stage": "rbfe", "status": "not_run_geometry_rejected"}
                reference_path = None
                receptor_path = None
                if status == "candidate_geometry_accepted":
                    reference_path = run_dir / "reference-ligand.sdf"
                    write_sdf(
                        EditResult(context.ligand, {"status": "reference"}),
                        reference_path,
                        name="reference-ligand",
                    )
                    receptor_path = context.write_receptor_pdb(
                        run_dir / "receptor-protein-only.pdb"
                    )
                    docking_adapter, _rbfe_adapter = configured_adapters(config_path, run_dir)
                    docking = docking_adapter.run(
                        candidate_path=candidate_path,
                        receptor_path=receptor_path,
                        reference_path=reference_path,
                        output_dir=run_dir / "docking",
                    )
                    rbfe = {
                        "stage": "rbfe",
                        "status": "deferred",
                        "message": "RBFE is intentionally deferred; this ablation stops after docking.",
                    }
                result = {
                    "decision": final_decision,
                    "validation": edit_result.report,
                    "candidate_path": str(candidate_path),
                    "reference_path": str(reference_path) if reference_path else None,
                    "receptor_path": str(receptor_path) if receptor_path else None,
                    "docking": docking,
                    "rbfe": rbfe,
                    "ready_gate": ready_gate,
                    "evaluation_scope": {
                        "site_evidence": "verified_by_ablation_gate" if require_site_evidence else "not_required",
                        "affinity": "not_evaluated",
                        "docking": docking["status"],
                        "fep": rbfe["status"],
                        "meaning": (
                            "Acceptance only means RDKit construction and the rigid-protein "
                            "clash threshold passed; it is not a binding or activity result."
                        ),
                    },
                }
            except Exception as exc:
                status = "edit_validation_failed"
                result = {"decision": final_decision, "validation_error": str(exc)}
        else:
            result = {}
    except SiteEvidenceGateError as exc:
        status = "site_evidence_gate_failed"
        error = str(exc)
        result = {
            "decision": exc.decision,
            "ready_gate": exc.report,
            "evaluation_scope": {
                "site_evidence": "required_and_failed",
                "affinity": "not_evaluated",
                "docking": "not_run_site_evidence_gate_failed",
                "fep": "deferred",
            },
        }
    except Exception as exc:
        error = str(exc)
        result = {}

    output = {
        "budget": budget,
        "status": status,
        "error": error,
        "tool_call_count": len(state["tool_calls"]),
        "decision_count": len(state["decisions"]),
        "state": state,
        "result": result,
    }
    write_json(run_dir / "result.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent tool-budget ablation for molecular design")
    parser.add_argument("--task", type=Path, default=Path("input/task.json"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/ablation-tool-budget"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--coordinate-scope", choices=["full", "pocket"], default="pocket")
    parser.add_argument("--pocket-radius", type=float, default=6.0)
    parser.add_argument(
        "--unbounded-budget",
        type=int,
        default=None,
        metavar="N",
        help="Run final budget-N with unlimited tool queries and strict site evidence gate.",
    )
    args = parser.parse_args()

    summaries = []
    for budget in args.budgets:
        if budget < 0:
            parser.error("budgets must be non-negative")
        print(f"[ablation] starting budget={budget}", flush=True)
        try:
            result = run_budget(
                args.task.resolve(), args.config.resolve(), args.output_root.resolve(),
                budget, args.coordinate_scope, args.pocket_radius,
                require_site_evidence=False,
            )
        except Exception as exc:
            result = {"budget": budget, "status": "run_failed", "error": str(exc)}
        summaries.append({
            "budget": budget,
            "status": result.get("status"),
            "error": result.get("error"),
            "tool_call_count": result.get("tool_call_count", 0),
            "decision_count": result.get("decision_count", 0),
            "candidate_path": result.get("result", {}).get("candidate_path"),
        })
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

    if args.unbounded_budget is not None:
        if args.unbounded_budget < 0:
            parser.error("unbounded budget must be non-negative")
        if args.unbounded_budget in args.budgets:
            parser.error("unbounded budget must not duplicate a limited budget")
        budget = args.unbounded_budget
        print(f"[ablation] starting unbounded final budget={budget}", flush=True)
        try:
            result = run_budget(
                args.task.resolve(), args.config.resolve(), args.output_root.resolve(),
                budget, args.coordinate_scope, args.pocket_radius,
                require_site_evidence=True,
                unbounded=True,
            )
        except Exception as exc:
            result = {"budget": budget, "status": "run_failed", "error": str(exc)}
        summaries.append({
            "budget": budget,
            "unbounded_tool_calls": True,
            "status": result.get("status"),
            "error": result.get("error"),
            "tool_call_count": result.get("tool_call_count", 0),
            "decision_count": result.get("decision_count", 0),
            "candidate_path": result.get("result", {}).get("candidate_path"),
        })
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

    write_json(args.output_root.resolve() / "summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
