"""Offline experience library for iterative molecular design.

The live workflow appends one evidence record per attempt to
``run_dir/trajectory-evidence.jsonl`` DURING the run. This module is independent
of the live loop: it reads that file, distills it into structured experience, and
stores/loads/matches it in an on-disk library so the next round can be seeded
with validated, ranked context (only the first design loop reads it).

Only host-verified facts are persisted: docking deltas, pose gates, seed
consistency, interaction changes and deterministic failures. LLM reasoning is
never stored, which keeps the experience reusable instead of overfit.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA = "simple-molecular-agent.trajectory-evidence.v1"
EXPERIENCE_SCHEMA = "simple-molecular-agent.experience.v1"

DEFAULT_LIBRARY_DIR = "experience-library"

#: Multi-way matching weights, higher wins.  A hit on the exact ligand dominates;
#: site/operation/fragment/pocket/failure refine the ranking within and across ligands.
MATCH_WEIGHTS: dict[str, float] = {
    "task": 1.0,
    "site": 0.8,
    "operation": 0.6,
    "fragment": 0.5,
    "pocket": 0.4,
    "failure": 0.3,
}

#: Only these outcomes count as productive evidence for "what to try".
PRODUCTIVE_OUTCOMES = {"new_best", "improved", "neutral"}
AVOID_OUTCOMES = {"pose_failed", "geometry_rejected", "duplicate_structure", "docking_failed"}


def ligand_key(smiles: str) -> str:
    """Stable on-disk key for one parent ligand."""
    return hashlib.sha256((smiles or "").encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_evidence_record(
    attempt: int,
    task: str,
    ligand_smiles: str,
    site: dict[str, Any],
    operation: str,
    fragment: dict[str, Any],
    pocket: dict[str, Any],
    evidence: dict[str, Any],
    outcome: str,
    failure: dict[str, Any] | None,
) -> dict[str, Any]:
    """One trajectory/evidence record, built by the live workflow per attempt."""
    return {
        "schema": EVIDENCE_SCHEMA,
        "attempt": attempt,
        "timestamp": now_iso(),
        "task": (task or "")[:200],
        "ligand_smiles": ligand_smiles,
        "site": site,
        "operation": operation,
        "fragment": fragment,
        "pocket": pocket,
        "evidence": evidence,
        "outcome": outcome,
        "failure": failure,
    }


def _fragment_key(fragment: dict[str, Any]) -> str:
    return fragment.get("fragment_id") or fragment.get("element") or "?"


def structure_experience(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Distill evidence records into structured site/fragment/operation/pocket/failure experience.

    Facts only. Deltas are candidate-minus-reference from the frozen docking
    protocol; they are not experimental activity.
    """
    site_agg: dict[tuple, dict[str, Any]] = {}
    fragment_agg: dict[str, dict[str, Any]] = {}
    operation_agg: dict[str, dict[str, Any]] = {}
    pocket_agg: dict[str, dict[str, Any]] = {}
    failure_agg: dict[str, dict[str, Any]] = {}
    ligand_smiles = records[0]["ligand_smiles"] if records else ""

    for rec in records:
        site = rec.get("site") or {}
        operation = rec.get("operation") or "?"
        fragment = rec.get("fragment") or {}
        pocket = rec.get("pocket") or {}
        evidence = rec.get("evidence") or {}
        failure = rec.get("failure") or {}
        outcome = rec.get("outcome") or "neutral"
        delta = evidence.get("delta_candidate_minus_reference")
        pose = evidence.get("pose_native_like")
        quality = evidence.get("quality")

        skey = (site.get("target_type"), site.get("target_id"), operation)
        cell = site_agg.setdefault(skey, {
            "target_type": site.get("target_type"),
            "target_id": site.get("target_id"),
            "operation": operation,
            "region": site.get("region"),
            "attempts": 0,
            "deltas": [],
            "new_best_count": 0,
            "pose_pass_count": 0,
            "pose_attempts": 0,
            "best_delta": None,
            "best_fragment": None,
        })
        cell["attempts"] += 1
        if isinstance(delta, (int, float)):
            cell["deltas"].append(delta)
            if cell["best_delta"] is None or delta < cell["best_delta"]:
                cell["best_delta"] = delta
                cell["best_fragment"] = _fragment_key(fragment)
        if outcome == "new_best":
            cell["new_best_count"] += 1
        if pose is not None:
            cell["pose_attempts"] += 1
            cell["pose_pass_count"] += 1 if pose else 0

        fkey = _fragment_key(fragment)
        fcell = fragment_agg.setdefault(fkey, {
            "fragment": _fragment_key(fragment),
            "fragment_smiles": fragment.get("fragment_smiles") or fragment.get("element"),
            "attempts": 0,
            "deltas": [],
            "pose_pass_count": 0,
            "pose_attempts": 0,
            "sites": set(),
            "best_delta": None,
        })
        fcell["attempts"] += 1
        if isinstance(delta, (int, float)):
            fcell["deltas"].append(delta)
            if fcell["best_delta"] is None or delta < fcell["best_delta"]:
                fcell["best_delta"] = delta
        if pose is not None:
            fcell["pose_attempts"] += 1
            fcell["pose_pass_count"] += 1 if pose else 0
        fcell["sites"].add(f"{site.get('target_type')}:{site.get('target_id')}")

        okey = operation
        ocell = operation_agg.setdefault(okey, {"operation": operation, "attempts": 0, "deltas": [], "new_best_count": 0})
        ocell["attempts"] += 1
        if isinstance(delta, (int, float)):
            ocell["deltas"].append(delta)
        if outcome == "new_best":
            ocell["new_best_count"] += 1

        for residue in pocket.get("residues") or []:
            pcell = pocket_agg.setdefault(residue, {"residue": residue, "fragments": set(), "attempts": 0})
            pcell["fragments"].add(_fragment_key(fragment))
            pcell["attempts"] += 1

        if outcome in AVOID_OUTCOMES or (failure or {}).get("failure_class"):
            fclass = (failure or {}).get("failure_class") or outcome
            fcell2 = failure_agg.setdefault(fclass, {"failure_class": fclass, "attempts": 0, "sites": set(), "fragments": set()})
            fcell2["attempts"] += 1
            fcell2["sites"].add(f"{site.get('target_type')}:{site.get('target_id')}")
            fcell2["fragments"].add(_fragment_key(fragment))

    def _avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    def _rate(num: int, den: int) -> float | None:
        return round(num / den, 3) if den else None

    site_exp = []
    for cell in site_agg.values():
        cell["avg_delta"] = _avg(cell["deltas"])
        cell["pose_pass_rate"] = _rate(cell["pose_pass_count"], cell["pose_attempts"])
        cell.pop("deltas", None)
        site_exp.append(cell)
    site_exp.sort(key=lambda c: (c["best_delta"] is not None, -(c["best_delta"] or 0)), reverse=True)

    fragment_exp = []
    for cell in fragment_agg.values():
        cell["avg_delta"] = _avg(cell["deltas"])
        cell["pose_pass_rate"] = _rate(cell["pose_pass_count"], cell["pose_attempts"])
        cell["sites"] = sorted(cell["sites"])
        cell.pop("deltas", None)
        fragment_exp.append(cell)
    fragment_exp.sort(key=lambda c: (c["best_delta"] is not None, -(c["best_delta"] or 0)), reverse=True)

    operation_exp = []
    for cell in operation_agg.values():
        cell["avg_delta"] = _avg(cell["deltas"])
        cell.pop("deltas", None)
        operation_exp.append(cell)
    operation_exp.sort(key=lambda c: c["attempts"], reverse=True)

    pocket_exp = []
    for cell in pocket_agg.values():
        cell["fragments"] = sorted(cell["fragments"])
        pocket_exp.append(cell)
    pocket_exp.sort(key=lambda c: c["attempts"], reverse=True)

    failure_exp = []
    for cell in failure_agg.values():
        cell["sites"] = sorted(cell["sites"])
        cell["fragments"] = sorted(cell["fragments"])
        failure_exp.append(cell)
    failure_exp.sort(key=lambda c: c["attempts"], reverse=True)

    return {
        "schema": EXPERIENCE_SCHEMA,
        "ligand_smiles": ligand_smiles,
        "generated_at": now_iso(),
        "record_count": len(records),
        "interpretation_boundary": (
            "Host-verified docking/pose facts only; deltas are candidate-minus-reference "
            "under the frozen protocol and are not experimental activity."
        ),
        "site_experience": site_exp,
        "fragment_experience": fragment_exp,
        "operation_experience": operation_exp,
        "pocket_experience": pocket_exp,
        "failure_experience": failure_exp,
    }


class ExperienceLibrary:
    """On-disk store + multi-way matcher for structured experience."""

    def __init__(self, library_dir: str | Path = DEFAULT_LIBRARY_DIR):
        self.library_dir = Path(library_dir)

    def path_for(self, ligand_smiles: str) -> Path:
        return self.library_dir / f"{ligand_key(ligand_smiles)}.json"

    def store(self, ligand_smiles: str, experience: dict[str, Any]) -> Path:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(ligand_smiles)
        path.write_text(json.dumps(experience, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, ligand_smiles: str) -> dict[str, Any] | None:
        path = self.path_for(ligand_smiles)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def ingest_run(self, run_dir: str | Path, ligand_smiles: str) -> dict[str, Any] | None:
        """Read a run's trajectory-evidence.jsonl, structure it, and store in the library."""
        evidence_path = Path(run_dir) / "trajectory-evidence.jsonl"
        if not evidence_path.is_file():
            return None
        records = []
        for line in evidence_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not records:
            return None
        experience = structure_experience(records)
        experience["source_runs"] = [str(Path(run_dir).resolve())]
        self.store(ligand_smiles, experience)
        return experience

    def match(self, ligand_smiles: str, query: dict[str, Any], top_k: int = 10) -> list[dict[str, Any]]:
        """Multi-way match + gate + weighted rank.

        ``query`` supplies the current round's task surface::

            {"sites": [{"target_type","target_id"}...],
             "fragments": [{"fragment_id"/"element", "fragment_smiles"}...],
             "operations": [...],
             "pocket_residues": [...],
             "failure_classes": [...]}
        """
        experience = self.load(ligand_smiles)
        if not experience:
            return []
        sites = {(s.get("target_type"), s.get("target_id")) for s in query.get("sites", [])}
        fragments = set()
        for f in query.get("fragments", []):
            fragments.add(f.get("fragment_id") or f.get("element"))
            if f.get("fragment_smiles"):
                fragments.add(f.get("fragment_smiles"))
        operations = set(query.get("operations", []))
        residues = set(query.get("pocket_residues", []))
        failures = set(query.get("failure_classes", []))

        ranked: list[dict[str, Any]] = []

        def push(score: float, kind: str, text: str, keys: dict[str, Any]) -> None:
            if score > 0:
                ranked.append({"score": round(score, 3), "kind": kind, "text": text, "keys": keys})

        for s in experience.get("site_experience", []):
            score = 0.0
            hits = []
            if (s.get("target_type"), s.get("target_id")) in sites:
                score += MATCH_WEIGHTS["site"]; hits.append("site")
            if s.get("operation") in operations:
                score += MATCH_WEIGHTS["operation"]; hits.append("operation")
            if score:
                push(score, "site",
                     f"site {s.get('target_type')}:{s.get('target_id')} ({s.get('operation')}): "
                     f"best_delta={s.get('best_delta')}, best_fragment={s.get('best_fragment')}, "
                     f"attempts={s.get('attempts')}, pose_pass_rate={s.get('pose_pass_rate')}",
                     {"site": s, "hits": hits})

        for f in experience.get("fragment_experience", []):
            score = 0.0
            hits = []
            if f.get("fragment") in fragments or f.get("fragment_smiles") in fragments:
                score += MATCH_WEIGHTS["fragment"]; hits.append("fragment")
            if score:
                push(score, "fragment",
                     f"fragment {f.get('fragment')} ({f.get('fragment_smiles')}): "
                     f"avg_delta={f.get('avg_delta')}, best_delta={f.get('best_delta')}, "
                     f"pose_pass_rate={f.get('pose_pass_rate')}, sites={f.get('sites')}",
                     {"fragment": f, "hits": hits})

        for o in experience.get("operation_experience", []):
            if o.get("operation") in operations:
                push(MATCH_WEIGHTS["operation"], "operation",
                     f"operation {o.get('operation')}: attempts={o.get('attempts')}, "
                     f"avg_delta={o.get('avg_delta')}, new_bests={o.get('new_best_count')}",
                     {"operation": o, "hits": ["operation"]})

        for p in experience.get("pocket_experience", []):
            if p.get("residue") in residues:
                push(MATCH_WEIGHTS["pocket"], "pocket",
                     f"pocket {p.get('residue')}: contacted by fragments {p.get('fragments')}",
                     {"pocket": p, "hits": ["pocket"]})

        for fl in experience.get("failure_experience", []):
            if fl.get("failure_class") in failures or failures == {"*"}:
                push(MATCH_WEIGHTS["failure"], "failure",
                     f"avoid {fl.get('failure_class')}: sites={fl.get('sites')}, fragments={fl.get('fragments')}",
                     {"failure": fl, "hits": ["failure"]})

        ranked.sort(key=lambda item: (-item["score"], item["kind"]))
        return ranked[:top_k]

    def context_text(self, ranked: list[dict[str, Any]]) -> str:
        """Serialize ranked experience into a compact structured text block."""
        if not ranked:
            return ""
        lines = [
            "Reusable validated experience from prior rounds (host-verified docking/pose facts only; "
            "deltas are candidate-minus-reference, not experimental activity):",
        ]
        for item in ranked:
            lines.append(f"- [{item['kind']}] {item['text']}")
        return "\n".join(lines)
