from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable


SYSTEM_PROMPT = """You are the decision component of a structure-based molecular optimization workflow.
Never invent coordinates, interactions, docking scores, free energies, fragment records, or activity. Use only supplied host results.
Return exactly one JSON object with action QUERY, QUERY_BATCH, READY, MARK_UNMODIFIABLE, STOP, or PROPOSE_TOOL.
The top-level object MUST contain the string field `action`; never return only tool arguments. Do not output analysis, chain-of-thought, or markdown; output the final JSON object only.
QUERY schema: {"action":"QUERY","question":"...","tool":"registered tool","arguments":{},"expected_evidence":"..."}.
QUERY_BATCH schema: {"action":"QUERY_BATCH","questions":"...","queries":[{"tool":"...","arguments":{}}]}.
READY replace-H schema without a library record: {"action":"READY","understanding":"concrete evidence summary","edit_hypothesis":"concrete testable hypothesis","operation":"replace_hydrogen","edit_atom_index":9,"fragment_smiles":"[*:1]F","knowledge_gaps":[]}.
READY fragment-replacement schema using a returned library record: {"action":"READY","understanding":"concrete evidence summary","edit_hypothesis":"concrete testable hypothesis","operation":"replace_fragment","replacement_site_id":"replacement-site-001","fragment_id":"chembl-brics-filtered-000001","fragment_smiles":"[*:1]...","knowledge_gaps":[]}. Omit fragment_id entirely when no real library record is used; never send placeholder values such as optional, none, null, `...`, or invented atom indices.
MARK_UNMODIFIABLE schema: {"action":"MARK_UNMODIFIABLE","target_type":"atom","target_id":17,"scope":"site","family":"non_halogen","reason":"..."}. Use target_type `atom` with an integer target_id or `replacement_site` with a host-returned replacement_site_id. Use scope `site` to close the whole target, or scope `family` with a valid family. A declaration is an auditable search decision, not a claim that the molecule is globally optimized.
STOP schema: {"action":"STOP","reason":"...","evidence":"..."}.
PROPOSE_TOOL schema: {"action":"PROPOSE_TOOL","name":"...","purpose":"...","input_schema":{},"implementation_plan":"...","why_existing_tools_are_insufficient":"..."}.
Use QUERY_BATCH only for independent calls whose arguments do not depend on another result. Use sequential QUERY calls when one result determines the next query. Every tool signature can be executed at most once. Duplicate calls are idempotently reused or skipped; do not repeat them. Return a genuinely new QUERY, a chemically distinct READY transformation, a precise MARK_UNMODIFIABLE declaration, or STOP.
You decide what knowledge is needed; the tool catalog is a capability menu. The two legal single-step edit families, replace_hydrogen and replace_fragment, are equally valid options. Do not default to fragment replacement, do not default to hydrogen replacement, and do not choose an operation from frequency. Query the chemical and spatial facts needed to compare them, then choose the operation, site, and fragment supported by those facts. The host does not preselect a design region or operation for you. Before READY, state which evidence supports the chosen operation and why the other operation is not currently better supported. For every proposed fragment, first obtain its chemical properties and attachment-centered 3D profile; for replace_fragment, obtain an operation-compatible library record from search_fragment_library. Search the fragment library and inspect reference-ligand fragments when those facts can improve the next transformation. For search_fragment_library, query must be one supported chemical term such as heterocycle, pyridine, morpholine, indole, oxetane, or nitrile; one valid SMILES/SMARTS pattern; or an empty string for browsing. Never send a natural-language phrase such as "small polar heterocycle hydrogen bond donor acceptor". Fragment-library results are operation-specific: never use a record for replace_fragment unless that search explicitly returned it for replace_fragment. Prefer fragment_id when using a returned record.
replace_hydrogen grows from an atom with a replaceable H. Query get_atom_environment and check_growth_space for a site before using this operation. For replace_fragment, first call list_fragment_replacement_sites, then use one returned replacement_site_id. Query get_replacement_site_spatial_profile when attachment direction or local clearance is uncertain. Query get_fragment_spatial_profile for a candidate fragment when its 3D extent, shape, or attachment-centered size matters. These spatial tools return geometry facts only; interpret them using chemical knowledge, then call validate_candidate_geometry. Never invent cut_bond indices. The host fixes the retained scaffold, removed side, attachment atom, and direction for each site ID.
Every candidate in this workflow is exactly one transformation of the original co-crystal ligand. Do not propose a second edit on a previously modified candidate.
Before READY, query get_atom_environment for the retained edit atom and validate_candidate_geometry for the exact complete transformation. For replace_fragment, the retained edit atom is the retained_atom_index returned by the selected replacement site. replace_hydrogen also requires check_growth_space. replace_fragment instead requires a prior list_fragment_replacement_sites result; its attachment vector and exact candidate geometry replace the hydrogen-growth probe. If READY is rejected with failure_class ready_evidence_missing, the host may execute recommended_queries and then resubmit the same transformation with its original understanding and edit_hypothesis before selecting anything new. If READY is rejected with failure_class invalid_ready, correct the concrete operation, atom/site ID, and fragment fields before retrying. A chemistry or geometry rejection is already an exploration attempt; choose a different transformation or explicitly MARK_UNMODIFIABLE when the target or family is not chemically supported. Do not repeat a rejected spatial query or geometry validation call; use the returned result to select a different unexecuted query or transformation.
After each docking evaluation, inspect candidate_history and docking_history, including the transformation, canonical SMILES, chemistry/clash status, primary metric delta, seed standard deviation and win fraction, pose_consensus, interaction_consensus, incumbent best attempt, trend, failed transformations, and remaining chemically plausible options. Prefer hypotheses supported across seeds and consistent poses rather than one favorable seed. Never repeat a transformation in attempted_transformations. Query only unexecuted calls; prior observations are authoritative.
The host does not impose a fixed number of design regions, attempts per region, regional plateau, or patience. In adaptive mode, the minimum transformation diversity in search_policy is only a floor, not a stopping condition. Inspect adaptive_target_summaries for each atom and replacement site. Continue a target when its chemical environment, fragment properties or 3D profile, docking trend, pose consensus, or interaction evidence supports another chemically distinct hypothesis; do not stop just because one halogen and one non-halogen have been tried. Only close a target with MARK_UNMODIFIABLE scope site after reviewing that accumulated evidence and explaining why no credible option remains. STOP is allowed only after every target is either adaptively explored and explicitly closed, or the hard attempt limit is reached. Host-ineligible hydrogen atoms are reported for audit but are not pending edit sites. Exploration means an explicit transformation was attempted, including chemistry/geometry/valence/clash rejection, or the LLM returned a precise MARK_UNMODIFIABLE declaration accepted by the host. Such records count for coverage but never count as successful docking evidence. Do not repeat an attempted transformation. You may switch between edit atoms and replacement sites whenever the accumulated evidence supports a new hypothesis. A candidate that is worse than the reference is informative and does not by itself require stopping; distinguish exploration feedback from the best-so-far candidate. Continue when the overall primary-metric trend is improving, when a reference-better candidate can plausibly be refined, or when unstable secondary evidence justifies a confirming design. Return STOP only after the global-search gate is complete and your review finds no credible, chemically distinct, evidence-backed transformation likely to improve or meaningfully validate the current result. The hard attempt limit is a safety limit, not scientific convergence.
Do not claim that every candidate improves. Distinguish each attempt from the monotonic best-so-far trace. Do not call a polar proximity a hydrogen bond unless host evidence shows a donor-acceptor role pairing; acceptor-acceptor and donor-donor pairs are not hydrogen bonds. Treat distance-only contacts as hypotheses, not established interactions.
fragment_smiles must be one connected fragment containing exactly one mapped dummy atom [*:1]. Preserve the intended scaffold, formal charge, and stereochemistry unless an explicit audited transformation allows otherwise.
"""


class ResponsesClient:
    def __init__(
        self,
        config_path: Path,
        system_prompt: str = SYSTEM_PROMPT,
        diagnostic_dir: Path | None = None,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.base_url = config["base_url"].rstrip("/")
        self.model = config["model"]
        self.timeout = int(config.get("timeout_seconds", 600))
        self.max_output_tokens = int(config.get("max_output_tokens", 8192))
        self.reasoning_effort = str(config.get("reasoning_effort", "medium"))
        self.repair_max_output_tokens = int(
            config.get("repair_max_output_tokens", min(self.max_output_tokens, 4096))
        )
        self.repair_reasoning_effort = str(
            config.get("repair_reasoning_effort", "low")
        )
        self.system_prompt = system_prompt
        self.diagnostic_dir = diagnostic_dir
        self.request_count = 0
        self.progress = progress
        key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.environ.get(key_env, "").strip()
        key_file = config.get("api_key_file")
        if not self.api_key and key_file:
            key_path = Path(str(key_file)).expanduser()
            if key_path.is_file():
                self.api_key = key_path.read_text(encoding="utf-8").strip()
        if not self.api_key:
            file_hint = f" or key file {Path(str(key_file)).expanduser()}" if key_file else ""
            raise ValueError(
                f"Missing API key: set environment variable {key_env}{file_hint}"
            )

    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._complete_json(payload, allow_repair=True)

    def _complete_json(
        self, payload: dict[str, Any], allow_repair: bool
    ) -> dict[str, Any]:
        self.request_count += 1
        if self.progress:
            self.progress("llm_request_started", {
                "request": self.request_count,
                "model": self.model,
                "mode": payload.get("mode"),
            })
        is_repair = payload.get("mode") == "json_output_repair"
        body = {
            "model": self.model,
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": self.system_prompt}]},
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
                },
            ],
            "reasoning": {
                "effort": self.repair_reasoning_effort if is_repair else self.reasoning_effort
            },
            "max_output_tokens": (
                self.repair_max_output_tokens if is_repair else self.max_output_tokens
            ),
            "text": {"format": {"type": "json_object"}},
        }
        with tempfile.TemporaryDirectory(prefix="simple-agent-http-") as tmp:
            request_path = Path(tmp) / "request.json"
            response_path = Path(tmp) / "response.json"
            request_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            command = [
                "curl",
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--http1.1",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "30",
                "--max-time",
                str(self.timeout),
                "-H",
                f"Authorization: Bearer {self.api_key}",
                "-H",
                "Content-Type: application/json",
                "-H",
                "Accept: application/json",
                "-H",
                "User-Agent: simple-molecular-agent/0.1",
                "--data-binary",
                f"@{request_path}",
                f"{self.base_url}/responses",
                "-o",
                str(response_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            raw_http_body = response_path.read_text(errors="replace") if response_path.exists() else ""
            if result.returncode != 0:
                detail = (result.stderr or raw_http_body).strip()
                self._write_diagnostic(payload, body, raw_http_body, None, None, f"curl_failed_{result.returncode}")
                raise RuntimeError(f"LLM curl request failed ({result.returncode}): {detail[:1500]}")
            try:
                data = json.loads(raw_http_body)
            except json.JSONDecodeError as error:
                self._write_diagnostic(payload, body, raw_http_body, None, None, "endpoint_invalid_json")
                raise RuntimeError("LLM endpoint returned invalid JSON") from error
        text = self._response_message_text(data)
        if data.get("status") == "incomplete":
            self._write_diagnostic(
                payload, body, raw_http_body, data, text, "assistant_output_truncated"
            )
            if allow_repair:
                return self._repair_incomplete_response(payload)
            raise RuntimeError(
                "LLM assistant output was truncated at the configured token limit; "
                "increase max_output_tokens or lower reasoning_effort"
            )
        if not text:
            self._write_diagnostic(payload, body, raw_http_body, data, text, "empty_message_output")
            raise RuntimeError("LLM response contained no complete assistant message")
        try:
            result = self._extract_json_object(text)
        except (TypeError, json.JSONDecodeError) as error:
            self._write_diagnostic(payload, body, raw_http_body, data, text, "assistant_content_incomplete_json")
            if allow_repair:
                return self._repair_incomplete_response(payload)
            raise RuntimeError(f"LLM did not return a complete JSON object: {text[:1500]}") from error
        if self.progress:
            self.progress("llm_request_completed", {
                "request": self.request_count,
                "action": result.get("action"),
            })
        return result

    @staticmethod
    def _response_message_text(data: dict[str, Any]) -> str:
        """Read final assistant text without treating reasoning as the workflow decision."""
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        chunks: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict) or item.get("type") not in {None, "message"}:
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") not in {"output_text", "text"}:
                    continue
                value = content.get("text")
                if isinstance(value, str):
                    chunks.append(value)
        return "".join(chunks)

    def _repair_incomplete_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.progress:
            self.progress("llm_json_repair_started", {
                "request": self.request_count,
                "mode": payload.get("mode"),
            })
        repair_payload = {
            "mode": "json_output_repair",
            "original_mode": payload.get("mode"),
            "state": payload.get("state"),
            "optimization_context": payload.get("optimization_context"),
            "instruction": (
                "The previous model response was incomplete or did not contain the final JSON decision. "
                "Ignore its reasoning and choose the next valid workflow action from the supplied state. "
                "Return exactly one compact JSON object with action QUERY, QUERY_BATCH, READY, "
                "MARK_UNMODIFIABLE, STOP, or PROPOSE_TOOL. Do not include analysis or markdown."
            ),
        }
        return self._complete_json(repair_payload, allow_repair=False)

    def _write_diagnostic(
        self,
        payload: dict[str, Any],
        request_body: dict[str, Any],
        raw_http_body: str,
        endpoint_json: Any,
        assistant_content: Any,
        failure: str,
    ) -> None:
        if self.diagnostic_dir is None:
            return
        path = self.diagnostic_dir / f"api-error-{self.request_count:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "failure": failure,
                    "payload": payload,
                    "request": request_body,
                    "raw_http_body": raw_http_body,
                    "endpoint_json": endpoint_json,
                    "assistant_content": assistant_content,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        """Extract the first complete JSON object from optional model narration."""
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise json.JSONDecodeError("No complete JSON object found", text, 0)
