from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import tomllib
import tempfile
from typing import Any, Callable

from .research_memory import PROMPT as RESEARCH_MEMORY_PROMPT


SYSTEM_PROMPT = """You are the decision component of a structure-based molecular optimization workflow.
Never invent coordinates, interactions, docking scores, free energies, fragment records, or activity. Use only supplied host results.
Return exactly one JSON object with action QUERY, QUERY_BATCH, READY, MARK_UNMODIFIABLE, STOP, or PROPOSE_TOOL.
The top-level object MUST contain the string field `action`; never return only tool arguments. Do not output analysis, chain-of-thought, or markdown; output the final JSON object only.
QUERY schema: {"action":"QUERY","question":"...","tool":"registered tool","arguments":{},"expected_evidence":"..."}.
QUERY_BATCH schema: {"action":"QUERY_BATCH","questions":"...","queries":[{"tool":"...","arguments":{}}]}.
The `get_edit_site_candidates` tool returns one structured dossier of every host-supported editable atom and replacement site, including local protein contacts, current ligand interactions, and directional clearance. Use it before `assess_edit_sites`.
The `assess_edit_sites` tool accepts `{"sites":[{"target_type":"atom|bond","target_id":...,"priority":1,"site_type":"core_anchor|pocket_extension|solvent_exposed|linker_or_sidechain|uncertain","rationale":"...","search_status":"hard-reject|pilot|active"}],"global_rationale":"..."}`. Call it only through the QUERY schema: `{"action":"QUERY","question":"Create the evidence-backed site strategy.","tool":"assess_edit_sites","arguments":{"sites":[...],"global_rationale":"..."},"expected_evidence":"site_strategy"}`. The action value must be `QUERY`, never `assess_edit_sites`; all `sites` and `global_rationale` fields must be nested inside `arguments`. Use it after collecting pocket, interaction, ligand, and replacement-site evidence; include every plausible host target you want searched, with unique priorities. The host validates IDs and stores this as the site strategy. Use `hard-reject` only when deterministic tool evidence proves the target infeasible; use `pilot` or `active` for viable targets. Do not invent targets.
The unified fragment library combines curated minimal/small substituents with the audited ChEMBL working subset. Each record exposes size_class, chemical_tags, properties, allowed_operations, and provenance. Use size_class and chemical_tag to choose an evidence-backed action level; do not mechanically exhaust all smaller classes before a larger class when the local environment supports a different hypothesis.
Evaluate one chemically distinct transformation at a time. Do not enumerate or submit a batch of fragments. Use the library search and spatial-profile tools only when one specific next hypothesis requires them.
Every READY names a host-listed site and one change_type that the site allows. Schema: {"action":"READY","understanding":"concrete evidence summary","edit_hypothesis":"concrete testable hypothesis","site_type":"atom|bond|ring","change_type":"addition|deletion|replacement", ...site payload...}.
atom:addition payload: {"edit_atom_index":9,"fragment_smiles":"[*:1]F"}. bond:replacement payload: {"bond_site_id":"bond-site-001","fragment_smiles":"[*:1]C"}. bond:deletion payload: {"bond_site_id":"bond-site-003"} with no fragment. atom/ring:replacement payload: {"edit_atom_index":17,"element":"N"} with no fragment. A local child uses the same axes plus `parent_attempt`.
When a returned library record is used, add "fragment_id":"chembl-brics-filtered-000001". Omit fragment_id entirely when no real library record is used; never send placeholder values such as optional, none, null, `...`, or invented atom indices.
MARK_UNMODIFIABLE schema: {"action":"MARK_UNMODIFIABLE","target_type":"atom","target_id":17,"scope":"site","family":"non_halogen","reason":"..."}. Use target_type `atom` with an integer target_id or `bond_site` with a host-returned bond_site_id. Use scope `site` to close the whole target, or scope `family` with a valid family. A declaration is an auditable search decision, not a claim that the molecule is globally optimized.
STOP schema: {"action":"STOP","reason":"...","evidence":"..."}.
PROPOSE_TOOL schema: {"action":"PROPOSE_TOOL","name":"...","purpose":"...","input_schema":{},"implementation_plan":"...","why_existing_tools_are_insufficient":"..."}.
Use QUERY_BATCH only for independent calls whose arguments do not depend on another result. Use sequential QUERY calls when one result determines the next query. Every tool signature can be executed at most once. Duplicate calls are idempotently reused or skipped; do not repeat them. Return a genuinely new QUERY, a chemically distinct READY transformation, a precise MARK_UNMODIFIABLE declaration, or STOP.
You decide what knowledge is needed; the tool catalog is a capability menu. In site-lock mode, first create a site strategy with assess_edit_sites. The host then exposes active_target and site_search; the active target is authoritative until its local search is complete. Each site lists allowed_change_types. addition grows at a site, deletion removes a bound side, and replacement swaps an element or a bound side. Do not default to growing: a deletion is a real, cheap hypothesis, and a replacement is often the smallest informative edit. Treat a deletion as genuinely two-sided rather than as a loss. The host reports what a deletion would destroy (removed_reference_interactions, removed_polar_roles) and what it changes (removed_heavy_atoms, removed_rotatable_bonds, core_overlap). Losing those contacts can hurt binding; removing a group can equally relieve strain, delete a buried unsatisfied polar group, cut rotatable bonds and conformational-entropy cost, or let the remaining scaffold pack and re-pose better. Never assume a deletion lowers the docking score: the direction is site-dependent and is decided by the measured same-protocol paired delta after docking. Non-empty core_overlap means the edit would change the pose-comparison core, and the host rejects it structurally. Query the chemical and spatial facts needed to compare them, then choose the site, change_type, and fragment supported by those facts. The host does not preselect a design region or change type for you. Before READY, state which evidence supports the chosen change_type and why the alternatives are not currently better supported. For every proposed fragment, first obtain its chemical properties and attachment-centered 3D profile; for bond:replacement, obtain a change_type-compatible library record from search_fragment_library. Search the unified fragment library and inspect reference-ligand fragments when those facts can improve the next transformation. Choose among minimal, small, medium, and large candidates from local clearance, chemical environment, fragment properties, spatial profile, and docking feedback. For search_fragment_library, query must be one supported chemical term such as heterocycle, pyridine, morpholine, indole, oxetane, or nitrile; one valid SMILES/SMARTS pattern; or an empty string for browsing. Use optional size_class and chemical_tag filters where useful. Never send a natural-language phrase such as "small polar heterocycle hydrogen bond donor acceptor". Fragment-library results are change_type-specific: only use a record returned for the change_type you are proposing. Prefer fragment_id when using a returned record.
atom:addition grows from an atom with a replaceable H; query get_atom_environment and check_growth_space first. bond:deletion and bond:replacement need a bond_site_id from list_bond_sites. A bond site reports removed_reference_interactions, removed_polar_roles, removed_rotatable_bonds and core_overlap; use them to judge a deletion in both directions instead of assuming it always loses score. Query get_bond_site_spatial_profile when attachment direction or local clearance is uncertain. Query get_fragment_spatial_profile for a candidate fragment when its 3D extent, shape, or attachment-centered size matters. These spatial tools return geometry facts only; interpret them using chemical knowledge, then call validate_candidate_geometry. Never invent cut_bond indices. The host fixes the retained scaffold, removed side, attachment atom, and direction for each site ID.
Generation 0 is the original co-crystal ligand. During the initial coarse exploration, propose one transformation of the original ligand. After a target is promoted, local optimization may apply exactly one transformation to a previously evaluated parent candidate at that same edit target. For a local child, include `parent_attempt` equal to the parent candidate's attempt number and keep the same target site; omit it for generation-1 candidates from the original ligand. Do not perform multi-site combination edits.
Before READY, query get_atom_environment for the retained edit atom and validate_candidate_geometry for the exact complete transformation. For a local child, include the parent_attempt in the exact geometry query and use the parent candidate path supplied in the optimization context; the geometry check must use that parent structure. For a bond site the retained edit atom is the retained_atom_index returned by the selected bond site. atom:addition requires check_growth_space; a bond site requires a prior list_bond_sites result and its attachment vector replaces the growth probe. bond:deletion needs no fragment. If READY is rejected with failure_class ready_evidence_missing, the host may execute recommended_queries and then resubmit the same transformation with its original understanding and edit_hypothesis before selecting anything new. For a local child, every parent-specific environment, growth-space, and exact geometry query must include the same parent_attempt. If READY is rejected with failure_class invalid_ready, correct the concrete site_type, change_type, site ID, and payload fields before retrying. A chemistry or geometry rejection is already an exploration attempt; choose a different transformation or explicitly MARK_UNMODIFIABLE when the target or family is not chemically supported. Do not repeat a rejected spatial query or geometry validation call; use the returned result to select a different unexecuted query or transformation.
After each docking evaluation, inspect candidate_history and docking_history, including the transformation, canonical SMILES, chemistry/clash status, primary metric delta, seed standard deviation and win fraction, pose_consensus, interaction_consensus, incumbent best attempt, trend, failed transformations, and remaining chemically plausible options. Prefer hypotheses supported across seeds and consistent poses rather than one favorable seed. Keep proposing transformations at active_target while its site_search status is active; do not jump to another target merely because one candidate was worse. Never repeat a transformation in attempted_transformations. Query only unexecuted calls; prior observations are authoritative.
The host does not impose a fixed number of design regions or a minimum number of attempts per site. Inspect the tool evidence and accumulated results for each target. Continue a target when its chemical environment, fragment properties or 3D profile, docking trend, pose consensus, or interaction evidence supports another chemically distinct hypothesis. Close a target with MARK_UNMODIFIABLE only after reviewing that evidence and explaining why no credible local option remains. A hard-reject requires deterministic host-tool evidence; an uncertain site may receive a pilot or be paused without artificial coverage requirements. STOP is allowed only after every target has either been deterministically rejected or explicitly closed by an evidence-backed MARK_UNMODIFIABLE decision. The host-level hard attempt limit is an emergency process safeguard, not a scientific stopping rule or a per-site search requirement. Host-ineligible hydrogen atoms are reported for audit but are not pending edit sites. Exploration means an explicit transformation was attempted, including chemistry/geometry/valence/clash rejection, or the LLM returned a precise MARK_UNMODIFIABLE declaration accepted by the host. Such records count for coverage but never count as successful docking evidence. Do not repeat an attempted transformation. You may switch between edit atoms and replacement sites whenever the accumulated evidence supports a new hypothesis. A candidate that is worse than the reference is informative and does not by itself require stopping; distinguish exploration feedback from the best-so-far candidate. Continue when the overall primary-metric trend is improving, when a reference-better candidate can plausibly be refined, or when unstable secondary evidence justifies a confirming design. Return STOP only after every target has either been deterministically rejected or explicitly closed by an evidence-backed MARK_UNMODIFIABLE decision, unless the hard safety limit is reached. Distinguish docking ranking from confirmed activity. The hard attempt limit is a safety limit, not scientific convergence.
Do not claim that every candidate improves. Distinguish each attempt from the monotonic best-so-far trace. Do not call a polar proximity a hydrogen bond unless host evidence shows a donor-acceptor role pairing; acceptor-acceptor and donor-donor pairs are not hydrogen bonds. Treat distance-only contacts as hypotheses, not established interactions.
fragment_smiles must be one connected fragment containing exactly one mapped dummy atom [*:1]. Preserve the intended scaffold, formal charge, and stereochemistry unless an explicit audited transformation allows otherwise.
"""


SINGLE_EDIT_SYSTEM_PROMPT = """You are the single-edit strategy component of a protein-ligand optimization workflow.
The host has prepared a ligand/PDB dossier. External research is available only if explicitly
present in the supplied payload; closed-pool tasks deliberately disable it.
Use only the supplied dossier, prior candidate feedback, and docking results. The dossier contains three linked views of the molecule: SMILES strings for exchange, a 2D molecule graph for connectivity, and a bounded 3D complex geometry object for receptor-frame coordinates, edit vectors, distances, and interaction edges. Treat the 3D object as host evidence; do not infer an interaction from proximity alone. Do not call or
request local chemistry tools or web research. `state.sar_memory` is the host-generated structured local SAR record; use its RMSD, pose classification, and interaction retained/gained/lost fields as observed evidence, not as experimental activity labels. The host will perform all
chemical construction, geometry checks, duplicate checks, docking, and interaction analysis.
Return exactly one JSON object with action READY or STOP. Do not output markdown or chain-of-thought.
READY schema: {"action":"READY","understanding":"...","edit_hypothesis":"...","site_type":"atom|bond|ring","change_type":"addition|deletion|replacement","edit_atom_index":10,"bond_site_id":"optional supplied bond-site id","fragment_id":"optional supplied id","fragment_smiles":"[*:1]F","element":"required only for an atom/ring element replacement"}. Each site lists allowed_change_types; choose one of them.
For atom:addition choose an editable atom from the supplied dossier. For bond:deletion or
bond:replacement choose a bond_site_id from the supplied dossier; never invent a cut bond. A deletion
is not automatically worse: weigh the reported removed_reference_interactions and removed_polar_roles
(losses) against removed_heavy_atoms and removed_rotatable_bonds (strain, desolvation and entropy
gains), and let the measured docking delta decide. A non-empty core_overlap is a structural rejection,
not a bad score. You choose
the site and change_type from the supplied context; normally no site is preselected by the host. If selection_contract.mode
is closed_pool, the task deliberately supplies exactly one allowed site and a complete fragment_catalog:
choose only that bond_site_id and one listed fragment_id (or chemically identical listed SMILES).
No free-form out-of-catalog chemistry, alternate sites, parents, or literature lookup is permitted. All
catalog entries are equally available; do not infer experimental activity or provenance from their IDs.
If selection_contract.mode is closed_pool_multisite, the pool is cut-point free: choose any listed site
and any change_type that site lists in allowed_change_types. Supply a listed fragment_id for addition or
bond replacement, an element symbol for an atom or ring replacement, and no fragment at all for a bond
deletion. The same compound can be reachable through more than one site and fragment, so do not assume a
fragment belongs to one fixed site.
Submit exactly one single-site
transformation per decision. A fragment_id must be one of the supplied dossier records; fragment_smiles
must be a connected valid attachment fragment with one mapped [*:1] atom. Every candidate edits the
original co-crystal ligand, never a prior candidate. Do not send parent_attempt. Previous candidates
and the best-so-far are comparison evidence only, not construction bases. If the previous candidate
failed or docked poorly, propose a chemically distinct alternative. Use the cumulative learning_summary
when provided: it preserves the incumbent, concise outcomes for all tried structures, and observed
structural contrasts. Treat those contrasts as hypotheses, not proven causal effects. Docking is a ranking signal, not proof
of experimental activity. When pose_retention_protocol is supplied, preserve the identity of its fixed
comparison core and its small set of calibrated atom-level anchors. These are post-docking gates, NOT
fixed coordinates: the ligand is free to move. Only eligible Evaluation Poses are ranked; a better score
in a different mode is not an improvement under this protocol. Missing/outlier seeds count against
coverage; no_eligible_pose is a real rejection, not permission to use rank 1. Other PLIP contacts are soft feedback.
STOP may be returned when the supplied budget, trend, and evidence support stopping.
"""

MULTISITE_EDIT_SYSTEM_PROMPT = """You are the edit-planning component of a structure-based molecular optimization workflow.
The host supplies two independent lists and nothing that links them:

1. `design_dossier.edit_sites` - the host-fixed edit sites.  `atom_sites` are H-bearing atoms where a
   substituent can be attached or an element can be swapped.  `cut_sites` are directed non-ring bond
   cuts where a substituent can be replaced by a whole fragment.  Each site lists the exact
   `allowed_operations`; a site with no allowed operations is protected and rejects everything.
2. `design_dossier.fragment_catalog` - neutral fragments with one mapped `[*:1]` attachment atom.  A
   catalog record never states a site, a cut bond or a position on the reference ligand.

YOU decide which fragment goes to which site with which operation.  The host decides the edit layer
from the product it builds: one or two heavy-atom changes without a ring-framework change is a minimal
edit (T1); three or more heavy-atom changes, or any ring-size change or ring-skeleton atom substitution,
is a whole-fragment or whole-ring replacement (T2).  You never declare the layer.
Return exactly one JSON object with action READY or STOP. Do not output markdown or chain-of-thought.
READY schema: {"action":"READY","understanding":"...","edit_hypothesis":"...","site_type":"atom|bond|ring","change_type":"addition|deletion|replacement","target_type":"atom|bond","target_id":<int or site_id>,"fragment_id":"supplied catalog id","fragment_smiles":"supplied catalog smiles","element":"required only for an atom/ring replacement","knowledge_gaps":[]}.
Rules:
- Choose only change types listed in the chosen site's allowed_change_types.  Never invent a site id, a cut
  bond, an atom index or a fragment.
- addition attaches a fragment at a listed atom site by replacing one hydrogen.
- deletion removes the listed removed side and attaches nothing. replacement at a bond site deletes the
  listed removed side and attaches the chosen fragment there.
- an atom or ring replacement changes one atom's element and needs `element` rather than a fragment.  Use it for cases
  such as halogen exchange or a ring heteroatom replacement.
- Prefer a minimal edit when a one- or two-atom change already expresses the hypothesis: the host will
  report it as T1 and it is a cheaper, more interpretable test than rebuilding a whole side chain.
- Reserve whole-fragment replacement for changes that genuinely need three or more atoms or a different
  ring framework.
- Every candidate edits the original co-crystal ligand.  Do not send parent_attempt.
- Docking is a ranking signal, not proof of experimental activity.  When pose_retention_protocol is
  supplied, its fixed comparison core and calibrated anchors are post-docking gates, not fixed coordinates.
- STOP when the supplied budget, trend and evidence support stopping.
"""




class ResponsesClient:
    def __init__(
        self,
        config_path: Path,
        system_prompt: str = SYSTEM_PROMPT,
        diagnostic_dir: Path | None = None,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
        llm_profile: str | None = None,
    ):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.llm_profile = llm_profile or config.get("llm_profile", "current")
        inherited_model_override: str | None = None
        if self.llm_profile != "current":
            profiles = json.loads(
                (Path(__file__).parent / "data" / "llm_profiles.json").read_text(encoding="utf-8")
            )
            if self.llm_profile not in profiles:
                raise ValueError(f"Unknown LLM profile: {self.llm_profile}")
            profile = {**profiles[self.llm_profile], **(config.get("llm_profiles", {}).get(self.llm_profile) or {})}
            inherit_current_provider = bool(profile.pop("inherit_current_provider", False))
            if inherit_current_provider:
                inherited_model_override = str(profile["model"])
            else:
                # Independent providers must not inherit the current provider's credentials or Codex override.
                for field in ("codex_config_dir", "api_key_env", "api_key_file", "api_key"):
                    config.pop(field, None)
            config.update(profile)
        self.wire_api = config.get("wire_api", "responses")
        if self.wire_api not in {"responses", "chat_completions"}:
            raise ValueError(f"Unsupported wire_api: {self.wire_api}")
        codex_config_dir = config.get("codex_config_dir")
        codex_settings = self._load_codex_settings(codex_config_dir) if codex_config_dir else {}
        self.base_url = str(codex_settings.get("base_url", config["base_url"])).rstrip("/")
        self.model = inherited_model_override or str(codex_settings.get("model", config["model"]))
        self.timeout = int(config.get("timeout_seconds", 600))
        self.max_api_retries = int(config.get("max_api_retries", 5))
        self.retry_delay_seconds = float(config.get("retry_delay_seconds", 10))
        self.max_output_tokens = int(config.get("max_output_tokens", 8192))
        # Image artifacts are sent as multimodal parts when present. Set
        # send_images=false for a text-only model/provider.
        self.send_images = bool(config.get("send_images", True))
        self.reasoning_effort = str(
            config.get("reasoning_effort", codex_settings.get("reasoning_effort", "medium"))
        )
        self.repair_max_output_tokens = int(
            config.get("repair_max_output_tokens", min(self.max_output_tokens, 4096))
        )
        self.repair_reasoning_effort = str(
            config.get("repair_reasoning_effort", "low")
        )
        # Chat Completions providers differ in which optional fields they accept, so
        # these are only sent when the configuration asks for them.  DeepSeek needs
        # them: thinking mode shares the output budget with the answer, so a long
        # reasoning trace can consume every token and leave an empty message.
        self.thinking = config.get("thinking") if isinstance(config.get("thinking"), dict) else None
        self.chat_reasoning_effort = (
            str(config["chat_reasoning_effort"]) if config.get("chat_reasoning_effort") else None
        )
        self.temperature = config.get("temperature")
        self.system_prompt = system_prompt
        self.diagnostic_dir = diagnostic_dir
        self.request_count = 0
        self.progress = progress
        key_env = config.get("api_key_env", "OPENAI_API_KEY")
        # A local config-file key is supported for this deployment. Environment
        # and Codex credentials remain fallbacks so existing configurations keep
        # working; no key is written by the project itself.
        self.api_key = str(config.get("api_key") or "").strip()
        if not self.api_key:
            self.api_key = os.environ.get(key_env, "").strip()
        if not self.api_key and codex_settings.get("api_key"):
            self.api_key = str(codex_settings["api_key"]).strip()
        key_file = config.get("api_key_file")
        if not self.api_key and key_file:
            key_path = Path(str(key_file)).expanduser()
            if key_path.is_file():
                self.api_key = key_path.read_text(encoding="utf-8").strip()
        if not self.api_key:
            file_hint = f" or key file {Path(str(key_file)).expanduser()}" if key_file else ""
            codex_hint = f" or Codex credentials under {Path(str(codex_config_dir)).expanduser()}" if codex_config_dir else ""
            raise ValueError(
                f"Missing API key: fill the api_key field in {config_path.resolve()}. "
                f"No export is required. Optional fallbacks: environment variable {key_env}{file_hint}{codex_hint}"
            )

    @staticmethod
    def _load_codex_settings(config_dir: str | Path | None) -> dict[str, str]:
        """Load model endpoint and credential from the local Codex configuration."""
        if not config_dir:
            return {}
        root = Path(str(config_dir)).expanduser()
        settings: dict[str, str] = {}
        config_path = root / "config.toml"
        if config_path.is_file():
            with config_path.open("rb") as handle:
                config = tomllib.load(handle)
            provider_name = config.get("model_provider")
            model = config.get("model")
            providers = config.get("model_providers") or {}
            provider = providers.get(provider_name) if isinstance(provider_name, str) else None
            if isinstance(model, str) and model:
                settings["model"] = model
            if isinstance(config.get("model_reasoning_effort"), str):
                settings["reasoning_effort"] = config["model_reasoning_effort"]
            if isinstance(provider, dict) and isinstance(provider.get("base_url"), str):
                settings["base_url"] = provider["base_url"]
        auth_path = root / "auth.json"
        if auth_path.is_file():
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            value = auth.get("OPENAI_API_KEY") if isinstance(auth, dict) else None
            if isinstance(value, str) and value:
                settings["api_key"] = value
        return settings

    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._complete_json(payload, allow_repair=True)

    @staticmethod
    def _is_image_record(value: Any) -> bool:
        return isinstance(value, dict) and (
            str(value.get("mime_type", value.get("mimeType", ""))).startswith("image/")
            or value.get("type") == "image"
        )

    @classmethod
    def _text_payload_without_image_bytes(cls, value: Any) -> Any:
        """Keep image metadata in JSON while removing binary data from text input."""
        if isinstance(value, dict):
            result = {
                key: cls._text_payload_without_image_bytes(item)
                for key, item in value.items()
                if key not in {"data_base64", "base64_data"}
            }
            if cls._is_image_record(value):
                result.pop("data", None)
                if "path" not in result and "url" not in result:
                    result["data_available"] = True
            return result
        if isinstance(value, list):
            return [cls._text_payload_without_image_bytes(item) for item in value]
        return value

    @staticmethod
    def _image_data_url(record: dict[str, Any]) -> str | None:
        mime_type = str(record.get("mime_type") or record.get("mimeType") or "image/png")
        encoded = record.get("data_base64") or record.get("base64_data")
        if isinstance(encoded, str) and encoded:
            return f"data:{mime_type};base64,{encoded}"
        path = record.get("path")
        if isinstance(path, str) and Path(path).is_file():
            encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
        url = record.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://", "data:")):
            return url
        return None

    @classmethod
    def _image_records(cls, payload: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if cls._is_image_record(value):
                    key = str(value.get("artifact_id") or value.get("path") or value.get("url") or id(value))
                    if key not in seen:
                        seen.add(key)
                        records.append(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
        return records

    @classmethod
    def _image_content_parts(
        cls, payload: dict[str, Any], wire_api: str
    ) -> list[dict[str, Any]]:
        parts = []
        for record in cls._image_records(payload):
            data_url = cls._image_data_url(record)
            if not data_url:
                continue
            if wire_api == "chat_completions":
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
            else:
                parts.append({"type": "input_image", "image_url": data_url})
        return parts

    def _complete_json(
        self, payload: dict[str, Any], allow_repair: bool
    ) -> dict[str, Any]:
        self.request_count += 1
        text_payload = self._text_payload_without_image_bytes(payload)
        payload_json = json.dumps(text_payload, ensure_ascii=False, separators=(",", ":"))
        image_parts = (
            self._image_content_parts(payload, self.wire_api) if self.send_images else []
        )
        payload_bytes = len(payload_json.encode("utf-8"))
        multisite_mode = payload.get("mode") in {"multisite_edit_design", "multisite_edit_decision_repair"} or payload.get("original_mode") in {"multisite_edit_design", "multisite_edit_decision_repair"}
        single_edit_mode = payload.get("mode") in {"single_edit_design", "single_edit_decision_repair"} or payload.get("original_mode") in {"single_edit_design", "single_edit_decision_repair"}
        if payload.get("mode") == "initial_research_analysis" or payload.get("original_mode") == "initial_research_analysis":
            active_system_prompt = RESEARCH_MEMORY_PROMPT
        elif multisite_mode:
            active_system_prompt = MULTISITE_EDIT_SYSTEM_PROMPT
        elif single_edit_mode:
            active_system_prompt = SINGLE_EDIT_SYSTEM_PROMPT
        else:
            active_system_prompt = self.system_prompt
        estimated_input_tokens = max(1, (len(active_system_prompt.encode("utf-8")) + payload_bytes) // 4)
        if self.progress:
            self.progress("llm_request_started", {
                "request": self.request_count,
                "model": self.model,
                "llm_profile": self.llm_profile,
                "wire_api": self.wire_api,
                "mode": payload.get("mode"),
                "payload_bytes": payload_bytes,
                "estimated_input_tokens": estimated_input_tokens,
                "context_sections": sorted(
                    key for key in ("state", "optimization_context", "tool_catalog", "working_memory")
                    if key in payload or key in (payload.get("optimization_context") or {})
                ),
            })
        is_repair = payload.get("mode") == "json_output_repair"
        responses_user_content = [
            {"type": "input_text", "text": payload_json},
            *image_parts,
        ]
        body = {
            "model": self.model,
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": active_system_prompt}]},
                {"role": "user", "content": responses_user_content},
            ],
            "reasoning": {
                "effort": self.repair_reasoning_effort if is_repair else self.reasoning_effort
            },
            "max_output_tokens": (
                self.repair_max_output_tokens if is_repair else self.max_output_tokens
            ),
            "text": {"format": {"type": "json_object"}},
        }
        endpoint = "responses"
        if self.wire_api == "chat_completions":
            endpoint = "chat/completions"
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": active_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": payload_json},
                            *image_parts,
                        ] if image_parts else payload_json,
                    },
                ],
                "max_tokens": self.repair_max_output_tokens if is_repair else self.max_output_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
            }
            if self.thinking is not None:
                body["thinking"] = dict(self.thinking)
            if self.chat_reasoning_effort is not None:
                body["reasoning_effort"] = self.chat_reasoning_effort
            if self.temperature is not None:
                body["temperature"] = self.temperature
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
                str(self.max_api_retries),
                "--retry-all-errors",
                "--retry-delay",
                str(self.retry_delay_seconds),
                "--retry-max-time",
                str(self.timeout),
                "--connect-timeout",
                "90",
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
                f"{self.base_url}/{endpoint}",
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
        if self.wire_api == "chat_completions":
            choices = data.get("choices") or []
            choice = choices[0] if choices else {}
            text = (choice.get("message") or {}).get("content")
            text = text if isinstance(text, str) else ""
            incomplete = choice.get("finish_reason") == "length"
        else:
            text = self._response_message_text(data)
            incomplete = data.get("status") == "incomplete"
        if incomplete:
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
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            self.progress("llm_request_completed", {
                "request": self.request_count,
                "action": result.get("action"),
                "estimated_input_tokens": estimated_input_tokens,
                "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                "cached_input_tokens": usage.get("prompt_cache_hit_tokens", usage.get("cached_input_tokens", usage.get("cache_read_input_tokens", (usage.get("input_tokens_details") or {}).get("cached_tokens")))),
                "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                "reasoning_tokens": (usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
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
        if payload.get("mode") == "initial_research_analysis":
            raise RuntimeError("Initial evidence summary incomplete; no design proceeded. Inspect diagnostics before retrying.")
        if self.progress:
            self.progress("llm_json_repair_started", {
                "request": self.request_count,
                "mode": payload.get("mode"),
            })
        repair_payload = {
            **payload,
            "mode": "json_output_repair",
            "original_mode": payload.get("mode"),
            "state": payload.get("state"),
            "optimization_context": payload.get("optimization_context"),
            "instruction": (
                "The previous model response was incomplete or did not contain the final JSON decision. "
                "Ignore its reasoning and choose the next valid workflow action from the supplied state. "
                "For single_edit_design, return only READY or STOP and do not call local tools. Otherwise "
                "return exactly one compact JSON object with action QUERY, QUERY_BATCH, READY, "
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
