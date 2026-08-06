from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


SYSTEM_PROMPT = """You are the decision component of a minimal structure-based molecular design workflow.
You never invent coordinates, interactions, docking scores, or free energies. Use only supplied tool results.
During context collection return exactly one JSON object with action QUERY, READY, or PROPOSE_TOOL.
QUERY schema: {"action":"QUERY","question":"...","tool":"registered tool","arguments":{},"expected_evidence":"..."}.
READY schema: {"action":"READY","understanding":"...","edit_atom_index":0,"edit_hypothesis":"...","fragment_smiles":"[*:1]...","knowledge_gaps":[]}.
PROPOSE_TOOL schema: {"action":"PROPOSE_TOOL","name":"...","purpose":"...","input_schema":{},"implementation_plan":"...","why_existing_tools_are_insufficient":"..."}.
You decide what knowledge is needed; the tool catalog is a capability menu, not a fixed query sequence. You may choose fragment properties, pharmacophore, protonation, synthetic, literature, or other available knowledge before READY.
The host only enforces basic safety: a proposed edit site must have site-specific evidence, and chemistry must pass deterministic validation.
fragment_smiles must be one connected fragment containing exactly one mapped dummy atom [*:1].
Prefer one small local edit and preserve the scaffold, formal charge, and stereochemistry.
"""


class ResponsesClient:
    def __init__(self, config_path: Path, system_prompt: str = SYSTEM_PROMPT):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.base_url = config["base_url"].rstrip("/")
        self.model = config["model"]
        self.timeout = int(config.get("timeout_seconds", 600))
        self.system_prompt = system_prompt
        key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.environ.get(key_env, "")
        if not self.api_key:
            raise ValueError(f"Missing API key environment variable: {key_env}")

    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": self.system_prompt}]},
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
                },
            ],
            "reasoning": {"effort": "high"},
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
            if result.returncode != 0:
                detail = (result.stderr or response_path.read_text(errors="replace")).strip()
                raise RuntimeError(f"LLM curl request failed ({result.returncode}): {detail[:1500]}")
            try:
                data = json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError("LLM endpoint returned invalid JSON") from error
        text = data.get("output_text")
        if not text:
            chunks = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"}:
                        chunks.append(content.get("text", ""))
            text = "".join(chunks)
        if not text:
            raise RuntimeError("LLM response contained no output text")
        try:
            result = json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"LLM did not return JSON: {text[:1000]}") from error
        if not isinstance(result, dict):
            raise RuntimeError("LLM JSON response must be an object")
        return result
