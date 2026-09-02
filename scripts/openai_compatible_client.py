from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tomllib
import tempfile
from typing import Any


class OpenAICompatibleChatClient:
    """OpenAI-compatible Chat Completions client used by independent model tests."""

    def __init__(self, config_path: Path, system_prompt: str):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        codex_settings = self._load_codex_settings(config.get("codex_config_dir"))
        self.base_url = str(codex_settings.get("base_url", config["base_url"])).rstrip("/")
        self.model = str(codex_settings.get("model", config["model"]))
        self.wire_api = str(codex_settings.get("wire_api", config.get("wire_api", "chat_completions")))
        self.timeout = int(config.get("timeout_seconds", 600))
        self.max_output_tokens = int(config.get("max_output_tokens", 8192))
        self.system_prompt = system_prompt
        key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.environ.get(key_env, "").strip()
        if not self.api_key and codex_settings.get("api_key"):
            self.api_key = str(codex_settings["api_key"]).strip()
        if not self.api_key:
            raise ValueError(f"Missing API key environment variable or Codex credential: {key_env}")

    @staticmethod
    def _load_codex_settings(config_dir: str | Path | None) -> dict[str, str]:
        if not config_dir:
            return {}
        root = Path(str(config_dir)).expanduser()
        settings: dict[str, str] = {}
        config_path = root / "config.toml"
        if config_path.is_file():
            with config_path.open("rb") as handle:
                data = tomllib.load(handle)
            provider_name = data.get("model_provider")
            provider = (data.get("model_providers") or {}).get(provider_name, {})
            if isinstance(data.get("model"), str):
                settings["model"] = data["model"]
            if isinstance(provider, dict):
                if isinstance(provider.get("base_url"), str):
                    settings["base_url"] = provider["base_url"]
                if isinstance(provider.get("wire_api"), str):
                    settings["wire_api"] = provider["wire_api"]
        auth_path = root / "auth.json"
        if auth_path.is_file():
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            if isinstance(auth, dict) and isinstance(auth.get("OPENAI_API_KEY"), str):
                settings["api_key"] = auth["OPENAI_API_KEY"]
        return settings

    def complete_json(
        self,
        payload: dict[str, Any],
        diagnostic_path: Path | None = None,
    ) -> dict[str, Any]:
        if self.wire_api == "responses":
            body = {
                "model": self.model,
                "input": [
                    {"role": "developer", "content": [{"type": "input_text", "text": self.system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]},
                ],
                "reasoning": {"effort": "medium"},
                "max_output_tokens": self.max_output_tokens,
                "text": {"format": {"type": "json_object"}},
            }
            endpoint = f"{self.base_url}/responses"
        else:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": self.max_output_tokens,
            }
            endpoint = f"{self.base_url}/chat/completions"
        with tempfile.TemporaryDirectory(prefix="simple-agent-chat-") as tmp:
            request_path = Path(tmp) / "request.json"
            response_path = Path(tmp) / "response.json"
            request_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            command = [
                "curl", "--silent", "--show-error", "--fail-with-body", "--http1.1",
                "--retry", "8", "--retry-all-errors", "--retry-delay", "2",
                "--connect-timeout", "30", "--max-time", str(self.timeout),
                "-H", f"Authorization: Bearer {self.api_key}",
                "-H", "Content-Type: application/json",
                "-H", "Accept: application/json",
                "-H", "User-Agent: simple-molecular-agent/ablation",
                "--data-binary", f"@{request_path}",
                endpoint, "-o", str(response_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                response_detail = response_path.read_text(errors="replace").strip()
                stderr_detail = result.stderr.strip()
                detail = response_detail or stderr_detail
                if diagnostic_path is not None:
                    self._write_diagnostic(
                        diagnostic_path,
                        payload,
                        body,
                        response_detail,
                        None,
                        None,
                        f"curl_failed_{result.returncode}",
                    )
                raise RuntimeError(f"Chat API curl request failed ({result.returncode}): {detail[:1500]}")
            raw_response = response_path.read_text(encoding="utf-8", errors="replace")
            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError as error:
                if diagnostic_path is not None:
                    self._write_diagnostic(
                        diagnostic_path, payload, body, raw_response, None, None,
                        "endpoint_invalid_json",
                    )
                raise RuntimeError(
                    f"Chat API endpoint returned invalid JSON: {raw_response[:1000]}"
                ) from error
        if self.wire_api == "responses":
            text = data.get("output_text")
            if not isinstance(text, str):
                chunks = []
                for item in data.get("output", []):
                    if not isinstance(item, dict) or item.get("type") not in {None, "message"}:
                        continue
                    for content in item.get("content") or []:
                        if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                            if isinstance(content.get("text"), str):
                                chunks.append(content["text"])
                text = "".join(chunks)
            finish_reason = "length" if data.get("status") == "incomplete" else None
        else:
            try:
                choice = data["choices"][0]
                text = choice["message"]["content"]
                finish_reason = choice.get("finish_reason")
            except (KeyError, IndexError, TypeError) as error:
                text = None
                finish_reason = None
                parse_error = error
            else:
                parse_error = None
        if self.wire_api == "responses":
            parse_error = None
        if parse_error is not None:
            if diagnostic_path is not None:
                self._write_diagnostic(
                    diagnostic_path, payload, body, raw_response, data, None,
                    "missing_assistant_message",
                )
            raise RuntimeError("Chat API response contained no assistant message") from error
        if finish_reason == "length":
            if diagnostic_path is not None:
                self._write_diagnostic(
                    diagnostic_path, payload, body, raw_response, data, text,
                    "assistant_output_truncated",
                )
            raise RuntimeError(
                "Chat API assistant output was truncated at the configured token limit; "
                "increase max_output_tokens"
            )
        if not text:
            if diagnostic_path is not None:
                self._write_diagnostic(
                    diagnostic_path, payload, body, raw_response, data, text,
                    "empty_assistant_content",
                )
            raise RuntimeError("Chat API response contained empty assistant content")
        normalized_text = self._normalize_json_content(text)
        try:
            parsed = json.loads(normalized_text)
        except json.JSONDecodeError as error:
            if diagnostic_path is not None:
                self._write_diagnostic(
                    diagnostic_path, payload, body, raw_response, data, text,
                    "assistant_content_invalid_json",
                )
            raise RuntimeError(f"Chat API did not return JSON: {text[:1000]}") from error
        if not isinstance(parsed, dict):
            if diagnostic_path is not None:
                self._write_diagnostic(
                    diagnostic_path, payload, body, raw_response, data, text,
                    f"assistant_json_type_{type(parsed).__name__}",
                )
            raise RuntimeError(
                "Chat API JSON response must be an object; "
                f"got {type(parsed).__name__}: {text[:1000]}"
            )
        return parsed

    @staticmethod
    def _normalize_json_content(text: str) -> str:
        """Accept JSON wrapped in a Markdown code fence, while rejecting truncation."""
        normalized = text.strip()
        if normalized.startswith("```"):
            first_newline = normalized.find("\n")
            if first_newline >= 0:
                normalized = normalized[first_newline + 1 :].strip()
            if normalized.endswith("```"):
                normalized = normalized[:-3].rstrip()
        return normalized

    @staticmethod
    def _write_diagnostic(
        path: Path,
        payload: dict[str, Any],
        request_body: dict[str, Any],
        raw_response: str,
        endpoint_json: Any,
        assistant_content: Any,
        failure: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_request = dict(request_body)
        path.write_text(
            json.dumps(
                {
                    "failure": failure,
                    "payload": payload,
                    "request": safe_request,
                    "raw_http_body": raw_response,
                    "endpoint_json": endpoint_json,
                    "assistant_content": assistant_content,
                    "assistant_content_type": type(assistant_content).__name__,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
