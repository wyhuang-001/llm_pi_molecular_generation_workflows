from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


class DeepSeekChatClient:
    """Official DeepSeek Chat Completions client used only by the ablation script."""

    def __init__(self, config_path: Path, system_prompt: str):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.base_url = config["base_url"].rstrip("/")
        self.model = config["model"]
        self.timeout = int(config.get("timeout_seconds", 600))
        self.system_prompt = system_prompt
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY environment variable")

    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        with tempfile.TemporaryDirectory(prefix="simple-agent-deepseek-") as tmp:
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
                f"{self.base_url}/chat/completions", "-o", str(response_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                detail = (result.stderr or response_path.read_text(errors="replace")).strip()
                raise RuntimeError(f"DeepSeek curl request failed ({result.returncode}): {detail[:1500]}")
            try:
                data = json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError("DeepSeek endpoint returned invalid JSON") from error
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("DeepSeek response contained no assistant message") from error
        if not text:
            raise RuntimeError("DeepSeek response contained empty assistant content")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"DeepSeek did not return JSON: {text[:1000]}") from error
        if not isinstance(parsed, dict):
            raise RuntimeError("DeepSeek JSON response must be an object")
        return parsed
