"""One-shot external research adapters used before LLM design decisions.

The workflow keeps web retrieval outside the local chemistry tools. A configured
command can be a simple Playwright/MCP bridge, or the adapter can speak the MCP
stdio JSON-RPC transport directly to a Playwright MCP server.
"""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
from pathlib import Path
import select
import shlex
import subprocess
import tempfile
import time
from typing import Any, Protocol


class ResearchAdapter(Protocol):
    def collect(self, request: dict[str, Any], output_dir: Path) -> dict[str, Any]: ...


class ExternalResearchError(RuntimeError):
    pass


class MCPStdioClient:
    """Small dependency-free MCP stdio client for one bounded research pass."""

    def __init__(
        self,
        command: list[str],
        timeout: float = 180.0,
        environment: dict[str, str] | None = None,
    ):
        self.command = command
        self.timeout = timeout
        self.environment = environment or {}
        self._next_id = 0
        self.process: subprocess.Popen[bytes] | None = None
        self._buffer = b""
        self._stderr = None

    def __enter__(self) -> "MCPStdioClient":
        self._stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, bufsize=0,
                env={**os.environ, **self.environment},
            )
            self._request("initialize", {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "simple-molecular-agent", "version": "0.1"},
            })
            self._notify("notifications/initialized", {})
            return self
        except OSError as error:
            self.__exit__()
            raise ExternalResearchError(f"MCP process could not start/initialize: {error}") from error
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_: Any) -> None:
        if self.process is None:
            if self._stderr:
                self._stderr.close()
            return
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.process.kill()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self.process.stdout:
            self.process.stdout.close()
        if self._stderr:
            self._stderr.close()

    def _send(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise ExternalResearchError("MCP stdio process is not running")
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode()
        try:
            while data:
                written = self.process.stdin.write(data)
                if not written:
                    raise ExternalResearchError("MCP stdin closed")
                data = data[written:]
            self.process.stdin.flush()
        except OSError as error:
            raise ExternalResearchError(f"MCP write failed: {error}") from error

    def _read(self, deadline: float) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise ExternalResearchError("MCP stdio process is not running")
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise ExternalResearchError(f"MCP response timed out after {self.timeout:g} seconds")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise ExternalResearchError("MCP stdout closed before response")
            self._buffer += data
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExternalResearchError(
                f"MCP stdio returned a non-JSON line: {line[:500]!r}"
            ) from error
        if not isinstance(value, dict):
            raise ExternalResearchError("MCP response must be a JSON object")
        return value

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + self.timeout
        while True:
            if time.monotonic() >= deadline:
                raise ExternalResearchError("MCP request deadline exceeded")
            message = self._read(deadline)
            if "method" in message and "id" in message:
                if message["method"] == "ping":
                    self._send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
                else:
                    self._send({"jsonrpc": "2.0", "id": message["id"],
                                "error": {"code": -32601, "message": "Unsupported client method"}})
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise ExternalResearchError(
                    f"MCP {method} failed: {json.dumps(message['error'], ensure_ascii=False)}"
                )
            result = message.get("result", {})
            return result if isinstance(result, dict) else {"value": result}

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def list_tools(self) -> dict[str, Any]:
        return self._request("tools/list", {})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )


class PlaywrightResearchAdapter:
    """Run a configured one-shot Playwright/MCP research pass.

    ``transport=stdio`` speaks MCP directly. The default ``command`` transport
    remains available for a project-specific bridge that reads one JSON request
    on stdin and returns one JSON object on stdout.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def required(self) -> bool:
        return bool(self.config.get("required", False))

    @staticmethod
    def _format_value(value: Any, request: dict[str, Any]) -> Any:
        if isinstance(value, str):
            try:
                return value.format_map({key: str(item) for key, item in request.items()})
            except (KeyError, ValueError):
                return value
        if isinstance(value, list):
            return [PlaywrightResearchAdapter._format_value(item, request) for item in value]
        if isinstance(value, dict):
            return {
                key: PlaywrightResearchAdapter._format_value(item, request)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _safe_extension(mime_type: str) -> str:
        mime = mime_type.split(";", 1)[0].lower()
        known = {
            "chemical/x-mdl-sdfile": ".sdf",
            "chemical/x-sdf": ".sdf",
            "chemical/x-mdl-molfile": ".mol",
            "chemical/molfile": ".mol",
            "application/json": ".json",
        }
        return known.get(mime) or mimetypes.guess_extension(mime) or ".bin"

    def _materialize_images(
        self, result: dict[str, Any], output_dir: Path
    ) -> dict[str, Any]:
        """Save inline image bytes and keep only paths in the research bundle."""
        images = result.get("images")
        if not isinstance(images, list):
            return result
        image_dir = output_dir / "artifacts"
        image_dir.mkdir(parents=True, exist_ok=True)
        normalized = []
        for index, item in enumerate(images, start=1):
            if not isinstance(item, dict):
                continue
            item = dict(item)
            mime_type = str(item.get("mime_type") or item.get("mimeType") or "image/png")
            encoded = item.pop("data_base64", None) or item.pop("data", None)
            if isinstance(encoded, str) and encoded and not encoded.startswith(("http://", "https://", "file://")):
                try:
                    data = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise ExternalResearchError("Research image contains invalid base64 data") from error
                import uuid
                path = image_dir / f"research-image-{uuid.uuid4().hex}{self._safe_extension(mime_type)}"
                path.write_bytes(data)
                item["path"] = str(path.resolve())
            item["mime_type"] = mime_type
            item.setdefault("artifact_id", f"research-image-{index:03d}")
            normalized.append(item)
        result["images"] = normalized
        return result

    def _normalize_mcp_result(
        self, tool: str, result: dict[str, Any], output_dir: Path, call_index: int
    ) -> dict[str, Any]:
        text: list[str] = []
        images: list[dict[str, Any]] = []
        resources: list[dict[str, Any]] = []
        structured = result.get("structuredContent")
        for item in result.get("content", []) if isinstance(result.get("content"), list) else []:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "text" and isinstance(item.get("text"), str):
                text.append(item["text"])
            elif kind == "image" and isinstance(item.get("data"), str):
                images.append({
                    "artifact_id": f"mcp-image-{call_index:03d}-{len(images) + 1:02d}",
                    "mime_type": item.get("mimeType", "image/png"),
                    "data_base64": item["data"],
                    "tool": tool,
                })
            elif kind == "resource":
                resource = item.get("resource")
                if isinstance(resource, dict):
                    resource = dict(resource)
                    blob = resource.pop("blob", None)
                    if isinstance(blob, str) and blob:
                        mime_type = str(resource.get("mimeType", "application/octet-stream"))
                        resource_dir = output_dir / "artifacts"
                        resource_dir.mkdir(parents=True, exist_ok=True)
                        suffix = self._safe_extension(mime_type)
                        resource_path = resource_dir / (
                            f"mcp-resource-{call_index:03d}-{len(resources) + 1:02d}{suffix}"
                        )
                        try:
                            resource_path.write_bytes(base64.b64decode(blob))
                        except (ValueError, binascii.Error) as error:
                            raise ExternalResearchError(
                                "MCP resource contains invalid base64 data"
                            ) from error
                        resource["path"] = str(resource_path.resolve())
                    if isinstance(resource.get("text"), str):
                        text.append(resource["text"])
                    resources.append(resource)
        output = {
            "tool": tool,
            "is_error": bool(result.get("isError", False)),
            "text": text,
            "resources": resources,
            "structured": structured,
        }
        if images:
            output["images"] = images
            self._materialize_images(output, output_dir)
        return output

    def _save_text_artifact(
        self,
        normalized: dict[str, Any],
        save_spec: Any,
        output_dir: Path,
        call_index: int,
    ) -> None:
        if not save_spec or not normalized.get("text"):
            return
        if isinstance(save_spec, str):
            filename = save_spec
            mime_type = "text/plain"
        elif isinstance(save_spec, dict):
            filename = str(save_spec.get("filename", ""))
            mime_type = str(save_spec.get("mime_type", "text/plain"))
        else:
            return
        filename = Path(filename).name
        if Path(filename).suffix.lower() in {".sdf", ".mol", ".pdb", ".cif"}:
            raise ExternalResearchError("MCP navigation text is not a structure download; save as .txt instead")
        if not filename:
            raise ExternalResearchError("save_text_as requires a filename")
        artifact_dir = output_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / filename
        path.write_text("\n\n".join(normalized["text"]), encoding="utf-8")
        normalized.setdefault("resources", []).append({
            "artifact_id": f"mcp-text-{call_index:03d}",
            "path": str(path.resolve()),
            "mime_type": mime_type,
            "source_tool": normalized.get("tool"),
        })

    def _collect_mcp_stdio(
        self, request: dict[str, Any], output_dir: Path
    ) -> dict[str, Any]:
        command = self.config.get("mcp_command") or self.config.get("command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list) or not command:
            raise ExternalResearchError(
                "MCP stdio research requires mcp_command or command"
            )
        calls = self._format_value(self.config.get("calls", []), request)
        if not isinstance(calls, list):
            raise ExternalResearchError("external_research.calls must be an array")
        bundle: dict[str, Any] = {
            "status": "complete",
            "source": "playwright_mcp_stdio",
            "command": [str(item) for item in command],
            "tool_catalog": None,
            "calls": [],
            "text": [],
            "images": [],
            "resources": [],
        }
        environment = self.config.get("environment", {})
        if not isinstance(environment, dict):
            raise ExternalResearchError("external_research.environment must be an object")
        current_url = None
        with MCPStdioClient(
            [str(item) for item in command],
            timeout=float(self.config.get("timeout_seconds", 180)),
            environment={str(key): str(value) for key, value in environment.items()},
        ) as client:
            if bool(self.config.get("include_tool_catalog", True)):
                bundle["tool_catalog"] = client.list_tools()
            for index, call in enumerate(calls, start=1):
                if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
                    raise ExternalResearchError("Each MCP research call requires a tool name")
                tool = call["tool"]
                arguments = call.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise ExternalResearchError(f"MCP arguments for {tool} must be an object")
                raw = client.call_tool(tool, arguments)
                normalized = self._normalize_mcp_result(tool, raw, output_dir, index)
                if tool == "browser_navigate":
                    current_url = arguments.get("url")
                normalized.update(source_id=f"call-{index:03d}", source_url=current_url)
                self._save_text_artifact(
                    normalized,
                    call.get("save_text_as"),
                    output_dir,
                    index,
                )
                bundle["calls"].append(normalized)
                bundle["text"].extend(normalized.get("text", []))
                bundle["images"].extend(normalized.get("images", []))
                bundle["resources"].extend(normalized.get("resources", []))
        if any(c["is_error"] for c in bundle["calls"]):
            bundle["status"] = "failed"
        return bundle

    def collect(self, request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        request_path = output_dir / "research-request.json"
        result_path = output_dir / "research-response.json"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        manifest_path = self.config.get("manifest_path")
        if manifest_path:
            path = Path(str(manifest_path)).expanduser()
            if not path.is_absolute():
                path = (output_dir.parent.parent / path).resolve()
            if not path.is_file():
                raise ExternalResearchError(f"Research manifest does not exist: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ExternalResearchError("Research manifest must contain a JSON object")
            result = self._materialize_images(result, output_dir)
            result.setdefault("status", "complete")
            result.setdefault("source", "configured_manifest")
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result

        if str(self.config.get("transport", "command")) in {"stdio", "mcp_stdio"}:
            result = self._collect_mcp_stdio(request, output_dir)
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if self.required and result["status"] != "complete":
                raise ExternalResearchError(f"Required MCP research failed; see {result_path}")
            return result

        command = self.config.get("command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list) or not command:
            if self.required:
                raise ExternalResearchError(
                    "external_research is required but no Playwright/MCP command or manifest_path is configured"
                )
            result = {
                "status": "skipped",
                "source": "playwright_mcp",
                "reason": "No research command or manifest_path configured.",
                "query_executed": False,
            }
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return result

        command = [str(item) for item in command]
        timeout = float(self.config.get("timeout_seconds", 180))
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ExternalResearchError(
                f"Playwright/MCP research command failed to run: {error}"
            ) from error

        stdout = completed.stdout.strip()
        stderr_path = output_dir / "research-stderr.txt"
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise ExternalResearchError(
                f"Playwright/MCP research command returned {completed.returncode}: "
                f"{completed.stderr[-1500:]}"
            )
        try:
            result = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError as error:
            raise ExternalResearchError(
                "Playwright/MCP research command did not return valid JSON"
            ) from error
        if not isinstance(result, dict):
            raise ExternalResearchError("Playwright/MCP research result must be a JSON object")
        result = self._materialize_images(result, output_dir)
        result.setdefault("status", "complete")
        result.setdefault("source", "playwright_mcp")
        result["command"] = command
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
