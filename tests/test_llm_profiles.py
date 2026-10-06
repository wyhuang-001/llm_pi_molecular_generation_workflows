"""Offline tests: no model endpoint or real credential is used."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from molecular_agent.cli import build_parser
from molecular_agent.llm import ResponsesClient


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://current.invalid/v1",
        "model": "existing-model",
        "api_key_env": "CURRENT_TEST_KEY",
        "max_output_tokens": 10000,
        "docking": {"enabled": True},
    }))
    return path


def test_current_remains_default_and_explicit_override_works(config, monkeypatch):
    monkeypatch.setenv("CURRENT_TEST_KEY", "fake-current")
    current = ResponsesClient(config)
    assert current.llm_profile == "current"
    assert current.model == "existing-model"
    assert current.wire_api == "responses"
    data = json.loads(config.read_text())
    data["llm_profile"] = "deepseek"
    config.write_text(json.dumps(data))
    assert ResponsesClient(config, llm_profile="current").model == "existing-model"


@pytest.mark.parametrize("profile", ["gpt-5.4-mini", "gpt-5.6-luna"])
def test_gpt_profiles_inherit_current_provider_and_credentials(config, monkeypatch, tmp_path, profile):
    monkeypatch.setenv("CURRENT_TEST_KEY", "fake-current")
    codex = tmp_path / "codex"
    codex.mkdir()
    (codex / "config.toml").write_text(
        'model = "codex-current-model"\n'
        'model_provider = "same-source"\n'
        '[model_providers.same-source]\n'
        'base_url = "https://codex-current.invalid/v1"\n'
    )
    data = json.loads(config.read_text())
    data["codex_config_dir"] = str(codex)
    config.write_text(json.dumps(data))

    current = ResponsesClient(config)
    lightweight = ResponsesClient(config, llm_profile=profile)

    assert current.model == "codex-current-model"
    assert lightweight.model == profile
    assert lightweight.base_url == current.base_url == "https://codex-current.invalid/v1"
    assert lightweight.api_key == current.api_key == "fake-current"
    assert lightweight.wire_api == current.wire_api == "responses"
    if profile == "gpt-5.6-luna":
        assert lightweight.timeout == 600
        assert lightweight.max_api_retries == 5
        assert lightweight.retry_delay_seconds == 10


def test_doubao_isolates_credentials_and_uses_ark_responses(config, monkeypatch, tmp_path):
    monkeypatch.setenv("CURRENT_TEST_KEY", "must-not-use")
    monkeypatch.setenv("ARK_API_KEY", "fake-ark")
    codex = tmp_path / "codex"
    codex.mkdir()
    (codex / "config.toml").write_text("invalid toml deliberately")
    data = json.loads(config.read_text())
    data["codex_config_dir"] = str(codex)
    config.write_text(json.dumps(data))

    client = ResponsesClient(config, llm_profile="doubao")

    assert client.model == "doubao-seed-evolving"
    assert client.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert client.api_key == "fake-ark"
    assert client.wire_api == "responses"


def test_deepseek_isolates_credentials_and_supports_overrides(config, monkeypatch, tmp_path):
    monkeypatch.setenv("CURRENT_TEST_KEY", "must-not-use")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
    codex = tmp_path / "codex"
    codex.mkdir()
    (codex / "config.toml").write_text("invalid toml deliberately")
    data = json.loads(config.read_text())
    data.update({"codex_config_dir": str(codex), "llm_profile": "deepseek"})
    data["llm_profiles"] = {"deepseek": {"model": "test-model-id"}}
    config.write_text(json.dumps(data))
    before = config.read_text()
    client = ResponsesClient(config)
    assert client.model == "test-model-id"
    assert client.base_url == "https://api.deepseek.com/v1"
    assert client.api_key == "fake-deepseek"
    assert client.wire_api == "chat_completions"
    assert config.read_text() == before


def test_deepseek_missing_key_does_not_fall_back_to_current(config, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("CURRENT_TEST_KEY", "must-not-use")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        ResponsesClient(config, llm_profile="deepseek")


def test_deepseek_reads_external_key_file(config, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    keyfile = tmp_path / ".config/simple-molecular-agent/deepseek-api-key"
    keyfile.parent.mkdir(parents=True)
    keyfile.write_text("fake-file-key\n")
    assert ResponsesClient(config, llm_profile="deepseek").api_key == "fake-file-key"


@pytest.mark.parametrize("truncated", [False, True])
def test_chat_completions_wire_format_and_repair(config, monkeypatch, truncated):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
    calls = []

    def fake_run(command, **kwargs):
        assert "https://api.deepseek.com/v1/chat/completions" in command
        assert "--retry-all-errors" in command
        assert command[command.index("--connect-timeout") + 1] == "90"
        body = json.loads(Path(command[command.index("--data-binary") + 1][1:]).read_text())
        assert "input" not in body and "reasoning" not in body
        assert body["messages"][0]["role"] == "system"
        assert body["response_format"] == {"type": "json_object"}
        calls.append(body)
        incomplete = truncated and len(calls) == 1
        Path(command[command.index("-o") + 1]).write_text(json.dumps({
            "choices": [{"finish_reason": "length" if incomplete else "stop",
                         "message": {"content": '{"action":' if incomplete else '{"action":"QUERY","tool":"get_ligand_info","arguments":{}}',
                                     "reasoning_content": "not a workflow decision"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 50},
        }))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("molecular_agent.llm.subprocess.run", fake_run)
    events = []
    client = ResponsesClient(config, llm_profile="deepseek", progress=lambda e, d: events.append((e, d)))
    assert client.complete_json({"mode": "test"})["action"] == "QUERY"
    assert len(calls) == (2 if truncated else 1)
    # DeepSeek reasoning is enabled by default; the profile keeps a large output
    # budget because the reasoning trace shares the answer budget.
    assert client.thinking == {"type": "enabled"}
    assert "thinking" in calls[0] and calls[0]["thinking"] == {"type": "enabled"}
    assert calls[0]["max_tokens"] == client.max_output_tokens >= 16384
    if truncated:
        assert calls[1]["max_tokens"] == client.repair_max_output_tokens >= 16384
    assert events[-1][1]["cached_input_tokens"] == 50


def test_cli_selector_and_script_argument_validation():
    assert build_parser().parse_args(["--llm", "deepseek", "--resume"]).llm == "deepseek"
    assert build_parser().parse_args(["--llm", "gpt-5.4-mini"]).llm == "gpt-5.4-mini"
    assert build_parser().parse_args(["--llm", "gpt-5.6-luna"]).llm == "gpt-5.6-luna"
    assert build_parser().parse_args(["--llm", "doubao"]).llm == "doubao"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "run_docking_loop_test.sh"), "--llm", "unknown"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires current, gpt-5.4-mini, gpt-5.6-luna, doubao, or deepseek" in result.stderr
