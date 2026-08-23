from types import SimpleNamespace

import pytest

import api.services.llm as llm
from api.config import DEFAULT_SYSTEM_PROMPT
from api.services.llm import _provider


# --- local endpoints win over everything ----------------------------------

def test_a_localhost_endpoint_is_local():
    assert _provider("http://localhost:11434/v1") == "local"


def test_a_loopback_ip_endpoint_is_local():
    assert _provider("http://127.0.0.1:8080/v1") == "local"


def test_local_takes_precedence_over_a_provider_name_in_the_path():
    # A local proxy serving a provider model is still "local": the check
    # order in _provider puts localhost first, deliberately.
    assert _provider("http://localhost:11434/mistral") == "local"


# --- hosted providers are labelled by substring ---------------------------

def test_known_providers_are_recognised_in_the_endpoint():
    assert _provider("https://api.mistral.ai/v1") == "mistral"
    assert _provider("https://api.openai.com/v1") == "openai"
    assert _provider("https://api.anthropic.com/v1/") == "anthropic"


def test_the_match_is_case_insensitive():
    assert _provider("https://API.OPENAI.COM/v1") == "openai"


def test_a_proxy_embedding_a_provider_name_gets_that_label():
    # Substring matching is the documented behaviour: a gateway named after
    # the provider it fronts is labelled as that provider.
    assert _provider("https://mistral-gateway.internal/v1") == "mistral"


# --- everything else falls back -------------------------------------------

def test_an_unknown_host_falls_back_to_openai_compatible():
    assert _provider("https://llm.example.com/v1") == "openai-compatible"


# --- ask: what is sent to the model, and what comes back -------------------

class _Recorder:
    """An OpenAI-compatible client that records the call and replies."""

    def __init__(self, answer="the catalog stores layer descriptors"):
        self.answer = answer
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.answer))])


@pytest.fixture
def recorder(monkeypatch):
    client = _Recorder()
    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-llm")
    monkeypatch.setenv("LLM_ENDPOINT", "http://localhost:11434/v1/")
    return client


def test_ask_returns_the_answer_of_the_model(recorder):
    assert llm.ask("Question?").answer == recorder.answer


def test_ask_reports_the_model_and_the_provider(recorder):
    response = llm.ask("Question?")

    assert response.model == "test-llm"
    assert response.provider == "local"


def test_ask_sends_the_system_instruction_then_the_prompt(recorder):
    llm.ask("Where is the catalog?")

    messages = recorder.calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert messages[1]["content"] == "Where is the catalog?"


def test_ask_honours_the_configured_system_instruction(recorder, monkeypatch):
    monkeypatch.setenv("LLM_PROMPT", "Answer in French.")

    llm.ask("Question?")

    assert recorder.calls[0]["messages"][0]["content"] == "Answer in French."


def test_ask_caps_the_answer_length(recorder, monkeypatch):
    monkeypatch.setenv("MAX_ANSWER_TOKENS", "128")

    llm.ask("Question?")

    assert recorder.calls[0]["max_tokens"] == 128
