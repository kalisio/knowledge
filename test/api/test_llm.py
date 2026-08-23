from types import SimpleNamespace

import pytest
from openai import APIConnectionError, OpenAIError

import api.clients.llm as llm
from api.config import DEFAULT_SYSTEM_PROMPT
from api.clients.llm import _provider


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


# --- ask: when the provider is down ----------------------------------------

# A client whose every call fails the way the openai SDK signals an outage.
@pytest.fixture
def unreachable(monkeypatch):
    def failing(**kwargs):
        raise APIConnectionError(request=None)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=failing)))
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-llm")
    monkeypatch.setenv("LLM_ENDPOINT", "http://llm.invalid/v1/")
    return client


def test_a_provider_that_cannot_be_reached_is_not_a_bare_500(unreachable):
    # An unreachable provider used to escape as an openai exception and come
    # back to the caller as "Internal Server Error" with a traceback in the
    # logs. It is an outage, and the API answers 503 for those.
    with pytest.raises(llm.LLMUnreachable):
        llm.ask("Question?")


def test_the_outage_names_the_endpoint_and_what_to_check(unreachable):
    with pytest.raises(llm.LLMUnreachable) as failure:
        llm.ask("Question?")

    message = str(failure.value)
    assert "http://llm.invalid/v1/" in message
    assert "LLM_ENDPOINT" in message
    assert "APIConnectionError" in message


def test_a_rejected_key_is_reported_the_same_way(monkeypatch):
    # A wrong LLM_API_KEY raises AuthenticationError, another OpenAIError:
    # the caller gets the same readable 503 rather than a 500.
    def failing(**kwargs):
        raise OpenAIError("invalid api key")

    monkeypatch.setattr(llm, "_get_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=failing))))
    monkeypatch.setenv("LLM_API_KEY", "wrong")
    monkeypatch.setenv("LLM_MODEL", "test-llm")
    monkeypatch.setenv("LLM_ENDPOINT", "https://api.anthropic.com/v1/")

    with pytest.raises(llm.LLMUnreachable):
        llm.ask("Question?")
