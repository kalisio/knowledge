import pytest

import api.config as api_config
from api.config import DEFAULT_SYSTEM_PROMPT


# The environment ApiConfig requires (RuntimeConfig + the LLM settings).
# A test overrides only what it exercises.
_BASE_ENV = {
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_COLLECTION_CODE": "knowledge_test_code",
    "QDRANT_COLLECTION_METADATA": "knowledge_test_metadata",
    "EMBEDDING_MODEL": "test-model",
    "LLM_API_KEY": "test-key",
    "LLM_MODEL": "test-llm",
    "LLM_ENDPOINT": "http://localhost:9999",
}

# Optional variables a developer shell may have set; cleared so the tests
# see the documented defaults.
_OPTIONAL = ("KNOWLEDGE_AUTH_ENABLED", "APP_SECRET", "HOST", "PORT",
             "JWT_ALGORITHM", "JWT_AUDIENCE", "JWT_ISSUER", "LLM_PROMPT")


# Build an ApiConfig from a known environment, bypassing the module cache.
@pytest.fixture
def api_env(monkeypatch):
    def build(**overrides):
        for name in _OPTIONAL:
            monkeypatch.delenv(name, raising=False)
        for name, value in {**_BASE_ENV, **overrides}.items():
            monkeypatch.setenv(name, str(value))
        return api_config.Config()
    return build


# --- auth_enabled: only the literal "false" disables auth -----------------

def test_auth_is_enabled_by_default(api_env):
    assert api_env().auth_enabled is True


def test_the_literal_false_disables_auth_in_any_case(api_env):
    assert api_env(KNOWLEDGE_AUTH_ENABLED="false").auth_enabled is False
    assert api_env(KNOWLEDGE_AUTH_ENABLED="False").auth_enabled is False
    assert api_env(KNOWLEDGE_AUTH_ENABLED="FALSE").auth_enabled is False


def test_any_other_value_keeps_auth_enabled(api_env):
    # Fail-safe direction: "0", "off", "no" are NOT recognised -- a typo in
    # the deployment env must leave auth on, never silently open the API.
    for value in ("0", "off", "no", "disabled", "true", "1"):
        config = api_env(KNOWLEDGE_AUTH_ENABLED=value)
        assert config.auth_enabled is True, value


def test_the_app_secret_defaults_to_none(api_env):
    # Optional by design: auth.py fails closed when auth is on without it.
    assert api_env().app_secret is None


# --- the LLM settings are required ----------------------------------------

def test_each_llm_setting_is_required(api_env, monkeypatch):
    for name in ("LLM_API_KEY", "LLM_MODEL", "LLM_ENDPOINT"):
        with pytest.raises(RuntimeError, match=name):
            api_env(**{name: ""})


# --- the default prompts hold their placeholders --------------------------

def test_the_default_prompt_template_formats_with_both_placeholders(api_env):
    # handlers.answer_question .format()s this at request time; a template
    # missing {context} or {question} would only fail in production.
    prompt = api_env().prompt_template.format(
        context="THE-CONTEXT", question="THE-QUESTION")

    assert "THE-CONTEXT" in prompt
    assert "THE-QUESTION" in prompt


def test_the_default_prompt_template_has_no_other_placeholder(api_env):
    # .format() raises KeyError on any placeholder beyond the two provided.
    api_env().prompt_template.format(context="c", question="q")


# --- service binding defaults ---------------------------------------------

def test_the_service_binds_loopback_8187_by_default(api_env):
    # Documented gotcha: inside a container this default is unreachable
    # from the host -- deployments (and the image smoke test) must override
    # HOST to 0.0.0.0.
    config = api_env()
    assert config.host == "127.0.0.1"
    assert config.port == 8187


# --- the system instruction ------------------------------------------------

def test_the_system_prompt_defaults_to_the_kalisio_instruction(api_env):
    assert api_env().system_prompt == DEFAULT_SYSTEM_PROMPT


def test_the_system_prompt_is_overridable_with_llm_prompt(api_env):
    config = api_env(LLM_PROMPT="Answer in French, quoting the sources.")

    assert config.system_prompt == "Answer in French, quoting the sources."


def test_an_empty_llm_prompt_falls_back_to_the_default(api_env):
    # The deployed environment ships LLM_PROMPT="" -- that must not leave the
    # model with no instruction at all.
    assert api_env(LLM_PROMPT="").system_prompt == DEFAULT_SYSTEM_PROMPT
