from api.llm import _provider


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
