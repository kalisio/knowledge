from collections import namedtuple


LLMResponse = namedtuple("LLMResponse", ["answer", "provider", "model"])


# Stub: canned answer until the real LLM provider is wired in
def ask(prompt):
    return LLMResponse(
        answer="(test answer - for test)",
        provider="test",
        model="test",
    )