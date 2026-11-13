"""Translating the query before it is embedded.

Retrieval is markedly weaker when the question and the corpus are not in
the same language. Measured against the kdk index, same question:

    comment grouper les marqueurs en grappes sur une carte   0.656, wrong section
    how to group markers into clusters on a map              0.760, right section

So the translation is not a nicety. What these tests guard is that it is
optional, that its direction is read from the model name, and above all
that it never costs a search: a missing model or a failed translation
leaves the query as typed.
"""

import api.clients.translation as translation
from api.config import Config


def configure(monkeypatch, model):
    monkeypatch.setenv("QUERY_TRANSLATION_MODEL", model)
    config = Config()
    monkeypatch.setattr(translation, "get_config", lambda: config)


# --- the direction is the model name -------------------------------------

# Helsinki-NLP names carry it, so serving a French corpus queried in English
# is one value away -- no second setting to keep in step with the first.
def test_the_direction_is_read_from_the_model_name(monkeypatch):
    configure(monkeypatch, "Helsinki-NLP/opus-mt-fr-en")
    assert translation.direction() == ("fr", "en")

    configure(monkeypatch, "Helsinki-NLP/opus-mt-en-fr")
    assert translation.direction() == ("en", "fr")


def test_no_model_means_no_translation(monkeypatch):
    configure(monkeypatch, "")
    assert translation.direction() is None


# A name we cannot read is reported as no direction rather than guessed at:
# translating the wrong way round would quietly ruin every query.
def test_an_unreadable_model_name_yields_no_direction(monkeypatch):
    configure(monkeypatch, "some-org/a-model-without-a-direction")
    assert translation.direction() is None


# --- which queries are translated ----------------------------------------

def test_a_french_query_is_recognised():
    assert translation.looks_like(
        "comment grouper les marqueurs sur une carte", "fr")


def test_an_english_query_is_not_taken_for_french():
    assert not translation.looks_like(
        "how to cluster markers on a map layer", "fr")


# The case that makes the whole feature worth having: a question written in
# French whose technical terms stay in English.
def test_a_french_query_carrying_english_terms_is_still_french():
    assert translation.looks_like(
        "comment personnaliser le style d'une couche GeoJSON", "fr")


# A query of bare keywords belongs to no language. Translating it would be
# a guess, so it is searched as typed.
def test_bare_keywords_are_claimed_by_no_language():
    assert not translation.looks_like("clustering geojson layer", "fr")
    assert not translation.looks_like("clustering geojson layer", "en")


def test_an_unknown_language_is_never_claimed():
    assert not translation.looks_like("wie gruppiere ich marker", "de")


# --- failure never costs a search ----------------------------------------

# The model is 300 MB fetched from the internet. When it is not there the
# search must still run, in the original language, rather than fail.
def test_an_unavailable_model_returns_nothing_rather_than_raising(monkeypatch):
    configure(monkeypatch, "")
    monkeypatch.setattr(translation, "_unavailable", False)
    monkeypatch.setattr(translation, "_model", None)

    assert translation.translate("comment creer un service") is None
