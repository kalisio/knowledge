"""Translating the query before it is embedded.

Retrieval is markedly weaker when the question and the corpus are not in
the same language. Measured against the kdk index, same question:

    comment grouper les marqueurs en grappes sur une carte   0.656, wrong section
    how to group markers into clusters on a map              0.760, right section

So the translation is not a nicety. What these tests guard is that it is
optional, that several directions can be served at once, that a direction
is read from the model name, that a model only ever runs on a query written
in its source language, and above all that it never costs a search: a
missing model or a failed translation leaves the query as typed.
"""

import api.clients.translation as translation
from api.config import Config

FR_EN = "Helsinki-NLP/opus-mt-fr-en"
EN_FR = "Helsinki-NLP/opus-mt-en-fr"


def configure(monkeypatch, models):
    monkeypatch.setenv("QUERY_TRANSLATION_MODEL", models)
    config = Config()
    monkeypatch.setattr(translation, "get_config", lambda: config)


# A stand-in for the Marian models: records what it was asked to translate,
# with which model, and answers a tagged text.
def fake_translator(monkeypatch):
    calls = []

    def translate(text, name=None):
        calls.append((name, text))
        return f"[{name}] {text}"
    monkeypatch.setattr(translation, "translate", translate)
    return calls


# --- the configured models ------------------------------------------------

def test_no_model_means_no_translation(monkeypatch):
    configure(monkeypatch, "")
    assert translation.models() == []
    assert translation.directions() == []


def test_one_model_is_one_direction(monkeypatch):
    configure(monkeypatch, FR_EN)
    assert translation.directions() == [("fr", "en", FR_EN)]


# A corpus mixing English and French pages needs both directions at once:
# the models are comma-separated, spaces tolerated.
def test_several_models_serve_several_directions(monkeypatch):
    configure(monkeypatch, f"{FR_EN}, {EN_FR}")
    assert translation.directions() == [("fr", "en", FR_EN),
                                        ("en", "fr", EN_FR)]


# --- the direction is the model name -------------------------------------

# Helsinki-NLP names carry it, so serving a French corpus queried in English
# is one value away -- no second setting to keep in step with the first.
def test_the_direction_is_read_from_the_model_name():
    assert translation.direction_of(FR_EN) == ("fr", "en")
    assert translation.direction_of(EN_FR) == ("en", "fr")


# A name we cannot read is reported as no direction rather than guessed at:
# translating the wrong way round would quietly ruin every query.
def test_an_unreadable_model_name_yields_no_direction(monkeypatch):
    assert translation.direction_of(
        "some-org/a-model-without-a-direction") is None

    configure(monkeypatch, f"some-org/a-model-without-a-direction,{FR_EN}")
    assert translation.directions() == [("fr", "en", FR_EN)]


# --- which language a query is written in --------------------------------

def test_a_french_query_is_recognised():
    assert translation.detect_language(
        "comment grouper les marqueurs sur une carte") == "fr"


def test_an_english_query_is_recognised():
    assert translation.detect_language(
        "how to cluster markers on a map layer") == "en"


# The case that makes the whole feature worth having: a question written in
# French whose technical terms stay in English.
def test_a_french_query_carrying_english_terms_is_still_french():
    assert translation.looks_like(
        "comment personnaliser le style d'une couche GeoJSON", "fr")


# Elided articles are among the strongest signals French gives, and the
# typographic apostrophe a phone keyboard types must count as well.
def test_elided_articles_count_as_french():
    assert translation.detect_language("où est l'API des catalogues") == "fr"
    assert translation.detect_language("où est l’API des catalogues") == "fr"


def test_the_other_european_languages_are_told_apart():
    assert translation.detect_language(
        "cómo añadir una capa en el mapa") == "es"
    assert translation.detect_language(
        "wie kann man eine Ebene auf der Karte hinzufügen") == "de"
    assert translation.detect_language(
        "come aggiungere un livello sulla mappa con il catalogo") == "it"


# A query of bare keywords belongs to no language. Translating it would be
# a guess, so it is searched as typed.
def test_bare_keywords_are_claimed_by_no_language():
    assert translation.detect_language("clustering geojson layer") is None
    assert not translation.looks_like("clustering geojson layer", "fr")
    assert not translation.looks_like("clustering geojson layer", "en")


# A query two languages claim equally is claimed by neither.
def test_a_tie_is_claimed_by_no_language():
    assert translation.detect_language("la de") is None


def test_an_unknown_language_is_never_claimed():
    assert translation.detect_language("hoe groepeer ik markers") is None
    assert not translation.looks_like("hoe groepeer ik markers", "nl")


# --- which translations a query gets -------------------------------------

# A French question over a mixed corpus is translated into English, never
# run through the English-to-French model it would only garble.
def test_a_french_query_runs_only_through_the_models_starting_from_french(
        monkeypatch):
    configure(monkeypatch, f"{FR_EN},{EN_FR}")
    calls = fake_translator(monkeypatch)

    found = translation.translations("comment créer une couche")

    assert calls == [(FR_EN, "comment créer une couche")]
    assert found == [("en", f"[{FR_EN}] comment créer une couche")]


# And the reverse: an English question reaches the French pages.
def test_an_english_query_is_translated_into_french(monkeypatch):
    configure(monkeypatch, f"{FR_EN},{EN_FR}")
    calls = fake_translator(monkeypatch)

    found = translation.translations("how to create a layer")

    assert calls == [(EN_FR, "how to create a layer")]
    assert found == [("fr", f"[{EN_FR}] how to create a layer")]


def test_a_query_no_model_starts_from_gets_no_translation(monkeypatch):
    configure(monkeypatch, FR_EN)
    calls = fake_translator(monkeypatch)

    assert translation.translations("how to create a layer") == []
    assert calls == []


def test_a_query_of_no_language_gets_no_translation(monkeypatch):
    configure(monkeypatch, f"{FR_EN},{EN_FR}")
    calls = fake_translator(monkeypatch)

    assert translation.translations("KLayerList clustering") == []
    assert calls == []


# A model that failed answers None, and its translation is simply left out.
def test_a_failed_translation_is_left_out(monkeypatch):
    configure(monkeypatch, FR_EN)
    monkeypatch.setattr(translation, "translate", lambda text, name=None: None)

    assert translation.translations("comment créer une couche") == []


# --- failure never costs a search ----------------------------------------

# The model is 300 MB fetched from the internet. When it is not there the
# search must still run, in the original language, rather than fail.
def test_no_configured_model_translates_nothing(monkeypatch):
    configure(monkeypatch, "")

    assert translation.translate("comment creer un service") is None


def test_an_unavailable_model_returns_nothing_rather_than_raising(
        monkeypatch):
    configure(monkeypatch, "Helsinki-NLP/opus-mt-xx-yy")
    monkeypatch.setattr(translation, "_unavailable", set())
    monkeypatch.setattr(translation, "_models", {})

    def unreachable(name):
        raise OSError(f"{name} is not on the hub")
    monkeypatch.setattr(translation.MarianTokenizer, "from_pretrained",
                        unreachable)

    assert translation.translate("comment creer un service") is None


# A missing model is tried once, not on every query: the failure is
# remembered.
def test_an_unavailable_model_is_not_tried_twice(monkeypatch):
    configure(monkeypatch, "Helsinki-NLP/opus-mt-xx-yy")
    monkeypatch.setattr(translation, "_unavailable", set())
    monkeypatch.setattr(translation, "_models", {})
    attempts = []

    def unreachable(name):
        attempts.append(name)
        raise OSError(f"{name} is not on the hub")
    monkeypatch.setattr(translation.MarianTokenizer, "from_pretrained",
                        unreachable)

    translation.translate("comment creer un service")
    translation.translate("comment creer une couche")

    assert attempts == ["Helsinki-NLP/opus-mt-xx-yy"]
