"""Translates a query into the languages the corpus is written in.

Retrieval is markedly weaker when the question and the documentation are
not in the same language, and the failure is lexical rather than semantic:
the embedding model is multilingual, and a French question carrying English
technical terms works well. It breaks down on plain French -- "grappes" is
close to "cluster" in no vector space trained at this size.

So the query is translated before it is embedded, never the corpus. A
dedicated translation model rather than an LLM: it is 75M parameters
against billions, it runs in tens of milliseconds on CPU, it needs no
prompt and no output parsing, and it cannot wander off and answer the
question instead of translating it.

Several models can be configured at once, one per direction. A corpus is
rarely written in a single language -- the Kalisio documentation is English
with French pages here and there -- so a French question is translated into
English to reach the English pages, and an English question into French to
reach the French ones. A model only ever runs on a query written in its
source language.
"""

import re

from transformers import MarianMTModel, MarianTokenizer

from api.config import get_config
from api.logger import get_logger

log = get_logger("translation")

# Loaded once on first use, then kept, keyed by model name.
_models = {}
_tokenizers = {}
_unavailable = set()

# Enough of each language to tell them apart on one sentence. A query is
# short, so this is counted on the whole of it -- no need for a language
# detector and the dependency it would add. The words several languages
# share ("la", "de", "in") count for each of them; it is the language with
# strictly the most hits that wins, so they only ever break a tie they
# cannot win alone.
_STOPWORDS = {
    "en": {"the", "a", "an", "and", "or", "is", "are", "in", "on", "for",
           "to", "of", "with", "how", "what", "where", "why", "when",
           "which", "do", "does", "can", "i", "you", "we", "it", "this",
           "that", "from", "by", "be", "my", "should", "into"},
    "fr": {"le", "la", "les", "un", "une", "des", "et", "ou", "est", "sont",
           "dans", "sur", "pour", "de", "du", "avec", "comment", "quoi",
           "où", "pourquoi", "quand", "quel", "quelle", "quels", "quelles",
           "que", "qui", "je", "tu", "il", "elle", "nous", "vous", "ce",
           "cette", "ces", "au", "aux", "par", "pas", "ne", "faut", "peut",
           "mon", "ma", "mes", "l'", "d'", "j'", "qu'", "n'", "s'", "c'"},
    "es": {"el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o",
           "es", "son", "en", "con", "para", "por", "de", "del", "cómo",
           "qué", "dónde", "cuándo", "cuál", "se", "al", "mi", "puedo",
           "hay", "lo"},
    "de": {"der", "die", "das", "ein", "eine", "einen", "und", "oder", "ist",
           "sind", "im", "auf", "für", "mit", "wie", "was", "wo", "warum",
           "wann", "welche", "welcher", "ich", "du", "wir", "es", "nicht",
           "den", "dem", "des", "zu", "von", "kann", "man"},
    "it": {"il", "lo", "gli", "un", "una", "e", "o", "è", "sono", "su",
           "per", "con", "di", "del", "della", "come", "cosa", "dove",
           "perché", "quando", "quale", "che", "non", "si", "al", "posso"},
}

# Words, with the elided articles of French and Italian ("l'", "d'") kept
# as tokens of their own: they are among the strongest signals there are,
# and dropping the apostrophe would turn "l'" into a bare "l".
_WORD = re.compile(r"[a-zà-ÿßñ]+['’]?")


# The models configured, in the order given. QUERY_TRANSLATION_MODEL takes
# one name or several, comma-separated.
def models():
    names = get_config().query_translation_model
    return [name.strip() for name in names.split(",") if name.strip()]


# The direction a model translates, read from its name: the Helsinki-NLP
# models are named opus-mt-<source>-<target>, so the name is the whole
# configuration. Returns (source, target), or None when the name does not
# carry a direction -- translating the wrong way round would quietly ruin
# every query, so a name we cannot read is never guessed at.
def direction_of(name):
    match = re.search(r"-([a-z]{2,3})-([a-z]{2,3})$", name)
    if not match:
        log.warning("cannot read a direction from '%s': "
                    "expected a name ending in -<source>-<target>", name)
        return None
    return match.group(1), match.group(2)


# Every configured direction as (source, target, model name), skipping the
# names that carry none.
def directions():
    found = []
    for name in models():
        way = direction_of(name)
        if way is not None:
            found.append((way[0], way[1], name))
    return found


# The language `text` is written in, or None when no language stands out.
# A query of bare keywords ("clustering geojson layer") belongs to none:
# translating it would be a guess, and the untranslated search still works.
def detect_language(text):
    words = {word.replace("’", "'")
             for word in _WORD.findall(text.lower())}
    scores = {code: len(words & stopwords)
              for code, stopwords in _STOPWORDS.items()}
    best = max(scores.values())
    if best == 0:
        return None
    winners = [code for code, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None


# Whether `text` looks like it is written in `language`.
def looks_like(text, language):
    return detect_language(text) == language


# The translations of `text` worth searching with, as [(target language,
# translated text)]: one per configured model whose source language is the
# one the query is written in. Empty when the language cannot be placed or
# no model starts from it.
def translations(text):
    language = detect_language(text)
    if language is None:
        return []
    found = []
    for source, target, name in directions():
        if source != language:
            continue
        translated = translate(text, name)
        if translated:
            found.append((target, translated))
    return found


# Translate `text` with the model `name` -- the first configured one when
# none is named -- or return None when translation is unavailable. Never
# raises: a search in the original language beats no search at all, and the
# caller keeps the original query either way.
def translate(text, name=None):
    if name is None:
        configured = models()
        if not configured:
            return None
        name = configured[0]
    model, tokenizer = _load(name)
    if model is None:
        return None
    try:
        batch = tokenizer([text], return_tensors="pt", truncation=True)
        generated = model.generate(**batch, max_new_tokens=128)
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    except Exception as exc:
        log.warning("translation with %s failed, keeping the original "
                    "query: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Load a model once. A failure is remembered, so a missing weight does not
# cost a download attempt on every single query.
def _load(name):
    if name in _unavailable:
        return None, None
    if name in _models:
        return _models[name], _tokenizers[name]
    try:
        log.info("loading translation model %s on cpu", name)
        tokenizer = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name)
        # The Marian configs ship a max_length of 512 that translate()
        # overrides with max_new_tokens: left in place, transformers warns
        # about the pair on every single query.
        model.generation_config.max_length = None
        log.info("translation model %s ready", name)
    except Exception as exc:
        log.warning("translation model %s unavailable, queries will be "
                    "searched without it: %s", name, exc)
        _unavailable.add(name)
        return None, None
    _models[name] = model
    _tokenizers[name] = tokenizer
    return model, tokenizer
