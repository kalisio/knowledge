"""Translates a query into the language the corpus is written in.

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
"""

import re

from transformers import MarianMTModel, MarianTokenizer

from api.config import get_config
from api.logger import get_logger

log = get_logger("translation")

# Loaded once on first use, then kept.
_model = None
_tokenizer = None
_unavailable = False

# Enough of each language to tell them apart on one sentence. A query is
# short, so this is counted on the whole of it -- no need for a language
# detector and the dependency it would add.
_STOPWORDS = {
    "en": {"the", "a", "an", "and", "or", "is", "are", "in", "on", "for",
           "to", "of", "with", "how", "what", "where", "why", "when",
           "which", "do", "does", "can", "i", "you", "we", "it", "this"},
    "fr": {"le", "la", "les", "un", "une", "des", "et", "ou", "est", "sont",
           "dans", "sur", "pour", "de", "du", "avec", "comment", "quoi",
           "où", "pourquoi", "quand", "quel", "quelle", "que", "qui",
           "je", "tu", "il", "elle", "nous", "vous", "ce", "cette"},
}


# The direction the configured model translates, read from its name: the
# Helsinki-NLP models are named opus-mt-<source>-<target>, so swapping
# opus-mt-fr-en for opus-mt-en-fr is all it takes to serve a French corpus
# queried in English. Returns (source, target), or None when no model is
# configured or its name does not carry a direction.
def direction():
    name = get_config().query_translation_model
    if not name:
        return None
    match = re.search(r"-([a-z]{2,3})-([a-z]{2,3})$", name)
    if not match:
        log.warning("cannot read a direction from '%s': "
                    "expected a name ending in -<source>-<target>", name)
        return None
    return match.group(1), match.group(2)


# Whether `text` looks like it is written in `language`. Unknown languages
# answer False: translating a query we cannot place would be a guess, and
# the untranslated search still works.
def looks_like(text, language):
    words = set(re.findall(r"[a-zà-ÿ']+", text.lower()))
    if language not in _STOPWORDS:
        return False
    scores = {code: len(words & stopwords)
              for code, stopwords in _STOPWORDS.items()}
    return scores[language] > 0 and scores[language] == max(scores.values())


# Translate `text`, or return None when translation is unavailable. Never
# raises: a search in the original language beats no search at all, and the
# caller keeps the original query either way.
def translate(text):
    model, tokenizer = _load()
    if model is None:
        return None
    try:
        batch = tokenizer([text], return_tensors="pt", truncation=True)
        generated = model.generate(**batch, max_new_tokens=128)
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    except Exception as exc:
        log.warning("translation failed, keeping the original query: %s", exc)
        return None


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------


# Load the model once. A failure is remembered, so a missing weight does not
# cost a download attempt on every single query.
def _load():
    global _model, _tokenizer, _unavailable
    if _unavailable:
        return None, None
    if _model is not None:
        return _model, _tokenizer
    name = get_config().query_translation_model
    if not name:
        _unavailable = True
        return None, None
    try:
        log.info("loading translation model %s on cpu", name)
        _tokenizer = MarianTokenizer.from_pretrained(name)
        _model = MarianMTModel.from_pretrained(name)
        log.info("translation model ready")
    except Exception as exc:
        log.warning("translation model %s unavailable, queries will be "
                    "searched as typed: %s", name, exc)
        _unavailable = True
        return None, None
    return _model, _tokenizer
