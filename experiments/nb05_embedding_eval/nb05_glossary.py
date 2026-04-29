"""Project glossary for the FR→EN query augmentation experiment in §13 of nb05.

This module is experiment-scoped and lives next to the other nb05 helpers.
Entries were curated from the per-query FR failure analysis on the Kalisio
JS/Vue corpus and intentionally map French concepts to project-specific
English identifiers (``mailer``, ``MapActivity``, ``EventLog``, …) rather
than generic word-level translations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GlossaryMatch:
    """One glossary hit applied to a query."""

    pattern: str
    expansion: str


@dataclass(frozen=True)
class GlossaryEntry:
    """French phrase pattern mapped to English project terminology."""

    pattern: str
    expansion: str


# Pattern -> EN augmentation. Order matters: most specific first.
# These entries come from nb05's FR failure analysis and intentionally map
# concepts to project identifiers rather than generic French-English words.
PROJECT_GLOSSARY: tuple[GlossaryEntry, ...] = (
    GlossaryEntry(r"\bjournal d['’]événements?\b", "EventLog event log"),
    GlossaryEntry(r"\bservice d['’]envoi d['’]emails?\b", "mailer service"),
    GlossaryEntry(r"\benvoi d['’]emails?\b", "mailer email"),
    GlossaryEntry(r"\bplugins? au démarrage\b", "boot kdk plugin"),
    GlossaryEntry(r"\bcharger? (?:ses )?plugins?\b", "boot kdk load plugin"),
    GlossaryEntry(r"\b(?:intégration|activité) (?:de )?carte\b", "MapActivity map"),
    GlossaryEntry(r"\bactivité catalogue\b", "CatalogActivity catalog"),
    GlossaryEntry(r"\bactivités? de démo\b", "ActivityLinks demo"),
    GlossaryEntry(r"\bsélection (?:des? )?(?:entités?|features?)\b", "feature selection mixin"),
    GlossaryEntry(r"\bmodèles? d['’]?événements?\b", "EventTemplate event template"),
    GlossaryEntry(r"\bmodèles? de plans?\b", "PlanTemplate plan template"),
    GlossaryEntry(r"\bobjectifs? de plans?\b", "plan objective"),
    GlossaryEntry(r"\bintercepter (?:une )?requête\b", "hook query intercept"),
    GlossaryEntry(r"\binternational(?:iser|isation)\b", "i18n internationalize locale"),
    GlossaryEntry(r"\blocalis(?:er|ation)\b", "localize locale i18n"),
    GlossaryEntry(r"\bauthentification\b", "authentication hook"),
    GlossaryEntry(r"\bautorisations?\b", "authorization authorisation"),
    GlossaryEntry(r"\brestreindre (?:un )?endpoint\b", "authorization role"),
    GlossaryEntry(r"\bformulaires?[^.?!]{0,40}schémas?\b", "KForm schema form"),
    GlossaryEntry(r"\b(?:importation|import) (?:de )?couches?\b", "KImportLayer import layer"),
    GlossaryEntry(r"\bcatalogue cartographique\b", "catalog mapping"),
    GlossaryEntry(r"\b(?:publication|release|publier|relâcher)\b", "publish release deploy"),
    GlossaryEntry(r"\broutage\b", "routing routes"),
    GlossaryEntry(r"\bdéploiement\b", "deploy deployment"),
    GlossaryEntry(r"\bcouche cartographique\b", "map layer"),
    GlossaryEntry(r"\bcomposant graphique\b", "KChart chart"),
    GlossaryEntry(r"\bhors ligne\b", "offline PWA"),
    GlossaryEntry(r"\bcouche météo\b", "weather probe layer"),
    GlossaryEntry(r"\broutes? de l[’']application\b", "router routes"),
    GlossaryEntry(r"\bvisites? guidées?\b", "tour KTour"),
    GlossaryEntry(r"\bévénements?.{0,40}plans?.{0,80}organisations?.{0,40}groupes?\b", "EventsActivity PlansActivity OrganisationMenu PlanMenu"),
    GlossaryEntry(r"\bcomposant KProfile\b", "KProfile profile account"),
    GlossaryEntry(r"\bcomposant KChart\b", "KChart chart"),
    GlossaryEntry(r"\bpage .{0,20}propos\b", "KAbout about"),
    GlossaryEntry(r"\bparamètres utilisateur\b", "KSettings user settings"),
    GlossaryEntry(r"\bactivité globe\b", "GlobeActivity globe"),
    GlossaryEntry(r"\bfiltre de collection\b", "collection-filter collection filter"),
    GlossaryEntry(r"\bcomposable catalogue\b", "catalog composable"),
    GlossaryEntry(r"\bhooks? API users\b", "users hooks users.hooks"),
)


_COMPILED_PROJECT_GLOSSARY = tuple(
    (entry, re.compile(entry.pattern, re.IGNORECASE)) for entry in PROJECT_GLOSSARY
)


def match_project_glossary(query: str) -> list[GlossaryMatch]:
    """Return glossary expansions that match ``query`` in configured order."""
    matches: list[GlossaryMatch] = []
    seen_expansions: set[str] = set()
    for entry, pattern in _COMPILED_PROJECT_GLOSSARY:
        if pattern.search(query) and entry.expansion not in seen_expansions:
            matches.append(GlossaryMatch(pattern=entry.pattern, expansion=entry.expansion))
            seen_expansions.add(entry.expansion)
    return matches


def augment_query_with_project_glossary(query: str) -> tuple[str, list[GlossaryMatch]]:
    """Append matched English project terms while preserving the original query."""
    matches = match_project_glossary(query)
    if not matches:
        return query, []
    expansions = " ".join(match.expansion for match in matches)
    return f"{query} ({expansions})", matches


def augment_queries_with_project_glossary(queries: Iterable[str]) -> list[str]:
    """Apply project glossary augmentation to a batch of queries."""
    return [augment_query_with_project_glossary(query)[0] for query in queries]
