from __future__ import annotations

from dataclasses import replace

from .models import FilterConfig


PROFILE_DEFAULT = "default"
PROFILE_JS_VUE_RAG = "js_vue_rag"


def build_default_profile() -> FilterConfig:
    return FilterConfig()


def build_js_vue_rag_profile() -> FilterConfig:
    cfg = FilterConfig()
    return replace(
        cfg,
        excluded_dirs=cfg.excluded_dirs
        | {
            "docs",
            "scripts",
            "tools",
            "examples",
            "sample",
            "samples",
            "tutorials",
            "public",
            "storybook-static",
        },
        included_extensions={".js", ".vue", ".mjs", ".cjs", ".ts", ".tsx", ".json"},
        max_file_size=200_000,
    )


def resolve_profile(profile: str) -> FilterConfig:
    if profile == PROFILE_DEFAULT:
        return build_default_profile()
    if profile == PROFILE_JS_VUE_RAG:
        return build_js_vue_rag_profile()
    raise ValueError(f"Unknown corpus filter profile: {profile!r}")


def available_profiles() -> tuple[str, ...]:
    return (PROFILE_DEFAULT, PROFILE_JS_VUE_RAG)
