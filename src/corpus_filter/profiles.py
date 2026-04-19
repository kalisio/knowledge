"""Configuration profiles for different corpus scanning scenarios.

This module provides pre-defined 'FilterConfig' objects tailored for 
specific experiments. For example, the 'js_vue_rag' profile is optimized for 
source code analysis in nb03, while the 'default' profile is used for 
general corpus inventory in nb01.
"""

from __future__ import annotations

from dataclasses import replace

from .models import FilterConfig


PROFILE_DEFAULT = "default"
PROFILE_JS_VUE_RAG = "js_vue_rag"


def build_default_profile() -> FilterConfig:
    """Standard inventory profile. 
    
    Includes most text files but excludes obvious noise (node_modules, .git).
    """
    return FilterConfig()


def build_js_vue_rag_profile() -> FilterConfig:
    """Specialized profile for nb03 (JS/Vue Chunking Experiment).
    
    This profile focuses exclusively on source code files. It aggressively 
    excludes non-source directories (docs, tools, examples) to ensure the 
    RAG experiments are measured against high-value production code only.
    """
    cfg = FilterConfig()
    return replace(
        cfg,
        # Exclude directories that contain documentation or secondary examples
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
        # Strict whitelist for source code analysis
        included_extensions={".js", ".vue", ".mjs", ".cjs", ".ts", ".tsx", ".json"},
        # Allow slightly larger files as some Vue SFCs can be substantial
        max_file_size=200_000,
    )


def resolve_profile(profile: str) -> FilterConfig:
    """Factory to fetch a profile config by its string name."""
    if profile == PROFILE_DEFAULT:
        return build_default_profile()
    if profile == PROFILE_JS_VUE_RAG:
        return build_js_vue_rag_profile()
    raise ValueError(f"Unknown corpus filter profile: {profile!r}")


def available_profiles() -> tuple[str, ...]:
    """Returns a list of all supported profile names."""
    return (PROFILE_DEFAULT, PROFILE_JS_VUE_RAG)
