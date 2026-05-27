"""Strategy E — ``D_sfc_plus_breadcrumb`` + expanded component-name hint.

D's breadcrumb only contains the raw path ``.../KZoomControl.vue``; a
bi-encoder treats that identifier as one opaque sub-word blob. E injects
the paraphrased component phrase into every chunk header so the query
(``"zoom control"``) and the chunk share surface tokens:

    <!-- kdk/.../KZoomControl.vue — zoom control [template] -->
    // kdk/.../KZoomControl.vue — zoom control [script]

When no phrase can be derived (e.g. short or single-word stems) E falls
back to D's original header.
"""

from __future__ import annotations

import vue_strategies as vuex
from paraphrase import component_phrase


def strategy_e(text: str, rel_path: str) -> list[vuex._Chunk]:
    phrase = component_phrase(rel_path)
    out: list[vuex._Chunk] = []
    for c in vuex.strategy_c(text, rel_path):
        if c.block == "script":
            header = (
                f"// {rel_path} — {phrase} [{c.block}]" if phrase
                else f"// {rel_path} [{c.block}]"
            )
        else:
            header = (
                f"<!-- {rel_path} — {phrase} [{c.block}] -->" if phrase
                else f"<!-- {rel_path} [{c.block}] -->"
            )
        out.append(vuex._Chunk(header + "\n" + c.text, rel_path, c.block))
    return out


def chunk_file(rel_path: str, data_root=None) -> list[vuex._Chunk]:
    """Single-file helper for qualitative inspection in the notebook."""
    from pathlib import Path
    root = data_root or (Path(__file__).resolve().parents[2] / "data")
    text = (root / rel_path).read_text(errors="ignore")
    return strategy_e(text, rel_path)
