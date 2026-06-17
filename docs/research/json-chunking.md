# nb04 · JSON chunking

**Notebook:** `notebooks/nb04_json_chunking.ipynb` · **Status:** done

## The question

JSON (configs, manifests, data) has structure but no prose. A naive whole-file or
fixed-size split loses the key→value relationships that make it searchable.

## Approach

**Category-aware key splitting** with a breadcrumb: group keys by category and split along
those boundaries, prefixing each chunk with its path so retrieval keeps the key context.

<!-- TODO: from the notebook — how categories are determined, and how it compares to a naive
     split on the gold set. -->

## In production

The JSON chunker is part of the [ingestion pipeline](/architecture/ingestion-pipeline).
