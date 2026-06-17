# nb01 · Corpus discovery

**Notebook:** `notebooks/nb01_corpus_discovery.ipynb` · **Status:** done

## The question

Of all the files across the Kalisio repositories, **which ones should actually be indexed?**
Indexing everything wastes embedding budget and pollutes retrieval with noise (lockfiles,
build artifacts, generated code, binaries).

## Approach

Profile-driven corpus discovery: a walker scans a repo tree and a **filtering profile**
decides what is kept (by extension, path, size, …), so the same engine can target different
corpora by swapping the profile rather than rewriting the walk.

<!-- TODO: from the notebook — the concrete profiles tried, and the keep/drop rules that
     survived. Add the before/after file counts if recorded. -->

## Outcome

A reusable discovery + filtering step that selects the files the rest of the pipeline chunks
and embeds. This logic feeds the file-selection stage of the
[ingestion pipeline](/architecture/ingestion-pipeline).

<!-- TODO: list the final profiles and where each is used. -->
