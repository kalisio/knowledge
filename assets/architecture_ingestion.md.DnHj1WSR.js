import{_ as c,c as l,b as o,w as n,al as s,V as d,m as e,a as t,E as h,o as a,J as r}from"./chunks/framework.GYYi-SO7.js";const w=JSON.parse('{"title":"Ingestion Job","description":"","frontmatter":{},"headers":[],"relativePath":"architecture/ingestion.md","filePath":"architecture/ingestion.md"}'),g={name:"architecture/ingestion.md"},u=d('<h1 id="ingestion-job" tabindex="-1">Ingestion Job <a class="header-anchor" href="#ingestion-job" aria-label="Permalink to &quot;Ingestion Job&quot;">​</a></h1><p>The ingestion job is responsible for building and maintaining the three knowledge layers exposed by the API:</p><ul><li><strong>Code index</strong>: indexes the Kalisio codebase into <strong>Qdrant</strong> to enable semantic code search.</li><li><strong>Git index</strong>: extracts Git history and engineering metrics (hotspots, co-changes, bus factor, etc.) into a <strong>SQLite</strong> database.</li><li><strong>Dependency graph</strong>: analyzes the codebase to build a graph of file dependencies and identify architectural relationships.</li></ul><h2 id="pipeline-stages" tabindex="-1">Pipeline stages <a class="header-anchor" href="#pipeline-stages" aria-label="Permalink to &quot;Pipeline stages&quot;">​</a></h2>',4),f=e("h2",{id:"incremental-ingestion",tabindex:"-1"},[t("Incremental ingestion "),e("a",{class:"header-anchor",href:"#incremental-ingestion","aria-label":'Permalink to "Incremental ingestion"'},"​")],-1),m=e("h2",{id:"dependency-graph",tabindex:"-1"},[t("Dependency graph "),e("a",{class:"header-anchor",href:"#dependency-graph","aria-label":'Permalink to "Dependency graph"'},"​")],-1),p=e("pre",null,[e("code",null,`# TODO incremental ingestion plan:
#
# 1. Clone / update repos via k-clone if needed:
#    k-clone <organization> <workspace|all>
#
# 2. Recover the last successful ingestion timestamp
#
# Store it in a dedicated metadata collection, separate from the code
# collection, with a single record such as:
#   {
#     "id": "collection_metadata",
#     "payload": {"last_ingestion": "2026-06-19T10:35:00Z"}
#   }
#
# Dates should be stored and read in ISO 8601 format. Read this value at
# the beginning of each run. On the first ingestion, the metadata record
# does not exist yet.
#
# 3. Build the candidate file list
#
# first_ingestion ?
# ├─ Yes:
# │    Scan every supported file in the selected repositories.
# │
# └─ No:
#      Use last_ingestion only as a recovery cursor to identify files that
#      may have changed since the previous successful run.
#      Example candidate source:
#          git log --since=<last_ingestion_iso8601> --name-only
#                  --pretty=format:
#
# Result:
#   candidate_files = files that may need reindexation
#
# 4. Confirm actual content changes with file_sha1
#
# For each candidate file:
#   - Read the current file content.
#   - Compute file_sha1 from the file content itself.
#   - Compare it with the file_sha1 already stored in Qdrant for the same
#     (repository, source_path).
#   - If the hash is unchanged, skip the file.
#   - If the hash changed, mark the file for reindexation.
#
# The final reindexation decision should rely on file_sha1, not on git log:
# git history is useful to reduce the scan perimeter and to enrich
# commit_history, but the hash is the reliable state-based check.
#
# 5. Synchronize the vector store
#
# For each file marked for reindexation:
#   - Delete the existing chunks for (repository, source_path) to avoid
#     stale versions remaining in the collection.
#   - Re-chunk the current file content.
#   - Recompute embeddings.
#   - Upsert the new chunks and metadata into the code collection.
#
# 6. Persist ingestion metadata
#
# Only after a successful run, update the metadata collection with the new
# last_ingestion timestamp. Do not update it at job start, otherwise a
# failed run could move the recovery cursor forward and miss files.
`)],-1);function _(b,y,k,x,D,v){const i=h("Mermaid");return a(),l("div",null,[u,(a(),o(s,null,{default:n(()=>[r(i,{id:"mermaid-26",class:"mermaid",graph:"flowchart%20LR%0A%20%20kli%5Bkli%5D%20--%3E%20clone%5Bclone%20repos%5D%20--%3E%20chunk%5Bchunk%5D%20--%3E%20embed%5Bembed%5D%20--%3E%20qdrant%5B(Qdrant)%5D%0A"})]),fallback:n(()=>[t(" Loading... ")]),_:1})),f,(a(),o(s,null,{default:n(()=>[r(i,{id:"mermaid-32",class:"mermaid",graph:"flowchart%20TD%0A%20%20first%5B%22First%20run%3A%20full%20index%22%5D%20-.-%3E%20store%5B(Index)%5D%0A%20%20next%5B%22Later%20runs%3A%20git%20diff%22%5D%20--%3E%20changed%5Bchanged%20files%20only%5D%20--%3E%20rechunk%5Btargeted%20re-chunk%5D%20--%3E%20store%0A"})]),fallback:n(()=>[t(" Loading... ")]),_:1})),m,p])}const E=c(g,[["render",_]]);export{w as __pageData,E as default};
