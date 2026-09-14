import{_ as d,c as l,b as a,w as e,al as i,V as c,m as s,a as n,E as h,o as t,J as r}from"./chunks/framework.GYYi-SO7.js";const I=JSON.parse('{"title":"Ingestion Job","description":"","frontmatter":{},"headers":[],"relativePath":"architecture/ingestion.md","filePath":"architecture/ingestion.md"}'),p={name:"architecture/ingestion.md"},u=c('<h1 id="ingestion-job" tabindex="-1">Ingestion Job <a class="header-anchor" href="#ingestion-job" aria-label="Permalink to &quot;Ingestion Job&quot;">​</a></h1><p>The ingestion job is responsible for building and maintaining the three knowledge layers exposed by the API:</p><ul><li><strong>Code index</strong>: indexes the Kalisio codebase into <strong>Qdrant</strong> to enable semantic code search.</li><li><strong>Git index</strong>: extracts Git history and engineering metrics (hotspots, co-changes, bus factor, etc.) into a <strong>SQLite</strong> database.</li><li><strong>Dependency graph</strong>: analyzes the codebase to build a graph of file dependencies and identify architectural relationships.</li></ul><h2 id="pipeline-stages" tabindex="-1">Pipeline stages <a class="header-anchor" href="#pipeline-stages" aria-label="Permalink to &quot;Pipeline stages&quot;">​</a></h2>',4),f=s("h2",{id:"incremental-ingestion",tabindex:"-1"},[n("Incremental ingestion "),s("a",{class:"header-anchor",href:"#incremental-ingestion","aria-label":'Permalink to "Incremental ingestion"'},"​")],-1),g=c(`<h2 id="dependency-graph" tabindex="-1">Dependency graph <a class="header-anchor" href="#dependency-graph" aria-label="Permalink to &quot;Dependency graph&quot;">​</a></h2><pre><code># TODO incremental ingestion plan:
#
# 1. Clone / update repos via k-clone if needed:
#    k-clone &lt;organization&gt; &lt;workspace|all&gt;
#
# 2. Recover the last successful ingestion timestamp
#
# Store it in a dedicated metadata collection, separate from the code
# collection, with a single record such as:
#   {
#     &quot;id&quot;: &quot;collection_metadata&quot;,
#     &quot;payload&quot;: {&quot;last_ingestion&quot;: &quot;2026-06-19T10:35:00Z&quot;}
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
#          git log --since=&lt;last_ingestion_iso8601&gt; --name-only
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
#     (repo, path).
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
#   - Delete the existing chunks for (repo, path) to avoid
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
</code></pre><h2 id="workspace-layout" tabindex="-1">Workspace layout <a class="header-anchor" href="#workspace-layout" aria-label="Permalink to &quot;Workspace layout&quot;">​</a></h2><p>The job scans <code>DEVELOPMENT_DIR</code>, the directory <code>k-clone</code> works from. It holds one directory per organisation, and each of those holds the cloned repositories:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code"><code><span class="line"><span>$DEVELOPMENT_DIR/</span></span>
<span class="line"><span>├── kalisio/      kdk, kano, development, kli, …</span></span>
<span class="line"><span>├── irsn/         planet, criter, …</span></span>
<span class="line"><span>└── airbus/       Gift-*, …</span></span></code></pre></div><p>In the cluster that workspace is an empty volume, so the tooling cannot live in it: the image ships <code>development</code> and <code>kli</code> under <code>/opt/kalisio</code>, and the job links them into <code>$KALISIO_DEVELOPMENT_DIR</code> before calling <code>k-clone</code>, which resolves everything through that directory. The image carries no credential — <code>KALISIO_GITHUB_TOKEN</code> (and <code>GITLAB_IRSN_TOKEN</code> for an IRSN workspace) is supplied at run time. A workspace that already holds the tooling is a developer&#39;s own checkout and is left untouched.</p><p>So a repository sits two levels down, and that is what the scan looks for (a repository sitting directly under the root is picked up too, which is how a hand-made workspace is laid out). A file is identified by the repository holding it and its path inside that repository — never by its path from the workspace, so <code>repo</code> stays <code>kano</code> and not <code>kalisio/kano</code>.</p><h2 id="commit-history" tabindex="-1">Commit history <a class="header-anchor" href="#commit-history" aria-label="Permalink to &quot;Commit history&quot;">​</a></h2><p>The commit history of a file is stored <strong>once per file</strong>, in its own collection (<code>QDRANT_COLLECTION_FILES</code>, derived from the code collection by default), and joined back onto every chunk of that file when a search returns it. Storing it on each chunk instead multiplies it by the number of chunks — eight times more on the kdk corpus.</p><p>What is kept is a sliding window: commits older than <code>COMMIT_HISTORY_MAX_AGE_DAYS</code> (180) drop off on their own and new ones come in, with a floor of <code>COMMIT_HISTORY_MIN_COMMITS</code> (5) kept whatever their age — without it, 84% of the kdk files would carry no history at all, and those are the stable files whose intent is hardest to recover from the code. <code>COMMIT_HISTORY_DEPTH</code> (0, no cap) can bound a very active file.</p><p>The window is rebuilt from git on every run, for every scanned file and not only the ones being reindexed, so a file nobody touched still lets its oldest commits go. It costs one <code>git log</code> pass per repository — 2.4 s over the 70 repositories of the workspace, against about four minutes if git were asked per file.</p>`,11);function m(_,k,y,b,w,T){const o=h("Mermaid");return t(),l("div",null,[u,(t(),a(i,null,{default:e(()=>[r(o,{id:"mermaid-26",class:"mermaid",graph:"flowchart%20LR%0A%20%20kli%5Bkli%5D%20--%3E%20clone%5Bclone%20repos%5D%20--%3E%20chunk%5Bchunk%5D%20--%3E%20embed%5Bembed%5D%20--%3E%20qdrant%5B(Qdrant)%5D%0A"})]),fallback:e(()=>[n(" Loading... ")]),_:1})),f,(t(),a(i,null,{default:e(()=>[r(o,{id:"mermaid-32",class:"mermaid",graph:"flowchart%20TD%0A%20%20first%5B%22First%20run%3A%20full%20index%22%5D%20-.-%3E%20store%5B(Index)%5D%0A%20%20next%5B%22Later%20runs%3A%20git%20diff%22%5D%20--%3E%20changed%5Bchanged%20files%20only%5D%20--%3E%20rechunk%5Btargeted%20re-chunk%5D%20--%3E%20store%0A"})]),fallback:e(()=>[n(" Loading... ")]),_:1})),g])}const D=d(p,[["render",m]]);export{I as __pageData,D as default};
