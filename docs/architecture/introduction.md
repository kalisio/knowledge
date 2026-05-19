Schéma Architecture globale : CI, dev machine, k8s etc.

Schéma Pipeline ingestion job : kli, clone, chunck, embed etc.

Schéma RAG : indexation (chunck, embed, qdrant) puis retrieval (Q,LLM, embed, top_k, raw chunck)

Schéma Endpoints API

sheam Tools MCP

Schéma Toutes les fonctionnalités


Schéma décision agent : agent reçoit une tâche → regarde la description des tools → décide lequel appeler. Montre les 3 chemins : "before reading" → search_code, "before editing" → get_file_context, "before refactoring" → get_dependents. C'est le schéma qui explique pourquoi la description des tools est du prompt engineering.


Schéma comparatif RAG vs sans RAG : côte à côte : sans RAG (agent lit 30 fichiers = 40k tokens input) vs avec RAG (1 call search_code = 5 chunks = 8k tokens input). Visuel direct pour justifier le projet.

Schéma incremental ingestion : diff entre première ingestion (tout reindexer) et les suivantes (git diff → fichiers modifiés uniquement → re-chunk ciblé + mise à jour SQLite partielle).

Schéma dependency graph : fichiers .js/.vue → tree-sitter AST → import edges → NetworkX DiGraph → PageRank + Louvain → graph.json → FastAPI /dependents → MCP get_dependents → agent.

Schéma git intelligence : git log + git blame → git_indexer → SQLite (3 tables) → FastAPI /file-context → MCP get_file_context → agent. Avec les métriques calculées (hotspot, cochange, bus factor) annotées sur les flèches.

Schéma du flux d'une session agent : ce qui se passe de "le dev pose une question" jusqu'à "l'agent répond" : Claude Code → SSE → MCP server → tool call → FastAPI → Qdrant → chunks → retour dans le contexte de l'agent. Montre concrètement pourquoi le token count baisse.