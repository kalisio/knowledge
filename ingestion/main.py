"""Command-line entry point for the repository indexing job.

Usage:
    python -m ingestion.main                 # index the whole corpus
    python -m ingestion.main <repo> [...]    # index only the named repos

With no arguments, every git repository directly under
KALISIO_DEVELOPMENT_DIR is indexed, except SKIP_REPOS (the Qdrant storage
dir and this service itself, whose docs/experiments would pollute the
corpus). The exact production corpus is still a team decision, so SKIP_REPOS
is the single knob that defines it.

With arguments, only the named repos (directory names under the same root)
are indexed -- handy for (re)indexing one repo at a time.

The selected repositories are chunked, embedded and upserted into the Qdrant
collection the API queries. IngestionConfig is built first so a misconfigured
run fails fast before any embedding work.
"""

import sys
from pathlib import Path

from ingestion.config import IngestionConfig
from ingestion.pipeline import run


# Directories under the workspace root that are not part of the indexed
# corpus: the Qdrant storage dir, and this service itself (its
# docs/experiments would pollute the corpus). Provisional -- the canonical
# production set is still a team decision; adjust this set to change it.
SKIP_REPOS = {"knowledge", "qdrant_data"}


def main(argv=None):
    config = IngestionConfig()
    root = Path(config.repos_dir)

    # TODO incremental ingestion plan:
    #
    # 1. Clone repos via k-clone:
    #    k-clone config.kli_organization config.kli_workspace => k-clone <organization> <workspace|all>
    #
    # 2. Déterminer les fichiers à indexer

    # is_first_ingestion ?
    # ├─ Oui :
    # │    Sélectionner tous les fichiers de DEVELOPMENT_DIR
    # │    dont l'extension appartient à SUPPORTED_EXTENSIONS.
    # │
    # └─ Non :
    #      Identifier les fichiers ajoutés ou modifiés au cours
    #      des dernières 24 heures.
    #
    #      Exemple :
    #      git log --since="24 hours ago" --name-only --pretty=format: | sort -u
    #      (à valider)
    #
    # Résultat :
    #   files_to_index = liste des fichiers à (ré)indexer
    #
    # 3. Synchroniser la base vectorielle

    # is_first_ingestion ?
    # ├─ Oui :
    # │    Aucune vérification nécessaire.
    # │
    # └─ Non :
    #      Pour chaque fichier de files_to_index :
    #        - Vérifier s'il existe déjà dans la base vectorielle.
    #        - Si présent :
    #              supprimer les embeddings associés afin d'éviter
    #              les doublons ou les versions obsolètes.
    #        - Sinon :
    #              ne rien faire.
    #
    # 4. Indexation

    # Pour chaque fichier de files_to_index :
    #   - Extraire le contenu.
    #   - Générer les embeddings.
    #   - Insérer les embeddings et les métadonnées
    #     dans la base vectorielle.

    repo_dirs = discover_repos(root)

    print(f"indexing {len(repo_dirs)} repo(s): "
          f"{', '.join(d.name for d in repo_dirs)}")
    count = run(repo_dirs)
    print(f"indexed {count} chunks into '{config.qdrant_collection}'")
    return 0


# Every git repository directly under `root`, minus SKIP_REPOS.
def discover_repos(root):
    return [
        path for path in sorted(root.iterdir())
        if path.name not in SKIP_REPOS and (path / ".git").exists()
    ]


if __name__ == "__main__":
    sys.exit(main())
