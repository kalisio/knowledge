"""Ingestion-job configuration (extends the shared RuntimeConfig)."""

from dataclasses import dataclass, field

from config import RuntimeConfig, require


@dataclass(frozen=True)
class IngestionConfig(RuntimeConfig):
    """Configuration for the repository indexing job.

    Inherits the shared vector-store and embedding settings so the job
    writes to the same Qdrant collection and uses the same model the API
    reads with. Further knobs (which repos to index, chunk sizes,
    incremental mode) are added as the pipeline is built and their
    variables are provisioned — not ported wholesale from the experiment.
    """

    # Root holding the local Kalisio repos to index. Comes from the
    # workspace env (KALISIO_DEVELOPMENT_DIR, set by k-dev / services.sh),
    # not from knowledge.enc.env. Required: can't scan without it.
    repos_dir: str = field(
        default_factory=lambda: require("KALISIO_DEVELOPMENT_DIR"))
