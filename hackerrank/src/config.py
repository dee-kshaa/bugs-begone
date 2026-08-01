"""
Central configuration for the WhatsApp Message Notification Router.

Every path, threshold, model name, cache location and tunable constant in the
project lives here. No other module should hardcode a filename or a magic
number; import from this module instead.

Environment variables consumed
------------------------------
ANTHROPIC_API_KEY   : required for any LLM call (read lazily, never logged)
CLAUDE_MODEL        : optional override for the reasoning model
WA_ROUTER_DATA_DIR  : optional override for the dataset directory
WA_ROUTER_LOG_LEVEL : optional log level (default INFO)
WA_ROUTER_STRICT    : "1" to make schema validation fatal

Dependencies
------------
Standard library only.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Project roots
# --------------------------------------------------------------------------- #

#: Repository root, i.e. the directory that contains ``src/`` and ``data/``.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Filesystem layout for the whole project.

    All directories are created on demand by :meth:`ensure`. Paths are frozen
    so that no module can mutate them at runtime.
    """

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    raw: Path = PROJECT_ROOT / "data" / "raw"
    media: Path = PROJECT_ROOT / "data" / "media"
    media_images: Path = PROJECT_ROOT / "data" / "media" / "images"
    media_voice: Path = PROJECT_ROOT / "data" / "media" / "voice"
    cache: Path = PROJECT_ROOT / "data" / "cache"
    outputs: Path = PROJECT_ROOT / "outputs"
    logs: Path = PROJECT_ROOT / "outputs" / "logs"
    config_dir: Path = PROJECT_ROOT / "src"

    # ---- Stage A artefacts (parquet caches) ------------------------------- #
    enriched_messages: Path = PROJECT_ROOT / "data" / "cache" / "enriched_messages.parquet"
    ocr_cache: Path = PROJECT_ROOT / "data" / "cache" / "ocr.parquet"
    asr_cache: Path = PROJECT_ROOT / "data" / "cache" / "asr.parquet"
    sender_affinity: Path = PROJECT_ROOT / "data" / "cache" / "sender_affinity.parquet"
    group_profile: Path = PROJECT_ROOT / "data" / "cache" / "group_profile.parquet"
    business_profile: Path = PROJECT_ROOT / "data" / "cache" / "business_profile.parquet"
    contact_relationships: Path = (
        PROJECT_ROOT / "data" / "cache" / "contact_relationships.parquet"
    )
    tfidf_index: Path = PROJECT_ROOT / "data" / "cache" / "tfidf_index.joblib"
    llm_cache: Path = PROJECT_ROOT / "data" / "cache" / "llm_cache.sqlite"

    # ---- YAML rule / weight files ----------------------------------------- #
    rules_yaml: Path = PROJECT_ROOT / "src" / "rules" / "rules.yaml"
    weights_yaml: Path = PROJECT_ROOT / "src" / "scoring" / "weights.yaml"
    lexicons_yaml: Path = PROJECT_ROOT / "src" / "relationship" / "lexicons.yaml"

    # ---- Fitted artefacts -------------------------------------------------- #
    thresholds_json: Path = PROJECT_ROOT / "outputs" / "thresholds.json"
    calibration_json: Path = PROJECT_ROOT / "outputs" / "calibration.json"

    # ---- Deliverables ------------------------------------------------------ #
    submission_csv: Path = PROJECT_ROOT / "outputs" / "submission.csv"
    traces_jsonl: Path = PROJECT_ROOT / "outputs" / "traces.jsonl"

    def ensure(self) -> None:
        """Create every writable directory used by the pipeline.

        Read-only directories (``raw``, ``media``) are created too so that a
        fresh clone does not explode on first run; they will simply be empty.
        """
        for directory in (
            self.data,
            self.raw,
            self.media,
            self.media_images,
            self.media_voice,
            self.cache,
            self.outputs,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of one input CSV.

    Attributes
    ----------
    key:
        Attribute name used on ``DataRepository`` (e.g. ``"messages"``).
    filename:
        File name relative to :attr:`Paths.raw`.
    required_columns:
        Columns that downstream modules assume exist. Missing ones are logged
        as errors and, when running in strict mode, raise.
    id_columns:
        Columns coerced to ``str`` after loading so that joins never fail
        because one file stored an id as int and another as text.
    datetime_columns:
        Columns parsed with :func:`pandas.to_datetime`.
    required_file:
        When ``False``, a missing file yields an empty DataFrame plus a
        warning instead of an error.
    """

    key: str
    filename: str
    required_columns: tuple[str, ...] = ()
    id_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    required_file: bool = True


#: The ten datasets described in the problem statement, in load order.
DATASETS: Final[tuple[DatasetSpec, ...]] = (
    DatasetSpec(
        key="users",
        filename="users.csv",
        required_columns=("user_id",),
        id_columns=("user_id",),
    ),
    DatasetSpec(
        key="groups",
        filename="groups.csv",
        required_columns=("group_id",),
        id_columns=("group_id",),
    ),
    DatasetSpec(
        key="group_members",
        filename="group_members.csv",
        required_columns=("group_id", "user_id"),
        id_columns=("group_id", "user_id"),
    ),
    DatasetSpec(
        key="business_accounts",
        filename="business_accounts.csv",
        required_columns=("business_id",),
        id_columns=("business_id",),
    ),
    DatasetSpec(
        key="user_business_history",
        filename="user_business_history.csv",
        required_columns=("user_id", "business_id"),
        id_columns=("user_id", "business_id"),
        datetime_columns=("timestamp",),
    ),
    DatasetSpec(
        key="messages",
        filename="messages.csv",
        required_columns=("message_id", "sender_id", "timestamp"),
        id_columns=(
            "message_id",
            "sender_id",
            "user_id",
            "group_id",
            "business_id",
            "conversation_id",
            "reply_to_id",
            "image_id",
            "voice_note_id",
        ),
        datetime_columns=("timestamp",),
    ),
    DatasetSpec(
        key="message_history",
        filename="message_history.csv",
        required_columns=("message_id",),
        id_columns=("message_id", "sender_id", "user_id", "group_id", "conversation_id"),
        datetime_columns=("timestamp",),
    ),
    DatasetSpec(
        key="message_events",
        filename="message_events.csv",
        required_columns=("message_id",),
        id_columns=("message_id", "user_id"),
        datetime_columns=("timestamp", "event_time"),
    ),
    DatasetSpec(
        key="images",
        filename="images.csv",
        required_columns=("image_id",),
        id_columns=("image_id", "message_id"),
        required_file=False,
    ),
    DatasetSpec(
        key="voice_notes",
        filename="voice_notes.csv",
        required_columns=("voice_note_id",),
        id_columns=("voice_note_id", "message_id"),
        required_file=False,
    ),
)

#: Canonical name for every column alias we have seen in the wild.
#: Keys are the *alias* (lower-cased, stripped); values are the canonical name.
#: Edit this dict once you have the real headers -- nothing else needs to move.
COLUMN_ALIASES: Final[dict[str, str]] = {
    # identifiers
    "msg_id": "message_id",
    "id": "message_id",
    "messageid": "message_id",
    "from_user": "sender_id",
    "from_id": "sender_id",
    "sender": "sender_id",
    "to_user": "recipient_user_id",
    "receiver_id": "recipient_user_id",
    "recipient_id": "recipient_user_id",
    "chat_id": "conversation_id",
    "thread_id": "conversation_id",
    "grp_id": "group_id",
    "biz_id": "business_id",
    "reply_to": "reply_to_id",
    "in_reply_to": "reply_to_id",
    "parent_message_id": "reply_to_id",
    "voice_id": "voice_note_id",
    "audio_id": "voice_note_id",
    "img_id": "image_id",
    # content
    "text": "message_text",
    "body": "message_text",
    "content": "message_text",
    "msg_text": "message_text",
    # time
    "sent_at": "timestamp",
    "created_at": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",
    # typing
    "type": "media_type",
    "msg_type": "media_type",
    "media": "media_type",
    # flags
    "is_forward": "is_forwarded",
    "forwarded": "is_forwarded",
    "mention_user_ids": "mentions",
    "mentioned_users": "mentions",
    # paths
    "file_path": "media_path",
    "path": "media_path",
    "image_path": "media_path",
    "audio_path": "media_path",
}


# --------------------------------------------------------------------------- #
# Media / OCR / ASR
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MediaConfig:
    """Settings for the offline OCR and speech-to-text stages."""

    #: Tesseract language pack(s), e.g. ``"eng"`` or ``"eng+hin"``.
    ocr_languages: str = "eng"
    #: Tesseract page-segmentation mode; 6 = assume a uniform block of text.
    ocr_psm: int = 6
    #: OCR results below this mean word confidence are treated as unreliable.
    ocr_min_confidence: float = 0.40
    #: Upscale factor applied to small images before OCR.
    ocr_upscale_factor: float = 2.0

    #: Whisper checkpoint. ``base`` is the hackathon sweet spot.
    whisper_model: str = "base"
    #: ``None`` lets Whisper auto-detect; set to "en" to force English.
    whisper_language: str | None = None
    #: Skip clips longer than this (seconds) to protect the time budget.
    whisper_max_duration_sec: float = 180.0
    #: Average token log-probability below which a transcript is unreliable.
    asr_min_avg_logprob: float = -1.0
    #: Number of worker processes for media enrichment.
    media_workers: int = 4


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LLMConfig:
    """Anthropic client settings and token budgets."""

    model: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout_sec: float = 90.0
    max_retries: int = 4
    retry_base_delay_sec: float = 2.0

    #: Messages per batched routing call. Beyond ~10 the model cross-contaminates.
    batch_size: int = 8
    #: Contacts per batched relationship-classification call (Stage A).
    relationship_batch_size: int = 20
    #: Hard cap on the rendered context packet for a single message.
    max_context_tokens: int = 1500
    #: Characters kept per evidence candidate line.
    candidate_snippet_chars: int = 140

    @property
    def api_key(self) -> str:
        """Return the Anthropic API key from the environment.

        Raises
        ------
        RuntimeError
            If ``ANTHROPIC_API_KEY`` is unset or empty. The key value is never
            logged anywhere in this project.
        """
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before running any "
                "stage that reaches the LLM, e.g. `export ANTHROPIC_API_KEY=sk-...`."
            )
        return key

    @property
    def api_key_available(self) -> bool:
        """``True`` when an API key is present, without raising."""
        return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrievalConfig:
    """Sizes for the three-tier evidence retrieval strategy."""

    #: Tier 1 -- most recent messages in the same conversation.
    structural_window: int = 5
    #: Tier 2 -- most recent messages from this sender to this user.
    relational_window: int = 3
    #: Tier 2b -- prior business interactions pulled in.
    business_history_window: int = 2
    #: Tier 3 -- lexical top-k, only used when tiers 1-2 are thin.
    lexical_top_k: int = 2
    #: Trigger tier 3 when the pool has fewer candidates than this.
    lexical_trigger_below: int = 4
    #: Absolute cap on the candidate pool handed to the LLM.
    max_candidates: int = 12
    #: Ignore context older than this when computing recency decay.
    recency_horizon_hours: float = 72.0
    #: Characters kept per evidence candidate line when rendering a snippet.
    #: Mirrors LLMConfig.candidate_snippet_chars; kept here too because
    #: src/retrieval/context.py and src/retrieval/lexical.py read it off
    #: this config, not off LLMConfig.
    candidate_snippet_chars: int = 140
    #: TF-IDF vectoriser settings.
    tfidf_max_features: int = 50_000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 2


# --------------------------------------------------------------------------- #
# Scoring / thresholds / confidence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScoringConfig:
    """Component caps for the six-part priority score.

    These are *defaults*. ``src/scoring/weights.yaml`` overrides them at
    runtime so that tuning never requires a code change.
    """

    cap_relationship: float = 30.0
    cap_urgency: float = 25.0
    cap_trust: float = 15.0
    cap_user_preference: float = 15.0
    cap_historical_behaviour: float = 15.0
    #: Safety is a penalty component; its "cap" is a floor.
    floor_safety: float = -30.0

    score_min: float = 0.0
    score_max: float = 100.0

    #: Relationship shrinkage pulls low-confidence categories toward this value.
    relationship_prior_mean: float = 8.0

    #: Extra points for a 1:1 conversation over a group conversation.
    direct_conversation_bonus: float = 8.0


@dataclass(frozen=True)
class ThresholdConfig:
    """Priority-score cut points and the grid search that fits them."""

    #: Defaults only -- ``outputs/thresholds.json`` wins when present.
    mute_digest_cut: float = 38.0
    digest_notify_cut: float = 71.0

    #: Grid search bounds and step for ``scoring/thresholds.py``.
    search_min: float = 0.0
    search_max: float = 100.0
    search_step: float = 2.0
    #: Minimum gap enforced between the two cut points during search.
    min_band_gap: float = 10.0


@dataclass(frozen=True)
class ConfidenceConfig:
    """Inputs to the blended confidence formula."""

    #: Score distance from the nearest band boundary that counts as "certain".
    margin_full_confidence: float = 20.0
    margin_floor: float = 0.25
    #: Base confidence when the LLM resolves a message.
    llm_base_intercept: float = 0.50
    llm_base_slope: float = 0.40
    #: Multiplier applied when rule/score/LLM verdicts disagree.
    disagreement_penalty: float = 0.75
    #: Multiplier applied when the LLM path failed and we fell back to rules.
    llm_fallback_penalty: float = 0.70
    #: Support term: 0.80 + 0.05 * min(n_evidence, 4).
    support_intercept: float = 0.80
    support_per_evidence: float = 0.05
    support_max_evidence: int = 4
    #: Media penalty floor / span for OCR and ASR quality.
    media_penalty_floor: float = 0.85
    media_penalty_span: float = 0.15
    #: Never emit a confidence outside this range.
    confidence_min: float = 0.05
    confidence_max: float = 0.97
    #: Deciles used by the calibration step.
    calibration_bins: int = 10


# --------------------------------------------------------------------------- #
# Routing behaviour
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RoutingConfig:
    """Escalation policy: who gets to talk to the LLM."""

    #: Scores at least this far from a boundary skip the LLM entirely.
    high_margin_skip_llm: float = 12.0
    #: Escalate anything whose relationship confidence is below this.
    escalate_below_relationship_confidence: float = 0.45
    #: Minimum messages a contact needs before LLM relationship escalation.
    relationship_escalation_min_messages: int = 5
    #: Burst suppression window and count.
    burst_window_minutes: int = 15
    burst_notify_threshold: int = 3
    #: Do-not-disturb window (local hours, inclusive start / exclusive end).
    dnd_start_hour: int = 23
    dnd_end_hour: int = 7
    #: Maximum evidence ids emitted per decision.
    max_evidence_ids: int = 4
    #: Evidence is never empty; backfill from the top pre-scored candidate.
    backfill_evidence: bool = True


#: Convenience constant used by :mod:`src.schema` validation.
MAX_EVIDENCE_IDS: Final[int] = RoutingConfig().max_evidence_ids

#: Closed vocabulary for the ``action`` field.
ACTIONS: Final[tuple[str, ...]] = ("notify", "digest", "mute")

#: Closed vocabulary for ``message_type``. Extend here, nowhere else.
MESSAGE_TYPES: Final[tuple[str, ...]] = (
    "personal",
    "group_chat",
    "work",
    "otp",
    "transactional",
    "promotional",
    "reminder",
    "media_share",
    "forward",
    "spam",
    "other",
)

#: Closed vocabulary for relationship categories.
RELATIONSHIP_CATEGORIES: Final[tuple[str, ...]] = (
    "Family",
    "Office",
    "College",
    "Close Friend",
    "Society",
    "Business",
    "Unknown",
)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%H:%M:%S"


def configure_logging(
    level: str | int | None = None,
    log_file: Path | None = None,
) -> None:
    """Configure root logging once, for stdout and optionally a file.

    Parameters
    ----------
    level:
        Log level name or number. Falls back to ``WA_ROUTER_LOG_LEVEL`` and
        then to ``INFO``.
    log_file:
        Optional path for a file handler. Parent directories are created.

    Notes
    -----
    Safe to call more than once; existing handlers are cleared so that repeated
    calls from notebooks do not duplicate output.
    """
    resolved = level if level is not None else os.environ.get("WA_ROUTER_LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(resolved)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Third-party noise we never want at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Aggregate application config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AppConfig:
    """Single object carrying every sub-configuration.

    Downstream modules should accept an ``AppConfig`` rather than reaching for
    module-level globals, which keeps them testable.
    """

    paths: Paths = field(default_factory=Paths)
    media: MediaConfig = field(default_factory=MediaConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    #: When True, schema problems raise instead of logging.
    strict_schema: bool = field(
        default_factory=lambda: os.environ.get("WA_ROUTER_STRICT", "0") == "1"
    )
    #: Global random seed for any sampling / splitting.
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Apply the dataset-directory environment override, if present."""
        override = os.environ.get("WA_ROUTER_DATA_DIR", "").strip()
        if override:
            raw = Path(override).expanduser().resolve()
            # ``Paths`` is frozen, so rebuild it rather than mutating.
            object.__setattr__(self, "paths", _paths_with_raw(self.paths, raw))


def _paths_with_raw(base: Paths, raw: Path) -> Paths:
    """Return a copy of ``base`` whose raw-CSV directory is ``raw``.

    Only the raw directory moves; caches and outputs stay under the project
    root so that a read-only dataset mount still works.
    """
    return Paths(
        root=base.root,
        data=base.data,
        raw=raw,
        media=base.media,
        media_images=base.media_images,
        media_voice=base.media_voice,
        cache=base.cache,
        outputs=base.outputs,
        logs=base.logs,
        config_dir=base.config_dir,
        enriched_messages=base.enriched_messages,
        ocr_cache=base.ocr_cache,
        asr_cache=base.asr_cache,
        sender_affinity=base.sender_affinity,
        group_profile=base.group_profile,
        business_profile=base.business_profile,
        contact_relationships=base.contact_relationships,
        tfidf_index=base.tfidf_index,
        llm_cache=base.llm_cache,
        rules_yaml=base.rules_yaml,
        weights_yaml=base.weights_yaml,
        lexicons_yaml=base.lexicons_yaml,
        thresholds_json=base.thresholds_json,
        calibration_json=base.calibration_json,
        submission_csv=base.submission_csv,
        traces_jsonl=base.traces_jsonl,
    )


#: Process-wide default configuration. Import this unless you need a variant.
CONFIG: Final[AppConfig] = AppConfig()


def get_config() -> AppConfig:
    """Return the process-wide :class:`AppConfig` singleton."""
    return CONFIG