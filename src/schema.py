"""
Typed domain objects shared by every stage of the router.

This module is the contract between components: the media stage fills in
``Message.ocr_text``, the relationship engine emits ``RelationshipResult``, the
scoring engine emits ``PriorityResult``, and the arbiter turns all of it into a
``Decision`` with an attached ``ScoreTrace``.

Design rules
------------
* Every class validates its own invariants in ``__post_init__``.
* Every class can serialise itself with ``to_dict()`` for JSONL traces.
* Nothing here touches pandas, the filesystem, or the network.

Dependencies
------------
``src.config`` (constants only). Standard library otherwise.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Iterable, Mapping

from src.config import MAX_EVIDENCE_IDS

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    return max(low, min(high, value))


def _validate_unit_interval(name: str, value: float) -> float:
    """Validate that ``value`` is a finite number in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If the value is NaN, infinite, or outside the unit interval.
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0.0, 1.0], got {value!r}")
    return float(value)


def _jsonable(value: Any) -> Any:
    """Recursively convert a value into something ``json.dumps`` accepts."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Action(str, Enum):
    """The three routing outcomes the model must predict."""

    NOTIFY = "notify"
    DIGEST = "digest"
    MUTE = "mute"

    @property
    def rank(self) -> int:
        """Ordinal priority: higher means more intrusive.

        Used by the override layer to compare a floor against a ceiling.
        """
        return {"mute": 0, "digest": 1, "notify": 2}[self.value]

    @classmethod
    def from_any(cls, value: str | "Action") -> "Action":
        """Parse loosely-cased text into an :class:`Action`.

        Raises
        ------
        ValueError
            If the text does not name a known action.
        """
        if isinstance(value, cls):
            return value
        normalised = str(value).strip().lower()
        try:
            return cls(normalised)
        except ValueError as exc:
            raise ValueError(f"Unknown action {value!r}") from exc


class MessageType(str, Enum):
    """Vocabulary for the predicted ``message_type`` field.

    The first block is the task's official closed vocabulary, which is what
    ``output.csv`` must contain. The second block is the internal taxonomy the
    rule and content layers were built against; it is retained for backward
    compatibility and mapped onto the official values at the output boundary
    by :func:`src.io.writers.to_official_message_type`.
    """

    # ---- official task vocabulary ---------------------------------------- #
    PERSONAL = "personal"
    URGENT = "urgent"
    EVENT = "event"
    PAYMENT = "payment"
    BUSINESS_UPDATE = "business_update"
    PROMOTION = "promotion"
    GREETING = "greeting"
    FORWARD = "forward"
    SPAM = "spam"
    SCAM = "scam"
    UNKNOWN = "unknown"

    # ---- internal taxonomy (mapped on output) ---------------------------- #
    GROUP_CHAT = "group_chat"
    WORK = "work"
    OTP = "otp"
    TRANSACTIONAL = "transactional"
    PROMOTIONAL = "promotional"
    REMINDER = "reminder"
    MEDIA_SHARE = "media_share"
    OTHER = "other"

    @classmethod
    def from_any(cls, value: str | "MessageType" | None) -> "MessageType":
        """Parse text into a :class:`MessageType`, defaulting to ``OTHER``."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.OTHER
        normalised = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return cls(normalised)
        except ValueError:
            return cls.OTHER


class RelationshipCategory(str, Enum):
    """Sender relationship categories produced by the relationship engine."""

    FAMILY = "Family"
    OFFICE = "Office"
    COLLEGE = "College"
    CLOSE_FRIEND = "Close Friend"
    SOCIETY = "Society"
    BUSINESS = "Business"
    UNKNOWN = "Unknown"

    @classmethod
    def from_any(cls, value: str | "RelationshipCategory" | None) -> "RelationshipCategory":
        """Parse text into a category, defaulting to :attr:`UNKNOWN`."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        normalised = str(value).strip().lower()
        for member in cls:
            if member.value.lower() == normalised:
                return member
        return cls.UNKNOWN


class ConversationType(str, Enum):
    """Surface a message arrived on."""

    DIRECT = "direct"
    GROUP = "group"
    BUSINESS = "business"
    BROADCAST = "broadcast"
    UNKNOWN = "unknown"

    @classmethod
    def from_any(cls, value: str | "ConversationType" | None) -> "ConversationType":
        """Parse text into a conversation type, defaulting to :attr:`UNKNOWN`."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        normalised = str(value).strip().lower()
        aliases = {
            "dm": cls.DIRECT,
            "1:1": cls.DIRECT,
            "one_to_one": cls.DIRECT,
            "personal": cls.DIRECT,
            "grp": cls.GROUP,
            "community": cls.GROUP,
            "biz": cls.BUSINESS,
        }
        if normalised in aliases:
            return aliases[normalised]
        try:
            return cls(normalised)
        except ValueError:
            return cls.UNKNOWN


class MediaType(str, Enum):
    """Payload type carried by a message."""

    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    DOCUMENT = "document"
    OTHER = "other"

    @classmethod
    def from_any(cls, value: str | "MediaType" | None) -> "MediaType":
        """Parse text into a media type, defaulting to :attr:`TEXT`."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.TEXT
        normalised = str(value).strip().lower()
        aliases = {
            "img": cls.IMAGE,
            "photo": cls.IMAGE,
            "picture": cls.IMAGE,
            "audio": cls.VOICE,
            "voice_note": cls.VOICE,
            "voicenote": cls.VOICE,
            "ptt": cls.VOICE,
            "doc": cls.DOCUMENT,
            "pdf": cls.DOCUMENT,
            "file": cls.DOCUMENT,
            "": cls.TEXT,
        }
        if normalised in aliases:
            return aliases[normalised]
        try:
            return cls(normalised)
        except ValueError:
            return cls.OTHER


class DecisionSource(str, Enum):
    """Which layer produced the final action. Recorded on every decision."""

    RULE = "rule"
    SCORE = "score"
    LLM = "llm"
    OVERRIDE = "override"
    FALLBACK = "fallback"


class OverrideEffect(str, Enum):
    """Direction of an override clamp."""

    FLOOR = "floor"
    CEILING = "ceiling"
    FORCE = "force"


# --------------------------------------------------------------------------- #
# Core message object
# --------------------------------------------------------------------------- #


@dataclass
class Message:
    """A single incoming message, enriched with media transcriptions.

    The raw CSV row supplies identifiers, timestamp and text. Stage A fills in
    :attr:`ocr_text` / :attr:`asr_text` and their quality signals.
    """

    message_id: str
    sender_id: str
    timestamp: datetime
    recipient_user_id: str | None = None
    conversation_id: str | None = None
    group_id: str | None = None
    business_id: str | None = None
    reply_to_id: str | None = None

    media_type: MediaType = MediaType.TEXT
    conversation_type: ConversationType = ConversationType.UNKNOWN

    message_text: str = ""
    ocr_text: str = ""
    asr_text: str = ""

    image_id: str | None = None
    voice_note_id: str | None = None
    media_path: str | None = None

    mentions: tuple[str, ...] = ()
    is_forwarded: bool = False
    is_from_business: bool = False

    ocr_confidence: float | None = None
    asr_avg_logprob: float | None = None
    voice_duration_sec: float | None = None

    #: Anything else from the source row, kept so nothing is silently lost.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise identifiers, enums and confidence fields."""
        if not str(self.message_id).strip():
            raise ValueError("Message.message_id must be a non-empty string")
        self.message_id = str(self.message_id).strip()
        self.sender_id = str(self.sender_id).strip()

        for attr in (
            "recipient_user_id",
            "conversation_id",
            "group_id",
            "business_id",
            "reply_to_id",
            "image_id",
            "voice_note_id",
        ):
            value = getattr(self, attr)
            setattr(self, attr, str(value).strip() if value not in (None, "") else None)

        self.media_type = MediaType.from_any(self.media_type)
        self.conversation_type = ConversationType.from_any(self.conversation_type)
        self.mentions = tuple(str(m).strip() for m in self.mentions if str(m).strip())

        self.message_text = (self.message_text or "").strip()
        self.ocr_text = (self.ocr_text or "").strip()
        self.asr_text = (self.asr_text or "").strip()

        if self.ocr_confidence is not None:
            self.ocr_confidence = _validate_unit_interval("ocr_confidence", self.ocr_confidence)
        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                f"Message.timestamp must be a datetime, got {type(self.timestamp).__name__}"
            )

    # ---- derived properties ------------------------------------------------ #

    @property
    def content(self) -> str:
        """Unified text used everywhere downstream.

        Concatenates the typed text with any OCR or ASR transcription, tagging
        transcribed segments so the prompt (and a human reader) can see where
        the words came from.
        """
        parts: list[str] = []
        if self.message_text:
            parts.append(self.message_text)
        if self.ocr_text:
            confidence = self.ocr_confidence if self.ocr_confidence is not None else 0.0
            parts.append(f"[image-ocr, conf={confidence:.2f}] {self.ocr_text}")
        if self.asr_text:
            duration = self.voice_duration_sec or 0.0
            parts.append(f"[voice-asr, {duration:.0f}s] {self.asr_text}")
        return "\n".join(parts).strip()

    @property
    def has_media(self) -> bool:
        """``True`` when the message carries an image or a voice note."""
        return self.media_type in (MediaType.IMAGE, MediaType.VOICE)

    @property
    def media_quality(self) -> float:
        """Quality of the transcription backing this message, in ``[0, 1]``.

        Text messages return ``1.0``. Images use the OCR confidence. Voice
        notes map the Whisper average log-probability onto ``[0, 1]`` with
        ``-1.0`` treated as the lower bound.
        """
        if self.media_type is MediaType.IMAGE:
            return _clamp(self.ocr_confidence if self.ocr_confidence is not None else 0.5, 0.0, 1.0)
        if self.media_type is MediaType.VOICE:
            if self.asr_avg_logprob is None:
                return 0.5
            return _clamp(1.0 + float(self.asr_avg_logprob), 0.0, 1.0)
        return 1.0

    @property
    def is_group_message(self) -> bool:
        """``True`` when the message arrived in a group conversation."""
        return self.group_id is not None or self.conversation_type is ConversationType.GROUP

    def mentions_user(self, user_id: str | None) -> bool:
        """Return whether ``user_id`` is explicitly @-mentioned."""
        if not user_id:
            return False
        return str(user_id).strip() in self.mentions

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return _jsonable(asdict(self))


# --------------------------------------------------------------------------- #
# Profiles (Stage A artefacts)
# --------------------------------------------------------------------------- #


@dataclass
class UserProfile:
    """Per-user preferences and behavioural aggregates.

    Populated once in Stage A from ``users.csv`` and ``message_events.csv``.
    """

    user_id: str
    display_name: str = ""
    timezone: str = "Asia/Kolkata"
    dnd_start_hour: int = 23
    dnd_end_hour: int = 7

    muted_contacts: frozenset[str] = frozenset()
    muted_groups: frozenset[str] = frozenset()
    pinned_contacts: frozenset[str] = frozenset()
    blocked_contacts: frozenset[str] = frozenset()

    #: Global open rate across all messages, used as a fallback prior.
    overall_open_rate: float = 0.5
    overall_reply_rate: float = 0.3
    median_reply_latency_sec: float | None = None
    messages_per_day: float = 0.0

    #: Free-form notification preferences, e.g. ``{"promotional": "mute"}``.
    notification_preferences: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce ids to strings and validate rate fields."""
        self.user_id = str(self.user_id).strip()
        self.muted_contacts = frozenset(str(x) for x in self.muted_contacts)
        self.muted_groups = frozenset(str(x) for x in self.muted_groups)
        self.pinned_contacts = frozenset(str(x) for x in self.pinned_contacts)
        self.blocked_contacts = frozenset(str(x) for x in self.blocked_contacts)
        self.overall_open_rate = _validate_unit_interval(
            "overall_open_rate", self.overall_open_rate
        )
        self.overall_reply_rate = _validate_unit_interval(
            "overall_reply_rate", self.overall_reply_rate
        )
        if not 0 <= self.dnd_start_hour <= 23 or not 0 <= self.dnd_end_hour <= 23:
            raise ValueError("DND hours must be in [0, 23]")

    def is_in_dnd(self, moment: datetime) -> bool:
        """Return whether ``moment``'s hour falls in this user's DND window.

        Handles windows that wrap past midnight (e.g. 23:00 to 07:00).
        """
        hour = moment.hour
        if self.dnd_start_hour <= self.dnd_end_hour:
            return self.dnd_start_hour <= hour < self.dnd_end_hour
        return hour >= self.dnd_start_hour or hour < self.dnd_end_hour

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return _jsonable(asdict(self))


@dataclass
class GroupProfile:
    """Per-group metadata plus this user's engagement with the group."""

    group_id: str
    name: str = ""
    size: int = 0
    member_ids: tuple[str, ...] = ()
    created_at: datetime | None = None

    #: Category hinted by the group name, e.g. Office from "Team Standup".
    category_hint: RelationshipCategory = RelationshipCategory.UNKNOWN
    category_hint_confidence: float = 0.0

    is_muted: bool = False
    user_read_rate: float = 0.5
    user_reply_rate: float = 0.1
    messages_per_day: float = 0.0

    def __post_init__(self) -> None:
        """Coerce ids and validate rate fields."""
        self.group_id = str(self.group_id).strip()
        self.member_ids = tuple(str(m) for m in self.member_ids)
        self.size = int(self.size) if self.size else len(self.member_ids)
        self.category_hint = RelationshipCategory.from_any(self.category_hint)
        self.category_hint_confidence = _validate_unit_interval(
            "category_hint_confidence", self.category_hint_confidence
        )
        self.user_read_rate = _validate_unit_interval("user_read_rate", self.user_read_rate)
        self.user_reply_rate = _validate_unit_interval("user_reply_rate", self.user_reply_rate)

    @property
    def is_large(self) -> bool:
        """``True`` for groups big enough that most traffic is not for you."""
        return self.size >= 25

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return _jsonable(asdict(self))


@dataclass
class BusinessProfile:
    """Business-account metadata and this user's transaction history with it."""

    business_id: str
    name: str = ""
    category: str = ""
    is_verified: bool = False

    user_txn_count: int = 0
    last_order_age_days: float | None = None
    has_active_order: bool = False

    #: Share of this business's messages that were promotional.
    promo_ratio: float = 0.0
    #: Share of this business's messages this user opened.
    user_open_rate: float = 0.0

    def __post_init__(self) -> None:
        """Coerce ids and validate ratio fields."""
        self.business_id = str(self.business_id).strip()
        self.user_txn_count = max(0, int(self.user_txn_count))
        self.promo_ratio = _validate_unit_interval("promo_ratio", self.promo_ratio)
        self.user_open_rate = _validate_unit_interval("user_open_rate", self.user_open_rate)

    @property
    def is_known_to_user(self) -> bool:
        """``True`` when the user has transacted with this business before."""
        return self.user_txn_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return _jsonable(asdict(self))


# --------------------------------------------------------------------------- #
# Relationship engine output
# --------------------------------------------------------------------------- #


@dataclass
class RelationshipResult:
    """Classification of one ``(user, peer)`` pair.

    Computed once in Stage A and looked up in O(1) on the hot path, so that the
    same contact never flips category between messages.
    """

    user_id: str
    peer_id: str
    category: RelationshipCategory = RelationshipCategory.UNKNOWN
    confidence: float = 0.30
    signals: tuple[str, ...] = ()
    evidence_message_ids: tuple[str, ...] = ()
    #: Which fusion branch fired: ``"business"``, ``"name_kinship"``,
    #: ``"group_pattern"``, ``"multi_signal"``, ``"behavioral"``,
    #: ``"llm"`` or ``"default"``.
    method: str = "default"

    def __post_init__(self) -> None:
        """Normalise ids, enum and confidence; cap evidence length."""
        self.user_id = str(self.user_id).strip()
        self.peer_id = str(self.peer_id).strip()
        self.category = RelationshipCategory.from_any(self.category)
        self.confidence = _validate_unit_interval("RelationshipResult.confidence", self.confidence)
        self.signals = tuple(str(s) for s in self.signals)
        self.evidence_message_ids = tuple(
            str(m) for m in self.evidence_message_ids[:MAX_EVIDENCE_IDS]
        )

    @property
    def is_confident(self) -> bool:
        """``True`` when the classification is strong enough to trust as-is."""
        return self.confidence >= 0.45

    def to_dict(self) -> dict[str, Any]:
        """Serialise in the exact shape agreed for the demo payload."""
        return {
            "relationship_category": self.category.value,
            "confidence": round(self.confidence, 4),
            "signals": list(self.signals),
            "evidence_message_ids": list(self.evidence_message_ids),
            "method": self.method,
            "user_id": self.user_id,
            "peer_id": self.peer_id,
        }


# --------------------------------------------------------------------------- #
# Scoring engine output
# --------------------------------------------------------------------------- #


@dataclass
class ScoreComponent:
    """One of the six additive contributions to the priority score."""

    name: str
    points: float
    cap: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Clamp points into the component's own range.

        Penalty components declare a negative ``cap`` (a floor); positive
        components clamp to ``[0, cap]``.
        """
        self.name = str(self.name).strip()
        self.points = float(self.points)
        self.cap = float(self.cap)
        if self.cap >= 0:
            self.points = _clamp(self.points, 0.0, self.cap)
        else:
            self.points = _clamp(self.points, self.cap, 0.0)
        self.reasons = tuple(str(r) for r in self.reasons)

    @property
    def utilisation(self) -> float:
        """Fraction of the component's range actually used, in ``[0, 1]``."""
        if self.cap == 0:
            return 0.0
        return _clamp(abs(self.points) / abs(self.cap), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "points": round(self.points, 3),
            "cap": self.cap,
            "utilisation": round(self.utilisation, 3),
            "reasons": list(self.reasons),
        }


@dataclass
class OverrideRecord:
    """A clamp applied after scoring: a floor, a ceiling, or a hard force."""

    rule_id: str
    effect: OverrideEffect
    bound: Action
    #: ``True`` when this clamp actually changed the outcome.
    binding: bool = False
    confidence: float = 0.90
    note: str = ""

    def __post_init__(self) -> None:
        """Normalise enums and validate confidence."""
        self.rule_id = str(self.rule_id).strip()
        self.effect = OverrideEffect(self.effect)
        self.bound = Action.from_any(self.bound)
        self.confidence = _validate_unit_interval("OverrideRecord.confidence", self.confidence)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "rule_id": self.rule_id,
            "effect": self.effect.value,
            "bound": self.bound.value,
            "binding": self.binding,
            "confidence": round(self.confidence, 4),
            "note": self.note,
        }


@dataclass
class PriorityResult:
    """Aggregate output of the six-component scoring engine."""

    message_id: str
    components: tuple[ScoreComponent, ...] = ()
    raw_score: float = 0.0
    final_score: float = 0.0
    band: Action = Action.DIGEST
    thresholds: tuple[float, float] = (38.0, 71.0)
    overrides: tuple[OverrideRecord, ...] = ()

    def __post_init__(self) -> None:
        """Validate score bounds and threshold ordering."""
        self.message_id = str(self.message_id).strip()
        self.components = tuple(self.components)
        self.raw_score = float(self.raw_score)
        self.final_score = _clamp(float(self.final_score), 0.0, 100.0)
        self.band = Action.from_any(self.band)
        low, high = float(self.thresholds[0]), float(self.thresholds[1])
        if low >= high:
            raise ValueError(f"thresholds must be increasing, got ({low}, {high})")
        self.thresholds = (low, high)
        self.overrides = tuple(self.overrides)

    @property
    def margin(self) -> float:
        """Distance from :attr:`final_score` to the nearest band boundary."""
        low, high = self.thresholds
        return min(abs(self.final_score - low), abs(self.final_score - high))

    @property
    def binding_overrides(self) -> tuple[OverrideRecord, ...]:
        """Only the overrides that actually changed the outcome."""
        return tuple(o for o in self.overrides if o.binding)

    @property
    def was_overridden(self) -> bool:
        """``True`` when at least one clamp changed the scored band."""
        return bool(self.binding_overrides)

    def component(self, name: str) -> ScoreComponent | None:
        """Return the component called ``name``, or ``None`` if absent."""
        for item in self.components:
            if item.name.lower() == name.lower():
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "message_id": self.message_id,
            "components": [c.to_dict() for c in self.components],
            "raw_score": round(self.raw_score, 3),
            "final_score": round(self.final_score, 3),
            "band": self.band.value,
            "thresholds": list(self.thresholds),
            "margin": round(self.margin, 3),
            "overrides": [o.to_dict() for o in self.overrides],
        }


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


@dataclass
class RetrievalCandidate:
    """One prior message offered to the LLM as possible evidence."""

    message_id: str
    sender_id: str
    timestamp: datetime
    snippet: str
    #: Which retrieval tier produced it: ``structural`` / ``relational`` /
    #: ``business`` / ``lexical``.
    tier: str = "structural"
    pre_score: float = 0.0
    age_hours: float = 0.0

    def __post_init__(self) -> None:
        """Normalise identifiers and snippet whitespace."""
        self.message_id = str(self.message_id).strip()
        self.sender_id = str(self.sender_id).strip()
        self.snippet = " ".join((self.snippet or "").split())
        self.pre_score = float(self.pre_score)

    def render(self, max_chars: int = 140) -> str:
        """Render a compact one-line form for the prompt."""
        body = self.snippet[:max_chars]
        return f"[{self.message_id} | {self.sender_id} | t-{self.age_hours:.1f}h] {body}"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return _jsonable(asdict(self))


@dataclass
class RetrievalContext:
    """The evidence pool assembled for a single message."""

    message_id: str
    candidates: tuple[RetrievalCandidate, ...] = ()
    used_lexical: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        """Sort candidates by pre-score, descending."""
        self.message_id = str(self.message_id).strip()
        self.candidates = tuple(
            sorted(self.candidates, key=lambda c: c.pre_score, reverse=True)
        )

    @property
    def candidate_ids(self) -> frozenset[str]:
        """Set of ids the LLM is permitted to cite."""
        return frozenset(c.message_id for c in self.candidates)

    @property
    def is_empty(self) -> bool:
        """``True`` when no prior context was found."""
        return not self.candidates

    def top(self, n: int) -> tuple[RetrievalCandidate, ...]:
        """Return the ``n`` highest pre-scored candidates."""
        return self.candidates[: max(0, n)]

    def render(self, max_chars: int = 140) -> str:
        """Render the whole pool as newline-separated prompt lines."""
        if not self.candidates:
            return "(no prior context found)"
        return "\n".join(c.render(max_chars) for c in self.candidates)

    def validate_ids(self, ids: Iterable[str]) -> tuple[list[str], list[str]]:
        """Split ``ids`` into those present in the pool and those invented.

        Returns
        -------
        tuple
            ``(valid_ids, hallucinated_ids)``, order preserved, duplicates
            removed from the valid list.
        """
        allowed = self.candidate_ids
        valid: list[str] = []
        invalid: list[str] = []
        for raw in ids:
            candidate_id = str(raw).strip()
            if candidate_id in allowed:
                if candidate_id not in valid:
                    valid.append(candidate_id)
            else:
                invalid.append(candidate_id)
        return valid, invalid

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "message_id": self.message_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "used_lexical": self.used_lexical,
            "truncated": self.truncated,
        }


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #


@dataclass
class ScoreTrace:
    """Everything needed to explain one routing decision on a demo screen.

    Emitted for *every* message, including rule-resolved ones, so that the fast
    path is never visibly thinner than the LLM path.
    """

    message_id: str
    relationship: RelationshipResult | None = None
    priority: PriorityResult | None = None
    rule_id: str | None = None
    llm_action: Action | None = None
    llm_confidence: float | None = None
    llm_agreed: bool | None = None
    final_action: Action = Action.DIGEST
    confidence: float = 0.5
    source: DecisionSource = DecisionSource.SCORE
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalise enums and validate confidences."""
        self.message_id = str(self.message_id).strip()
        self.final_action = Action.from_any(self.final_action)
        self.source = DecisionSource(self.source)
        self.confidence = _validate_unit_interval("ScoreTrace.confidence", self.confidence)
        if self.llm_confidence is not None:
            self.llm_confidence = _validate_unit_interval(
                "ScoreTrace.llm_confidence", self.llm_confidence
            )
        if self.llm_action is not None:
            self.llm_action = Action.from_any(self.llm_action)
        self.notes = tuple(str(n) for n in self.notes)

    @property
    def disagreement(self) -> bool:
        """``True`` when the scored band and the LLM verdict differ.

        This set is both the best error-analysis surface and the most
        interesting thing to show a judge.
        """
        if self.priority is None or self.llm_action is None:
            return False
        return self.priority.band is not self.llm_action

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full trace for ``outputs/traces.jsonl``."""
        return {
            "message_id": self.message_id,
            "relationship": self.relationship.to_dict() if self.relationship else None,
            "priority": self.priority.to_dict() if self.priority else None,
            "rule_id": self.rule_id,
            "llm": {
                "action": self.llm_action.value if self.llm_action else None,
                "confidence": self.llm_confidence,
                "agreed": self.llm_agreed,
            },
            "disagreement": self.disagreement,
            "final_action": self.final_action.value,
            "confidence": round(self.confidence, 4),
            "source": self.source.value,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Final decision
# --------------------------------------------------------------------------- #


@dataclass
class Decision:
    """The routed outcome for one message: the row we actually submit."""

    message_id: str
    action: Action
    message_type: MessageType
    reason: str
    confidence: float
    evidence_message_ids: tuple[str, ...] = ()
    source: DecisionSource = DecisionSource.SCORE
    trace: ScoreTrace | None = None

    def __post_init__(self) -> None:
        """Validate the submission contract.

        Enforces the closed vocabularies, the confidence range, the evidence
        cap, and a non-empty reason string.
        """
        self.message_id = str(self.message_id).strip()
        if not self.message_id:
            raise ValueError("Decision.message_id must be non-empty")

        self.action = Action.from_any(self.action)
        self.message_type = MessageType.from_any(self.message_type)
        self.source = DecisionSource(self.source)

        self.reason = " ".join((self.reason or "").split())
        if not self.reason:
            raise ValueError(f"Decision.reason must be non-empty for {self.message_id}")

        self.confidence = _validate_unit_interval("Decision.confidence", self.confidence)

        deduped: list[str] = []
        for raw in self.evidence_message_ids:
            evidence_id = str(raw).strip()
            if evidence_id and evidence_id not in deduped:
                deduped.append(evidence_id)
        self.evidence_message_ids = tuple(deduped[:MAX_EVIDENCE_IDS])

    def to_submission_row(self) -> dict[str, Any]:
        """Return the flat dictionary written to ``submission.csv``.

        Evidence ids are pipe-joined so the CSV stays single-valued per cell.
        """
        return {
            "message_id": self.message_id,
            "action": self.action.value,
            "message_type": self.message_type.value,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "evidence_message_ids": "|".join(self.evidence_message_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the decision plus its trace."""
        payload = self.to_submission_row()
        payload["evidence_message_ids"] = list(self.evidence_message_ids)
        payload["source"] = self.source.value
        payload["trace"] = self.trace.to_dict() if self.trace else None
        return payload

    @classmethod
    def fallback(
        cls,
        message_id: str,
        reason: str = "Fallback decision: routing pipeline could not resolve this message.",
    ) -> "Decision":
        """Build a safe default decision for a message that blew up.

        Defaults to ``digest`` at low confidence, because the cost asymmetry
        favours over-delivering to a digest over silently dropping a message.
        """
        return cls(
            message_id=message_id,
            action=Action.DIGEST,
            message_type=MessageType.OTHER,
            reason=reason,
            confidence=0.20,
            evidence_message_ids=(),
            source=DecisionSource.FALLBACK,
        )


__all__ = [
    "Action",
    "BusinessProfile",
    "ConversationType",
    "Decision",
    "DecisionSource",
    "GroupProfile",
    "MediaType",
    "Message",
    "MessageType",
    "OverrideEffect",
    "OverrideRecord",
    "PriorityResult",
    "RelationshipCategory",
    "RelationshipResult",
    "RetrievalCandidate",
    "RetrievalContext",
    "ScoreComponent",
    "ScoreTrace",
    "UserProfile",
]