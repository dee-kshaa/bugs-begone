"""
Relationship signal extraction.

This module produces the *signals* the relationship engine fuses; it does not
itself decide a category. ``src/relationship/classifier.py`` consumes what is
produced here and emits a :class:`~src.schema.RelationshipResult`.

Three signal families live here:

* **Lexical name signals** -- kinship and organisational tokens in a contact's
  display name (Amma, Appa, Anna, Mom, Sir, HR, Prof).
* **Group name signals** -- category hints from a group's title
  ("Team Standup" -> Office, "8th Sem CSE" -> College).
* **Behavioural signals** -- reply rate, median latency, initiation ratio,
  volume and night-hour share for a ``(user, peer)`` pair.

Everything is a pure function over pandas frames and dataclasses. Nothing here
touches the network or the LLM.

Dependencies
------------
``pandas``, ``src.config``, ``src.schema``. ``PyYAML`` is optional: when
``src/relationship/lexicons.yaml`` exists and PyYAML is installed, its contents
override the built-in lexicons below.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.config import AppConfig, get_config
from src.schema import RelationshipCategory

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Built-in lexicons
# --------------------------------------------------------------------------- #

#: Kinship terms that appear in saved contact names. Covers English plus the
#: Indian-language terms most likely in a WhatsApp address book. Matched as
#: whole words against the normalised contact name.
KINSHIP_TERMS: dict[str, tuple[str, ...]] = {
    "parent": (
        "mom", "mum", "mummy", "mother", "amma", "aai", "maa", "mataji",
        "dad", "daddy", "father", "appa", "papa", "baba", "pitaji", "abbu", "ammi",
    ),
    "sibling": (
        "bro", "brother", "sis", "sister", "anna", "akka", "akkaa", "thambi",
        "thangai", "bhai", "bhaiya", "didi", "dada", "chettan", "chechi",
    ),
    "extended": (
        "uncle", "aunty", "aunt", "mama", "mami", "chacha", "chachi", "kaka",
        "kaki", "athai", "chithi", "chithappa", "periyamma", "periyappa",
        "maama", "maasi", "bua", "phupha", "nana", "nani", "dada ji", "dadi",
        "thatha", "paati", "ajji", "ajja", "grandma", "grandpa", "granny",
    ),
    "spouse_child": (
        "wife", "husband", "hubby", "beta", "beti", "kanna", "son", "daughter",
    ),
    "in_law": ("bhabhi", "jiju", "devar", "nanad", "saas", "sasur"),
}

#: Tokens that mark a contact or group as work-related.
OFFICE_TERMS: tuple[str, ...] = (
    "office", "work", "team", "standup", "scrum", "sprint", "manager", "lead",
    "hr", "admin", "boss", "client", "project", "eng", "engineering", "dev",
    "qa", "ops", "sales", "finance", "intern", "onsite", "sir", "madam",
    "ma'am", "colleague", "corp", "corporate", "desk", "shift", "roster",
)

#: Tokens that mark a contact or group as college-related.
COLLEGE_TERMS: tuple[str, ...] = (
    "college", "campus", "class", "sem", "semester", "batch", "sec", "section",
    "hostel", "prof", "professor", "faculty", "lab", "dept", "department",
    "cse", "ece", "eee", "mech", "civil", "ise", "aiml", "placement", "tnp",
    "assignment", "exam", "viva", "seminar", "project group", "ieee", "acm",
    "club", "chapter", "fest", "hackathon", "juniors", "seniors", "alumni",
)

#: Tokens that mark a contact or group as residential-society related.
SOCIETY_TERMS: tuple[str, ...] = (
    "apartment", "apt", "flat", "society", "residents", "resident", "owners",
    "tower", "block", "wing", "association", "rwa", "maintenance", "security",
    "guard", "watchman", "plumber", "electrician", "milkman", "cook", "maid",
    "driver", "neighbours", "neighbors", "gated", "layout", "colony", "enclave",
)

#: Tokens that mark a contact or group as a close-friend circle.
FRIEND_TERMS: tuple[str, ...] = (
    "friends", "gang", "squad", "buddies", "bffs", "besties", "crew",
    "the boys", "the girls", "chill", "adda", "trip", "goa", "plan",
)

#: Tokens that mark a contact as a commercial account.
BUSINESS_TERMS: tuple[str, ...] = (
    "pvt", "ltd", "llp", "inc", "store", "shop", "mart", "bazaar", "cafe",
    "restaurant", "hotel", "salon", "clinic", "hospital", "pharmacy", "bank",
    "insurance", "support", "care", "helpdesk", "service", "delivery",
    "courier", "logistics", "official", "verified", "noreply", "no-reply",
)

#: Address terms found *inside* message bodies, mapped to the category they hint.
ADDRESS_TERM_HINTS: dict[str, RelationshipCategory] = {
    "beta": RelationshipCategory.FAMILY,
    "kanna": RelationshipCategory.FAMILY,
    "putta": RelationshipCategory.FAMILY,
    "chellam": RelationshipCategory.FAMILY,
    "sir": RelationshipCategory.OFFICE,
    "madam": RelationshipCategory.OFFICE,
    "ma'am": RelationshipCategory.OFFICE,
    "team": RelationshipCategory.OFFICE,
    "folks": RelationshipCategory.OFFICE,
    "all": RelationshipCategory.OFFICE,
    "da": RelationshipCategory.CLOSE_FRIEND,
    "machan": RelationshipCategory.CLOSE_FRIEND,
    "macha": RelationshipCategory.CLOSE_FRIEND,
    "bruh": RelationshipCategory.CLOSE_FRIEND,
    "dude": RelationshipCategory.CLOSE_FRIEND,
    "guys": RelationshipCategory.CLOSE_FRIEND,
    "bandhu": RelationshipCategory.CLOSE_FRIEND,
}

#: Ordered mapping from a category to its token list, used for name matching.
CATEGORY_TERMS: dict[RelationshipCategory, tuple[str, ...]] = {
    RelationshipCategory.OFFICE: OFFICE_TERMS,
    RelationshipCategory.COLLEGE: COLLEGE_TERMS,
    RelationshipCategory.SOCIETY: SOCIETY_TERMS,
    RelationshipCategory.CLOSE_FRIEND: FRIEND_TERMS,
    RelationshipCategory.BUSINESS: BUSINESS_TERMS,
}

#: Confidence attached to a single strong lexical hit, per family.
LEXICAL_CONFIDENCE: dict[str, float] = {
    "kinship": 0.90,
    "group_name": 0.85,
    "org_name": 0.70,
    "address_term": 0.45,
}


# --------------------------------------------------------------------------- #
# Lexicon overrides from YAML
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def load_lexicons(config_path: str | None = None) -> dict[str, Any]:
    """Load lexicon overrides from ``src/relationship/lexicons.yaml``.

    The YAML file is optional. When present, its top-level keys
    (``kinship_terms``, ``office_terms``, ``college_terms``, ``society_terms``,
    ``friend_terms``, ``business_terms``) *replace* the corresponding built-in
    tuples so that tuning never requires a code change.

    Parameters
    ----------
    config_path:
        Explicit path to a YAML file. Defaults to the configured location.

    Returns
    -------
    dict
        Merged lexicon dictionary. Always usable, even when no file exists.
    """
    merged: dict[str, Any] = {
        "kinship_terms": tuple(term for group in KINSHIP_TERMS.values() for term in group),
        "office_terms": OFFICE_TERMS,
        "college_terms": COLLEGE_TERMS,
        "society_terms": SOCIETY_TERMS,
        "friend_terms": FRIEND_TERMS,
        "business_terms": BUSINESS_TERMS,
    }

    path = config_path or str(get_config().paths.lexicons_yaml)
    try:
        import yaml  # Imported lazily: PyYAML is optional.
    except ImportError:
        logger.debug("PyYAML not installed; using built-in lexicons only.")
        return merged

    from pathlib import Path

    yaml_path = Path(path)
    if not yaml_path.exists():
        logger.debug("No lexicon override at %s; using built-in lexicons.", yaml_path)
        return merged

    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception as error:  # noqa: BLE001 - a bad YAML must not kill the run
        logger.error("Failed to parse %s (%s); using built-in lexicons.", yaml_path, error)
        return merged

    for key, value in payload.items():
        if key in merged and isinstance(value, (list, tuple)):
            merged[key] = tuple(str(item).strip().lower() for item in value if str(item).strip())
            logger.info("Lexicon override applied for %r (%d terms).", key, len(merged[key]))

    return merged


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

_NON_WORD = re.compile(r"[^a-z0-9\s']+")
_WHITESPACE = re.compile(r"\s+")


def normalise_name(name: str | None) -> str:
    """Lower-case a contact or group name and strip punctuation and emoji.

    Parameters
    ----------
    name:
        Raw display name, possibly containing emoji, dots or underscores.

    Returns
    -------
    str
        Whitespace-separated lower-case tokens, or ``""`` for empty input.
    """
    if not name:
        return ""
    lowered = str(name).strip().lower()
    cleaned = _NON_WORD.sub(" ", lowered)
    return _WHITESPACE.sub(" ", cleaned).strip()


def tokenise(text: str | None) -> tuple[str, ...]:
    """Split normalised text into word tokens."""
    normalised = normalise_name(text)
    return tuple(token for token in normalised.split(" ") if token)


def _match_terms(tokens: Sequence[str], normalised: str, terms: Iterable[str]) -> list[str]:
    """Return every term that matches, as a whole token or a phrase.

    Single-word terms must match a token exactly, which avoids "ma" firing on
    "mahesh". Multi-word terms are matched as substrings of the normalised name.
    """
    token_set = set(tokens)
    hits: list[str] = []
    for term in terms:
        term_clean = str(term).strip().lower()
        if not term_clean:
            continue
        if " " in term_clean:
            if term_clean in normalised:
                hits.append(term_clean)
        elif term_clean in token_set:
            hits.append(term_clean)
    return hits


# --------------------------------------------------------------------------- #
# Lexical signals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LexicalSignal:
    """One category hint derived from a name or a message body."""

    category: RelationshipCategory
    confidence: float
    #: Human-readable signal string, e.g. ``"name_kinship:amma"``.
    label: str
    #: Which family produced it: kinship / group_name / org_name / address_term.
    family: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "category": self.category.value,
            "confidence": round(self.confidence, 4),
            "label": self.label,
            "family": self.family,
        }


def extract_name_signals(display_name: str | None) -> list[LexicalSignal]:
    """Derive category hints from a contact's saved display name.

    Kinship terms are checked first and carry the highest confidence, because a
    contact saved as "Amma" is almost never anything but family.

    Parameters
    ----------
    display_name:
        Saved contact name, e.g. ``"Amma"``, ``"Rahul HR"``, ``"Swiggy"``.

    Returns
    -------
    list of LexicalSignal
        Possibly empty, ordered strongest-first.
    """
    normalised = normalise_name(display_name)
    if not normalised:
        return []

    tokens = tuple(normalised.split(" "))
    lexicons = load_lexicons()
    signals: list[LexicalSignal] = []

    kinship_hits = _match_terms(tokens, normalised, lexicons["kinship_terms"])
    for hit in kinship_hits:
        signals.append(
            LexicalSignal(
                category=RelationshipCategory.FAMILY,
                confidence=LEXICAL_CONFIDENCE["kinship"],
                label=f"name_kinship:{hit}",
                family="kinship",
            )
        )

    term_sources = {
        RelationshipCategory.OFFICE: lexicons["office_terms"],
        RelationshipCategory.COLLEGE: lexicons["college_terms"],
        RelationshipCategory.SOCIETY: lexicons["society_terms"],
        RelationshipCategory.CLOSE_FRIEND: lexicons["friend_terms"],
        RelationshipCategory.BUSINESS: lexicons["business_terms"],
    }
    for category, terms in term_sources.items():
        for hit in _match_terms(tokens, normalised, terms):
            signals.append(
                LexicalSignal(
                    category=category,
                    confidence=LEXICAL_CONFIDENCE["org_name"],
                    label=f"name_org:{hit}",
                    family="org_name",
                )
            )

    signals.sort(key=lambda item: item.confidence, reverse=True)
    return signals


def extract_group_name_signals(group_name: str | None) -> list[LexicalSignal]:
    """Derive category hints from a group's title.

    Group titles are more informative than contact names because people name
    groups after their purpose ("Flat 302 Owners", "8th Sem ECE", "Sprint 14").

    Parameters
    ----------
    group_name:
        Raw group title.

    Returns
    -------
    list of LexicalSignal
        Possibly empty, ordered strongest-first.
    """
    normalised = normalise_name(group_name)
    if not normalised:
        return []

    tokens = tuple(normalised.split(" "))
    lexicons = load_lexicons()
    signals: list[LexicalSignal] = []

    kinship_hits = _match_terms(tokens, normalised, lexicons["kinship_terms"])
    if kinship_hits or "family" in tokens or "parivar" in tokens or "kutumb" in tokens:
        label = kinship_hits[0] if kinship_hits else "family"
        signals.append(
            LexicalSignal(
                category=RelationshipCategory.FAMILY,
                confidence=LEXICAL_CONFIDENCE["group_name"],
                label=f"group_name:{label}",
                family="group_name",
            )
        )

    term_sources = {
        RelationshipCategory.OFFICE: lexicons["office_terms"],
        RelationshipCategory.COLLEGE: lexicons["college_terms"],
        RelationshipCategory.SOCIETY: lexicons["society_terms"],
        RelationshipCategory.CLOSE_FRIEND: lexicons["friend_terms"],
    }
    for category, terms in term_sources.items():
        hits = _match_terms(tokens, normalised, terms)
        if hits:
            signals.append(
                LexicalSignal(
                    category=category,
                    confidence=LEXICAL_CONFIDENCE["group_name"],
                    label=f"group_name:{hits[0]}",
                    family="group_name",
                )
            )

    # A group title that is purely numeric or a bare name gives nothing away.
    signals.sort(key=lambda item: item.confidence, reverse=True)
    return signals


def extract_address_term_signals(texts: Iterable[str]) -> list[LexicalSignal]:
    """Derive weak category hints from how the sender addresses the user.

    "beta" implies family; "sir" implies office; "machan" implies a close
    friend. Individually weak, but they stack with behavioural evidence.

    Parameters
    ----------
    texts:
        Message bodies from this sender.

    Returns
    -------
    list of LexicalSignal
        One signal per distinct category found, confidence scaled by hit count.
    """
    counts: dict[RelationshipCategory, int] = {}
    examples: dict[RelationshipCategory, str] = {}

    for text in texts:
        tokens = set(tokenise(text))
        if not tokens:
            continue
        for term, category in ADDRESS_TERM_HINTS.items():
            if term in tokens:
                counts[category] = counts.get(category, 0) + 1
                examples.setdefault(category, term)

    signals: list[LexicalSignal] = []
    for category, count in counts.items():
        # Two or more occurrences is meaningfully stronger than one.
        confidence = min(LEXICAL_CONFIDENCE["address_term"] + 0.05 * (count - 1), 0.65)
        signals.append(
            LexicalSignal(
                category=category,
                confidence=confidence,
                label=f"address_term:{examples[category]}x{count}",
                family="address_term",
            )
        )

    signals.sort(key=lambda item: item.confidence, reverse=True)
    return signals


# --------------------------------------------------------------------------- #
# Behavioural signals
# --------------------------------------------------------------------------- #


@dataclass
class BehaviouralSignals:
    """Interaction statistics for one ``(user, peer)`` pair.

    Computed once in Stage A over the full message history. These are the only
    signals available for contacts saved under a bare personal name, which is
    most of a real address book.
    """

    user_id: str
    peer_id: str
    total_messages: int = 0
    inbound_messages: int = 0
    outbound_messages: int = 0
    reply_rate: float = 0.0
    median_reply_latency_sec: float | None = None
    initiation_ratio: float = 0.0
    messages_per_week: float = 0.0
    night_share: float = 0.0
    span_days: float = 0.0
    last_interaction: datetime | None = None
    evidence_message_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce identifiers to strings and clamp rate fields."""
        self.user_id = str(self.user_id)
        self.peer_id = str(self.peer_id)
        self.reply_rate = max(0.0, min(1.0, float(self.reply_rate)))
        self.initiation_ratio = max(0.0, min(1.0, float(self.initiation_ratio)))
        self.night_share = max(0.0, min(1.0, float(self.night_share)))
        self.evidence_message_ids = tuple(str(m) for m in self.evidence_message_ids)

    @property
    def is_high_engagement(self) -> bool:
        """``True`` when the pair looks like an active personal relationship.

        The thresholds are deliberately conservative: fast replies *and* real
        volume, so that a single burst of messages does not qualify.
        """
        fast = (
            self.median_reply_latency_sec is not None
            and self.median_reply_latency_sec <= 1800.0
        )
        return self.reply_rate >= 0.55 and self.messages_per_week >= 3.0 and fast

    @property
    def is_dormant(self) -> bool:
        """``True`` when there is barely any two-way history."""
        return self.total_messages < 3 or self.messages_per_week < 0.25

    def to_signals(self) -> list[str]:
        """Render as human-readable signal strings for the trace."""
        signals = [
            f"behavior:reply_rate={self.reply_rate:.2f}",
            f"behavior:msgs_per_week={self.messages_per_week:.1f}",
            f"behavior:total_messages={self.total_messages}",
        ]
        if self.median_reply_latency_sec is not None:
            signals.append(
                f"behavior:median_latency={self.median_reply_latency_sec / 60.0:.0f}m"
            )
        if self.night_share > 0.25:
            signals.append(f"behavior:night_share={self.night_share:.2f}")
        return signals

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "user_id": self.user_id,
            "peer_id": self.peer_id,
            "total_messages": self.total_messages,
            "inbound_messages": self.inbound_messages,
            "outbound_messages": self.outbound_messages,
            "reply_rate": round(self.reply_rate, 4),
            "median_reply_latency_sec": self.median_reply_latency_sec,
            "initiation_ratio": round(self.initiation_ratio, 4),
            "messages_per_week": round(self.messages_per_week, 4),
            "night_share": round(self.night_share, 4),
            "span_days": round(self.span_days, 3),
            "last_interaction": (
                self.last_interaction.isoformat() if self.last_interaction else None
            ),
            "evidence_message_ids": list(self.evidence_message_ids),
        }


def _pair_slice(
    history: pd.DataFrame,
    user_id: str,
    peer_id: str,
) -> pd.DataFrame:
    """Return the rows of ``history`` exchanged between two participants.

    Works whether or not the frame carries a ``recipient_user_id`` column: when
    it does, both directions are matched; when it does not, only messages *from*
    the peer are available and the pair is treated as inbound-only.
    """
    if history.empty or "sender_id" not in history.columns:
        return history.iloc[0:0]

    sender = history["sender_id"].astype(str)
    inbound = sender == str(peer_id)

    if "recipient_user_id" in history.columns:
        recipient = history["recipient_user_id"].astype(str)
        inbound = inbound & ((recipient == str(user_id)) | (recipient == ""))
        outbound = (sender == str(user_id)) & (recipient == str(peer_id))
        return history[inbound | outbound]

    return history[inbound]


def compute_pair_behaviour(
    history: pd.DataFrame,
    user_id: str,
    peer_id: str,
    now: datetime | None = None,
    max_evidence: int = 2,
) -> BehaviouralSignals:
    """Compute interaction statistics for one ``(user, peer)`` pair.

    Parameters
    ----------
    history:
        Message history frame with at least ``sender_id`` and ``timestamp``.
        ``message_id`` and ``recipient_user_id`` are used when present.
    user_id:
        The receiving user whose perspective we model.
    peer_id:
        The other participant (a contact id, not a group id).
    now:
        Reference time for recency. Defaults to the latest timestamp in
        ``history``, which keeps the statistics stable across reruns.
    max_evidence:
        How many message ids to keep as illustrative evidence.

    Returns
    -------
    BehaviouralSignals
        Zero-filled when the pair has no shared history.
    """
    signals = BehaviouralSignals(user_id=str(user_id), peer_id=str(peer_id))
    pair = _pair_slice(history, user_id, peer_id)
    if pair.empty or "timestamp" not in pair.columns:
        return signals

    pair = pair.dropna(subset=["timestamp"]).sort_values("timestamp")
    if pair.empty:
        return signals

    sender = pair["sender_id"].astype(str)
    inbound_mask = sender == str(peer_id)
    outbound_mask = sender == str(user_id)

    signals.total_messages = int(len(pair))
    signals.inbound_messages = int(inbound_mask.sum())
    signals.outbound_messages = int(outbound_mask.sum())

    timestamps = pd.to_datetime(pair["timestamp"])
    first_seen = timestamps.iloc[0].to_pydatetime()
    last_seen = timestamps.iloc[-1].to_pydatetime()
    signals.last_interaction = last_seen

    reference = now or last_seen
    span_seconds = max((last_seen - first_seen).total_seconds(), 0.0)
    signals.span_days = span_seconds / 86400.0
    weeks = max(signals.span_days / 7.0, 1.0 / 7.0)
    signals.messages_per_week = signals.total_messages / weeks

    if signals.total_messages:
        signals.initiation_ratio = signals.outbound_messages / signals.total_messages

    night_hours = timestamps.dt.hour
    signals.night_share = float(((night_hours >= 23) | (night_hours < 7)).mean())

    # Reply behaviour: for each inbound message, did the user answer, and how fast?
    ordered = list(zip(sender.tolist(), timestamps.tolist()))
    latencies: list[float] = []
    answered = 0
    inbound_total = 0
    for index, (who, when) in enumerate(ordered):
        if who != str(peer_id):
            continue
        inbound_total += 1
        for later_who, later_when in ordered[index + 1 :]:
            if later_who == str(user_id):
                answered += 1
                latencies.append(max((later_when - when).total_seconds(), 0.0))
                break
            if later_who == str(peer_id):
                # Peer sent again before the user answered: not a reply.
                break

    if inbound_total:
        signals.reply_rate = answered / inbound_total
    if latencies:
        signals.median_reply_latency_sec = float(pd.Series(latencies).median())

    if "message_id" in pair.columns:
        recent_inbound = pair[inbound_mask].tail(max_evidence)
        signals.evidence_message_ids = tuple(
            str(value) for value in recent_inbound.get("message_id", pd.Series(dtype=str))
        )

    # ``reference`` is retained for callers that want recency; expose via span.
    if reference and signals.last_interaction:
        logger.debug(
            "Pair (%s, %s): %d msgs, reply_rate=%.2f",
            user_id,
            peer_id,
            signals.total_messages,
            signals.reply_rate,
        )
    return signals


def compute_all_pair_behaviour(
    history: pd.DataFrame,
    user_ids: Iterable[str] | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Compute behavioural signals for every ``(user, peer)`` pair in bulk.

    This is a Stage A helper: run it once, persist the result to
    ``data/cache/sender_affinity.parquet``, and look pairs up in O(1) afterwards.

    Parameters
    ----------
    history:
        Message history frame.
    user_ids:
        Restrict to these users. ``None`` means every user seen as a recipient
        (or, absent a recipient column, every sender).
    now:
        Reference time passed through to :func:`compute_pair_behaviour`.

    Returns
    -------
    pandas.DataFrame
        One row per pair, columns matching :meth:`BehaviouralSignals.to_dict`.
    """
    if history.empty or "sender_id" not in history.columns:
        logger.warning("compute_all_pair_behaviour: empty or malformed history frame.")
        return pd.DataFrame()

    has_recipient = "recipient_user_id" in history.columns
    if user_ids is not None:
        targets = [str(uid) for uid in user_ids]
    elif has_recipient:
        targets = sorted({str(v) for v in history["recipient_user_id"] if str(v).strip()})
    else:
        targets = sorted({str(v) for v in history["sender_id"] if str(v).strip()})

    rows: list[dict[str, Any]] = []
    for user_id in targets:
        if has_recipient:
            visible = history[
                (history["recipient_user_id"].astype(str) == user_id)
                | (history["sender_id"].astype(str) == user_id)
            ]
        else:
            visible = history
        peers = sorted(
            {
                str(value)
                for value in visible["sender_id"]
                if str(value).strip() and str(value) != user_id
            }
        )
        for peer_id in peers:
            signals = compute_pair_behaviour(visible, user_id, peer_id, now=now)
            if signals.total_messages:
                rows.append(signals.to_dict())

    logger.info("Computed behavioural signals for %d pair(s).", len(rows))
    return pd.DataFrame(rows)


def behaviour_to_signal(behaviour: BehaviouralSignals) -> LexicalSignal | None:
    """Convert strong behavioural engagement into a Close Friend hint.

    Behaviour alone cannot distinguish Family from Close Friend, so this only
    fires for the friend category and only at modest confidence. The classifier
    lets any lexical hit outrank it.

    Returns
    -------
    LexicalSignal or None
        ``None`` when engagement is not high enough to say anything.
    """
    if not behaviour.is_high_engagement:
        return None
    return LexicalSignal(
        category=RelationshipCategory.CLOSE_FRIEND,
        confidence=0.55,
        label=(
            f"behavior:high_engagement(reply_rate={behaviour.reply_rate:.2f},"
            f"msgs_per_week={behaviour.messages_per_week:.1f})"
        ),
        family="behavioral",
    )


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #


def relationship_features(
    category: RelationshipCategory,
    confidence: float,
    behaviour: BehaviouralSignals | None = None,
    is_direct: bool = True,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Assemble the ``rel_*`` feature block consumed by the scoring engine.

    Parameters
    ----------
    category:
        Fused relationship category for this sender.
    confidence:
        Confidence attached to that category, in ``[0, 1]``.
    behaviour:
        Behavioural statistics for the pair, when available.
    is_direct:
        Whether the message arrived in a 1:1 conversation.
    config:
        Application configuration; defaults to the singleton.

    Returns
    -------
    dict
        Feature dictionary with the ``rel_`` prefix.
    """
    cfg = config or get_config()
    features: dict[str, Any] = {
        "rel_category": category.value,
        "rel_confidence": float(max(0.0, min(1.0, confidence))),
        "rel_is_direct": bool(is_direct),
        "rel_is_personal": category
        in (
            RelationshipCategory.FAMILY,
            RelationshipCategory.CLOSE_FRIEND,
        ),
        "rel_is_institutional": category
        in (
            RelationshipCategory.OFFICE,
            RelationshipCategory.COLLEGE,
            RelationshipCategory.SOCIETY,
        ),
        "rel_is_business": category is RelationshipCategory.BUSINESS,
        "rel_is_unknown": category is RelationshipCategory.UNKNOWN,
        "rel_prior_mean": cfg.scoring.relationship_prior_mean,
        "rel_reply_rate": 0.0,
        "rel_median_latency_sec": None,
        "rel_messages_per_week": 0.0,
        "rel_total_messages": 0,
        "rel_high_engagement": False,
        "rel_dormant": True,
    }

    if behaviour is not None:
        features.update(
            {
                "rel_reply_rate": behaviour.reply_rate,
                "rel_median_latency_sec": behaviour.median_reply_latency_sec,
                "rel_messages_per_week": behaviour.messages_per_week,
                "rel_total_messages": behaviour.total_messages,
                "rel_high_engagement": behaviour.is_high_engagement,
                "rel_dormant": behaviour.is_dormant,
            }
        )

    return features


__all__ = [
    "ADDRESS_TERM_HINTS",
    "BUSINESS_TERMS",
    "BehaviouralSignals",
    "COLLEGE_TERMS",
    "FRIEND_TERMS",
    "KINSHIP_TERMS",
    "LEXICAL_CONFIDENCE",
    "LexicalSignal",
    "OFFICE_TERMS",
    "SOCIETY_TERMS",
    "behaviour_to_signal",
    "compute_all_pair_behaviour",
    "compute_pair_behaviour",
    "extract_address_term_signals",
    "extract_group_name_signals",
    "extract_name_signals",
    "load_lexicons",
    "normalise_name",
    "relationship_features",
    "tokenise",
]