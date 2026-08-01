"""
Lexical retrieval utilities: pandas-only evidence lookup, no embeddings.

Every function here returns :class:`~src.schema.RetrievalCandidate` objects so
that :mod:`src.retrieval.context` can merge results from different lookups
into one ranked pool without caring which one produced them.

Similarity is computed with plain term-frequency cosine over
:class:`collections.Counter`, not TF-IDF or any vector-database technology, per
the no-embeddings constraint.

Dependencies
------------
``pandas``, ``src.config``, ``src.schema``, ``src.features.content`` (for the
shared forward-marker pattern), ``src.features.relationship`` (for the shared
tokeniser).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable, Sequence

import pandas as pd

from src.features.content import FORWARD_PATTERN
from src.features.relationship import tokenise
from src.schema import RetrievalCandidate

logger = logging.getLogger(__name__)

#: Column candidates for message body text, in preference order.
_TEXT_COLUMNS = ("message_text", "content", "text", "body")

#: Weights for the candidate pre-score, matching the frozen retrieval design:
#: score = 0.35*participant_match + 0.25*thread_link + 0.20*recency + 0.20*lexical_sim
_WEIGHT_PARTICIPANT = 0.35
_WEIGHT_THREAD = 0.25
_WEIGHT_RECENCY = 0.20
_WEIGHT_LEXICAL = 0.20


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def normalise_for_matching(text: str | None) -> str:
    """Lower-case and collapse whitespace for exact / duplicate comparisons.

    Parameters
    ----------
    text:
        Raw message body.

    Returns
    -------
    str
        Normalised text, or ``""`` for empty input.
    """
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def term_frequency(text: str | None) -> Counter[str]:
    """Build a term-frequency vector for a message body.

    Reuses the tokeniser already validated for relationship-name matching, so
    stopwords and punctuation are handled identically everywhere in the project.

    Parameters
    ----------
    text:
        Raw message body.

    Returns
    -------
    collections.Counter
        Token -> count. Empty for empty input.
    """
    return Counter(tokenise(text))


def cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    """Cosine similarity between two term-frequency vectors, in ``[0, 1]``.

    Pure pandas/stdlib implementation -- no sklearn, no embeddings.

    Parameters
    ----------
    a, b:
        Term-frequency counters, typically from :func:`term_frequency`.

    Returns
    -------
    float
        ``0.0`` when either vector is empty.
    """
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in shared)
    norm_a = math.sqrt(sum(count * count for count in a.values()))
    norm_b = math.sqrt(sum(count * count for count in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _text_column(frame: pd.DataFrame) -> str | None:
    """Return the first usable body-text column in ``frame``."""
    for candidate in _TEXT_COLUMNS:
        if candidate in frame.columns:
            return candidate
    return None


def _row_text(row: pd.Series, text_column: str | None) -> str:
    """Extract the body text of one row, tolerating a missing column."""
    if text_column is None:
        return ""
    value = row.get(text_column)
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _row_timestamp(row: pd.Series) -> datetime | None:
    """Extract a usable ``datetime`` from a row's ``timestamp`` field."""
    value = row.get("timestamp")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _age_hours(timestamp: datetime, reference: datetime) -> float:
    """Return the age of ``timestamp`` relative to ``reference``, in hours."""
    return max((reference - timestamp).total_seconds() / 3600.0, 0.0)


def _recency_weight(age_hours: float, horizon_hours: float) -> float:
    """Linear recency decay, matching :func:`src.features.temporal.recency_decay`.

    Duplicated locally (rather than imported) to keep this module free of a
    dependency on the temporal feature module, which is not part of the
    retrieval layer's declared interface.
    """
    if horizon_hours <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (age_hours / horizon_hours)))


def score_candidate(
    participant_match: float,
    thread_link: float,
    recency: float,
    lexical_sim: float,
) -> float:
    """Combine the four retrieval sub-scores into one pre-score.

    Weights follow the frozen retrieval design: participant match matters most,
    then thread linkage, then recency and lexical similarity equally.

    Parameters
    ----------
    participant_match:
        ``1.0`` when the candidate involves the same people as the query.
    thread_link:
        ``1.0`` when the candidate shares a conversation thread with the query.
    recency:
        Freshness weight in ``[0, 1]`` from :func:`_recency_weight`.
    lexical_sim:
        Cosine similarity to the query text, in ``[0, 1]``.

    Returns
    -------
    float
        Pre-score in ``[0, 1]``.
    """
    return (
        _WEIGHT_PARTICIPANT * max(0.0, min(1.0, participant_match))
        + _WEIGHT_THREAD * max(0.0, min(1.0, thread_link))
        + _WEIGHT_RECENCY * max(0.0, min(1.0, recency))
        + _WEIGHT_LEXICAL * max(0.0, min(1.0, lexical_sim))
    )


def _make_candidate(
    row: pd.Series,
    text_column: str | None,
    tier: str,
    pre_score: float,
    age_hours: float,
    id_column: str,
    max_snippet_chars: int,
) -> RetrievalCandidate | None:
    """Build one :class:`RetrievalCandidate` from a DataFrame row, or ``None``."""
    message_id = row.get(id_column)
    sender_id = row.get("sender_id")
    timestamp = _row_timestamp(row)
    if message_id is None or sender_id is None or timestamp is None:
        return None

    text = _row_text(row, text_column)
    return RetrievalCandidate(
        message_id=str(message_id),
        sender_id=str(sender_id),
        timestamp=timestamp,
        snippet=text[:max_snippet_chars],
        tier=tier,
        pre_score=pre_score,
        age_hours=age_hours,
    )


# --------------------------------------------------------------------------- #
# Keyword matching
# --------------------------------------------------------------------------- #


def keyword_match(
    frame: pd.DataFrame,
    keywords: Iterable[str],
    limit: int = 50,
) -> pd.DataFrame:
    """Return rows whose body text contains any of ``keywords``.

    Matching is on normalised whole tokens, so "otp" does not match "adopt".

    Parameters
    ----------
    frame:
        Message frame with a recognisable text column.
    keywords:
        Terms to search for. Case-insensitive.
    limit:
        Maximum number of rows to return, most recent first when a timestamp
        column exists.

    Returns
    -------
    pandas.DataFrame
        Subset of ``frame``; empty when nothing matches or the frame is unusable.
    """
    if frame.empty:
        return frame.iloc[0:0]

    text_column = _text_column(frame)
    if text_column is None:
        logger.debug("keyword_match: no text column found in frame.")
        return frame.iloc[0:0]

    keyword_set = {str(k).strip().lower() for k in keywords if str(k).strip()}
    if not keyword_set:
        return frame.iloc[0:0]

    def _hit(text: object) -> bool:
        tokens = set(tokenise(str(text)))
        return bool(tokens & keyword_set)

    matched = frame[frame[text_column].fillna("").map(_hit)]
    if "timestamp" in matched.columns:
        matched = matched.sort_values("timestamp", ascending=False)
    return matched.head(limit)


# --------------------------------------------------------------------------- #
# Sender / group / business history lookups
# --------------------------------------------------------------------------- #


def sender_history_lookup(
    messages: pd.DataFrame,
    sender_id: str,
    reference_time: datetime,
    window: int,
    exclude_message_id: str | None = None,
    horizon_hours: float = 72.0,
    max_snippet_chars: int = 140,
) -> list[RetrievalCandidate]:
    """Return the most recent prior messages from one sender to one user.

    This is retrieval tier 2 ("relational") from the frozen architecture: the
    last ``window`` messages from this sender, regardless of which conversation
    they landed in.

    Parameters
    ----------
    messages:
        Combined message frame (``messages.csv`` + ``message_history.csv``),
        must carry ``sender_id`` and ``timestamp``.
    sender_id:
        Sender to look up.
    reference_time:
        Timestamp of the message being routed; only strictly earlier rows
        qualify.
    window:
        Maximum number of prior messages to return.
    exclude_message_id:
        The current message's own id, excluded from its own evidence pool.
    horizon_hours:
        Recency horizon used for the pre-score.
    max_snippet_chars:
        Snippet truncation length.

    Returns
    -------
    list of RetrievalCandidate
        Ordered most-recent-first; empty when the frame lacks required columns.
    """
    if messages.empty or "sender_id" not in messages.columns or "timestamp" not in messages.columns:
        return []

    subset = messages[messages["sender_id"].astype(str) == str(sender_id)].copy()
    if "message_id" in subset.columns and exclude_message_id:
        subset = subset[subset["message_id"].astype(str) != str(exclude_message_id)]

    subset["_ts"] = pd.to_datetime(subset["timestamp"], errors="coerce")
    subset = subset.dropna(subset=["_ts"])
    subset = subset[subset["_ts"] < reference_time]
    if subset.empty:
        return []

    subset = subset.sort_values("_ts", ascending=False).head(window)
    text_column = _text_column(subset)
    id_column = "message_id" if "message_id" in subset.columns else subset.columns[0]

    candidates: list[RetrievalCandidate] = []
    for _, row in subset.iterrows():
        timestamp = row["_ts"].to_pydatetime()
        age = _age_hours(timestamp, reference_time)
        pre_score = score_candidate(
            participant_match=1.0,
            thread_link=0.0,
            recency=_recency_weight(age, horizon_hours),
            lexical_sim=0.0,
        )
        candidate = _make_candidate(
            row, text_column, "relational", pre_score, age, id_column, max_snippet_chars
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def group_history_lookup(
    messages: pd.DataFrame,
    conversation_id: str,
    reference_time: datetime,
    window: int,
    exclude_message_id: str | None = None,
    peer_id: str | None = None,
    horizon_hours: float = 72.0,
    max_snippet_chars: int = 140,
) -> list[RetrievalCandidate]:
    """Return the most recent prior messages in the same thread or group.

    This is retrieval tier 1 ("structural") from the frozen architecture: the
    last ``window`` messages in the same ``conversation_id`` (which for a group
    message is typically the group id).

    Parameters
    ----------
    messages:
        Combined message frame, must carry ``conversation_id`` and ``timestamp``.
        Falls back to ``group_id`` when ``conversation_id`` is absent.
    conversation_id:
        Thread or group identifier to look up.
    reference_time:
        Timestamp of the message being routed.
    window:
        Maximum number of prior messages to return.
    exclude_message_id:
        The current message's own id, excluded from its own evidence pool.
    peer_id:
        When given, candidates sent by this id are scored with a higher
        participant-match, since they involve the same two people.
    horizon_hours:
        Recency horizon used for the pre-score.
    max_snippet_chars:
        Snippet truncation length.

    Returns
    -------
    list of RetrievalCandidate
        Ordered most-recent-first; empty when the frame lacks required columns.
    """
    key_column = "conversation_id" if "conversation_id" in messages.columns else "group_id"
    if messages.empty or key_column not in messages.columns or "timestamp" not in messages.columns:
        return []

    subset = messages[messages[key_column].astype(str) == str(conversation_id)].copy()
    if "message_id" in subset.columns and exclude_message_id:
        subset = subset[subset["message_id"].astype(str) != str(exclude_message_id)]

    subset["_ts"] = pd.to_datetime(subset["timestamp"], errors="coerce")
    subset = subset.dropna(subset=["_ts"])
    subset = subset[subset["_ts"] < reference_time]
    if subset.empty:
        return []

    subset = subset.sort_values("_ts", ascending=False).head(window)
    text_column = _text_column(subset)
    id_column = "message_id" if "message_id" in subset.columns else subset.columns[0]

    candidates: list[RetrievalCandidate] = []
    for _, row in subset.iterrows():
        timestamp = row["_ts"].to_pydatetime()
        age = _age_hours(timestamp, reference_time)
        participant_match = (
            1.0 if peer_id and str(row.get("sender_id")) == str(peer_id) else 0.5
        )
        pre_score = score_candidate(
            participant_match=participant_match,
            thread_link=1.0,
            recency=_recency_weight(age, horizon_hours),
            lexical_sim=0.0,
        )
        candidate = _make_candidate(
            row, text_column, "structural", pre_score, age, id_column, max_snippet_chars
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def business_history_lookup(
    user_business_history: pd.DataFrame,
    messages: pd.DataFrame,
    user_id: str,
    business_id: str,
    reference_time: datetime,
    window: int = 2,
    horizon_hours: float = 72.0,
    max_snippet_chars: int = 140,
) -> list[RetrievalCandidate]:
    """Return recent prior messages from a business account to this user.

    Combines two sources: ``user_business_history.csv`` rows that carry a
    ``message_id`` (transaction-linked messages) and, as a fallback, the
    business's own message traffic to this user from the message frame.

    Parameters
    ----------
    user_business_history:
        ``user_business_history.csv`` frame.
    messages:
        Combined message frame, used as a fallback source of business messages.
    user_id:
        The receiving user.
    business_id:
        The business account to look up.
    reference_time:
        Timestamp of the message being routed.
    window:
        Maximum number of prior interactions to return.
    horizon_hours:
        Recency horizon used for the pre-score.
    max_snippet_chars:
        Snippet truncation length.

    Returns
    -------
    list of RetrievalCandidate
        Ordered most-recent-first; empty when no linkage is available.
    """
    candidates: list[RetrievalCandidate] = []

    if not user_business_history.empty and "message_id" in user_business_history.columns:
        subset = user_business_history.copy()
        if "user_id" in subset.columns:
            subset = subset[subset["user_id"].astype(str) == str(user_id)]
        if "business_id" in subset.columns:
            subset = subset[subset["business_id"].astype(str) == str(business_id)]

        if not subset.empty and "message_id" in subset.columns:
            time_col = "timestamp" if "timestamp" in subset.columns else None
            if time_col:
                subset["_ts"] = pd.to_datetime(subset[time_col], errors="coerce")
                subset = subset.dropna(subset=["_ts"])
                subset = subset[subset["_ts"] < reference_time]
                subset = subset.sort_values("_ts", ascending=False)

            for _, row in subset.head(window).iterrows():
                message_id = row.get("message_id")
                if message_id is None:
                    continue
                timestamp = row.get("_ts")
                ts = (
                    timestamp.to_pydatetime()
                    if isinstance(timestamp, pd.Timestamp) and pd.notna(timestamp)
                    else reference_time
                )
                age = _age_hours(ts, reference_time)
                pre_score = score_candidate(
                    participant_match=1.0,
                    thread_link=0.0,
                    recency=_recency_weight(age, horizon_hours),
                    lexical_sim=0.0,
                )
                candidates.append(
                    RetrievalCandidate(
                        message_id=str(message_id),
                        sender_id=str(business_id),
                        timestamp=ts,
                        snippet="(prior business transaction)",
                        tier="business",
                        pre_score=pre_score,
                        age_hours=age,
                    )
                )

    if candidates:
        return candidates[:window]

    # Fallback: business's own message traffic to this user, same shape as
    # sender_history_lookup but scoped to conversations involving this user.
    if messages.empty or "business_id" not in messages.columns:
        return candidates

    subset = messages[messages["business_id"].astype(str) == str(business_id)].copy()
    if "recipient_user_id" in subset.columns:
        subset = subset[subset["recipient_user_id"].astype(str) == str(user_id)]
    if "timestamp" not in subset.columns:
        return candidates

    subset["_ts"] = pd.to_datetime(subset["timestamp"], errors="coerce")
    subset = subset.dropna(subset=["_ts"])
    subset = subset[subset["_ts"] < reference_time].sort_values("_ts", ascending=False).head(window)

    text_column = _text_column(subset)
    id_column = "message_id" if "message_id" in subset.columns else None
    if id_column is None:
        return candidates

    for _, row in subset.iterrows():
        timestamp = row["_ts"].to_pydatetime()
        age = _age_hours(timestamp, reference_time)
        pre_score = score_candidate(
            participant_match=1.0,
            thread_link=0.0,
            recency=_recency_weight(age, horizon_hours),
            lexical_sim=0.0,
        )
        candidate = _make_candidate(
            row, text_column, "business", pre_score, age, id_column, max_snippet_chars
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


# --------------------------------------------------------------------------- #
# Duplicate / similarity / forward detection
# --------------------------------------------------------------------------- #


def duplicate_detection(
    messages: pd.DataFrame,
    sender_id: str,
    body_text: str,
    reference_time: datetime,
    window_hours: float = 24.0,
    exclude_message_id: str | None = None,
) -> tuple[bool, int, tuple[str, ...]]:
    """Detect whether this exact message was already seen from this sender.

    Used by the rules layer for the "duplicate / forward already seen in the
    last 24h" mute rule, and by burst suppression.

    Parameters
    ----------
    messages:
        Combined message frame.
    sender_id:
        Sender of the current message.
    body_text:
        Current message's unified content.
    reference_time:
        Timestamp of the current message.
    window_hours:
        Lookback window for prior duplicates.
    exclude_message_id:
        The current message's own id.

    Returns
    -------
    tuple
        ``(is_duplicate, prior_count, evidence_message_ids)``.
    """
    normalised_target = normalise_for_matching(body_text)
    if not normalised_target or messages.empty:
        return False, 0, ()

    text_column = _text_column(messages)
    if text_column is None or "sender_id" not in messages.columns or "timestamp" not in messages.columns:
        return False, 0, ()

    subset = messages[messages["sender_id"].astype(str) == str(sender_id)].copy()
    if "message_id" in subset.columns and exclude_message_id:
        subset = subset[subset["message_id"].astype(str) != str(exclude_message_id)]

    subset["_ts"] = pd.to_datetime(subset["timestamp"], errors="coerce")
    subset = subset.dropna(subset=["_ts"])
    cutoff = reference_time - timedelta(hours=window_hours)
    subset = subset[(subset["_ts"] >= cutoff) & (subset["_ts"] < reference_time)]
    if subset.empty:
        return False, 0, ()

    subset["_normalised"] = subset[text_column].fillna("").map(normalise_for_matching)
    matches = subset[subset["_normalised"] == normalised_target]
    if matches.empty:
        return False, 0, ()

    id_column = "message_id" if "message_id" in matches.columns else None
    evidence = (
        tuple(str(value) for value in matches[id_column].tolist())
        if id_column
        else ()
    )
    return True, len(matches), evidence[:4]


def similar_message_lookup(
    candidate_frame: pd.DataFrame,
    query_text: str,
    reference_time: datetime,
    top_k: int = 2,
    min_similarity: float = 0.15,
    horizon_hours: float = 72.0,
    max_snippet_chars: int = 140,
) -> list[RetrievalCandidate]:
    """Rank prior messages by lexical similarity to the query text.

    This is retrieval tier 3 ("lexical") from the frozen architecture,
    implemented with term-frequency cosine similarity -- no TF-IDF, no
    embeddings, no vector index.

    Parameters
    ----------
    candidate_frame:
        Pool to search, typically the sender's or group's recent history.
    query_text:
        Unified content of the message being routed.
    reference_time:
        Timestamp of the message being routed, used for the recency term.
    top_k:
        Number of top-similarity candidates to return.
    min_similarity:
        Candidates scoring below this are dropped even if they would fill
        ``top_k``, since a weak match adds noise to the LLM prompt.
    horizon_hours:
        Recency horizon used for the pre-score.
    max_snippet_chars:
        Snippet truncation length.

    Returns
    -------
    list of RetrievalCandidate
        Ordered by similarity, descending; empty when nothing clears the bar.
    """
    if candidate_frame.empty or not query_text.strip():
        return []

    text_column = _text_column(candidate_frame)
    if text_column is None or "timestamp" not in candidate_frame.columns:
        return []

    query_vector = term_frequency(query_text)
    if not query_vector:
        return []

    id_column = "message_id" if "message_id" in candidate_frame.columns else None
    if id_column is None:
        return []

    scored: list[tuple[float, pd.Series, datetime]] = []
    for _, row in candidate_frame.iterrows():
        timestamp = _row_timestamp(row)
        if timestamp is None or timestamp >= reference_time:
            continue
        body = _row_text(row, text_column)
        similarity = cosine_similarity(query_vector, term_frequency(body))
        if similarity >= min_similarity:
            scored.append((similarity, row, timestamp))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)

    candidates: list[RetrievalCandidate] = []
    for similarity, row, timestamp in scored[:top_k]:
        age = _age_hours(timestamp, reference_time)
        pre_score = score_candidate(
            participant_match=0.5,
            thread_link=0.0,
            recency=_recency_weight(age, horizon_hours),
            lexical_sim=similarity,
        )
        candidate = _make_candidate(
            row, text_column, "lexical", pre_score, age, id_column, max_snippet_chars
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def forwarded_pattern_lookup(
    messages: pd.DataFrame,
    sender_id: str,
    reference_time: datetime,
    window_hours: float = 24.0,
    limit: int = 5,
) -> list[RetrievalCandidate]:
    """Find recent messages from this sender that look like chain forwards.

    Used by content-adjacent rules that want to see whether a sender has been
    forwarding broadcast material recently, independent of the current
    message's own content.

    Parameters
    ----------
    messages:
        Combined message frame.
    sender_id:
        Sender to inspect.
    reference_time:
        Timestamp of the message being routed.
    window_hours:
        Lookback window.
    limit:
        Maximum number of matches to return.

    Returns
    -------
    list of RetrievalCandidate
        Tier ``"lexical"``, most-recent-first.
    """
    if messages.empty or "sender_id" not in messages.columns or "timestamp" not in messages.columns:
        return []

    subset = messages[messages["sender_id"].astype(str) == str(sender_id)].copy()
    subset["_ts"] = pd.to_datetime(subset["timestamp"], errors="coerce")
    subset = subset.dropna(subset=["_ts"])
    cutoff = reference_time - timedelta(hours=window_hours)
    subset = subset[(subset["_ts"] >= cutoff) & (subset["_ts"] < reference_time)]
    if subset.empty:
        return []

    text_column = _text_column(subset)
    id_column = "message_id" if "message_id" in subset.columns else None
    if text_column is None or id_column is None:
        return []

    is_forwarded_flagged = (
        subset["is_forwarded"].astype(str).str.lower().isin({"1", "true", "yes"})
        if "is_forwarded" in subset.columns
        else pd.Series(False, index=subset.index)
    )
    matches_pattern = subset[text_column].fillna("").map(
        lambda text: bool(FORWARD_PATTERN.search(str(text)))
    )
    forwarded = subset[is_forwarded_flagged | matches_pattern]
    if forwarded.empty:
        return []

    forwarded = forwarded.sort_values("_ts", ascending=False).head(limit)
    candidates: list[RetrievalCandidate] = []
    for _, row in forwarded.iterrows():
        timestamp = row["_ts"].to_pydatetime()
        age = _age_hours(timestamp, reference_time)
        pre_score = score_candidate(
            participant_match=1.0,
            thread_link=0.0,
            recency=_recency_weight(age, 72.0),
            lexical_sim=0.0,
        )
        candidate = _make_candidate(
            row, text_column, "lexical", pre_score, age, id_column, 140
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def deduplicate_candidates(
    candidates: Sequence[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    """Remove duplicate candidates, keeping the highest-scored occurrence.

    Different tiers can independently surface the same message id; the merge
    step in :mod:`src.retrieval.context` relies on this to avoid double-counting.

    Parameters
    ----------
    candidates:
        Candidates from one or more lookups.

    Returns
    -------
    list of RetrievalCandidate
        One entry per distinct ``message_id``, highest pre-score kept.
    """
    best: dict[str, RetrievalCandidate] = {}
    for candidate in candidates:
        existing = best.get(candidate.message_id)
        if existing is None or candidate.pre_score > existing.pre_score:
            best[candidate.message_id] = candidate
    return sorted(best.values(), key=lambda item: item.pre_score, reverse=True)


__all__ = [
    "business_history_lookup",
    "cosine_similarity",
    "deduplicate_candidates",
    "duplicate_detection",
    "forwarded_pattern_lookup",
    "group_history_lookup",
    "keyword_match",
    "normalise_for_matching",
    "score_candidate",
    "sender_history_lookup",
    "similar_message_lookup",
    "term_frequency",
]