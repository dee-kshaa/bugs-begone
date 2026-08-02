"""
Context assembly: turn one incoming message into everything downstream needs.

:class:`ContextRetriever` is the single entry point. For each message it
builds:

* a :class:`MessageContext` -- user, group and business profiles, membership
  and admin info, recent notification-relevant events, duplicate/repeat
  status, and reports -- for the scoring and rules layers, and
* a :class:`~src.schema.RetrievalContext` -- the ranked evidence pool, built
  from the lexical lookups in :mod:`src.retrieval.lexical` -- for the LLM
  prompt and for evidence_message_ids.

``MessageContext`` is local to this module rather than added to
``src/schema.py``: the frozen ``RetrievalContext`` dataclass is specifically
the evidence pool, and extending it would change a frozen interface.

Dependencies
------------
``pandas``, ``src.config``, ``src.schema``, ``src.io.loaders``,
``src.features.group``, ``src.features.business``, ``src.features.temporal``,
``src.retrieval.lexical``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.config import AppConfig, get_config
from src.features.business import build_business_profiles
from src.features.group import build_group_profiles
from src.features.temporal import conversation_timestamp_index
from src.io.loaders import DataRepository
from src.retrieval import lexical
from src.schema import (
    BusinessProfile,
    GroupProfile,
    Message,
    RetrievalCandidate,
    RetrievalContext,
    UserProfile,
)

logger = logging.getLogger(__name__)

#: Column candidates for a user's display name.
_USER_NAME_COLUMNS = ("display_name", "name", "username")

#: Column candidates for a user's timezone.
_USER_TIMEZONE_COLUMNS = ("timezone", "tz")

#: Column candidates for DND window hours.
_DND_START_COLUMNS = ("dnd_start_hour", "quiet_hours_start", "dnd_start")
_DND_END_COLUMNS = ("dnd_end_hour", "quiet_hours_end", "dnd_end")

#: Column candidates for pipe/comma-separated id lists on the users frame.
_MUTED_CONTACTS_COLUMNS = ("muted_contacts", "muted_users")
_MUTED_GROUPS_COLUMNS = ("muted_groups",)
_PINNED_CONTACTS_COLUMNS = ("pinned_contacts", "pinned_users", "starred_contacts")
_BLOCKED_CONTACTS_COLUMNS = ("blocked_contacts", "blocked_users")

#: Column candidates for an event-type column in message_events.csv.
_EVENT_TYPE_COLUMNS = ("event_type", "event", "action", "status")

#: Event values, normalised, that count toward each summary bucket.
_OPEN_EVENTS = {"read", "open", "opened", "seen", "viewed"}
_REPLY_EVENTS = {"reply", "replied", "responded", "react", "reacted"}
_DISMISS_EVENTS = {"dismiss", "dismissed", "delete", "deleted", "archive", "archived", "swipe_away"}
_REPORT_EVENTS = {"report", "reported", "flag", "flagged", "spam_report", "block", "blocked"}

#: Column candidates for group-membership role/admin info.
_ADMIN_FLAG_COLUMNS = ("is_admin", "role", "member_role")
_ADMIN_TRUE_VALUES = {"admin", "owner", "true", "1", "yes"}


def _split_id_list(raw: Any) -> tuple[str, ...]:
    """Parse a pipe- or comma-separated id list cell into a tuple of strings."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    separator = "|" if "|" in text else ","
    return tuple(part.strip() for part in text.split(separator) if part.strip())


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first column from ``candidates`` present in ``frame``."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _combined_message_frame(repo: DataRepository) -> pd.DataFrame:
    """Union ``messages.csv`` and ``message_history.csv``, deduped by id.

    The retrieval layer needs one frame to search across; splitting lookups
    across two frames would double every query and risk inconsistent ordering.

    Parameters
    ----------
    repo:
        Loaded dataset repository.

    Returns
    -------
    pandas.DataFrame
        Empty when both source frames are empty.
    """
    frames = [frame for frame in (repo.messages, repo.message_history) if not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "message_id" in combined.columns:
        combined = combined.drop_duplicates(subset="message_id", keep="first")
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
        combined = combined.sort_values("timestamp")
    return combined.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Local context objects (evidence pool stays in schema.RetrievalContext)
# --------------------------------------------------------------------------- #


@dataclass
class EventSummary:
    """Counts of a user's recent notification-relevant behaviour.

    Covers "recent opens/replies/dismissals" from the batch requirements.
    Scoped to a single sender or business by the caller.
    """

    opens: int = 0
    replies: int = 0
    dismissals: int = 0
    reports: int = 0
    total_events: int = 0
    last_event_at: datetime | None = None

    @property
    def open_rate(self) -> float:
        """Share of tracked events that were opens, in ``[0, 1]``."""
        return self.opens / self.total_events if self.total_events else 0.0

    @property
    def reply_rate(self) -> float:
        """Share of tracked events that were replies, in ``[0, 1]``."""
        return self.replies / self.total_events if self.total_events else 0.0

    @property
    def dismissal_rate(self) -> float:
        """Share of tracked events that were dismissals, in ``[0, 1]``."""
        return self.dismissals / self.total_events if self.total_events else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "opens": self.opens,
            "replies": self.replies,
            "dismissals": self.dismissals,
            "reports": self.reports,
            "total_events": self.total_events,
            "open_rate": round(self.open_rate, 4),
            "reply_rate": round(self.reply_rate, 4),
            "dismissal_rate": round(self.dismissal_rate, 4),
            "last_event_at": self.last_event_at.isoformat() if self.last_event_at else None,
        }


@dataclass
class MessageContext:
    """Everything the scoring and rules layers need for one message.

    Not part of ``src/schema.py`` by design: this bundles profile lookups for
    internal pipeline use, while :attr:`retrieval` is the frozen
    :class:`~src.schema.RetrievalContext` used for LLM evidence.
    """

    message_id: str
    user: UserProfile | None = None
    group: GroupProfile | None = None
    business: BusinessProfile | None = None

    group_admin_ids: tuple[str, ...] = ()
    sender_is_group_admin: bool = False

    sender_event_summary: EventSummary = field(default_factory=EventSummary)
    business_event_summary: EventSummary = field(default_factory=EventSummary)

    is_duplicate: bool = False
    duplicate_count: int = 0
    duplicate_evidence_ids: tuple[str, ...] = ()

    is_reported_recently: bool = False
    report_count: int = 0

    conversation_history_ids: tuple[str, ...] = ()

    retrieval: RetrievalContext = field(default_factory=lambda: RetrievalContext(message_id=""))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary for the explanation trace."""
        return {
            "message_id": self.message_id,
            "user": self.user.to_dict() if self.user else None,
            "group": self.group.to_dict() if self.group else None,
            "business": self.business.to_dict() if self.business else None,
            "group_admin_ids": list(self.group_admin_ids),
            "sender_is_group_admin": self.sender_is_group_admin,
            "sender_event_summary": self.sender_event_summary.to_dict(),
            "business_event_summary": self.business_event_summary.to_dict(),
            "is_duplicate": self.is_duplicate,
            "duplicate_count": self.duplicate_count,
            "duplicate_evidence_ids": list(self.duplicate_evidence_ids),
            "is_reported_recently": self.is_reported_recently,
            "report_count": self.report_count,
            "conversation_history_ids": list(self.conversation_history_ids),
            "retrieval": self.retrieval.to_dict(),
        }


# --------------------------------------------------------------------------- #
# User profile building
# --------------------------------------------------------------------------- #


def _event_summary_for(
    events: pd.DataFrame,
    message_ids: set[str],
    user_id: str | None,
) -> EventSummary:
    """Summarise a user's events over a specific set of message ids."""
    summary = EventSummary()
    if events.empty or not message_ids or "message_id" not in events.columns:
        return summary

    subset = events[events["message_id"].astype(str).isin(message_ids)]
    if user_id and "user_id" in subset.columns:
        subset = subset[subset["user_id"].astype(str) == str(user_id)]
    if subset.empty:
        return summary

    event_column = _first_present(subset, _EVENT_TYPE_COLUMNS)
    if event_column is None:
        return summary

    values = subset[event_column].astype(str).str.lower()
    summary.opens = int(values.isin(_OPEN_EVENTS).sum())
    summary.replies = int(values.isin(_REPLY_EVENTS).sum())
    summary.dismissals = int(values.isin(_DISMISS_EVENTS).sum())
    summary.reports = int(values.isin(_REPORT_EVENTS).sum())
    summary.total_events = int(len(subset))

    time_column = _first_present(subset, ("timestamp", "event_time"))
    if time_column is not None:
        stamps = pd.to_datetime(subset[time_column], errors="coerce").dropna()
        if not stamps.empty:
            summary.last_event_at = stamps.max().to_pydatetime()

    return summary


def _build_user_profile(
    users: pd.DataFrame,
    events: pd.DataFrame,
    user_id: str,
    config: AppConfig,
) -> UserProfile:
    """Build a :class:`UserProfile` for one user from the users and events frames.

    Falls back to a neutral default profile when ``user_id`` has no row in
    ``users.csv``, so an unseen recipient never breaks the pipeline.

    Parameters
    ----------
    users:
        ``users.csv`` frame.
    events:
        ``message_events.csv`` frame, used for overall engagement priors.
    user_id:
        User to build a profile for.
    config:
        Application configuration, for DND defaults.

    Returns
    -------
    UserProfile
    """
    defaults = UserProfile(
        user_id=user_id,
        dnd_start_hour=config.routing.dnd_start_hour,
        dnd_end_hour=config.routing.dnd_end_hour,
    )

    if users.empty or "user_id" not in users.columns:
        return defaults

    rows = users[users["user_id"].astype(str) == str(user_id)]
    if rows.empty:
        return defaults

    row = rows.iloc[0]
    name_column = _first_present(users, _USER_NAME_COLUMNS)
    timezone_column = _first_present(users, _USER_TIMEZONE_COLUMNS)
    dnd_start_column = _first_present(users, _DND_START_COLUMNS)
    dnd_end_column = _first_present(users, _DND_END_COLUMNS)

    def _hour(column: str | None, fallback: int) -> int:
        if column is None or pd.isna(row.get(column)):
            return fallback
        try:
            return int(row[column]) % 24
        except (TypeError, ValueError):
            return fallback

    muted_contacts = _split_id_list(row.get(_first_present(users, _MUTED_CONTACTS_COLUMNS) or ""))
    muted_groups = _split_id_list(row.get(_first_present(users, _MUTED_GROUPS_COLUMNS) or ""))
    pinned_contacts = _split_id_list(row.get(_first_present(users, _PINNED_CONTACTS_COLUMNS) or ""))
    blocked_contacts = _split_id_list(row.get(_first_present(users, _BLOCKED_CONTACTS_COLUMNS) or ""))

    overall_open_rate = defaults.overall_open_rate
    overall_reply_rate = defaults.overall_reply_rate
    median_latency: float | None = None
    messages_per_day = 0.0

    if not events.empty and "user_id" in events.columns:
        user_events = events[events["user_id"].astype(str) == str(user_id)]
        event_column = _first_present(user_events, _EVENT_TYPE_COLUMNS)
        if event_column is not None and not user_events.empty:
            values = user_events[event_column].astype(str).str.lower()
            total = len(user_events)
            overall_open_rate = min(int(values.isin(_OPEN_EVENTS).sum()) / total, 1.0)
            overall_reply_rate = min(int(values.isin(_REPLY_EVENTS).sum()) / total, 1.0)

        time_column = _first_present(user_events, ("timestamp", "event_time"))
        if time_column is not None and not user_events.empty:
            stamps = pd.to_datetime(user_events[time_column], errors="coerce").dropna()
            if len(stamps) >= 2:
                span_days = max(
                    (stamps.max() - stamps.min()).total_seconds() / 86400.0, 1.0
                )
                messages_per_day = len(stamps) / span_days

    return UserProfile(
        user_id=str(user_id),
        display_name=str(row[name_column]) if name_column and pd.notna(row.get(name_column)) else "",
        timezone=str(row[timezone_column]) if timezone_column and pd.notna(row.get(timezone_column)) else "Asia/Kolkata",
        dnd_start_hour=_hour(dnd_start_column, config.routing.dnd_start_hour),
        dnd_end_hour=_hour(dnd_end_column, config.routing.dnd_end_hour),
        muted_contacts=frozenset(muted_contacts),
        muted_groups=frozenset(muted_groups),
        pinned_contacts=frozenset(pinned_contacts),
        blocked_contacts=frozenset(blocked_contacts),
        overall_open_rate=overall_open_rate,
        overall_reply_rate=overall_reply_rate,
        median_reply_latency_sec=median_latency,
        messages_per_day=messages_per_day,
    )


def _group_admin_ids(group_members: pd.DataFrame, group_id: str) -> tuple[str, ...]:
    """Return the ids of a group's admins, when membership carries a role column."""
    if group_members.empty or "group_id" not in group_members.columns:
        return ()
    rows = group_members[group_members["group_id"].astype(str) == str(group_id)]
    if rows.empty:
        return ()

    role_column = _first_present(rows, _ADMIN_FLAG_COLUMNS)
    if role_column is None or "user_id" not in rows.columns:
        return ()

    values = rows[role_column].astype(str).str.lower()
    admins = rows[values.isin(_ADMIN_TRUE_VALUES)]
    return tuple(str(value) for value in admins["user_id"] if str(value).strip())


# --------------------------------------------------------------------------- #
# Context retriever
# --------------------------------------------------------------------------- #


class ContextRetriever:
    """Builds :class:`MessageContext` and :class:`~src.schema.RetrievalContext`
    for each incoming message.

    Expensive lookups (profile tables, timestamp indices) are built once at
    construction time from the full :class:`~src.io.loaders.DataRepository`
    and reused across every message, matching the Stage A / Stage B split in
    the frozen architecture.

    Parameters
    ----------
    repo:
        Fully loaded dataset repository.
    config:
        Application configuration. Defaults to the process-wide singleton.

    Notes
    -----
    Every profile lookup degrades gracefully: a missing dataset yields
    ``None`` or a neutral default rather than raising, so the retriever never
    blocks the pipeline on incomplete data.
    """

    def __init__(self, repo: DataRepository, config: AppConfig | None = None) -> None:
        self._repo = repo
        self._config = config or get_config()

        self._messages = _combined_message_frame(repo)
        logger.info("ContextRetriever: combined message frame has %d row(s).", len(self._messages))

        self._sender_index = conversation_timestamp_index(self._messages, "sender_id")
        key_column = "conversation_id" if "conversation_id" in self._messages.columns else "group_id"
        self._conversation_index = (
            conversation_timestamp_index(self._messages, key_column)
            if key_column in self._messages.columns
            else {}
        )

        self._group_profiles = build_group_profiles(
            repo.groups, repo.group_members, self._messages, repo.message_events
        )
        self._business_profiles = build_business_profiles(
            repo.business_accounts,
            repo.user_business_history,
            self._messages,
            repo.message_events,
        )

        self._user_profile_cache: dict[str, UserProfile] = {}

    # ---- profile accessors -------------------------------------------------- #

    def get_user_profile(self, user_id: str | None) -> UserProfile | None:
        """Return the :class:`UserProfile` for ``user_id``, building it on first use.

        Parameters
        ----------
        user_id:
            User to look up. ``None`` returns ``None``.

        Returns
        -------
        UserProfile or None
        """
        if not user_id:
            return None
        cleaned = str(user_id)
        if cleaned not in self._user_profile_cache:
            self._user_profile_cache[cleaned] = _build_user_profile(
                self._repo.users, self._repo.message_events, cleaned, self._config
            )
        return self._user_profile_cache[cleaned]

    def get_group_profile(self, group_id: str | None) -> GroupProfile | None:
        """Return the cached :class:`GroupProfile` for ``group_id``, if any."""
        if not group_id:
            return None
        return self._group_profiles.get(str(group_id))

    def get_business_profile(self, business_id: str | None) -> BusinessProfile | None:
        """Return the cached :class:`BusinessProfile` for ``business_id``, if any."""
        if not business_id:
            return None
        return self._business_profiles.get(str(business_id))

    # ---- evidence pool ------------------------------------------------------- #

    def _build_retrieval_context(
        self,
        message: Message,
        peer_id: str | None,
    ) -> RetrievalContext:
        """Assemble the three-tier evidence pool for one message.

        Tier 1 (structural) and tier 2 (relational) always run; tier 2b
        (business) runs when the message has a business id; tier 3 (lexical)
        only runs when the pool is thin, per the frozen retrieval design.
        """
        cfg = self._config.retrieval
        reference_time = message.timestamp

        structural: list[RetrievalCandidate] = []
        if message.group_id or message.conversation_id:
            key = message.conversation_id or message.group_id
            structural = lexical.group_history_lookup(
                self._messages,
                conversation_id=str(key),
                reference_time=reference_time,
                window=cfg.structural_window,
                exclude_message_id=message.message_id,
                peer_id=peer_id,
                horizon_hours=cfg.recency_horizon_hours,
                max_snippet_chars=cfg.candidate_snippet_chars,
            )

        relational: list[RetrievalCandidate] = []
        if peer_id:
            relational = lexical.sender_history_lookup(
                self._messages,
                sender_id=peer_id,
                reference_time=reference_time,
                window=cfg.relational_window,
                exclude_message_id=message.message_id,
                horizon_hours=cfg.recency_horizon_hours,
                max_snippet_chars=cfg.candidate_snippet_chars,
            )

        business: list[RetrievalCandidate] = []
        if message.business_id and message.recipient_user_id:
            business = lexical.business_history_lookup(
                self._repo.user_business_history,
                self._messages,
                user_id=message.recipient_user_id,
                business_id=message.business_id,
                reference_time=reference_time,
                window=cfg.business_history_window,
                horizon_hours=cfg.recency_horizon_hours,
                max_snippet_chars=cfg.candidate_snippet_chars,
            )

        pool = lexical.deduplicate_candidates(structural + relational + business)

        lexical_hits: list[RetrievalCandidate] = []
        used_lexical = False
        if len(pool) < cfg.lexical_trigger_below and message.content.strip():
            search_frame = self._messages
            if peer_id and "sender_id" in search_frame.columns:
                search_frame = search_frame[
                    search_frame["sender_id"].astype(str) == str(peer_id)
                ]
            lexical_hits = lexical.similar_message_lookup(
                search_frame,
                query_text=message.content,
                reference_time=reference_time,
                top_k=cfg.lexical_top_k,
                horizon_hours=cfg.recency_horizon_hours,
                max_snippet_chars=cfg.candidate_snippet_chars,
            )
            used_lexical = bool(lexical_hits)
            pool = lexical.deduplicate_candidates(pool + lexical_hits)

        truncated = len(pool) > cfg.max_candidates
        pool = pool[: cfg.max_candidates]

        return RetrievalContext(
            message_id=message.message_id,
            candidates=tuple(pool),
            used_lexical=used_lexical,
            truncated=truncated,
        )

    # ---- main entry point ----------------------------------------------------- #

    def gather(self, message: Message) -> MessageContext:
        """Assemble the full :class:`MessageContext` for one message.

        Parameters
        ----------
        message:
            The message being routed.

        Returns
        -------
        MessageContext
            Never raises on missing data; every field degrades to a neutral
            default when the underlying dataset is absent or thin.
        """
        cfg = self._config
        recipient = message.recipient_user_id
        peer_id = message.sender_id

        user = self.get_user_profile(recipient)
        group = self.get_group_profile(message.group_id)
        business = self.get_business_profile(message.business_id)

        admin_ids = self._group_admin_ids(message.group_id)
        sender_is_admin = peer_id in admin_ids if admin_ids else False

        sender_message_ids = self._message_ids_for_sender(peer_id)
        sender_events = _event_summary_for(self._repo.message_events, sender_message_ids, recipient)

        business_events = EventSummary()
        if message.business_id:
            business_message_ids = self._message_ids_for_business(message.business_id)
            business_events = _event_summary_for(
                self._repo.message_events, business_message_ids, recipient
            )

        is_duplicate, duplicate_count, duplicate_ids = lexical.duplicate_detection(
            self._messages,
            sender_id=peer_id,
            body_text=message.content,
            reference_time=message.timestamp,
            window_hours=float(cfg.routing.burst_window_minutes) * 4 / 60.0 or 24.0,
            exclude_message_id=message.message_id,
        )

        report_count = sender_events.reports
        is_reported_recently = report_count > 0

        conversation_key = message.conversation_id or message.group_id
        conversation_history_ids = self._recent_conversation_ids(
            conversation_key, message.timestamp, message.message_id
        )

        retrieval_context = self._build_retrieval_context(message, peer_id)

        return MessageContext(
            message_id=message.message_id,
            user=user,
            group=group,
            business=business,
            group_admin_ids=admin_ids,
            sender_is_group_admin=sender_is_admin,
            sender_event_summary=sender_events,
            business_event_summary=business_events,
            is_duplicate=is_duplicate,
            duplicate_count=duplicate_count,
            duplicate_evidence_ids=duplicate_ids,
            is_reported_recently=is_reported_recently,
            report_count=report_count,
            conversation_history_ids=conversation_history_ids,
            retrieval=retrieval_context,
        )

    # ---- internal helpers ---------------------------------------------------- #

    def _group_admin_ids(self, group_id: str | None) -> tuple[str, ...]:
        """Return admin ids for a group, cached implicitly via small dataset size."""
        if not group_id:
            return ()
        return _group_admin_ids(self._repo.group_members, str(group_id))

    def _message_ids_for_sender(self, sender_id: str | None) -> set[str]:
        """Return every message id sent by ``sender_id`` in the combined frame."""
        if not sender_id or self._messages.empty or "sender_id" not in self._messages.columns:
            return set()
        if "message_id" not in self._messages.columns:
            return set()
        subset = self._messages[self._messages["sender_id"].astype(str) == str(sender_id)]
        return {str(value) for value in subset["message_id"]}

    def _message_ids_for_business(self, business_id: str | None) -> set[str]:
        """Return every message id sent by ``business_id`` in the combined frame."""
        if not business_id or self._messages.empty or "business_id" not in self._messages.columns:
            return set()
        if "message_id" not in self._messages.columns:
            return set()
        subset = self._messages[self._messages["business_id"].astype(str) == str(business_id)]
        return {str(value) for value in subset["message_id"]}

    def _recent_conversation_ids(
        self,
        conversation_key: str | None,
        reference_time: datetime,
        exclude_message_id: str,
        limit: int = 10,
    ) -> tuple[str, ...]:
        """Return recent message ids in the same conversation, oldest-first.

        Used for "conversation history" in the context bundle -- a plain list
        of ids, distinct from the scored evidence pool in ``retrieval``.
        """
        if not conversation_key:
            return ()
        key_column = "conversation_id" if "conversation_id" in self._messages.columns else "group_id"
        if key_column not in self._messages.columns or "message_id" not in self._messages.columns:
            return ()

        subset = self._messages[self._messages[key_column].astype(str) == str(conversation_key)]
        subset = subset[subset["message_id"].astype(str) != str(exclude_message_id)]
        if "timestamp" not in subset.columns:
            return ()

        subset = subset.copy()
        subset["_ts"] = pd.to_datetime(subset["timestamp"], errors="coerce")
        subset = subset.dropna(subset=["_ts"])
        subset = subset[subset["_ts"] < reference_time].sort_values("_ts").tail(limit)
        return tuple(str(value) for value in subset["message_id"])

    def stats(self) -> dict[str, Any]:
        """Return a small diagnostic summary, useful for a startup log line."""
        return {
            "combined_messages": len(self._messages),
            "group_profiles": len(self._group_profiles),
            "business_profiles": len(self._business_profiles),
            "sender_index_keys": len(self._sender_index),
            "conversation_index_keys": len(self._conversation_index),
        }


__all__ = [
    "ContextRetriever",
    "EventSummary",
    "MessageContext",
]