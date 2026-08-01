"""
Group features: how much of a group's traffic is actually for this user.

Two responsibilities:

* **Profile building** (Stage A) -- turn ``groups.csv``, ``group_members.csv``
  and ``message_events.csv`` into :class:`~src.schema.GroupProfile` objects,
  including the category hint read off the group's title.
* **Feature extraction** (hot path) -- given a message and the relevant
  profiles, emit the ``group_*`` feature block.

Dependencies
------------
``pandas``, ``src.schema``, ``src.features.relationship`` (for the group-name
lexicons, which are shared rather than duplicated).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

from src.features.relationship import extract_group_name_signals
from src.schema import GroupProfile, Message, RelationshipCategory, UserProfile

logger = logging.getLogger(__name__)

#: Group size above which most traffic is assumed not to concern any one member.
LARGE_GROUP_SIZE = 25

#: Group size below which a group behaves much like a direct conversation.
INTIMATE_GROUP_SIZE = 5

#: Message volume treated as saturating for the noise-ratio calculation.
NOISE_VOLUME_SATURATION = 50.0

#: Column candidates for a group's display title, in preference order.
_GROUP_NAME_COLUMNS = ("group_name", "name", "title", "subject")

#: Column candidates for a group's mute flag.
_GROUP_MUTED_COLUMNS = ("is_muted", "muted", "notifications_muted")

#: Column candidates for an explicit member count.
_GROUP_SIZE_COLUMNS = ("member_count", "size", "num_members", "members")

#: Column candidates for a group's creation timestamp.
_GROUP_CREATED_COLUMNS = ("created_at", "created", "timestamp")

#: Column candidates for an event-type column in ``message_events.csv``.
_EVENT_TYPE_COLUMNS = ("event_type", "event", "action", "status")

#: Event values that count as the user having read a message.
_READ_EVENTS = {"read", "open", "opened", "seen", "viewed"}

#: Event values that count as the user having engaged with a message.
_REPLY_EVENTS = {"reply", "replied", "responded", "react", "reacted"}

#: Values treated as boolean true in loosely-typed CSV columns.
_TRUTHY = {"1", "true", "yes", "y", "muted", "t"}


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first column from ``candidates`` present in ``frame``."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def infer_group_category(
    group_name: str | None,
) -> tuple[RelationshipCategory, float, tuple[str, ...]]:
    """Infer a category from a group's title.

    Group titles are more informative than contact names, because people name
    groups after their purpose ("Flat 302 Owners", "8th Sem ECE", "Sprint 14").

    Parameters
    ----------
    group_name:
        Raw group title.

    Returns
    -------
    tuple
        ``(category, confidence, signals)``. Returns
        ``(UNKNOWN, 0.0, ())`` when the title says nothing useful.
    """
    signals = extract_group_name_signals(group_name)
    if not signals:
        return RelationshipCategory.UNKNOWN, 0.0, ()

    best = signals[0]
    # Two hits pointing at different categories should lower confidence, not raise it.
    distinct = {signal.category for signal in signals}
    confidence = best.confidence if len(distinct) == 1 else best.confidence * 0.75

    return best.category, confidence, tuple(signal.label for signal in signals[:3])


def _member_ids_for_group(group_members: pd.DataFrame, group_id: str) -> tuple[str, ...]:
    """Return the member ids of one group, or an empty tuple.

    Parameters
    ----------
    group_members:
        ``group_members.csv`` frame.
    group_id:
        Group whose membership to look up.
    """
    if group_members.empty:
        return ()
    if "group_id" not in group_members.columns or "user_id" not in group_members.columns:
        return ()
    rows = group_members[group_members["group_id"].astype(str) == str(group_id)]
    return tuple(str(value) for value in rows["user_id"] if str(value).strip())


def _engagement_rates(
    events: pd.DataFrame,
    message_ids: set[str],
    user_id: str | None,
) -> tuple[float, float]:
    """Estimate ``(read_rate, reply_rate)`` for a user over a message set.

    Falls back to neutral priors when ``message_events.csv`` lacks the columns
    needed, so a missing dataset degrades quality rather than crashing.

    Parameters
    ----------
    events:
        ``message_events.csv`` frame.
    message_ids:
        Message ids belonging to the group under consideration.
    user_id:
        Restrict to this user's events when a ``user_id`` column exists.

    Returns
    -------
    tuple
        ``(read_rate, reply_rate)``, both in ``[0, 1]``.
    """
    if events.empty or not message_ids or "message_id" not in events.columns:
        return 0.5, 0.1

    subset = events[events["message_id"].astype(str).isin(message_ids)]
    if user_id and "user_id" in subset.columns:
        subset = subset[subset["user_id"].astype(str) == str(user_id)]
    if subset.empty:
        return 0.5, 0.1

    event_column = _first_present(subset, _EVENT_TYPE_COLUMNS)
    if event_column is None:
        return 0.5, 0.1

    values = subset[event_column].astype(str).str.lower()
    total = max(len(message_ids), 1)
    read_like = int(values.isin(_READ_EVENTS).sum())
    reply_like = int(values.isin(_REPLY_EVENTS).sum())

    return min(read_like / total, 1.0), min(reply_like / total, 1.0)


def _messages_per_day(group_messages: pd.DataFrame) -> float:
    """Return the average daily message volume of a group."""
    if group_messages.empty or "timestamp" not in group_messages.columns:
        return 0.0
    stamps = pd.to_datetime(group_messages["timestamp"], errors="coerce").dropna()
    if len(stamps) >= 2:
        span_days = max((stamps.max() - stamps.min()).total_seconds() / 86400.0, 1.0)
        return len(stamps) / span_days
    return 1.0 if len(stamps) == 1 else 0.0


def build_group_profiles(
    groups: pd.DataFrame,
    group_members: pd.DataFrame,
    messages: pd.DataFrame,
    events: pd.DataFrame,
    user_id: str | None = None,
) -> dict[str, GroupProfile]:
    """Build a :class:`GroupProfile` for every group, keyed by ``group_id``.

    This is a Stage A helper. Persist the result to
    ``data/cache/group_profile.parquet`` and look groups up in O(1) afterwards.

    Parameters
    ----------
    groups:
        ``groups.csv`` frame.
    group_members:
        ``group_members.csv`` frame.
    messages:
        Message frame used to compute traffic volume, typically the union of
        ``messages.csv`` and ``message_history.csv``.
    events:
        ``message_events.csv`` frame used for read and reply rates.
    user_id:
        Restrict engagement rates to this user. ``None`` averages over everyone.

    Returns
    -------
    dict
        ``group_id -> GroupProfile``. Empty when ``groups`` is unusable.
    """
    profiles: dict[str, GroupProfile] = {}
    if groups.empty or "group_id" not in groups.columns:
        logger.warning("build_group_profiles: groups frame is empty or missing group_id.")
        return profiles

    name_column = _first_present(groups, _GROUP_NAME_COLUMNS)
    muted_column = _first_present(groups, _GROUP_MUTED_COLUMNS)
    size_column = _first_present(groups, _GROUP_SIZE_COLUMNS)
    created_column = _first_present(groups, _GROUP_CREATED_COLUMNS)

    has_group_messages = not messages.empty and "group_id" in messages.columns

    for _, row in groups.iterrows():
        group_id = str(row["group_id"]).strip()
        if not group_id:
            continue

        raw_name = str(row[name_column]) if name_column and pd.notna(row.get(name_column)) else ""
        category, confidence, _signals = infer_group_category(raw_name)
        members = _member_ids_for_group(group_members, group_id)

        size = len(members)
        if size_column is not None and pd.notna(row.get(size_column)):
            try:
                size = max(size, int(float(row[size_column])))
            except (TypeError, ValueError):
                logger.debug("Group %s: unparseable size %r", group_id, row.get(size_column))

        group_messages = (
            messages[messages["group_id"].astype(str) == group_id]
            if has_group_messages
            else messages.iloc[0:0]
        )
        message_ids = (
            {str(value) for value in group_messages["message_id"]}
            if "message_id" in group_messages.columns
            else set()
        )

        read_rate, reply_rate = _engagement_rates(events, message_ids, user_id)

        is_muted = False
        if muted_column is not None:
            is_muted = str(row.get(muted_column)).strip().lower() in _TRUTHY

        created_at = None
        if created_column is not None and pd.notna(row.get(created_column)):
            parsed = pd.to_datetime(row[created_column], errors="coerce")
            if pd.notna(parsed):
                created_at = parsed.to_pydatetime()

        profiles[group_id] = GroupProfile(
            group_id=group_id,
            name=raw_name,
            size=size,
            member_ids=members,
            created_at=created_at,
            category_hint=category,
            category_hint_confidence=confidence,
            is_muted=is_muted,
            user_read_rate=read_rate,
            user_reply_rate=reply_rate,
            messages_per_day=_messages_per_day(group_messages),
        )

    logger.info("Built %d group profile(s).", len(profiles))
    return profiles


def group_profiles_to_frame(profiles: Mapping[str, GroupProfile]) -> pd.DataFrame:
    """Flatten group profiles into a DataFrame for parquet persistence.

    Member id tuples are pipe-joined so the frame stays flat.

    Parameters
    ----------
    profiles:
        Mapping produced by :func:`build_group_profiles`.

    Returns
    -------
    pandas.DataFrame
    """
    if not profiles:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for profile in profiles.values():
        payload = profile.to_dict()
        payload["member_ids"] = "|".join(profile.member_ids)
        rows.append(payload)
    return pd.DataFrame(rows)


def group_profiles_from_frame(frame: pd.DataFrame) -> dict[str, GroupProfile]:
    """Rebuild group profiles from a persisted parquet frame.

    Inverse of :func:`group_profiles_to_frame`.

    Parameters
    ----------
    frame:
        Frame previously written by :func:`group_profiles_to_frame`.

    Returns
    -------
    dict
        ``group_id -> GroupProfile``.
    """
    profiles: dict[str, GroupProfile] = {}
    if frame.empty or "group_id" not in frame.columns:
        return profiles

    for _, row in frame.iterrows():
        members = str(row.get("member_ids") or "")
        created = row.get("created_at")
        parsed_created = pd.to_datetime(created, errors="coerce") if created else None

        profiles[str(row["group_id"])] = GroupProfile(
            group_id=str(row["group_id"]),
            name=str(row.get("name") or ""),
            size=int(row.get("size") or 0),
            member_ids=tuple(part for part in members.split("|") if part),
            created_at=(
                parsed_created.to_pydatetime()
                if parsed_created is not None and pd.notna(parsed_created)
                else None
            ),
            category_hint=RelationshipCategory.from_any(row.get("category_hint")),
            category_hint_confidence=float(row.get("category_hint_confidence") or 0.0),
            is_muted=bool(row.get("is_muted")),
            user_read_rate=float(row.get("user_read_rate") or 0.5),
            user_reply_rate=float(row.get("user_reply_rate") or 0.1),
            messages_per_day=float(row.get("messages_per_day") or 0.0),
        )
    return profiles


def group_features(
    message: Message,
    group: GroupProfile | None,
    user: UserProfile | None = None,
) -> dict[str, Any]:
    """Assemble the ``group_*`` feature block for one message.

    For direct messages the block is still emitted, with ``group_is_group``
    false and neutral values elsewhere, so the scoring engine never has to
    branch on presence.

    Parameters
    ----------
    message:
        The message being routed.
    group:
        Profile of the group it arrived in, or ``None`` for a direct message.
    user:
        The receiving user's profile, used for the per-user mute flag.

    Returns
    -------
    dict
        Feature dictionary with the ``group_`` prefix.
    """
    recipient = message.recipient_user_id or (user.user_id if user else None)
    mentions_user = message.mentions_user(recipient)

    features: dict[str, Any] = {
        "group_is_group": message.is_group_message,
        "group_id": message.group_id,
        "group_name": "",
        "group_size": 0,
        "group_is_large": False,
        "group_is_intimate": False,
        "group_is_muted": False,
        "group_mentions_user": mentions_user,
        "group_category_hint": RelationshipCategory.UNKNOWN.value,
        "group_category_confidence": 0.0,
        "group_user_read_rate": 0.5,
        "group_user_reply_rate": 0.1,
        "group_messages_per_day": 0.0,
        "group_user_is_member": True,
        "group_noise_ratio": 0.0,
    }

    if not message.is_group_message or group is None:
        return features

    muted_by_user = bool(
        user is not None and message.group_id and message.group_id in user.muted_groups
    )

    features.update(
        {
            "group_name": group.name,
            "group_size": group.size,
            "group_is_large": group.size >= LARGE_GROUP_SIZE,
            "group_is_intimate": 0 < group.size <= INTIMATE_GROUP_SIZE,
            "group_is_muted": group.is_muted or muted_by_user,
            "group_category_hint": group.category_hint.value,
            "group_category_confidence": group.category_hint_confidence,
            "group_user_read_rate": group.user_read_rate,
            "group_user_reply_rate": group.user_reply_rate,
            "group_messages_per_day": group.messages_per_day,
            "group_user_is_member": (
                True
                if not group.member_ids or not recipient
                else str(recipient) in group.member_ids
            ),
        }
    )

    # Noise ratio: high-volume, low-read groups are where mute lives.
    if group.messages_per_day > 0:
        volume_pressure = min(group.messages_per_day / NOISE_VOLUME_SATURATION, 1.0)
        features["group_noise_ratio"] = volume_pressure * (1.0 - group.user_read_rate)

    return features


__all__ = [
    "INTIMATE_GROUP_SIZE",
    "LARGE_GROUP_SIZE",
    "NOISE_VOLUME_SATURATION",
    "build_group_profiles",
    "group_features",
    "group_profiles_from_frame",
    "group_profiles_to_frame",
    "infer_group_category",
]