"""
Temporal features: when a message arrived and what else arrived around it.

Time carries real signal that content does not. A work message at 02:00 is
different from the same message at 14:00; the fourth message from one sender in
ten minutes is different from the first.

Everything here is a pure function over timestamps, so it is trivially testable
and has no dependency on the rest of the pipeline beyond configuration.

Dependencies
------------
``pandas``, ``src.config``, ``src.schema``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Sequence

import pandas as pd

from src.config import AppConfig, get_config
from src.schema import Message, UserProfile

logger = logging.getLogger(__name__)

#: Boundaries used to bucket an hour into a part of the day.
PART_OF_DAY_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("night", 0, 6),
    ("early_morning", 6, 9),
    ("morning", 9, 12),
    ("afternoon", 12, 17),
    ("evening", 17, 21),
    ("late_evening", 21, 24),
)

#: Working hours used for the office-hours flag.
WORK_START_HOUR = 9
WORK_END_HOUR = 19

#: Thread gap below which a conversation counts as actively in progress.
ACTIVE_THREAD_GAP_HOURS = 2.0


def part_of_day(moment: datetime) -> str:
    """Bucket a timestamp into a named part of the day.

    Parameters
    ----------
    moment:
        Timestamp to bucket.

    Returns
    -------
    str
        One of the labels in :data:`PART_OF_DAY_BOUNDS`.
    """
    hour = moment.hour
    for label, start, end in PART_OF_DAY_BOUNDS:
        if start <= hour < end:
            return label
    return "night"


def is_in_dnd_window(moment: datetime, start_hour: int, end_hour: int) -> bool:
    """Return whether ``moment`` falls inside a do-not-disturb window.

    Handles windows that wrap past midnight, which the default 23:00-07:00 does.

    Parameters
    ----------
    moment:
        Timestamp to test.
    start_hour:
        Inclusive start hour, 0-23.
    end_hour:
        Exclusive end hour, 0-23.

    Returns
    -------
    bool
    """
    hour = moment.hour
    if start_hour <= end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def is_working_hours(moment: datetime) -> bool:
    """Return whether ``moment`` falls in weekday working hours."""
    if moment.weekday() >= 5:
        return False
    return WORK_START_HOUR <= moment.hour < WORK_END_HOUR


def recency_decay(age_hours: float, horizon_hours: float) -> float:
    """Map an age in hours onto a ``[0, 1]`` freshness weight.

    Linear decay to zero at the horizon. Linear rather than exponential because
    it is easier to read off an explanation chart, and the difference does not
    matter at this scale.

    Parameters
    ----------
    age_hours:
        How old the item is.
    horizon_hours:
        Age at which the weight reaches zero.

    Returns
    -------
    float
        Weight in ``[0, 1]``.
    """
    if horizon_hours <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (max(age_hours, 0.0) / horizon_hours)))


def count_within_window(
    timestamps: Sequence[datetime],
    reference: datetime,
    window_minutes: int,
) -> int:
    """Count timestamps falling in the window immediately before ``reference``.

    Parameters
    ----------
    timestamps:
        Prior timestamps from the same source.
    reference:
        The current message's timestamp.
    window_minutes:
        Length of the lookback window.

    Returns
    -------
    int
        Number of timestamps in ``(reference - window, reference]``.
    """
    if not timestamps or window_minutes <= 0:
        return 0
    cutoff = reference - timedelta(minutes=window_minutes)
    return sum(1 for stamp in timestamps if cutoff < stamp <= reference)


def detect_burst(
    timestamps: Sequence[datetime],
    reference: datetime,
    window_minutes: int,
    threshold: int,
) -> tuple[bool, int]:
    """Detect whether a source is currently flooding the user.

    Parameters
    ----------
    timestamps:
        Prior timestamps from the same sender or group.
    reference:
        The current message's timestamp.
    window_minutes:
        Lookback window.
    threshold:
        Count at or above which the source counts as bursting.

    Returns
    -------
    tuple
        ``(is_bursting, count_in_window)``.
    """
    count = count_within_window(timestamps, reference, window_minutes)
    return count >= threshold, count


def time_since_last(
    timestamps: Sequence[datetime],
    reference: datetime,
) -> float | None:
    """Return hours since the most recent prior timestamp, or ``None``.

    Only timestamps strictly before ``reference`` are considered, so a
    duplicated timestamp does not produce a spurious zero-hour gap.

    Parameters
    ----------
    timestamps:
        Prior timestamps from the same source.
    reference:
        The current message's timestamp.

    Returns
    -------
    float or None
        Hours elapsed, or ``None`` when there is no prior contact.
    """
    prior = [stamp for stamp in timestamps if stamp < reference]
    if not prior:
        return None
    return (reference - max(prior)).total_seconds() / 3600.0


def extract_timestamps(frame: pd.DataFrame, column: str = "timestamp") -> list[datetime]:
    """Pull a clean, sorted list of Python datetimes out of a DataFrame column.

    Returns an empty list when the column is absent or entirely unparseable, so
    callers never need a guard.

    Parameters
    ----------
    frame:
        Source frame.
    column:
        Timestamp column name.

    Returns
    -------
    list of datetime
    """
    if frame.empty or column not in frame.columns:
        return []
    stamps = pd.to_datetime(frame[column], errors="coerce").dropna()
    return sorted(stamp.to_pydatetime() for stamp in stamps)


def temporal_features(
    message: Message,
    user: UserProfile | None = None,
    sender_timestamps: Sequence[datetime] | None = None,
    conversation_timestamps: Sequence[datetime] | None = None,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Assemble the ``time_*`` feature block for one message.

    Parameters
    ----------
    message:
        The message being routed; its ``timestamp`` is the reference point.
    user:
        The receiving user's profile, whose DND window overrides the default.
    sender_timestamps:
        Prior message timestamps from the same sender, used for burst detection
        and for the gap since last contact.
    conversation_timestamps:
        Prior message timestamps in the same conversation, used for thread
        activity.
    config:
        Application configuration; defaults to the singleton.

    Returns
    -------
    dict
        Feature dictionary with the ``time_`` prefix.
    """
    cfg = config or get_config()
    moment = message.timestamp

    dnd_start = user.dnd_start_hour if user else cfg.routing.dnd_start_hour
    dnd_end = user.dnd_end_hour if user else cfg.routing.dnd_end_hour

    sender_stamps = list(sender_timestamps or ())
    conversation_stamps = list(conversation_timestamps or ())

    is_bursting, burst_count = detect_burst(
        sender_stamps,
        moment,
        cfg.routing.burst_window_minutes,
        cfg.routing.burst_notify_threshold,
    )

    gap_hours = time_since_last(sender_stamps, moment)
    thread_gap_hours = time_since_last(conversation_stamps, moment)

    return {
        "time_hour": moment.hour,
        "time_weekday": moment.weekday(),
        "time_is_weekend": moment.weekday() >= 5,
        "time_part_of_day": part_of_day(moment),
        "time_is_dnd": is_in_dnd_window(moment, dnd_start, dnd_end),
        "time_is_working_hours": is_working_hours(moment),
        "time_is_odd_hour": moment.hour < 6 or moment.hour >= 23,
        "time_burst_count": burst_count,
        "time_is_bursting": is_bursting,
        "time_sender_gap_hours": gap_hours,
        "time_thread_gap_hours": thread_gap_hours,
        "time_is_first_contact": not sender_stamps,
        "time_thread_is_active": (
            thread_gap_hours is not None and thread_gap_hours <= ACTIVE_THREAD_GAP_HOURS
        ),
        "time_sender_recency_weight": (
            recency_decay(gap_hours, cfg.retrieval.recency_horizon_hours)
            if gap_hours is not None
            else 0.0
        ),
        "time_messages_last_hour": count_within_window(sender_stamps, moment, 60),
        "time_messages_last_day": count_within_window(sender_stamps, moment, 60 * 24),
    }


def conversation_timestamp_index(
    messages: pd.DataFrame,
    key_column: str,
) -> dict[str, list[datetime]]:
    """Build a ``key -> sorted timestamps`` index for O(1) hot-path lookups.

    Run once in Stage A for both ``sender_id`` and ``conversation_id``, then
    pass the relevant list into :func:`temporal_features`.

    Parameters
    ----------
    messages:
        Message frame containing ``key_column`` and ``timestamp``.
    key_column:
        Column to group by, e.g. ``"sender_id"`` or ``"conversation_id"``.

    Returns
    -------
    dict
        ``key -> sorted list of datetimes``. Empty when inputs are unusable.
    """
    index: dict[str, list[datetime]] = {}
    if messages.empty or key_column not in messages.columns:
        logger.warning("conversation_timestamp_index: missing %r column.", key_column)
        return index
    if "timestamp" not in messages.columns:
        logger.warning("conversation_timestamp_index: missing timestamp column.")
        return index

    frame = messages[[key_column, "timestamp"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])

    for key, group in frame.groupby(frame[key_column].astype(str)):
        cleaned = str(key).strip()
        if not cleaned:
            continue
        index[cleaned] = sorted(stamp.to_pydatetime() for stamp in group["timestamp"])

    logger.info("Built timestamp index over %r with %d key(s).", key_column, len(index))
    return index


__all__ = [
    "ACTIVE_THREAD_GAP_HOURS",
    "PART_OF_DAY_BOUNDS",
    "WORK_END_HOUR",
    "WORK_START_HOUR",
    "conversation_timestamp_index",
    "count_within_window",
    "detect_burst",
    "extract_timestamps",
    "is_in_dnd_window",
    "is_working_hours",
    "part_of_day",
    "recency_decay",
    "temporal_features",
    "time_since_last",
]