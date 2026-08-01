"""
Business features: separating a shipping notification from a flash sale.

The distinction that matters is not "is this a business" but "does this user
have a live relationship with this business". A delivery update for an order
placed this morning is a notify; the same account's weekend sale is a mute.

Two responsibilities:

* **Profile building** (Stage A) -- turn ``business_accounts.csv`` and
  ``user_business_history.csv`` into :class:`~src.schema.BusinessProfile`
  objects, including the account's observed promotional ratio.
* **Feature extraction** (hot path) -- emit the ``biz_*`` feature block.

Dependencies
------------
``pandas``, ``src.schema``, ``src.features.content`` (for promo and
transactional detection, shared rather than duplicated).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from src.features.content import ContentVerdict, analyse_content
from src.schema import BusinessProfile, Message

logger = logging.getLogger(__name__)

#: An order younger than this is treated as active, so its updates are wanted.
ACTIVE_ORDER_WINDOW_DAYS = 14.0

#: Cap on how many of a business's messages are scanned for the promo ratio.
PROMO_SAMPLE_LIMIT = 200

#: Columns that might carry a transaction timestamp, in preference order.
_TXN_TIME_COLUMNS = (
    "timestamp",
    "order_date",
    "transaction_date",
    "created_at",
    "last_order_at",
)

#: Columns that might carry an interaction type.
_TXN_TYPE_COLUMNS = ("interaction_type", "event_type", "type", "status", "action")

#: Values in a type column that count as a real transaction.
_TRANSACTION_VALUES = {
    "order",
    "ordered",
    "purchase",
    "purchased",
    "payment",
    "paid",
    "booking",
    "booked",
    "delivered",
    "shipped",
    "transaction",
    "checkout",
}

#: Column candidates for a business display name.
_BIZ_NAME_COLUMNS = ("business_name", "name", "display_name")

#: Column candidates for a business category.
_BIZ_CATEGORY_COLUMNS = ("category", "sector", "vertical", "type")

#: Column candidates for a verification flag.
_BIZ_VERIFIED_COLUMNS = ("is_verified", "verified", "verification_status")

#: Column candidates for message body text.
_TEXT_COLUMNS = ("message_text", "content", "text", "body")

#: Column candidates for an event-type column.
_EVENT_TYPE_COLUMNS = ("event_type", "event", "action", "status")

#: Event values that count as the user having opened a message.
_OPEN_EVENTS = {"read", "open", "opened", "seen", "viewed", "clicked"}

#: Values treated as boolean true in loosely-typed CSV columns.
_TRUTHY = {"1", "true", "yes", "y", "verified", "t"}


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first column from ``candidates`` present in ``frame``."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _promo_ratio_for_business(messages: pd.DataFrame, business_id: str) -> float:
    """Estimate what share of a business's messages are promotional.

    Uses the shared content detectors so that "promotional" means exactly the
    same thing here as it does in the scoring engine.

    Parameters
    ----------
    messages:
        Message frame carrying a ``business_id`` column.
    business_id:
        Account to profile.

    Returns
    -------
    float
        Ratio in ``[0, 1]``; ``0.0`` when nothing can be measured.
    """
    if messages.empty or "business_id" not in messages.columns:
        return 0.0

    subset = messages[messages["business_id"].astype(str) == str(business_id)]
    if subset.empty:
        return 0.0

    text_column = _first_present(subset, _TEXT_COLUMNS)
    if text_column is None:
        return 0.0

    bodies = [str(value) for value in subset[text_column].fillna("") if str(value).strip()]
    if not bodies:
        return 0.0

    # Cap the sample: a chatty account does not need every message scanned.
    sample = bodies[:PROMO_SAMPLE_LIMIT]
    promotional = sum(1 for body in sample if analyse_content(body).is_promotional)
    return promotional / len(sample)


def _open_rate_for_business(
    events: pd.DataFrame,
    messages: pd.DataFrame,
    business_id: str,
    user_id: str | None,
) -> float:
    """Estimate how often this user opens messages from this business.

    Parameters
    ----------
    events:
        ``message_events.csv`` frame.
    messages:
        Message frame carrying ``business_id`` and ``message_id``.
    business_id:
        Account to profile.
    user_id:
        Restrict to this user's events when a ``user_id`` column exists.

    Returns
    -------
    float
        Ratio in ``[0, 1]``; ``0.0`` when nothing can be measured.
    """
    if events.empty or messages.empty:
        return 0.0
    if "business_id" not in messages.columns or "message_id" not in messages.columns:
        return 0.0

    subset = messages[messages["business_id"].astype(str) == str(business_id)]
    message_ids = {str(value) for value in subset["message_id"]}
    if not message_ids or "message_id" not in events.columns:
        return 0.0

    relevant = events[events["message_id"].astype(str).isin(message_ids)]
    if user_id and "user_id" in relevant.columns:
        relevant = relevant[relevant["user_id"].astype(str) == str(user_id)]
    if relevant.empty:
        return 0.0

    event_column = _first_present(relevant, _EVENT_TYPE_COLUMNS)
    if event_column is None:
        return 0.0

    values = relevant[event_column].astype(str).str.lower()
    opened = int(values.isin(_OPEN_EVENTS).sum())
    return min(opened / max(len(message_ids), 1), 1.0)


def build_business_profiles(
    business_accounts: pd.DataFrame,
    user_business_history: pd.DataFrame,
    messages: pd.DataFrame,
    events: pd.DataFrame,
    user_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, BusinessProfile]:
    """Build a :class:`BusinessProfile` for every business account.

    Parameters
    ----------
    business_accounts:
        ``business_accounts.csv`` frame.
    user_business_history:
        ``user_business_history.csv`` frame.
    messages:
        Message frame used to estimate the promotional ratio.
    events:
        ``message_events.csv`` frame used to estimate the open rate.
    user_id:
        Restrict transaction and open-rate statistics to this user.
    now:
        Reference time for order recency. Defaults to the latest transaction
        timestamp so that reruns are stable.

    Returns
    -------
    dict
        ``business_id -> BusinessProfile``. Empty when the account frame is
        unusable.
    """
    profiles: dict[str, BusinessProfile] = {}
    if business_accounts.empty or "business_id" not in business_accounts.columns:
        logger.warning("build_business_profiles: business_accounts is empty or malformed.")
        return profiles

    name_column = _first_present(business_accounts, _BIZ_NAME_COLUMNS)
    category_column = _first_present(business_accounts, _BIZ_CATEGORY_COLUMNS)
    verified_column = _first_present(business_accounts, _BIZ_VERIFIED_COLUMNS)

    history = user_business_history
    time_column = _first_present(history, _TXN_TIME_COLUMNS) if not history.empty else None
    type_column = _first_present(history, _TXN_TYPE_COLUMNS) if not history.empty else None

    if user_id and not history.empty and "user_id" in history.columns:
        history = history[history["user_id"].astype(str) == str(user_id)]

    reference = now
    if reference is None and time_column and not history.empty:
        stamps = pd.to_datetime(history[time_column], errors="coerce").dropna()
        reference = stamps.max().to_pydatetime() if not stamps.empty else None

    for _, row in business_accounts.iterrows():
        business_id = str(row["business_id"]).strip()
        if not business_id:
            continue

        verified = False
        if verified_column is not None:
            verified = str(row.get(verified_column)).strip().lower() in _TRUTHY

        txn_count = 0
        last_age_days: float | None = None
        has_active = False

        if not history.empty and "business_id" in history.columns:
            rows = history[history["business_id"].astype(str) == business_id]
            if not rows.empty:
                if type_column is not None:
                    types = rows[type_column].astype(str).str.lower()
                    matched = int(types.isin(_TRANSACTION_VALUES).sum())
                    txn_count = matched if matched else len(rows)
                else:
                    txn_count = len(rows)

                if time_column is not None and reference is not None:
                    stamps = pd.to_datetime(rows[time_column], errors="coerce").dropna()
                    if not stamps.empty:
                        latest = stamps.max().to_pydatetime()
                        last_age_days = max(
                            (reference - latest).total_seconds() / 86400.0, 0.0
                        )
                        has_active = last_age_days <= ACTIVE_ORDER_WINDOW_DAYS

        profiles[business_id] = BusinessProfile(
            business_id=business_id,
            name=(
                str(row[name_column])
                if name_column and pd.notna(row.get(name_column))
                else ""
            ),
            category=(
                str(row[category_column])
                if category_column and pd.notna(row.get(category_column))
                else ""
            ),
            is_verified=verified,
            user_txn_count=txn_count,
            last_order_age_days=last_age_days,
            has_active_order=has_active,
            promo_ratio=_promo_ratio_for_business(messages, business_id),
            user_open_rate=_open_rate_for_business(events, messages, business_id, user_id),
        )

    logger.info("Built %d business profile(s).", len(profiles))
    return profiles


def business_profiles_to_frame(profiles: Mapping[str, BusinessProfile]) -> pd.DataFrame:
    """Flatten business profiles into a DataFrame for parquet persistence.

    Parameters
    ----------
    profiles:
        Mapping produced by :func:`build_business_profiles`.

    Returns
    -------
    pandas.DataFrame
    """
    if not profiles:
        return pd.DataFrame()
    return pd.DataFrame([profile.to_dict() for profile in profiles.values()])


def business_profiles_from_frame(frame: pd.DataFrame) -> dict[str, BusinessProfile]:
    """Rebuild business profiles from a persisted parquet frame.

    Inverse of :func:`business_profiles_to_frame`.

    Parameters
    ----------
    frame:
        Frame previously written by :func:`business_profiles_to_frame`.

    Returns
    -------
    dict
        ``business_id -> BusinessProfile``.
    """
    profiles: dict[str, BusinessProfile] = {}
    if frame.empty or "business_id" not in frame.columns:
        return profiles

    for _, row in frame.iterrows():
        raw_age = row.get("last_order_age_days")
        profiles[str(row["business_id"])] = BusinessProfile(
            business_id=str(row["business_id"]),
            name=str(row.get("name") or ""),
            category=str(row.get("category") or ""),
            is_verified=bool(row.get("is_verified")),
            user_txn_count=int(row.get("user_txn_count") or 0),
            last_order_age_days=(
                float(raw_age) if raw_age is not None and pd.notna(raw_age) else None
            ),
            has_active_order=bool(row.get("has_active_order")),
            promo_ratio=float(row.get("promo_ratio") or 0.0),
            user_open_rate=float(row.get("user_open_rate") or 0.0),
        )
    return profiles


def business_features(
    message: Message,
    business: BusinessProfile | None,
    content_verdict: ContentVerdict | None = None,
) -> dict[str, Any]:
    """Assemble the ``biz_*`` feature block for one message.

    Emitted for every message, with neutral values for non-business senders, so
    the scoring engine never branches on presence.

    Parameters
    ----------
    message:
        The message being routed.
    business:
        Profile of the sending business account, or ``None``.
    content_verdict:
        Optional pre-computed :class:`~src.features.content.ContentVerdict`.
        Recomputed from the message when omitted, so callers may skip it.

    Returns
    -------
    dict
        Feature dictionary with the ``biz_`` prefix.
    """
    verdict = content_verdict or analyse_content(message.content, message=message)

    features: dict[str, Any] = {
        "biz_is_business": bool(message.business_id) or message.is_from_business,
        "biz_id": message.business_id,
        "biz_name": "",
        "biz_category": "",
        "biz_is_verified": False,
        "biz_is_known_to_user": False,
        "biz_txn_count": 0,
        "biz_last_order_age_days": None,
        "biz_has_active_order": False,
        "biz_promo_ratio": 0.0,
        "biz_user_open_rate": 0.0,
        "biz_message_is_transactional": verdict.is_transactional,
        "biz_message_is_promotional": verdict.is_promotional,
        "biz_is_cold_promo": False,
        "biz_is_wanted_update": False,
    }

    if business is None:
        # An unrecognised sender pushing promotional copy is the cold-promo case
        # even when we have no profile for it.
        features["biz_is_cold_promo"] = (
            features["biz_is_business"] and features["biz_message_is_promotional"]
        )
        return features

    features.update(
        {
            "biz_is_business": True,
            "biz_name": business.name,
            "biz_category": business.category,
            "biz_is_verified": business.is_verified,
            "biz_is_known_to_user": business.is_known_to_user,
            "biz_txn_count": business.user_txn_count,
            "biz_last_order_age_days": business.last_order_age_days,
            "biz_has_active_order": business.has_active_order,
            "biz_promo_ratio": business.promo_ratio,
            "biz_user_open_rate": business.user_open_rate,
        }
    )

    # A promotional message from an account the user has never bought from.
    features["biz_is_cold_promo"] = (
        features["biz_message_is_promotional"] and not business.is_known_to_user
    )

    # A transactional message tied to a live order, or from a trusted account.
    features["biz_is_wanted_update"] = features["biz_message_is_transactional"] and (
        business.has_active_order or business.is_known_to_user
    )

    return features


__all__ = [
    "ACTIVE_ORDER_WINDOW_DAYS",
    "PROMO_SAMPLE_LIMIT",
    "build_business_profiles",
    "business_features",
    "business_profiles_from_frame",
    "business_profiles_to_frame",
]   