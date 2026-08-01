"""
Deterministic rule engine for the WhatsApp notification router.

This module evaluates a message against a battery of declarative rules and
returns a structured :class:`RuleEvaluation` -- triggered rules, their
suggested weights and confidences, category signals, safety overrides, and a
metadata bundle for downstream scoring. It never emits ``notify``/``digest``/
``mute``: that resolution belongs to the scoring engine and arbiter, which
consume this output.

Rule families
-------------
Emergency, Family, Office, College, Friends, Society, Healthcare, Finance,
Travel, Business, Promotions, Spam, Scam -- plus a ``System`` family for
structural behaviours (mute state, quiet hours, duplicates, admin status)
that cut across every domain category.

Precedence
----------
Structured signals always outrank inferred ones. A :class:`~src.schema.RelationshipResult`
already encodes this precedence via its ``method`` field (business match and
kinship-name match rank above behavioural inference); this engine reads that
field rather than re-deriving it. When no ``RelationshipResult`` is supplied,
the engine falls back to a group's name-inferred category hint, clearly
labelled as inferred in the resulting category signal.

Conflict resolution between overrides (floor vs. ceiling vs. force) is
intentionally *not* performed here -- it is recorded as a set of constraints
for the arbiter to resolve.

Dependencies
------------
``src.config``, ``src.schema``, ``src.features.content``,
``src.retrieval.context`` (for :class:`~src.retrieval.context.MessageContext`
and :class:`~src.retrieval.context.EventSummary`). ``src.media.ocr`` /
``src.media.asr`` result types are accepted optionally for richer trace
metadata.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.config import AppConfig, get_config
from src.features.content import ContentVerdict, analyse_content
from src.media.asr import AsrResult
from src.media.ocr import OcrResult
from src.retrieval.context import EventSummary, MessageContext
from src.schema import (
    Action,
    GroupProfile,
    Message,
    MessageType,
    OverrideEffect,
    OverrideRecord,
    RelationshipCategory,
    RelationshipResult,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Domain keyword patterns not already covered by src.features.content
# --------------------------------------------------------------------------- #
#
# src.features.content owns generic OTP / transactional / promotional / spam
# detection. The patterns below are domain-specific refinements used only by
# this engine to route a transactional message into Healthcare, Finance, or
# Travel rather than leaving it generically "transactional".

HEALTHCARE_PATTERN = re.compile(
    r"\b(appointment|doctor|dr\.?|clinic|hospital|prescription|lab report|"
    r"test result|diagnosis|medicine|pharmacy|vaccination|vaccine|surgery|"
    r"consultation|follow[\s-]?up|health checkup|blood test|scan report)\b",
    re.IGNORECASE,
)

FINANCE_PATTERN = re.compile(
    r"\b(debited|credited|account balance|available balance|emi|due amount|"
    r"minimum due|credit card|debit card|neft|imps|upi|autopay|auto[\s-]?debit|"
    r"loan|interest rate|overdue|late fee|statement generated|bill payment)\b",
    re.IGNORECASE,
)

TRAVEL_PATTERN = re.compile(
    r"\b(flight|boarding pass|pnr|itinerary|check[\s-]?in|gate closes|"
    r"train ticket|coach|seat number|departure|arrival|layover|hotel booking|"
    r"cab booking|ride confirmed|trip confirmed|travel itinerary)\b",
    re.IGNORECASE,
)

SOCIETY_CONTENT_PATTERN = re.compile(
    r"\b(maintenance|water supply|power cut|electricity|society meeting|"
    r"rwa|watchman|security guard|visitor|gate pass|parking|garbage|"
    r"lift (?:not working|maintenance)|society notice)\b",
    re.IGNORECASE,
)

SCAM_HIGH_CONFIDENCE_PATTERN = re.compile(
    r"\b(kyc (?:suspend|expire|block)|account (?:suspend|block)ed|"
    r"click this link to verify|verify (?:immediately|now) to avoid|"
    r"your account (?:will be|has been) (?:blocked|suspended|locked)|"
    r"congratulations you (?:have )?won|lucky winner|claim your prize)\b",
    re.IGNORECASE,
)

#: Message-directed emergency language, distinct from generic urgency terms.
EMERGENCY_PATTERN = re.compile(
    r"\b(emergency|accident|hospitalized|hospitalised|call me now|"
    r"call immediately|need help now|urgent help|please call|sos)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Enumerations and output dataclasses
# --------------------------------------------------------------------------- #


class RuleFamily(str, Enum):
    """The rule categories this engine organises checks under."""

    SYSTEM = "system"
    EMERGENCY = "emergency"
    FAMILY = "family"
    OFFICE = "office"
    COLLEGE = "college"
    FRIENDS = "friends"
    SOCIETY = "society"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    TRAVEL = "travel"
    BUSINESS = "business"
    PROMOTIONS = "promotions"
    SPAM = "spam"
    SCAM = "scam"


@dataclass(frozen=True)
class TriggeredRule:
    """One rule that fired for a message.

    Attributes
    ----------
    rule_id:
        Stable identifier, e.g. ``"emergency_otp"``. Used for provenance in
        the explanation trace and for tests.
    family:
        Which :class:`RuleFamily` this rule belongs to.
    description:
        Human-readable explanation, suitable for the demo trace.
    weight:
        Suggested signed contribution for downstream scoring. Positive values
        push toward higher priority (notify), negative toward lower (mute).
        This is a *suggestion*; the scoring engine owns final weighting and
        caps.
    confidence:
        How certain this rule is about its own applicability, in ``[0, 1]``.
    evidence_message_ids:
        Message ids that support this rule firing, if any.
    signals:
        Human-readable signal strings, in the same style used across the
        feature-extraction layer.
    """

    rule_id: str
    family: RuleFamily
    description: str
    weight: float
    confidence: float
    evidence_message_ids: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate confidence bounds without mutating a frozen instance improperly."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"TriggeredRule.confidence must be in [0,1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "rule_id": self.rule_id,
            "family": self.family.value,
            "description": self.description,
            "weight": round(self.weight, 3),
            "confidence": round(self.confidence, 4),
            "evidence_message_ids": list(self.evidence_message_ids),
            "signals": list(self.signals),
        }


@dataclass
class RuleEvaluation:
    """Structured output of running the rule engine over one message.

    Deliberately carries no ``action`` field: this is an intermediate
    artefact consumed by the scoring engine and arbiter, not a decision.

    Attributes
    ----------
    message_id:
        The message this evaluation covers.
    triggered_rules:
        Every rule that fired, across all families.
    overrides:
        Floor / ceiling / force constraints for the arbiter to resolve. Not
        resolved against each other here.
    category_signals:
        Human-readable relationship/category hints surfaced by rule checks,
        e.g. ``"relationship:Family(conf=0.90,method=name_kinship)"``.
    suggested_message_type:
        Best-guess :class:`~src.schema.MessageType`, derived from whichever
        domain-specific rule fired with the highest confidence.
    metadata:
        Flat dictionary of booleans/floats/strings for the scoring engine to
        read without re-deriving them (mute state, DND, engagement rates,
        business trust signals, and so on).
    """

    message_id: str
    triggered_rules: tuple[TriggeredRule, ...] = ()
    overrides: tuple[OverrideRecord, ...] = ()
    category_signals: tuple[str, ...] = ()
    suggested_message_type: MessageType = MessageType.OTHER
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_weight(self) -> float:
        """Sum of every triggered rule's suggested weight.

        A convenience aggregate; the scoring engine is free to weight rules
        differently and should not treat this as authoritative.
        """
        return sum(rule.weight for rule in self.triggered_rules)

    @property
    def families_triggered(self) -> tuple[RuleFamily, ...]:
        """Distinct families with at least one triggered rule, insertion order."""
        seen: list[RuleFamily] = []
        for rule in self.triggered_rules:
            if rule.family not in seen:
                seen.append(rule.family)
        return tuple(seen)

    @property
    def has_override(self) -> bool:
        """``True`` when at least one safety override was recorded."""
        return bool(self.overrides)

    def rules_in(self, family: RuleFamily) -> tuple[TriggeredRule, ...]:
        """Return every triggered rule belonging to ``family``."""
        return tuple(rule for rule in self.triggered_rules if rule.family is family)

    def highest_confidence_rule(self) -> TriggeredRule | None:
        """Return the single most confident triggered rule, or ``None``."""
        if not self.triggered_rules:
            return None
        return max(self.triggered_rules, key=lambda rule: rule.confidence)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary for the explanation trace."""
        return {
            "message_id": self.message_id,
            "triggered_rules": [rule.to_dict() for rule in self.triggered_rules],
            "overrides": [override.to_dict() for override in self.overrides],
            "category_signals": list(self.category_signals),
            "suggested_message_type": self.suggested_message_type.value,
            "total_weight": round(self.total_weight, 3),
            "families_triggered": [family.value for family in self.families_triggered],
            "metadata": self.metadata,
        }


# --------------------------------------------------------------------------- #
# Weight constants
# --------------------------------------------------------------------------- #
#
# These are engine-local suggestions, deliberately not imported from
# src.config.ScoringConfig: rule weighting is this module's concern, final
# component caps and blending are the scoring engine's.


class _W:
    """Namespace of suggested rule weights, roughly on a [-30, 30] scale."""

    OTP = 28.0
    EMERGENCY_KEYWORD = 24.0
    MENTION = 14.0
    REPLY_TO_USER = 10.0
    DIRECT_FAMILY = 18.0
    DIRECT_CLOSE_FRIEND = 12.0
    ADMIN_ANNOUNCEMENT = 8.0
    PINNED_CONTACT = 10.0
    HIGH_REPLY_HISTORY = 6.0
    HIGH_DISMISSAL_HISTORY = -6.0
    OFFICE_WORK_REQUEST = 10.0
    OFFICE_AFTER_HOURS = -4.0
    COLLEGE_DEADLINE = 9.0
    COLLEGE_LARGE_GROUP_NOISE = -5.0
    SOCIETY_URGENT_SECURITY = 12.0
    SOCIETY_NOTICE = -2.0
    HEALTHCARE_APPOINTMENT = 14.0
    HEALTHCARE_CRITICAL = 20.0
    FINANCE_PAYMENT_DUE = 13.0
    FINANCE_TRANSACTION_ALERT = 11.0
    TRAVEL_BOOKING = 12.0
    TRAVEL_CHECKIN = 15.0
    BUSINESS_ACTIVE_ORDER = 12.0
    BUSINESS_WANTED_UPDATE = 8.0
    BUSINESS_UNVERIFIED_COLD = -8.0
    PROMOTIONS_COLD = -14.0
    PROMOTIONS_REPEATED = -22.0
    SPAM = -20.0
    FORWARD_CHAIN = -10.0
    SCAM = -28.0
    DUPLICATE = -12.0
    REPORTED_SENDER = -18.0
    BLOCKED_SENDER = -30.0
    MUTED = -15.0
    QUIET_HOURS = -6.0
    BURST = -8.0
    FRIENDS_ENGAGEMENT = 9.0
    MEDIA_LOW_CONFIDENCE = -3.0


# --------------------------------------------------------------------------- #
# Rule engine
# --------------------------------------------------------------------------- #


class RuleEngine:
    """Evaluates a message against every rule family and returns a :class:`RuleEvaluation`.

    Parameters
    ----------
    config:
        Application configuration. Defaults to the process-wide singleton.
        Used for do-not-disturb hours, burst thresholds, and active-order
        windows via their respective sub-configs.

    Notes
    -----
    Every check degrades gracefully: a missing profile, missing relationship
    result, or missing OCR/ASR result simply causes the corresponding rule to
    be skipped rather than raising. The engine never fails a message.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or get_config()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        message: Message,
        context: MessageContext,
        relationship: RelationshipResult | None = None,
        content_verdict: ContentVerdict | None = None,
        ocr_result: OcrResult | None = None,
        asr_result: AsrResult | None = None,
    ) -> RuleEvaluation:
        """Evaluate every rule family for one message.

        Parameters
        ----------
        message:
            The message being routed.
        context:
            Assembled :class:`~src.retrieval.context.MessageContext` -- user,
            group, business profiles, event summaries, duplicate/report
            status, and the evidence pool.
        relationship:
            Fused relationship result for the sender, if the relationship
            engine has already run. When omitted, category rules fall back to
            the group's name-inferred hint and business presence.
        content_verdict:
            Pre-computed content analysis. Recomputed from the message when
            omitted.
        ocr_result:
            Raw OCR result for richer trace metadata on image messages.
        asr_result:
            Raw ASR result for richer trace metadata on voice messages.

        Returns
        -------
        RuleEvaluation
        """
        verdict = content_verdict or analyse_content(
            message.content,
            mentions_user=message.mentions_user(self._recipient_id(message, context)),
            is_reply_to_user=self._is_reply_to_user(message, context),
            message=message,
        )

        triggered: list[TriggeredRule] = []
        overrides: list[OverrideRecord] = []
        category_signals: list[str] = []

        recipient = self._recipient_id(message, context)
        mentions_user = message.mentions_user(recipient)
        is_reply_to_user = self._is_reply_to_user(message, context)

        category, category_confidence, category_source = self._resolve_category(
            relationship, context.group
        )
        if category is not RelationshipCategory.UNKNOWN:
            category_signals.append(
                f"relationship:{category.value}(conf={category_confidence:.2f},"
                f"source={category_source})"
            )

        # ---- System (structural) rules, evaluated first ------------------ #
        self._check_blocked_sender(message, context, triggered, overrides)
        self._check_muted_contact(message, context, mentions_user, triggered, overrides)
        self._check_muted_group(message, context, mentions_user, triggered, overrides)
        self._check_mention_and_reply(mentions_user, is_reply_to_user, triggered, overrides)
        self._check_pinned_contact(message, context, triggered, overrides)
        self._check_quiet_hours(message, context, triggered, overrides)
        self._check_trusted_admin(message, context, triggered)
        self._check_duplicate(context, triggered, overrides)
        self._check_reported_sender(context, triggered, overrides)
        self._check_engagement_history(context, triggered)
        self._check_media_quality(message, ocr_result, asr_result, triggered)

        # ---- Emergency ------------------------------------------------- #
        self._check_otp(message, verdict, triggered, overrides)
        self._check_emergency_keyword(message, verdict, category, triggered, overrides)

        # ---- Relationship-anchored families ----------------------------- #
        self._check_family(message, category, category_confidence, triggered, overrides)
        self._check_office(message, verdict, category, triggered)
        self._check_college(message, verdict, category, context, triggered)
        self._check_friends(message, category, context, triggered)
        self._check_society(message, verdict, category, triggered)

        # ---- Content-domain families -------------------------------- #
        self._check_healthcare(message, verdict, context, triggered, overrides)
        self._check_finance(message, verdict, context, triggered, overrides)
        self._check_travel(message, verdict, triggered, overrides)

        # ---- Business / promotions / spam / scam --------------------- #
        self._check_business(message, verdict, context, triggered, overrides)
        self._check_promotions(message, verdict, context, triggered, overrides)
        self._check_spam(message, verdict, triggered, overrides)
        self._check_scam(message, verdict, triggered, overrides)
        self._check_forward_chain(message, verdict, category, triggered)

        suggested_type = self._suggest_message_type(verdict, triggered, message)
        metadata = self._build_metadata(
            message, context, verdict, category, category_confidence,
            mentions_user, is_reply_to_user, ocr_result, asr_result,
        )

        evaluation = RuleEvaluation(
            message_id=message.message_id,
            triggered_rules=tuple(triggered),
            overrides=tuple(overrides),
            category_signals=tuple(dict.fromkeys(category_signals)),
            suggested_message_type=suggested_type,
            metadata=metadata,
        )

        logger.debug(
            "RuleEngine: message_id=%s triggered=%d overrides=%d total_weight=%.1f",
            message.message_id,
            len(evaluation.triggered_rules),
            len(evaluation.overrides),
            evaluation.total_weight,
        )
        return evaluation

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _recipient_id(message: Message, context: MessageContext) -> str | None:
        """Resolve the receiving user's id from the message or its profile."""
        if message.recipient_user_id:
            return message.recipient_user_id
        if context.user is not None:
            return context.user.user_id
        return None

    @staticmethod
    def _is_reply_to_user(message: Message, context: MessageContext) -> bool:
        """Return whether this message replies to something the recipient wrote.

        Looks up ``reply_to_id`` in the retrieval evidence pool; if the
        replied-to message's sender is the recipient, this message is a
        direct reply to them.
        """
        if not message.reply_to_id:
            return False
        recipient = message.recipient_user_id
        if not recipient:
            return False
        for candidate in context.retrieval.candidates:
            if candidate.message_id == message.reply_to_id:
                return candidate.sender_id == str(recipient)
        return False

    def _resolve_category(
        self,
        relationship: RelationshipResult | None,
        group: GroupProfile | None,
    ) -> tuple[RelationshipCategory, float, str]:
        """Resolve the sender's relationship category, honouring precedence.

        Structured signals (business match, kinship-name match, as already
        encoded in ``relationship.method``) outrank inferred ones. When no
        fused :class:`RelationshipResult` is supplied, falls back to the
        group's name-inferred hint, explicitly labelled as inferred.

        Parameters
        ----------
        relationship:
            Fused relationship result, if available.
        group:
            Group profile for the conversation, if this is a group message.

        Returns
        -------
        tuple
            ``(category, confidence, source_label)``.
        """
        if relationship is not None and relationship.category is not RelationshipCategory.UNKNOWN:
            return relationship.category, relationship.confidence, relationship.method

        if group is not None and group.category_hint is not RelationshipCategory.UNKNOWN:
            return (
                group.category_hint,
                group.category_hint_confidence,
                "group_name_hint(inferred)",
            )

        return RelationshipCategory.UNKNOWN, 0.0, "unresolved"

    @staticmethod
    def _suggest_message_type(
        verdict: ContentVerdict,
        triggered: list[TriggeredRule],
        message: Message,
    ) -> MessageType:
        """Refine the content-level type suggestion using domain rule hits.

        Domain-specific rules (healthcare/finance/travel) carry more precise
        information than the generic content verdict, so a firing rule from
        one of those families overrides the generic ``TRANSACTIONAL`` guess.
        """
        family_priority = (
            RuleFamily.SCAM,
            RuleFamily.SPAM,
            RuleFamily.EMERGENCY,
            RuleFamily.HEALTHCARE,
            RuleFamily.FINANCE,
            RuleFamily.TRAVEL,
            RuleFamily.PROMOTIONS,
        )
        firing_families = {rule.family for rule in triggered}
        for family in family_priority:
            if family in firing_families:
                mapped = {
                    RuleFamily.SCAM: MessageType.SPAM,
                    RuleFamily.SPAM: MessageType.SPAM,
                    RuleFamily.EMERGENCY: MessageType.OTP if verdict.is_otp else MessageType.PERSONAL,
                    RuleFamily.HEALTHCARE: MessageType.TRANSACTIONAL,
                    RuleFamily.FINANCE: MessageType.TRANSACTIONAL,
                    RuleFamily.TRAVEL: MessageType.TRANSACTIONAL,
                    RuleFamily.PROMOTIONS: MessageType.PROMOTIONAL,
                }
                return mapped[family]
        return verdict.suggested_type if verdict.suggested_type != MessageType.OTHER else (
            MessageType.from_any(None) if False else verdict.suggested_type
        )

    def _build_metadata(
        self,
        message: Message,
        context: MessageContext,
        verdict: ContentVerdict,
        category: RelationshipCategory,
        category_confidence: float,
        mentions_user: bool,
        is_reply_to_user: bool,
        ocr_result: OcrResult | None,
        asr_result: AsrResult | None,
    ) -> dict[str, Any]:
        """Assemble the flat metadata dictionary for downstream scoring."""
        user = context.user
        group = context.group
        business = context.business
        sender_events: EventSummary = context.sender_event_summary

        metadata: dict[str, Any] = {
            "relationship_category": category.value,
            "relationship_confidence": round(category_confidence, 4),
            "mentions_user": mentions_user,
            "is_reply_to_user": is_reply_to_user,
            "is_group_message": message.is_group_message,
            "is_direct_message": not message.is_group_message,
            "is_duplicate": context.is_duplicate,
            "duplicate_count": context.duplicate_count,
            "is_reported_recently": context.is_reported_recently,
            "report_count": context.report_count,
            "sender_is_group_admin": context.sender_is_group_admin,
            "sender_reply_rate": round(sender_events.reply_rate, 4),
            "sender_open_rate": round(sender_events.open_rate, 4),
            "sender_dismissal_rate": round(sender_events.dismissal_rate, 4),
            "content_is_otp": verdict.is_otp,
            "content_is_transactional": verdict.is_transactional,
            "content_is_promotional": verdict.is_promotional,
            "content_is_spam": verdict.is_spam,
            "content_urgency_score": round(verdict.urgency_score, 4),
            "content_promo_score": round(verdict.promo_score, 4),
            "content_spam_score": round(verdict.spam_score, 4),
            "has_forward_marker": verdict.has_forward_marker or message.is_forwarded,
            "media_type": message.media_type.value,
            "media_quality": round(message.media_quality, 4),
            "is_media_only": message.has_media and not message.message_text.strip(),
        }

        if user is not None:
            metadata.update(
                {
                    "user_is_blocked_sender": message.sender_id in user.blocked_contacts,
                    "user_muted_sender": message.sender_id in user.muted_contacts,
                    "user_pinned_sender": message.sender_id in user.pinned_contacts,
                    "user_dnd_active": user.is_in_dnd(message.timestamp),
                }
            )

        if group is not None:
            metadata.update(
                {
                    "group_size": group.size,
                    "group_is_large": group.is_large,
                    "group_is_muted": group.is_muted,
                    "group_category_hint": group.category_hint.value,
                    "group_user_read_rate": round(group.user_read_rate, 4),
                }
            )

        if business is not None:
            metadata.update(
                {
                    "business_is_verified": business.is_verified,
                    "business_is_known_to_user": business.is_known_to_user,
                    "business_has_active_order": business.has_active_order,
                    "business_promo_ratio": round(business.promo_ratio, 4),
                    "business_txn_count": business.user_txn_count,
                }
            )

        if ocr_result is not None:
            metadata["ocr_success"] = ocr_result.success
            metadata["ocr_confidence"] = round(ocr_result.confidence, 4)
        if asr_result is not None:
            metadata["asr_success"] = asr_result.success
            metadata["asr_avg_logprob"] = round(asr_result.avg_logprob, 4)

        return metadata

    # ------------------------------------------------------------------ #
    # System family
    # ------------------------------------------------------------------ #

    def _check_blocked_sender(
        self,
        message: Message,
        context: MessageContext,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Force-mute a sender the user has explicitly blocked."""
        user = context.user
        if user is None or message.sender_id not in user.blocked_contacts:
            return
        triggered.append(
            TriggeredRule(
                rule_id="sys_blocked_sender",
                family=RuleFamily.SYSTEM,
                description="Sender is explicitly blocked by the user.",
                weight=_W.BLOCKED_SENDER,
                confidence=0.98,
                signals=("system:blocked_sender",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="sys_blocked_sender",
                effect=OverrideEffect.FORCE,
                bound=Action.MUTE,
                binding=True,
                confidence=0.98,
                note="User has blocked this sender.",
            )
        )

    def _check_muted_contact(
        self,
        message: Message,
        context: MessageContext,
        mentions_user: bool,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Ceiling-mute a muted 1:1 contact, unless the user is @-mentioned."""
        user = context.user
        if user is None or message.sender_id not in user.muted_contacts:
            return
        if mentions_user:
            return
        triggered.append(
            TriggeredRule(
                rule_id="sys_muted_contact",
                family=RuleFamily.SYSTEM,
                description="Sender is a contact the user has muted.",
                weight=_W.MUTED,
                confidence=0.90,
                signals=("system:muted_contact",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="sys_muted_contact",
                effect=OverrideEffect.CEILING,
                bound=Action.MUTE,
                binding=True,
                confidence=0.85,
                note="User has muted this contact and is not mentioned.",
            )
        )

    def _check_muted_group(
        self,
        message: Message,
        context: MessageContext,
        mentions_user: bool,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Ceiling-mute a muted group, unless the user is @-mentioned."""
        user = context.user
        group = context.group
        is_muted = bool(group and group.is_muted) or bool(
            user and message.group_id and message.group_id in user.muted_groups
        )
        if not message.is_group_message or not is_muted or mentions_user:
            return
        triggered.append(
            TriggeredRule(
                rule_id="sys_muted_group",
                family=RuleFamily.SYSTEM,
                description="Group is muted and the user is not mentioned.",
                weight=_W.MUTED,
                confidence=0.85,
                signals=("system:muted_group",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="sys_muted_group",
                effect=OverrideEffect.CEILING,
                bound=Action.MUTE,
                binding=True,
                confidence=0.80,
                note="Group is muted and the user is not mentioned.",
            )
        )

    def _check_mention_and_reply(
        self,
        mentions_user: bool,
        is_reply_to_user: bool,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Floor-digest any message that directly addresses the user."""
        if mentions_user:
            triggered.append(
                TriggeredRule(
                    rule_id="sys_mention_floor",
                    family=RuleFamily.SYSTEM,
                    description="User is explicitly @-mentioned.",
                    weight=_W.MENTION,
                    confidence=0.95,
                    signals=("system:mention_of_user",),
                )
            )
            overrides.append(
                OverrideRecord(
                    rule_id="sys_mention_floor",
                    effect=OverrideEffect.FLOOR,
                    bound=Action.DIGEST,
                    binding=True,
                    confidence=0.90,
                    note="User is @-mentioned; should not be fully muted.",
                )
            )
        if is_reply_to_user:
            triggered.append(
                TriggeredRule(
                    rule_id="sys_reply_to_user_floor",
                    family=RuleFamily.SYSTEM,
                    description="Message directly replies to something the user wrote.",
                    weight=_W.REPLY_TO_USER,
                    confidence=0.85,
                    signals=("system:reply_to_user",),
                )
            )
            overrides.append(
                OverrideRecord(
                    rule_id="sys_reply_to_user_floor",
                    effect=OverrideEffect.FLOOR,
                    bound=Action.DIGEST,
                    binding=True,
                    confidence=0.80,
                    note="Direct reply to the user's own message.",
                )
            )

    def _check_pinned_contact(
        self,
        message: Message,
        context: MessageContext,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Floor-digest a message from a contact the user has pinned or starred."""
        user = context.user
        if user is None or message.sender_id not in user.pinned_contacts:
            return
        triggered.append(
            TriggeredRule(
                rule_id="sys_pinned_contact",
                family=RuleFamily.SYSTEM,
                description="Sender is a contact the user has pinned or starred.",
                weight=_W.PINNED_CONTACT,
                confidence=0.80,
                signals=("system:pinned_contact",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="sys_pinned_contact",
                effect=OverrideEffect.FLOOR,
                bound=Action.DIGEST,
                binding=True,
                confidence=0.75,
                note="User has pinned this contact.",
            )
        )

    def _check_quiet_hours(
        self,
        message: Message,
        context: MessageContext,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Ceiling-digest messages arriving during the user's quiet hours.

        Recorded as a ceiling only; whether an emergency floor should win is
        left to the arbiter's documented conflict rule (floors beat ceilings).
        """
        user = context.user
        routing = self._config.routing
        start_hour = user.dnd_start_hour if user else routing.dnd_start_hour
        end_hour = user.dnd_end_hour if user else routing.dnd_end_hour
        in_dnd = user.is_in_dnd(message.timestamp) if user else _in_window(
            message.timestamp.hour, start_hour, end_hour
        )
        if not in_dnd:
            return
        triggered.append(
            TriggeredRule(
                rule_id="sys_quiet_hours",
                family=RuleFamily.SYSTEM,
                description=f"Message arrived during quiet hours ({start_hour:02d}:00-{end_hour:02d}:00).",
                weight=_W.QUIET_HOURS,
                confidence=0.70,
                signals=(f"system:quiet_hours({start_hour:02d}-{end_hour:02d})",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="sys_quiet_hours",
                effect=OverrideEffect.CEILING,
                bound=Action.DIGEST,
                binding=True,
                confidence=0.65,
                note="Arrived during the user's configured quiet hours.",
            )
        )

    def _check_trusted_admin(
        self,
        message: Message,
        context: MessageContext,
        triggered: list[TriggeredRule],
    ) -> None:
        """Weight-boost group announcements from a trusted group admin."""
        if not message.is_group_message or not context.sender_is_group_admin:
            return
        triggered.append(
            TriggeredRule(
                rule_id="sys_trusted_admin",
                family=RuleFamily.SYSTEM,
                description="Sender is an admin of this group.",
                weight=_W.ADMIN_ANNOUNCEMENT,
                confidence=0.65,
                signals=("system:trusted_admin",),
            )
        )

    def _check_duplicate(
        self,
        context: MessageContext,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Ceiling-mute an exact duplicate of a recently seen message."""
        if not context.is_duplicate:
            return
        triggered.append(
            TriggeredRule(
                rule_id="sys_duplicate_message",
                family=RuleFamily.SYSTEM,
                description=(
                    f"Identical message already seen {context.duplicate_count} time(s) recently."
                ),
                weight=_W.DUPLICATE,
                confidence=0.75,
                evidence_message_ids=context.duplicate_evidence_ids,
                signals=(f"system:duplicate_count={context.duplicate_count}",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="sys_duplicate_message",
                effect=OverrideEffect.CEILING,
                bound=Action.DIGEST,
                binding=True,
                confidence=0.70,
                note="Exact duplicate seen recently from the same sender.",
            )
        )

    def _check_reported_sender(
        self,
        context: MessageContext,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Ceiling-mute a sender the user has reported or blocked in event history."""
        if not context.is_reported_recently:
            return
        triggered.append(
            TriggeredRule(
                rule_id="sys_reported_sender",
                family=RuleFamily.SYSTEM,
                description=f"Sender has been reported {context.report_count} time(s) by this user.",
                weight=_W.REPORTED_SENDER,
                confidence=0.85,
                signals=(f"system:report_count={context.report_count}",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="sys_reported_sender",
                effect=OverrideEffect.CEILING,
                bound=Action.MUTE,
                binding=True,
                confidence=0.80,
                note="User has previously reported this sender.",
            )
        )

    def _check_engagement_history(
        self,
        context: MessageContext,
        triggered: list[TriggeredRule],
    ) -> None:
        """Weight messages by the user's historical reply/dismissal pattern with this sender."""
        summary = context.sender_event_summary
        if summary.total_events < 3:
            return
        if summary.reply_rate >= 0.5:
            triggered.append(
                TriggeredRule(
                    rule_id="sys_high_reply_history",
                    family=RuleFamily.SYSTEM,
                    description=f"User replies to this sender {summary.reply_rate:.0%} of the time.",
                    weight=_W.HIGH_REPLY_HISTORY,
                    confidence=0.60,
                    signals=(f"system:reply_rate={summary.reply_rate:.2f}",),
                )
            )
        if summary.dismissal_rate >= 0.5:
            triggered.append(
                TriggeredRule(
                    rule_id="sys_high_dismissal_history",
                    family=RuleFamily.SYSTEM,
                    description=f"User dismisses this sender's messages {summary.dismissal_rate:.0%} of the time.",
                    weight=_W.HIGH_DISMISSAL_HISTORY,
                    confidence=0.60,
                    signals=(f"system:dismissal_rate={summary.dismissal_rate:.2f}",),
                )
            )

    def _check_media_quality(
        self,
        message: Message,
        ocr_result: OcrResult | None,
        asr_result: AsrResult | None,
        triggered: list[TriggeredRule],
    ) -> None:
        """Flag low-confidence image-only or voice-only messages for cautious handling."""
        if not message.has_media:
            return
        low_quality = message.media_quality < 0.4
        no_text_extracted = not message.ocr_text.strip() and not message.asr_text.strip()
        if not (low_quality or no_text_extracted):
            return

        source = "image" if message.media_type.value == "image" else "voice"
        detail = []
        if ocr_result is not None and not ocr_result.success:
            detail.append(f"ocr_error={ocr_result.error}")
        if asr_result is not None and not asr_result.success:
            detail.append(f"asr_error={asr_result.error}")

        triggered.append(
            TriggeredRule(
                rule_id="sys_media_low_confidence",
                family=RuleFamily.SYSTEM,
                description=(
                    f"{source.capitalize()} message could not be reliably transcribed "
                    "(low confidence or no text extracted)."
                ),
                weight=_W.MEDIA_LOW_CONFIDENCE,
                confidence=0.50,
                signals=tuple([f"system:media_quality={message.media_quality:.2f}"] + detail),
            )
        )

    # ------------------------------------------------------------------ #
    # Emergency family
    # ------------------------------------------------------------------ #

    def _check_otp(
        self,
        message: Message,
        verdict: ContentVerdict,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Floor-notify a detected one-time password, regardless of source."""
        if not verdict.is_otp:
            return
        source = "typed text"
        if message.media_type.value == "image":
            source = "image (OCR)"
        elif message.media_type.value == "voice":
            source = "voice note (ASR)"

        triggered.append(
            TriggeredRule(
                rule_id="emergency_otp",
                family=RuleFamily.EMERGENCY,
                description=f"One-time password detected in {source}.",
                weight=_W.OTP,
                confidence=0.95,
                evidence_message_ids=(message.message_id,),
                signals=tuple(f"content:{term}" for term in verdict.matched_terms[:3]),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="emergency_otp",
                effect=OverrideEffect.FLOOR,
                bound=Action.NOTIFY,
                binding=True,
                confidence=0.95,
                note="OTP / verification code detected.",
            )
        )

    def _check_emergency_keyword(
        self,
        message: Message,
        verdict: ContentVerdict,
        category: RelationshipCategory,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Floor-notify explicit emergency language from a personal contact."""
        match = EMERGENCY_PATTERN.search(message.content)
        if not match:
            return

        is_personal = category in (RelationshipCategory.FAMILY, RelationshipCategory.CLOSE_FRIEND)
        confidence = 0.85 if is_personal else 0.55

        triggered.append(
            TriggeredRule(
                rule_id="emergency_keyword",
                family=RuleFamily.EMERGENCY,
                description=f"Emergency language detected ('{match.group(0).lower()}').",
                weight=_W.EMERGENCY_KEYWORD if is_personal else _W.EMERGENCY_KEYWORD * 0.5,
                confidence=confidence,
                evidence_message_ids=(message.message_id,),
                signals=(f"content:emergency_term={match.group(0).lower()}",),
            )
        )

        if is_personal:
            overrides.append(
                OverrideRecord(
                    rule_id="emergency_keyword",
                    effect=OverrideEffect.FLOOR,
                    bound=Action.NOTIFY,
                    binding=True,
                    confidence=0.80,
                    note="Emergency language from a family member or close friend.",
                )
            )

    # ------------------------------------------------------------------ #
    # Family
    # ------------------------------------------------------------------ #

    def _check_family(
        self,
        message: Message,
        category: RelationshipCategory,
        category_confidence: float,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Floor-digest direct messages from a Family-category sender."""
        if category is not RelationshipCategory.FAMILY:
            return
        weight = _W.DIRECT_FAMILY if not message.is_group_message else _W.DIRECT_FAMILY * 0.6
        triggered.append(
            TriggeredRule(
                rule_id="family_direct_message",
                family=RuleFamily.FAMILY,
                description="Sender is classified as Family.",
                weight=weight,
                confidence=category_confidence,
                evidence_message_ids=(message.message_id,),
                signals=("relationship:family",),
            )
        )
        if not message.is_group_message and category_confidence >= 0.45:
            overrides.append(
                OverrideRecord(
                    rule_id="family_direct_message",
                    effect=OverrideEffect.FLOOR,
                    bound=Action.DIGEST,
                    binding=True,
                    confidence=min(category_confidence, 0.85),
                    note="Direct message from a family member.",
                )
            )

    # ------------------------------------------------------------------ #
    # Office
    # ------------------------------------------------------------------ #

    def _check_office(
        self,
        message: Message,
        verdict: ContentVerdict,
        category: RelationshipCategory,
        triggered: list[TriggeredRule],
    ) -> None:
        """Weight office-category requests during working hours; deprioritise idle chatter after hours."""
        if category is not RelationshipCategory.OFFICE:
            return

        from src.features.temporal import is_working_hours  # local import: avoids a hard module-level

        working = is_working_hours(message.timestamp)
        if working and (verdict.is_request or verdict.is_question or verdict.has_deadline):
            triggered.append(
                TriggeredRule(
                    rule_id="office_working_hours_request",
                    family=RuleFamily.OFFICE,
                    description="Work-related request or question during working hours.",
                    weight=_W.OFFICE_WORK_REQUEST,
                    confidence=0.65,
                    evidence_message_ids=(message.message_id,),
                    signals=("relationship:office", "content:request_or_question"),
                )
            )
        elif not working and not verdict.has_deadline and verdict.urgency_score < 0.3:
            triggered.append(
                TriggeredRule(
                    rule_id="office_after_hours_deprioritize",
                    family=RuleFamily.OFFICE,
                    description="Non-urgent office message arriving outside working hours.",
                    weight=_W.OFFICE_AFTER_HOURS,
                    confidence=0.50,
                    signals=("relationship:office", "system:after_hours"),
                )
            )

    # ------------------------------------------------------------------ #
    # College
    # ------------------------------------------------------------------ #

    def _check_college(
        self,
        message: Message,
        verdict: ContentVerdict,
        category: RelationshipCategory,
        context: MessageContext,
        triggered: list[TriggeredRule],
    ) -> None:
        """Weight deadline-bearing college messages up; large noisy class groups down."""
        if category is not RelationshipCategory.COLLEGE:
            return

        if verdict.has_deadline or verdict.is_reminder:
            triggered.append(
                TriggeredRule(
                    rule_id="college_deadline",
                    family=RuleFamily.COLLEGE,
                    description="College-related deadline or reminder detected.",
                    weight=_W.COLLEGE_DEADLINE,
                    confidence=0.60,
                    evidence_message_ids=(message.message_id,),
                    signals=("relationship:college", "content:deadline_or_reminder"),
                )
            )

        group = context.group
        if group is not None and group.is_large and not message.mentions_user(
            context.user.user_id if context.user else None
        ):
            triggered.append(
                TriggeredRule(
                    rule_id="college_large_group_noise",
                    family=RuleFamily.COLLEGE,
                    description=f"Large college group ({group.size} members) with no direct mention.",
                    weight=_W.COLLEGE_LARGE_GROUP_NOISE,
                    confidence=0.55,
                    signals=(f"group:size={group.size}",),
                )
            )

    # ------------------------------------------------------------------ #
    # Friends
    # ------------------------------------------------------------------ #

    def _check_friends(
        self,
        message: Message,
        category: RelationshipCategory,
        context: MessageContext,
        triggered: list[TriggeredRule],
    ) -> None:
        """Weight-boost messages from a high-engagement close friend."""
        if category is not RelationshipCategory.CLOSE_FRIEND:
            return

        summary = context.sender_event_summary
        weight = _W.DIRECT_CLOSE_FRIEND
        confidence = 0.55
        signals = ["relationship:close_friend"]

        if summary.reply_rate >= 0.5 and summary.total_events >= 3:
            weight += _W.FRIENDS_ENGAGEMENT * 0.5
            confidence = 0.65
            signals.append(f"engagement:reply_rate={summary.reply_rate:.2f}")

        triggered.append(
            TriggeredRule(
                rule_id="friends_high_engagement",
                family=RuleFamily.FRIENDS,
                description="Sender is a close friend with an active conversation history.",
                weight=weight,
                confidence=confidence,
                evidence_message_ids=(message.message_id,),
                signals=tuple(signals),
            )
        )

    # ------------------------------------------------------------------ #
    # Society
    # ------------------------------------------------------------------ #

    def _check_society(
        self,
        message: Message,
        verdict: ContentVerdict,
        category: RelationshipCategory,
        triggered: list[TriggeredRule],
    ) -> None:
        """Weight urgent society/security notices up, routine notices down."""
        if category is not RelationshipCategory.SOCIETY:
            return

        society_match = SOCIETY_CONTENT_PATTERN.search(message.content)
        if not society_match:
            return

        is_urgent = verdict.urgency_score >= 0.4 or bool(
            re.search(r"\b(security|guard|urgent|alert|emergency)\b", message.content, re.IGNORECASE)
        )
        if is_urgent:
            triggered.append(
                TriggeredRule(
                    rule_id="society_urgent_security",
                    family=RuleFamily.SOCIETY,
                    description=f"Urgent society/security notice ('{society_match.group(0).lower()}').",
                    weight=_W.SOCIETY_URGENT_SECURITY,
                    confidence=0.60,
                    evidence_message_ids=(message.message_id,),
                    signals=("relationship:society", "content:security_notice"),
                )
            )
        else:
            triggered.append(
                TriggeredRule(
                    rule_id="society_routine_notice",
                    family=RuleFamily.SOCIETY,
                    description="Routine society notice (maintenance, utilities, etc.).",
                    weight=_W.SOCIETY_NOTICE,
                    confidence=0.50,
                    signals=("relationship:society", "content:routine_notice"),
                )
            )

    # ------------------------------------------------------------------ #
    # Healthcare
    # ------------------------------------------------------------------ #

    def _check_healthcare(
        self,
        message: Message,
        verdict: ContentVerdict,
        context: MessageContext,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Floor-digest healthcare appointments/results; boost weight for critical ones."""
        match = HEALTHCARE_PATTERN.search(message.content)
        if not match:
            return

        is_critical = verdict.urgency_score >= 0.5 or bool(
            re.search(r"\b(critical|abnormal|urgent|immediately)\b", message.content, re.IGNORECASE)
        )
        weight = _W.HEALTHCARE_CRITICAL if is_critical else _W.HEALTHCARE_APPOINTMENT
        rule_id = "healthcare_critical_result" if is_critical else "healthcare_appointment_reminder"

        triggered.append(
            TriggeredRule(
                rule_id=rule_id,
                family=RuleFamily.HEALTHCARE,
                description=f"Healthcare-related message detected ('{match.group(0).lower()}').",
                weight=weight,
                confidence=0.65 if is_critical else 0.55,
                evidence_message_ids=(message.message_id,),
                signals=(f"content:healthcare_term={match.group(0).lower()}",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id=rule_id,
                effect=OverrideEffect.FLOOR,
                bound=Action.NOTIFY if is_critical else Action.DIGEST,
                binding=True,
                confidence=0.60,
                note="Healthcare-related message should not be silently muted.",
            )
        )

    # ------------------------------------------------------------------ #
    # Finance
    # ------------------------------------------------------------------ #

    def _check_finance(
        self,
        message: Message,
        verdict: ContentVerdict,
        context: MessageContext,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Floor-digest payment-due and transaction-alert messages from finance sources."""
        match = FINANCE_PATTERN.search(message.content)
        if not match:
            return

        is_payment_due = bool(
            re.search(r"\b(due|overdue|last date|emi|minimum due)\b", message.content, re.IGNORECASE)
        ) and verdict.has_deadline
        rule_id = "finance_payment_reminder" if is_payment_due else "finance_transaction_alert"
        weight = _W.FINANCE_PAYMENT_DUE if is_payment_due else _W.FINANCE_TRANSACTION_ALERT

        triggered.append(
            TriggeredRule(
                rule_id=rule_id,
                family=RuleFamily.FINANCE,
                description=f"Finance-related message detected ('{match.group(0).lower()}').",
                weight=weight,
                confidence=0.65,
                evidence_message_ids=(message.message_id,),
                signals=(f"content:finance_term={match.group(0).lower()}",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id=rule_id,
                effect=OverrideEffect.FLOOR,
                bound=Action.DIGEST,
                binding=True,
                confidence=0.60,
                note="Financial message should not be silently muted.",
            )
        )

    # ------------------------------------------------------------------ #
    # Travel
    # ------------------------------------------------------------------ #

    def _check_travel(
        self,
        message: Message,
        verdict: ContentVerdict,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Floor-digest travel bookings; floor-notify imminent check-in deadlines."""
        match = TRAVEL_PATTERN.search(message.content)
        if not match:
            return

        is_checkin = bool(
            re.search(r"\bcheck[\s-]?in|gate closes|boarding\b", message.content, re.IGNORECASE)
        ) and verdict.has_deadline
        rule_id = "travel_checkin_reminder" if is_checkin else "travel_booking_confirmation"
        weight = _W.TRAVEL_CHECKIN if is_checkin else _W.TRAVEL_BOOKING

        triggered.append(
            TriggeredRule(
                rule_id=rule_id,
                family=RuleFamily.TRAVEL,
                description=f"Travel-related message detected ('{match.group(0).lower()}').",
                weight=weight,
                confidence=0.65,
                evidence_message_ids=(message.message_id,),
                signals=(f"content:travel_term={match.group(0).lower()}",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id=rule_id,
                effect=OverrideEffect.FLOOR,
                bound=Action.NOTIFY if is_checkin else Action.DIGEST,
                binding=True,
                confidence=0.60,
                note="Time-sensitive travel information should not be silently muted.",
            )
        )

    # ------------------------------------------------------------------ #
    # Business
    # ------------------------------------------------------------------ #

    def _check_business(
        self,
        message: Message,
        verdict: ContentVerdict,
        context: MessageContext,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Floor-digest verified/trusted business updates; penalise unverified cold outreach."""
        business = context.business
        if business is None and not message.business_id and not message.is_from_business:
            return

        if business is not None and business.has_active_order and verdict.is_transactional:
            triggered.append(
                TriggeredRule(
                    rule_id="business_active_order_update",
                    family=RuleFamily.BUSINESS,
                    description="Transactional update from a business with an active order.",
                    weight=_W.BUSINESS_ACTIVE_ORDER,
                    confidence=0.75,
                    evidence_message_ids=(message.message_id,),
                    signals=("business:active_order", "business:transactional"),
                )
            )
            overrides.append(
                OverrideRecord(
                    rule_id="business_active_order_update",
                    effect=OverrideEffect.FLOOR,
                    bound=Action.DIGEST,
                    binding=True,
                    confidence=0.70,
                    note="Active order with this business; update should not be muted.",
                )
            )
        elif business is not None and business.is_known_to_user and verdict.is_transactional:
            triggered.append(
                TriggeredRule(
                    rule_id="business_known_wanted_update",
                    family=RuleFamily.BUSINESS,
                    description="Transactional update from a business the user has purchased from before.",
                    weight=_W.BUSINESS_WANTED_UPDATE,
                    confidence=0.60,
                    evidence_message_ids=(message.message_id,),
                    signals=("business:known_to_user", "business:transactional"),
                )
            )
        elif business is None or not business.is_known_to_user:
            if verdict.is_promotional or verdict.is_transactional:
                triggered.append(
                    TriggeredRule(
                        rule_id="business_unverified_cold_outreach",
                        family=RuleFamily.BUSINESS,
                        description="Message from a business account the user has no history with.",
                        weight=_W.BUSINESS_UNVERIFIED_COLD,
                        confidence=0.55,
                        signals=("business:unknown_to_user",),
                    )
                )

    # ------------------------------------------------------------------ #
    # Promotions
    # ------------------------------------------------------------------ #

    def _check_promotions(
        self,
        message: Message,
        verdict: ContentVerdict,
        context: MessageContext,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Ceiling-mute repeated cold promotions; ceiling-digest a first-time promo."""
        if not verdict.is_promotional:
            return

        business = context.business
        is_repeated = bool(business is not None and business.promo_ratio >= 0.5) or context.is_duplicate
        is_cold = business is None or not business.is_known_to_user

        if is_repeated and is_cold:
            triggered.append(
                TriggeredRule(
                    rule_id="promotions_repeated_cold",
                    family=RuleFamily.PROMOTIONS,
                    description="Repeated promotional content from an account with no purchase history.",
                    weight=_W.PROMOTIONS_REPEATED,
                    confidence=0.75,
                    evidence_message_ids=(message.message_id,),
                    signals=("content:promotional", "business:repeated_cold"),
                )
            )
            overrides.append(
                OverrideRecord(
                    rule_id="promotions_repeated_cold",
                    effect=OverrideEffect.CEILING,
                    bound=Action.MUTE,
                    binding=True,
                    confidence=0.65,
                    note="Repeated cold promotional traffic.",
                )
            )
        else:
            triggered.append(
                TriggeredRule(
                    rule_id="promotions_cold",
                    family=RuleFamily.PROMOTIONS,
                    description="Promotional content detected.",
                    weight=_W.PROMOTIONS_COLD,
                    confidence=0.60,
                    evidence_message_ids=(message.message_id,),
                    signals=("content:promotional",),
                )
            )
            overrides.append(
                OverrideRecord(
                    rule_id="promotions_cold",
                    effect=OverrideEffect.CEILING,
                    bound=Action.DIGEST,
                    binding=True,
                    confidence=0.55,
                    note="Promotional content capped below notify.",
                )
            )

    # ------------------------------------------------------------------ #
    # Spam
    # ------------------------------------------------------------------ #

    def _check_spam(
        self,
        message: Message,
        verdict: ContentVerdict,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Ceiling-mute detected spam that does not reach scam-level confidence."""
        if not verdict.is_spam:
            return
        if SCAM_HIGH_CONFIDENCE_PATTERN.search(message.content):
            # Handled by _check_scam with a stronger override; avoid double-counting.
            return

        triggered.append(
            TriggeredRule(
                rule_id="spam_detected",
                family=RuleFamily.SPAM,
                description="Message matches spam patterns.",
                weight=_W.SPAM,
                confidence=0.70,
                evidence_message_ids=(message.message_id,),
                signals=tuple(f"content:{term}" for term in verdict.matched_terms[:3]),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="spam_detected",
                effect=OverrideEffect.CEILING,
                bound=Action.MUTE,
                binding=True,
                confidence=0.65,
                note="Spam pattern match.",
            )
        )

    def _check_forward_chain(
        self,
        message: Message,
        verdict: ContentVerdict,
        category: RelationshipCategory,
        triggered: list[TriggeredRule],
    ) -> None:
        """Weight-penalise chain-forward messages, softened for trusted personal contacts."""
        if not (verdict.has_forward_marker or message.is_forwarded):
            return
        is_trusted = category in (RelationshipCategory.FAMILY, RelationshipCategory.CLOSE_FRIEND)
        weight = _W.FORWARD_CHAIN * (0.4 if is_trusted else 1.0)

        triggered.append(
            TriggeredRule(
                rule_id="spam_forward_chain",
                family=RuleFamily.SPAM,
                description="Message shows forward-chain markers.",
                weight=weight,
                confidence=0.55 if is_trusted else 0.70,
                evidence_message_ids=(message.message_id,),
                signals=("content:forward_marker",),
            )
        )

    # ------------------------------------------------------------------ #
    # Scam
    # ------------------------------------------------------------------ #

    def _check_scam(
        self,
        message: Message,
        verdict: ContentVerdict,
        triggered: list[TriggeredRule],
        overrides: list[OverrideRecord],
    ) -> None:
        """Force-mute high-confidence scam patterns, regardless of any floor.

        This is the one family that records a :attr:`~src.schema.OverrideEffect.FORCE`
        rather than a ceiling: the arbiter's documented conflict rule treats
        force as outranking a floor, which is the intended behaviour here --
        an OTP-shaped message that is *also* a KYC-suspension scam should not
        be forced to notify.
        """
        match = SCAM_HIGH_CONFIDENCE_PATTERN.search(message.content)
        if not match and verdict.spam_score < 0.75:
            return

        matched_text = match.group(0).lower() if match else "high_spam_score"
        triggered.append(
            TriggeredRule(
                rule_id="scam_high_confidence",
                family=RuleFamily.SCAM,
                description=f"High-confidence scam pattern detected ('{matched_text}').",
                weight=_W.SCAM,
                confidence=0.85,
                evidence_message_ids=(message.message_id,),
                signals=(f"content:scam_pattern={matched_text}",),
            )
        )
        overrides.append(
            OverrideRecord(
                rule_id="scam_high_confidence",
                effect=OverrideEffect.FORCE,
                bound=Action.MUTE,
                binding=True,
                confidence=0.85,
                note="High-confidence scam pattern; forced mute overrides other floors.",
            )
        )


def _in_window(hour: int, start_hour: int, end_hour: int) -> bool:
    """Standalone DND-window check, mirroring :meth:`UserProfile.is_in_dnd`.

    Used only as a fallback when no :class:`UserProfile` is available, so this
    module does not require importing the temporal feature module for a
    one-line calculation.
    """
    if start_hour <= end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


__all__ = [
    "RuleEngine",
    "RuleEvaluation",
    "RuleFamily",
    "TriggeredRule",
]