"""
The arbiter: converts a PriorityAssessment into a submission-ready Decision.

This is the only module in the pipeline that chooses ``notify``/``digest``/
``mute``. It does not re-score: it reads the priority engine's 0-100 score and
the rule engine's override constraints, resolves them deterministically, and
applies a small set of named contextual exceptions (quiet hours, trusted
admins, verified businesses, direct mentions) before producing a
:class:`~src.schema.Decision`.

Conflict resolution
--------------------
1. A :attr:`~src.schema.OverrideEffect.FORCE` constraint is absolute and short-
   circuits everything else.
2. Otherwise, the priority score is banded by threshold, then a
   :attr:`~src.schema.OverrideEffect.CEILING` is applied, then a
   :attr:`~src.schema.OverrideEffect.FLOOR` is applied. Applying the floor
   last means a floor always wins a floor-vs-ceiling conflict, matching the
   frozen architecture's documented "floors beat ceilings" rule.
3. A small set of named contextual exceptions run last and only ever *relax*
   a ``mute`` to ``digest`` or a score-derived ``notify`` to ``digest`` during
   quiet hours -- they never strengthen a decision past what step 2 produced,
   and never run at all when a FORCE override is present.

Dependencies
------------
``src.config``, ``src.schema``, ``src.retrieval.context``, ``src.rules.engine``,
``src.scoring.priority``, ``src.decision.evidence``, ``src.decision.confidence``,
``src.media.ocr``, ``src.media.asr``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from src.config import AppConfig, get_config
from src.decision.confidence import CalibratedConfidence, ConfidenceCalibrator
from src.decision.evidence import EvidenceSelection, select_evidence
from src.media.asr import AsrResult
from src.media.ocr import OcrResult
from src.retrieval.context import MessageContext
from src.rules.engine import RuleEvaluation, RuleFamily
from src.schema import (
    Action,
    Decision,
    DecisionSource,
    Message,
    OverrideEffect,
    OverrideRecord,
    RelationshipCategory,
    ScoreTrace,
)
from src.scoring.priority import PriorityAssessment

logger = logging.getLogger(__name__)

#: Human-readable noun phrase per relationship category, for reason text.
_CATEGORY_NOUN: dict[str, str] = {
    "Family": "family member",
    "Close Friend": "close friend",
    "Office": "office contact",
    "College": "college contact",
    "Society": "society contact",
    "Business": "business contact",
    "Unknown": "sender",
}


def _band_from_score(score: float, thresholds: tuple[float, float]) -> Action:
    """Map a 0-100 priority score onto an :class:`~src.schema.Action` band.

    Parameters
    ----------
    score:
        Priority score, expected in ``[0, 100]``.
    thresholds:
        ``(mute_digest_cut, digest_notify_cut)``.

    Returns
    -------
    Action
    """
    low, high = thresholds
    if score >= high:
        return Action.NOTIFY
    if score >= low:
        return Action.DIGEST
    return Action.MUTE


@dataclass
class BandResolution:
    """Result of resolving the priority score against the rule constraints.

    Attributes
    ----------
    band:
        The resolved action, before contextual exceptions.
    threshold_band:
        The band the raw score alone would have produced.
    applied_overrides:
        The :class:`~src.schema.OverrideRecord` objects that changed the band.
    has_force:
        ``True`` when a FORCE override determined the band.
    """

    band: Action
    threshold_band: Action
    applied_overrides: tuple[OverrideRecord, ...]
    has_force: bool


def resolve_band(
    priority_score: float,
    overrides: tuple[OverrideRecord, ...],
    thresholds: tuple[float, float],
) -> BandResolution:
    """Resolve the final band from the score and the rule engine's constraints.

    Parameters
    ----------
    priority_score:
        The priority engine's final 0-100 score.
    overrides:
        Constraints recorded by the rule engine.
    thresholds:
        ``(mute_digest_cut, digest_notify_cut)`` for threshold banding.

    Returns
    -------
    BandResolution
    """
    threshold_band = _band_from_score(priority_score, thresholds)
    applied: list[OverrideRecord] = []

    forces = [o for o in overrides if o.effect is OverrideEffect.FORCE]
    if forces:
        strongest = max(forces, key=lambda o: o.confidence)
        return BandResolution(
            band=strongest.bound,
            threshold_band=threshold_band,
            applied_overrides=(strongest,),
            has_force=True,
        )

    band = threshold_band

    ceilings = [o for o in overrides if o.effect is OverrideEffect.CEILING]
    if ceilings:
        strongest_ceiling = min(ceilings, key=lambda o: o.bound.rank)
        if strongest_ceiling.bound.rank < band.rank:
            band = strongest_ceiling.bound
            applied.append(strongest_ceiling)

    # Floors are applied last: a floor always wins a floor-vs-ceiling conflict.
    floors = [o for o in overrides if o.effect is OverrideEffect.FLOOR]
    if floors:
        strongest_floor = max(floors, key=lambda o: o.bound.rank)
        if strongest_floor.bound.rank > band.rank:
            band = strongest_floor.bound
            applied.append(strongest_floor)

    return BandResolution(
        band=band,
        threshold_band=threshold_band,
        applied_overrides=tuple(applied),
        has_force=False,
    )


def apply_context_exceptions(
    band: Action,
    rules: RuleEvaluation,
    has_force: bool,
) -> tuple[Action, tuple[str, ...]]:
    """Apply named contextual exceptions on top of the resolved band.

    These exceptions only ever relax a decision -- they never escalate one --
    and never run when a FORCE override already decided the band, since a
    force (e.g. a scam suppression) must be absolute.

    Supported exceptions
    ---------------------
    * Quiet hours: a purely score-derived ``notify`` (no explicit notify
      floor) arriving during the user's quiet hours is capped to ``digest``.
    * Trusted admin: a ``mute`` from a trusted group admin is relaxed to
      ``digest``.
    * Verified business: a ``mute`` on a verified business's transactional
      message is relaxed to ``digest``.
    * Direct mention: a ``mute`` on a message that @-mentions the user is
      relaxed to ``digest``, as a final guard even though the rule engine
      already records a mention floor.

    Parameters
    ----------
    band:
        The band produced by :func:`resolve_band`.
    rules:
        The rule evaluation, whose metadata and triggered rules drive the
        exception checks.
    has_force:
        Whether a FORCE override already determined ``band``.

    Returns
    -------
    tuple
        ``(final_band, notes)`` -- ``notes`` describes any exception applied.
    """
    if has_force:
        return band, ()

    notes: list[str] = []
    meta = rules.metadata

    quiet_hours_active = bool(meta.get("user_dnd_active")) or any(
        rule.rule_id == "sys_quiet_hours" for rule in rules.triggered_rules
    )
    notify_floor_present = any(
        override.effect is OverrideEffect.FLOOR and override.bound is Action.NOTIFY
        for override in rules.overrides
    )
    if band is Action.NOTIFY and quiet_hours_active and not notify_floor_present:
        band = Action.DIGEST
        notes.append("Quiet hours in effect: notify capped to digest.")

    if band is Action.MUTE and bool(meta.get("sender_is_group_admin")):
        band = Action.DIGEST
        notes.append("Sender is a trusted group admin: mute relaxed to digest.")

    is_verified_business = bool(meta.get("business_is_verified"))
    is_transactional = bool(meta.get("content_is_transactional"))
    if band is Action.MUTE and is_verified_business and is_transactional:
        band = Action.DIGEST
        notes.append("Verified business transactional update: mute relaxed to digest.")

    if band is Action.MUTE and bool(meta.get("mentions_user")):
        band = Action.DIGEST
        notes.append("User is directly mentioned: mute relaxed to digest.")

    return band, tuple(notes)


# --------------------------------------------------------------------------- #
# Reason generation
# --------------------------------------------------------------------------- #

#: A reason rule: a predicate over (rules, metadata, band) and the text it
#: produces. Evaluated top-down; the first match wins. Ordered most specific
#: and most severe first.
_ReasonPredicate = Callable[["RuleEvaluation", dict, Action], bool]
_ReasonTemplate = Callable[["RuleEvaluation", dict, Action], str]


def _has_rule(rules: RuleEvaluation, rule_id: str) -> bool:
    """Return whether a specific rule id fired."""
    return rules.has_rule(rule_id) if hasattr(rules, "has_rule") else any(
        r.rule_id == rule_id for r in rules.triggered_rules
    )


def _category_noun(meta: dict) -> str:
    """Return a human noun phrase for the sender's relationship category."""
    category = str(meta.get("relationship_category", "Unknown"))
    return _CATEGORY_NOUN.get(category, "sender")


def _reason_rules() -> list[tuple[_ReasonPredicate, _ReasonTemplate]]:
    """Build the ordered (predicate, template) list for reason generation.

    Rebuilt per call rather than module-level so lambdas close over nothing
    mutable; the cost is negligible relative to a single decision.

    Returns
    -------
    list of (predicate, template) pairs, most specific first.
    """
    return [
        (
            lambda r, m, b: any(rule.family is RuleFamily.SCAM for rule in r.triggered_rules),
            lambda r, m, b: "Potential phishing attempt detected.",
        ),
        (
            lambda r, m, b: _has_rule(r, "sys_blocked_sender"),
            lambda r, m, b: "Sender is blocked by the user.",
        ),
        (
            lambda r, m, b: _has_rule(r, "emergency_otp"),
            lambda r, m, b: "Time-sensitive verification code requires immediate attention.",
        ),
        (
            lambda r, m, b: _has_rule(r, "emergency_keyword")
            and str(m.get("relationship_category")) in ("Family", "Close Friend"),
            lambda r, m, b: f"Trusted {_category_noun(m)} sent a time-sensitive update.",
        ),
        (
            lambda r, m, b: _has_rule(r, "finance_payment_reminder")
            and bool(m.get("business_is_verified")),
            lambda r, m, b: "Verified business sent a payment reminder.",
        ),
        (
            lambda r, m, b: _has_rule(r, "finance_transaction_alert")
            and bool(m.get("business_is_verified")),
            lambda r, m, b: "Verified business sent a transaction alert.",
        ),
        (
            lambda r, m, b: _has_rule(r, "finance_payment_reminder"),
            lambda r, m, b: "Payment reminder detected.",
        ),
        (
            lambda r, m, b: _has_rule(r, "healthcare_critical_result"),
            lambda r, m, b: "Critical healthcare update requires attention.",
        ),
        (
            lambda r, m, b: _has_rule(r, "healthcare_appointment_reminder"),
            lambda r, m, b: "Healthcare appointment reminder detected.",
        ),
        (
            lambda r, m, b: _has_rule(r, "travel_checkin_reminder"),
            lambda r, m, b: "Time-sensitive travel check-in reminder.",
        ),
        (
            lambda r, m, b: _has_rule(r, "travel_booking_confirmation"),
            lambda r, m, b: "Travel booking confirmation received.",
        ),
        (
            lambda r, m, b: _has_rule(r, "promotions_repeated_cold"),
            lambda r, m, b: "Repeated promotional message previously dismissed.",
        ),
        (
            lambda r, m, b: any(rule.family is RuleFamily.PROMOTIONS for rule in r.triggered_rules),
            lambda r, m, b: "Promotional message from an unfamiliar sender.",
        ),
        (
            lambda r, m, b: _has_rule(r, "spam_detected"),
            lambda r, m, b: "Message matches known spam patterns.",
        ),
        (
            lambda r, m, b: _has_rule(r, "sys_duplicate_message"),
            lambda r, m, b: "Duplicate message already seen recently.",
        ),
        (
            lambda r, m, b: str(m.get("relationship_category")) == "Family"
            and b is not Action.MUTE,
            lambda r, m, b: "Trusted family member sent a message.",
        ),
        (
            lambda r, m, b: str(m.get("relationship_category")) == "Close Friend"
            and b is not Action.MUTE,
            lambda r, m, b: "Message from a close friend with an active conversation.",
        ),
        (
            lambda r, m, b: _has_rule(r, "office_working_hours_request"),
            lambda r, m, b: "Work-related request received during working hours.",
        ),
        (
            lambda r, m, b: _has_rule(r, "college_deadline"),
            lambda r, m, b: "College-related deadline or reminder detected.",
        ),
        (
            lambda r, m, b: _has_rule(r, "society_urgent_security"),
            lambda r, m, b: "Urgent society or security notice.",
        ),
        (
            lambda r, m, b: _has_rule(r, "business_active_order_update"),
            lambda r, m, b: "Business update related to an active order.",
        ),
        (
            lambda r, m, b: bool(m.get("mentions_user")),
            lambda r, m, b: "User was directly mentioned in the conversation.",
        ),
        (
            lambda r, m, b: b is Action.MUTE
            and (_has_rule(r, "sys_muted_contact") or _has_rule(r, "sys_muted_group")),
            lambda r, m, b: "Message muted per user preference.",
        ),
    ]


#: Fallback reason text keyed by message type value, used when no rule-based
#: template matched.
_FALLBACK_BY_TYPE: dict[str, str] = {
    "otp": "Verification code message detected.",
    "transactional": "Transactional update received.",
    "promotional": "Promotional content detected.",
    "reminder": "Reminder message detected.",
    "forward": "Forwarded message detected.",
    "spam": "Message flagged as spam.",
    "media_share": "Media message received.",
    "work": "Work-related message received.",
    "group_chat": "Group conversation activity.",
    "personal": "Personal message received.",
    "other": "Message received.",
}

#: Fallback reason text keyed by final band, used when even the type-based
#: fallback has nothing to say.
_FALLBACK_BY_BAND: dict[Action, str] = {
    Action.NOTIFY: "Message flagged as high priority.",
    Action.DIGEST: "Message queued for later review.",
    Action.MUTE: "Message classified as low priority.",
}


def build_reason(
    rules: RuleEvaluation,
    band: Action,
) -> str:
    """Generate a deterministic, one-sentence, human-readable reason.

    No LLM involved: a fixed, ordered set of rule-driven templates, falling
    back to a message-type-based sentence and finally a band-based sentence.

    Parameters
    ----------
    rules:
        The rule evaluation for this message.
    band:
        The final resolved action.

    Returns
    -------
    str
        A single sentence.
    """
    metadata = rules.metadata
    for predicate, template in _reason_rules():
        try:
            if predicate(rules, metadata, band):
                return template(rules, metadata, band)
        except Exception as error:  # noqa: BLE001 - a bad template must not break the decision
            logger.warning("build_reason: template evaluation failed (%s)", error)
            continue

    type_key = rules.suggested_message_type.value
    if type_key in _FALLBACK_BY_TYPE:
        return _FALLBACK_BY_TYPE[type_key]

    return _FALLBACK_BY_BAND[band]


# --------------------------------------------------------------------------- #
# Arbiter
# --------------------------------------------------------------------------- #


class Arbiter:
    """Converts a :class:`~src.scoring.priority.PriorityAssessment` into a
    submission-ready :class:`~src.schema.Decision`.

    Parameters
    ----------
    config:
        Application configuration; defaults to the process-wide singleton.
        Supplies the default routing thresholds.
    thresholds:
        Explicit ``(mute_digest_cut, digest_notify_cut)`` override, e.g. from
        a fitted ``outputs/thresholds.json``. Defaults to
        ``config.thresholds``.
    confidence_calibrator:
        Calibrator instance; built from ``config`` when omitted.
    max_evidence:
        Maximum evidence ids per decision, mirrors
        :attr:`~src.config.RoutingConfig.max_evidence_ids`.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        thresholds: tuple[float, float] | None = None,
        confidence_calibrator: ConfidenceCalibrator | None = None,
        max_evidence: int | None = None,
    ) -> None:
        self._config = config or get_config()
        self._thresholds = thresholds or (
            self._config.thresholds.mute_digest_cut,
            self._config.thresholds.digest_notify_cut,
        )
        self._confidence_calibrator = confidence_calibrator or ConfidenceCalibrator(self._config)
        self._max_evidence = max_evidence or self._config.routing.max_evidence_ids

    @property
    def thresholds(self) -> tuple[float, float]:
        """The ``(mute_digest_cut, digest_notify_cut)`` pair in use."""
        return self._thresholds

    def decide(
        self,
        message: Message,
        context: MessageContext,
        rules: RuleEvaluation,
        priority: PriorityAssessment,
        ocr_result: OcrResult | None = None,
        asr_result: AsrResult | None = None,
    ) -> Decision:
        """Produce the final :class:`~src.schema.Decision` for one message.

        Parameters
        ----------
        message:
            The message being decided.
        context:
            Assembled :class:`~src.retrieval.context.MessageContext`.
        rules:
            The rule evaluation for this message.
        priority:
            The priority engine's assessment; this method never re-scores it.
        ocr_result:
            Raw OCR result, for image messages, used only for confidence.
        asr_result:
            Raw ASR result, for voice messages, used only for confidence.

        Returns
        -------
        Decision
        """
        resolution = resolve_band(priority.priority_score, rules.overrides, self._thresholds)
        band, exception_notes = apply_context_exceptions(
            resolution.band, rules, resolution.has_force
        )

        evidence = select_evidence(message, context, rules, max_evidence=self._max_evidence)

        calibrated = self._confidence_calibrator.calibrate(
            message,
            context,
            rules,
            priority,
            evidence_confidence=evidence.confidence,
            ocr_result=ocr_result,
            asr_result=asr_result,
        )

        reason = build_reason(rules, band)
        source = self._resolve_source(resolution, exception_notes)
        trace = self._build_trace(message, priority, rules, band, source, calibrated)

        decision = Decision(
            message_id=message.message_id,
            action=band,
            message_type=rules.suggested_message_type,
            reason=reason,
            confidence=calibrated.value,
            evidence_message_ids=evidence.evidence_message_ids,
            source=source,
            trace=trace,
        )

        logger.info(
            "Arbiter: message_id=%s action=%s type=%s confidence=%.2f "
            "score=%.1f threshold_band=%s overrides_applied=%d exceptions=%d",
            message.message_id,
            band.value,
            rules.suggested_message_type.value,
            calibrated.value,
            priority.priority_score,
            resolution.threshold_band.value,
            len(resolution.applied_overrides),
            len(exception_notes),
        )
        return decision

    def decide_many(
        self,
        items: list[tuple[Message, MessageContext, RuleEvaluation, PriorityAssessment]],
    ) -> list[Decision]:
        """Decide many messages, reusing this arbiter's configuration.

        Parameters
        ----------
        items:
            Sequence of ``(message, context, rules, priority)`` tuples.

        Returns
        -------
        list[Decision]
            One decision per input item, in order. A single item raising an
            unexpected error yields a :meth:`~src.schema.Decision.fallback`
            decision rather than aborting the batch.
        """
        decisions: list[Decision] = []
        for message, context, rules, priority in items:
            try:
                decisions.append(self.decide(message, context, rules, priority))
            except Exception as error:  # noqa: BLE001 - one bad message must not stop the batch
                logger.error(
                    "Arbiter.decide failed for message_id=%s (%s); using fallback decision.",
                    message.message_id,
                    error,
                )
                decisions.append(
                    Decision.fallback(
                        message.message_id,
                        reason=f"Arbiter error: {error}",
                    )
                )
        return decisions

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_source(
        resolution: BandResolution,
        exception_notes: tuple[str, ...],
    ) -> DecisionSource:
        """Determine the :class:`~src.schema.DecisionSource` for the trace.

        FORCE and applied floor/ceiling overrides, as well as contextual
        exceptions, are all reported as ``OVERRIDE``; a purely threshold-
        derived band is reported as ``SCORE``.
        """
        if resolution.has_force or resolution.applied_overrides or exception_notes:
            return DecisionSource.OVERRIDE
        return DecisionSource.SCORE

    def _build_trace(
        self,
        message: Message,
        priority: PriorityAssessment,
        rules: RuleEvaluation,
        band: Action,
        source: DecisionSource,
        calibrated: CalibratedConfidence,
    ) -> ScoreTrace:
        """Build the :class:`~src.schema.ScoreTrace` attached to the decision.

        Uses :meth:`~src.scoring.priority.PriorityAssessment.to_schema_result`
        to construct the frozen schema's ``PriorityResult`` now that a band has
        been chosen.
        """
        strongest_rule = rules.highest_confidence_rule()
        schema_priority = priority.to_schema_result(self._thresholds, band)

        return ScoreTrace(
            message_id=message.message_id,
            relationship=None,
            priority=schema_priority,
            rule_id=strongest_rule.rule_id if strongest_rule else None,
            llm_action=None,
            llm_confidence=None,
            llm_agreed=None,
            final_action=band,
            confidence=calibrated.value,
            source=source,
            notes=tuple(
                f"{contribution.name}: {contribution.explanation}"
                for contribution in priority.top_contributions(3)
            ),
        )


__all__ = [
    "Arbiter",
    "BandResolution",
    "apply_context_exceptions",
    "build_reason",
    "resolve_band",
]