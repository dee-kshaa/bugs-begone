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

# --------------------------------------------------------------------------- #
# Reason generation
# --------------------------------------------------------------------------- #
#
# Reasons are composed rather than looked up, from four parts:
#
#     <source>  <content>  [<evidence>]  <justification for this action>
#
# e.g. "A verified business sent an order or service update matching the
#       user's recent order history, so it is worth surfacing now."
#
# This mirrors the reference style in ``dataset/sample_messages.csv``, which
# names the actor, describes the content, and -- crucially -- states why the
# chosen action follows. The justification clause is selected from the final
# band, so a reason can never contradict the action it explains.
#
# Everything is deterministic: no LLM, no sampling, no wall-clock reads.

#: Upper bound on reason length, in words. The reference reasons in
#: ``dataset/sample_messages.csv`` average about fourteen words.
MAX_REASON_WORDS = 18

#: Source clause keyed by relationship category, used when no more specific
#: sender description applies.
_SOURCE_BY_CATEGORY: dict[str, str] = {
    "Family": "A trusted family member",
    "Close Friend": "A close contact",
    "Office": "A work contact",
    "College": "A college contact",
    "Society": "A society contact",
    "Business": "A business account",
    "Unknown": "An unfamiliar sender",
}

#: Content clause keyed by the official message type.
_CONTENT_BY_TYPE: dict[str, str] = {
    "urgent": "a time-sensitive update",
    "event": "a scheduled update",
    "payment": "a payment reminder",
    "business_update": "an order update",
    "promotion": "a promotional offer",
    "greeting": "a routine greeting",
    "forward": "a forwarded chain message",
    "personal": "a personal message",
    "spam": "bulk promotional content",
    "scam": "a suspicious verification request",
    "unknown": "a message with no clear purpose",
}

#: Justification clause keyed by the final action. Selected from the band that
#: was actually chosen, so the reason cannot contradict the decision.
_JUSTIFICATION: dict[Action, str] = {
    Action.NOTIFY: "so it should reach the user now",
    Action.DIGEST: "but it is not urgent enough to interrupt",
    Action.MUTE: "so it is held back",
}

#: Risk reasons, which replace the composed sentence entirely. A safety
#: finding is the whole explanation and should not be diluted by relationship
#: or history commentary.
_RISK_REASONS: tuple[tuple[str, str], ...] = (
    (
        "scam_high_confidence",
        "The message asks for urgent OTP or account verification through a "
        "suspicious flow, so it is muted regardless of the sender.",
    ),
    (
        "sys_blocked_sender",
        "The user has blocked this sender, so the message is muted without "
        "further consideration.",
    ),
)


def _source_clause(metadata: dict, band: Action) -> str:
    """Describe who sent the message, most specific description first."""
    category = str(metadata.get("relationship_category", "Unknown"))

    if metadata.get("sender_is_group_admin"):
        return "A trusted group admin"

    if metadata.get("business_is_verified"):
        return "A verified business"

    if metadata.get("business_has_active_order"):
        return "A business with an open order"

    if metadata.get("is_business_message") or metadata.get("business_txn_count"):
        if metadata.get("business_is_known_to_user"):
            return "A familiar business"
        return "An unknown business"

    if metadata.get("business_promo_ratio") is not None and category == "Business":
        return "An unknown business"

    if metadata.get("mentions_user"):
        return "A group member"

    return _SOURCE_BY_CATEGORY.get(category, "A sender")


def _content_clause(message_type: str, metadata: dict) -> str:
    """Describe what the message contains."""
    base = _CONTENT_BY_TYPE.get(str(message_type).lower(), "a message")

    if metadata.get("is_media_only"):
        media = str(metadata.get("media_type", "")).lower()
        if media == "image":
            return f"{base} as an image"
        if media == "voice":
            return f"{base} as a voice note"
    return base


def _evidence_clause(metadata: dict) -> str:
    """Reference the strongest supporting history signal, if any.

    Only one clause is emitted, chosen by how strongly it bears on the
    decision, so the sentence stays to a single readable line.
    """
    if metadata.get("is_reported_recently"):
        count = int(metadata.get("report_count", 0) or 0)
        if count > 1:
            return f" from a sender reported {count} times"
        return " from a previously reported sender"

    if metadata.get("is_duplicate"):
        count = int(metadata.get("duplicate_count", 0) or 0)
        if count > 1:
            return f" repeating content seen {count} times recently"
        return " repeating content seen recently"

    dismissal = float(metadata.get("sender_dismissal_rate", 0.0) or 0.0)
    if dismissal >= 0.4:
        return " of a kind the user repeatedly dismisses"

    if metadata.get("business_has_active_order"):
        return " matching the user's recent order"

    reply_rate = float(metadata.get("sender_reply_rate", 0.0) or 0.0)
    if reply_rate >= 0.5:
        return " from a sender the user regularly replies to"

    open_rate = float(metadata.get("sender_open_rate", 0.0) or 0.0)
    if open_rate >= 0.6:
        return " the user usually opens"

    if metadata.get("group_is_muted"):
        return " in a group the user has muted"

    if metadata.get("user_dnd_active"):
        return " arriving in the user's quiet hours"

    if metadata.get("has_forward_marker"):
        return " carrying forward-chain markers"

    return ""


def _mute_justification(metadata: dict, message_type: str) -> str:
    """Pick the mute justification that best matches why it was muted."""
    if metadata.get("group_is_muted") or metadata.get("user_muted_sender"):
        return "and the user muted this conversation, so it is held back"

    dismissal = float(metadata.get("sender_dismissal_rate", 0.0) or 0.0)
    if dismissal >= 0.4 or str(message_type).lower() in {"forward", "greeting"}:
        return "which the user usually ignores, so it is held back"

    if str(message_type).lower() in {"promotion", "spam"}:
        return "the user never opted into, so it is held back"

    return _JUSTIFICATION[Action.MUTE]


def build_reason(
    rules: RuleEvaluation,
    band: Action,
    message_type: str | None = None,
    evidence: EvidenceSelection | None = None,
) -> str:
    """Generate a deterministic, one-sentence explanation for a decision.

    Composes ``<source> <content>[<evidence>], <justification>`` so that every
    reason names who sent the message, what it was, the strongest supporting
    history signal where one exists, and why the chosen action follows. No
    LLM is involved.

    Parameters
    ----------
    rules:
        The rule evaluation, whose metadata supplies every signal used.
    band:
        The final resolved action. The justification clause is selected from
        this, so a reason can never contradict the decision it explains.
    message_type:
        The official message type. Falls back to the rule engine's suggestion.
    evidence:
        The selected evidence, used only to note when a decision rests on no
        historical support at all.

    Returns
    -------
    str
        A single sentence.
    """
    metadata = rules.metadata

    # Safety findings are the whole explanation.
    for rule_id, text in _RISK_REASONS:
        if any(rule.rule_id == rule_id for rule in rules.triggered_rules):
            return text

    resolved_type = str(
        message_type if message_type is not None else rules.suggested_message_type.value
    ).lower()

    if resolved_type == "scam":
        return _RISK_REASONS[0][1]

    source = _source_clause(metadata, band)
    content = _content_clause(resolved_type, metadata)
    evidence_text = _evidence_clause(metadata)

    if band is Action.MUTE:
        justification = _mute_justification(metadata, resolved_type)
    else:
        justification = _JUSTIFICATION[band]

    # An OTP or explicit mention deserves a sharper notify justification.
    if band is Action.NOTIFY:
        if metadata.get("content_is_otp"):
            justification = "carrying a verification code needed right away"
        elif metadata.get("mentions_user"):
            justification = "addressed to the user directly, so it interrupts now"
        elif float(metadata.get("content_urgency_score", 0.0) or 0.0) >= 0.5:
            justification = "time-critical enough to interrupt the user"

    if band is Action.DIGEST and not evidence_text and resolved_type in {
        "greeting",
        "personal",
    }:
        justification = "safe casual content the user can read later"

    sentence = f"{source} sent {content}{evidence_text}, {justification}."
    sentence = " ".join(sentence.split())

    # Concision is a hard property, not a preference: the reference reasons
    # average about fourteen words. When the supporting-evidence clause pushes
    # the sentence past the cap, it is dropped -- the source, the content and
    # the justification matter more than the corroborating detail.
    if len(sentence.split()) > MAX_REASON_WORDS and evidence_text:
        sentence = " ".join(f"{source} sent {content}, {justification}.".split())

    return sentence


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

        reason = build_reason(
            rules,
            band,
            message_type=rules.suggested_message_type.value,
            evidence=evidence,
        )
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