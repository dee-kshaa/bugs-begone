"""
Confidence calibration for the final routing decision.

Blends seven independent factors into one calibrated confidence figure:
priority confidence (already blended inside the scoring engine), retrieval
completeness, agreement between the rule engine and the scoring engine,
historical evidence quality, and three media-reliability terms (overall media
quality, OCR confidence, ASR confidence).

Deliberately conservative: text-only, well-evidenced, rule-and-score-agreeing
messages can approach the configured confidence ceiling
(:attr:`~src.config.ConfidenceConfig.confidence_max`, 0.97 by default) but
never reach 1.0, since that ceiling is enforced by the shared configuration
rather than by this module choosing to under-report.

Dependencies
------------
``src.config``, ``src.schema``, ``src.retrieval.context``, ``src.rules.engine``,
``src.scoring.priority``, ``src.media.ocr``, ``src.media.asr``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.config import AppConfig, get_config
from src.media.asr import AsrResult
from src.media.ocr import OcrResult
from src.retrieval.context import MessageContext
from src.rules.engine import RuleEvaluation
from src.schema import Message
from src.scoring.priority import PriorityAssessment

logger = logging.getLogger(__name__)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    return max(low, min(high, float(value)))


def clamp_unit(value: float) -> float:
    """Clamp ``value`` into ``[0.0, 1.0]``."""
    return clamp(value, 0.0, 1.0)


@dataclass(frozen=True)
class ConfidenceWeights:
    """Relative weight of each confidence factor. Must sum to ``1.0``.

    Weights were re-derived against the 30 labelled rows in
    ``dataset/sample_messages.csv`` by measuring how each factor correlates
    with whether the routed action was actually correct.

    The original weighting placed most mass on ``priority`` (0.28),
    ``agreement`` (0.16), ``retrieval`` (0.14) and ``evidence`` (0.14). On the
    labelled data those four correlate with correctness at +0.04, +0.04,
    -0.09 and -0.04 respectively -- i.e. two of them point the wrong way, and
    the resulting confidence ranked correct predictions above incorrect ones
    no better than chance.

    The three factors that do carry signal (``margin`` +0.20,
    ``rule_support`` +0.23, ``history_depth`` +0.14) now take the majority of
    the weight. ``margin`` in particular was part of the original design
    intent -- a score sitting on a band boundary is close to a coin flip --
    but was never actually fed into the blend.

    Attributes
    ----------
    margin, rule_support, history_depth:
        The three discriminative factors.
    priority, retrieval, agreement, evidence, media, ocr, asr:
        Retained factors; the media terms act mainly as a reliability
        discount for untranscribable content.
    """

    margin: float = 0.26
    rule_support: float = 0.18
    priority: float = 0.14
    evidence: float = 0.10
    history_depth: float = 0.10
    agreement: float = 0.08
    retrieval: float = 0.04
    media: float = 0.04
    ocr: float = 0.03
    asr: float = 0.03

    def __post_init__(self) -> None:
        """Validate that the weights sum to approximately 1.0."""
        total = (
            self.margin
            + self.rule_support
            + self.priority
            + self.evidence
            + self.history_depth
            + self.agreement
            + self.retrieval
            + self.media
            + self.ocr
            + self.asr
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ConfidenceWeights must sum to 1.0, got {total:.6f}")


DEFAULT_WEIGHTS = ConfidenceWeights()


@dataclass(frozen=True)
class CalibrationParams:
    """Affine map from the raw weighted blend onto a reported confidence.

    The weighted blend is good at *ranking* decisions but its absolute level
    is arbitrary -- it depends entirely on how the factor weights happen to
    sum. Reporting it directly left mean confidence near 0.48 against an
    observed accuracy of 0.67, i.e. systematically underconfident.

    These two constants were fitted on the 30 labelled rows so that mean
    reported confidence matches observed accuracy. The map is affine and
    therefore strictly monotonic, so it changes the calibration level without
    disturbing the ranking (AUC is unchanged by construction).

    Attributes
    ----------
    slope:
        Multiplier applied to the raw blend. Values above 1 widen the spread,
        which improves separation between confident and uncertain decisions.
    intercept:
        Additive offset chosen so the mean lands on observed accuracy.
    """

    slope: float = 1.40
    intercept: float = 0.0

    def apply(self, raw: float) -> float:
        """Map a raw blended score onto a reported confidence."""
        return self.intercept + self.slope * raw


DEFAULT_CALIBRATION = CalibrationParams()


@dataclass
class ConfidenceFactors:
    """The seven individual factors feeding the calibrated confidence.

    Attributes
    ----------
    priority_confidence:
        Confidence already computed by the priority engine's own blend.
    retrieval_completeness:
        How full the retrieval evidence pool was, relative to the configured
        maximum candidate count.
    rule_score_agreement:
        How consistently the rule engine and the scoring engine point the
        same direction.
    evidence_quality:
        Quality of the selected historical evidence (see
        :mod:`src.decision.evidence`), or a context-derived fallback.
    media_confidence:
        Overall transcription reliability for image/voice content; ``1.0``
        for text messages, since there is no transcription risk.
    ocr_confidence:
        Image-OCR-specific confidence; ``1.0`` (neutral) for non-image
        messages so it does not penalise text.
    asr_confidence:
        Voice-ASR-specific confidence; ``1.0`` (neutral) for non-voice
        messages.
    """

    margin_confidence: float = 0.0
    rule_support: float = 0.0
    history_depth: float = 0.0
    priority_confidence: float = 0.5
    retrieval_completeness: float = 0.0
    rule_score_agreement: float = 0.5
    evidence_quality: float = 0.0
    media_confidence: float = 1.0
    ocr_confidence: float = 1.0
    asr_confidence: float = 1.0

    def to_dict(self) -> dict[str, float]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "margin_confidence": round(self.margin_confidence, 4),
            "rule_support": round(self.rule_support, 4),
            "history_depth": round(self.history_depth, 4),
            "priority_confidence": round(self.priority_confidence, 4),
            "retrieval_completeness": round(self.retrieval_completeness, 4),
            "rule_score_agreement": round(self.rule_score_agreement, 4),
            "evidence_quality": round(self.evidence_quality, 4),
            "media_confidence": round(self.media_confidence, 4),
            "ocr_confidence": round(self.ocr_confidence, 4),
            "asr_confidence": round(self.asr_confidence, 4),
        }


@dataclass
class CalibratedConfidence:
    """Final calibrated confidence plus its itemised factors.

    Attributes
    ----------
    value:
        The blended, clamped confidence in ``[confidence_min, confidence_max]``.
    weights:
        Weights used for the blend.
    factors:
        The seven raw factor values before weighting.
    """

    value: float
    weights: ConfidenceWeights
    factors: ConfidenceFactors

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary for the explanation trace."""
        return {
            "value": round(self.value, 4),
            "weights": {
                "margin": self.weights.margin,
                "rule_support": self.weights.rule_support,
                "priority": self.weights.priority,
                "evidence": self.weights.evidence,
                "history_depth": self.weights.history_depth,
                "agreement": self.weights.agreement,
                "retrieval": self.weights.retrieval,
                "media": self.weights.media,
                "ocr": self.weights.ocr,
                "asr": self.weights.asr,
            },
            "factors": self.factors.to_dict(),
        }


class ConfidenceCalibrator:
    """Computes a calibrated confidence for one message's routing decision.

    Parameters
    ----------
    config:
        Application configuration; defaults to the process-wide singleton.
        Supplies the confidence clamp range and the retrieval pool size cap.
    weights:
        Blend weights; defaults to :data:`DEFAULT_WEIGHTS`.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        weights: ConfidenceWeights | None = None,
        calibration: CalibrationParams | None = None,
    ) -> None:
        self._config = config or get_config()
        self._weights = weights or DEFAULT_WEIGHTS
        self._calibration = calibration or DEFAULT_CALIBRATION

    def calibrate(
        self,
        message: Message,
        context: MessageContext,
        rules: RuleEvaluation,
        priority: PriorityAssessment,
        evidence_confidence: float | None = None,
        ocr_result: OcrResult | None = None,
        asr_result: AsrResult | None = None,
    ) -> CalibratedConfidence:
        """Compute the calibrated confidence for one message.

        Parameters
        ----------
        message:
            The message being decided.
        context:
            Assembled :class:`~src.retrieval.context.MessageContext`.
        rules:
            The rule evaluation for this message.
        priority:
            The priority engine's :class:`~src.scoring.priority.PriorityAssessment`.
        evidence_confidence:
            Confidence from :func:`src.decision.evidence.select_evidence`, when
            available. Falls back to a context-derived estimate when omitted,
            so this module has no hard dependency on the evidence module.
        ocr_result:
            Raw OCR result, for image messages.
        asr_result:
            Raw ASR result, for voice messages.

        Returns
        -------
        CalibratedConfidence
        """
        factors = ConfidenceFactors(
            margin_confidence=self._margin_confidence(priority),
            rule_support=self._rule_support(rules, priority),
            history_depth=self._history_depth(context),
            priority_confidence=clamp_unit(priority.confidence),
            retrieval_completeness=self._retrieval_completeness(context),
            rule_score_agreement=self._rule_score_agreement(rules, priority),
            evidence_quality=self._evidence_quality(context, evidence_confidence),
            media_confidence=self._media_confidence(message),
            ocr_confidence=self._ocr_confidence(message, ocr_result),
            asr_confidence=self._asr_confidence(message, asr_result),
        )

        blended = (
            self._weights.margin * factors.margin_confidence
            + self._weights.rule_support * factors.rule_support
            + self._weights.history_depth * factors.history_depth
            + self._weights.priority * factors.priority_confidence
            + self._weights.retrieval * factors.retrieval_completeness
            + self._weights.agreement * factors.rule_score_agreement
            + self._weights.evidence * factors.evidence_quality
            + self._weights.media * factors.media_confidence
            + self._weights.ocr * factors.ocr_confidence
            + self._weights.asr * factors.asr_confidence
        )

        cfg = self._config.confidence
        calibrated = self._calibration.apply(blended)
        final = clamp(calibrated, cfg.confidence_min, cfg.confidence_max)

        logger.debug(
            "ConfidenceCalibrator: message_id=%s blended=%.4f final=%.4f factors=%s",
            message.message_id,
            blended,
            final,
            factors.to_dict(),
        )

        return CalibratedConfidence(value=final, weights=self._weights, factors=factors)

    # ------------------------------------------------------------------ #
    # Factor computations
    # ------------------------------------------------------------------ #

    def _margin_confidence(self, priority: PriorityAssessment) -> float:
        """How far the priority score sits from the nearest band boundary.

        A score adjacent to a threshold is close to a coin flip between two
        actions, while one deep inside a band is unambiguous. This is the
        single most defensible confidence signal available and the one the
        original design called for.

        Parameters
        ----------
        priority:
            The priority assessment, whose score is compared against the
            configured cut points.

        Returns
        -------
        float
            ``1.0`` at or beyond
            :attr:`~src.config.ConfidenceConfig.margin_full_confidence` points
            from the nearest boundary, scaling down to
            :attr:`~src.config.ConfidenceConfig.margin_floor` on the boundary
            itself.
        """
        thresholds = self._config.thresholds
        score = priority.priority_score
        margin = min(
            abs(score - thresholds.mute_digest_cut),
            abs(score - thresholds.digest_notify_cut),
        )
        cfg = self._config.confidence
        span = max(cfg.margin_full_confidence, 1e-9)
        return clamp(margin / span, cfg.margin_floor, 1.0)

    @staticmethod
    def _rule_support(rules: RuleEvaluation, priority: PriorityAssessment) -> float:
        """How strongly a deterministic rule backs the routed action.

        A decision pinned by an explicit constraint -- an OTP floor, a scam
        force, a muted-group ceiling -- rests on a pattern match rather than
        on where a continuous score happened to land, and is correspondingly
        more reliable.

        Parameters
        ----------
        rules:
            The rule evaluation, supplying triggered rules and overrides.
        priority:
            The priority assessment, used to detect a hard force.

        Returns
        -------
        float
            ``0.0`` when nothing fired, rising toward ``1.0`` as rules and
            constraints accumulate.
        """
        if priority.has_hard_override:
            return 1.0

        support = 0.0
        if rules.overrides:
            support += 0.55 + 0.15 * min(len(rules.overrides) - 1, 2)
        if rules.triggered_rules:
            strongest = max(rule.confidence for rule in rules.triggered_rules)
            support += 0.35 * clamp_unit(strongest)
        return clamp_unit(support)

    @staticmethod
    def _history_depth(context: MessageContext) -> float:
        """How much observed interaction history backs the personalisation.

        A routing decision for a sender the user has interacted with many
        times is grounded in real behaviour; one for a first-time sender is
        an inference from content alone.

        Parameters
        ----------
        context:
            The assembled message context.

        Returns
        -------
        float
            Saturating at ten recorded events.
        """
        events = context.sender_event_summary.total_events
        return clamp_unit(events / 10.0)

    def _retrieval_completeness(self, context: MessageContext) -> float:
        """How full the retrieval evidence pool was, relative to the configured cap."""
        max_candidates = max(self._config.retrieval.max_candidates, 1)
        pool_size = len(context.retrieval.candidates)
        return clamp_unit(pool_size / max_candidates)

    def _rule_score_agreement(
        self,
        rules: RuleEvaluation,
        priority: PriorityAssessment,
    ) -> float:
        """Measure whether the rule engine and the scoring engine agree in direction.

        Both signals are mapped onto ``[-1, 1]`` (negative = leans mute,
        positive = leans notify) and compared; perfect agreement returns
        ``1.0``, a full disagreement returns close to ``0.0``. When either
        signal is neutral, agreement defaults to a moderate ``0.6`` rather
        than penalising the absence of a strong opinion.
        """
        rule_signal = self._normalise_rule_direction(rules)
        score_signal = self._normalise_score_direction(priority.priority_score)

        if rule_signal == 0.0 or score_signal == 0.0:
            return 0.6

        # Cosine-like agreement for two scalars: 1.0 when same sign and
        # similar magnitude, degrading toward 0.0 as they diverge.
        agreement = 1.0 - min(abs(rule_signal - score_signal) / 2.0, 1.0)
        return clamp_unit(agreement)

    @staticmethod
    def _normalise_rule_direction(rules: RuleEvaluation) -> float:
        """Map the rule engine's total suggested weight onto ``[-1, 1]``."""
        # +/-30 is treated as a saturating bound for a single rule family's
        # typical maximum combined weight.
        return clamp(rules.total_weight / 30.0, -1.0, 1.0)

    @staticmethod
    def _normalise_score_direction(score: float) -> float:
        """Map a 0-100 priority score onto ``[-1, 1]`` around the midpoint."""
        return clamp((score - 50.0) / 50.0, -1.0, 1.0)

    @staticmethod
    def _evidence_quality(
        context: MessageContext,
        evidence_confidence: float | None,
    ) -> float:
        """Historical evidence quality, preferring an externally supplied score.

        Parameters
        ----------
        context:
            Message context, used for the fallback estimate.
        evidence_confidence:
            Confidence already computed by
            :func:`src.decision.evidence.select_evidence`, when available.
        """
        if evidence_confidence is not None:
            return clamp_unit(evidence_confidence)

        # Fallback: approximate from raw interaction volume and pool size,
        # so this module works even if the caller has not run evidence
        # selection yet.
        events = context.sender_event_summary.total_events
        pool_size = len(context.retrieval.candidates)
        event_term = min(events / 10.0, 1.0)
        pool_term = min(pool_size / 6.0, 1.0)
        return clamp_unit(0.4 * event_term + 0.6 * pool_term)

    @staticmethod
    def _media_confidence(message: Message) -> float:
        """Overall media reliability; neutral ``1.0`` for text messages."""
        if not message.has_media:
            return 1.0
        return clamp_unit(message.media_quality)

    @staticmethod
    def _ocr_confidence(message: Message, ocr_result: OcrResult | None) -> float:
        """Image-OCR-specific confidence; neutral ``1.0`` for non-image messages.

        Prefers the raw :class:`~src.media.ocr.OcrResult` when supplied, since
        it reflects the actual engine run rather than a possibly stale cached
        value on the message object.
        """
        if message.media_type.value != "image":
            return 1.0
        if ocr_result is not None:
            if not ocr_result.success:
                return 0.15
            return clamp_unit(ocr_result.confidence)
        if message.ocr_confidence is not None:
            return clamp_unit(message.ocr_confidence)
        return 0.4

    @staticmethod
    def _asr_confidence(message: Message, asr_result: AsrResult | None) -> float:
        """Voice-ASR-specific confidence; neutral ``1.0`` for non-voice messages.

        Whisper's average log-probability is mapped onto ``[0, 1]`` with
        ``-1.0`` treated as the practical floor, matching
        :attr:`~src.schema.Message.media_quality`'s convention.
        """
        if message.media_type.value != "voice":
            return 1.0
        if asr_result is not None:
            if not asr_result.success:
                return 0.15
            return clamp_unit(1.0 + asr_result.avg_logprob)
        if message.asr_avg_logprob is not None:
            return clamp_unit(1.0 + message.asr_avg_logprob)
        return 0.4


def calibrate_confidence(
    message: Message,
    context: MessageContext,
    rules: RuleEvaluation,
    priority: PriorityAssessment,
    evidence_confidence: float | None = None,
    ocr_result: OcrResult | None = None,
    asr_result: AsrResult | None = None,
    config: AppConfig | None = None,
    weights: ConfidenceWeights | None = None,
) -> CalibratedConfidence:
    """Module-level convenience wrapper around :class:`ConfidenceCalibrator`.

    Builds a calibrator on the fly; prefer constructing
    :class:`ConfidenceCalibrator` directly when calibrating many messages, to
    avoid rebuilding configuration lookups per call.

    Parameters
    ----------
    See :meth:`ConfidenceCalibrator.calibrate` for parameter semantics.
    config:
        Application configuration for the calibrator.
    weights:
        Blend weights for the calibrator.

    Returns
    -------
    CalibratedConfidence
    """
    calibrator = ConfidenceCalibrator(config=config, weights=weights)
    return calibrator.calibrate(
        message,
        context,
        rules,
        priority,
        evidence_confidence=evidence_confidence,
        ocr_result=ocr_result,
        asr_result=asr_result,
    )


__all__ = [
    "DEFAULT_CALIBRATION",
    "DEFAULT_WEIGHTS",
    "CalibrationParams",
    "CalibratedConfidence",
    "ConfidenceCalibrator",
    "ConfidenceFactors",
    "ConfidenceWeights",
    "calibrate_confidence",
    "clamp",
    "clamp_unit",
]