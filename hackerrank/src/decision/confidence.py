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

    Attributes
    ----------
    priority, retrieval, agreement, evidence, media, ocr, asr:
        Weight of each corresponding factor in :class:`ConfidenceFactors`.
    """

    priority: float = 0.28
    retrieval: float = 0.14
    agreement: float = 0.16
    evidence: float = 0.14
    media: float = 0.10
    ocr: float = 0.09
    asr: float = 0.09

    def __post_init__(self) -> None:
        """Validate that the weights sum to approximately 1.0."""
        total = (
            self.priority
            + self.retrieval
            + self.agreement
            + self.evidence
            + self.media
            + self.ocr
            + self.asr
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ConfidenceWeights must sum to 1.0, got {total:.6f}")


DEFAULT_WEIGHTS = ConfidenceWeights()


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
                "priority": self.weights.priority,
                "retrieval": self.weights.retrieval,
                "agreement": self.weights.agreement,
                "evidence": self.weights.evidence,
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
    ) -> None:
        self._config = config or get_config()
        self._weights = weights or DEFAULT_WEIGHTS

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
            priority_confidence=clamp_unit(priority.confidence),
            retrieval_completeness=self._retrieval_completeness(context),
            rule_score_agreement=self._rule_score_agreement(rules, priority),
            evidence_quality=self._evidence_quality(context, evidence_confidence),
            media_confidence=self._media_confidence(message),
            ocr_confidence=self._ocr_confidence(message, ocr_result),
            asr_confidence=self._asr_confidence(message, asr_result),
        )

        blended = (
            self._weights.priority * factors.priority_confidence
            + self._weights.retrieval * factors.retrieval_completeness
            + self._weights.agreement * factors.rule_score_agreement
            + self._weights.evidence * factors.evidence_quality
            + self._weights.media * factors.media_confidence
            + self._weights.ocr * factors.ocr_confidence
            + self._weights.asr * factors.asr_confidence
        )

        cfg = self._config.confidence
        final = clamp(blended, cfg.confidence_min, cfg.confidence_max)

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
    "DEFAULT_WEIGHTS",
    "CalibratedConfidence",
    "ConfidenceCalibrator",
    "ConfidenceFactors",
    "ConfidenceWeights",
    "calibrate_confidence",
    "clamp",
    "clamp_unit",
]