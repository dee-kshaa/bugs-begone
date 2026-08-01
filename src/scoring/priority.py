"""
Priority aggregation engine.

Consumes the :class:`~src.rules.engine.RuleEvaluation` and the per-component
:class:`~src.scoring.components.ScoreContribution` objects for one message and
aggregates them into a single explainable :class:`PriorityAssessment`.

What this module does *not* do
------------------------------
It never decides ``notify``/``digest``/``mute``. It produces a 0-100 priority
score, a magnitude-descriptive :class:`PriorityLevel`, and a set of unresolved
override constraints. Mapping those onto a routing action -- including
resolving a floor against a ceiling -- belongs to the decision layer.

Naming note
-----------
``src/schema.py`` already defines a frozen ``PriorityResult`` that carries a
``band: Action`` field. Since this engine must not choose a band, it emits
:class:`PriorityAssessment` instead and offers
:meth:`PriorityAssessment.to_schema_result` so the decision layer can build the
frozen type once it has selected a band. ``PriorityResult`` is aliased to
:class:`PriorityAssessment` in this module's namespace for call-site
convenience; the frozen schema class is unchanged.

Dependencies
------------
``src.config``, ``src.schema``, ``src.rules.engine``, ``src.scoring.components``,
``src.retrieval.context``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from src.config import AppConfig, get_config
from src.rules.engine import RuleEvaluation, RuleFamily, TriggeredRule
from src.scoring.components import (
    ComponentCaps,
    ScoreContribution,
    ScoringComponent,
    ScoringInput,
    build_components,
    clamp,
    clamp_unit,
    score_all,
)
from src.schema import (
    Action,
    MessageType,
    OverrideEffect,
    OverrideRecord,
    PriorityResult as SchemaPriorityResult,
    ScoreComponent,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Priority level
# --------------------------------------------------------------------------- #


class PriorityLevel(str, Enum):
    """Descriptive magnitude of a priority score.

    This is a *description of score strength*, not a routing action. The
    decision layer maps levels and overrides onto notify/digest/mute; this
    enum deliberately carries no such semantics.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

    @property
    def rank(self) -> int:
        """Ordinal magnitude, higher meaning a stronger priority signal."""
        return {"minimal": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}[self.value]

    @classmethod
    def from_score(cls, score: float) -> "PriorityLevel":
        """Bucket a 0-100 score into a descriptive level.

        The cut points here are fixed descriptive bands, deliberately
        independent of the tunable routing thresholds in
        :class:`~src.config.ThresholdConfig`. Retuning routing thresholds must
        not silently change what "high priority" means in an explanation.

        Parameters
        ----------
        score:
            Priority score in ``[0, 100]``.

        Returns
        -------
        PriorityLevel
        """
        if score >= 85.0:
            return cls.CRITICAL
        if score >= 65.0:
            return cls.HIGH
        if score >= 40.0:
            return cls.MEDIUM
        if score >= 20.0:
            return cls.LOW
        return cls.MINIMAL


class OverrideKind(str, Enum):
    """Classification of the rule constraints this engine recognises."""

    HARD_OVERRIDE = "hard_override"
    SCORE_FLOOR = "score_floor"
    SCORE_CEILING = "score_ceiling"
    MANDATORY_PENALTY = "mandatory_penalty"
    EMERGENCY_ESCALATION = "emergency_escalation"
    SCAM_SUPPRESSION = "scam_suppression"
    QUIET_HOUR_ADJUSTMENT = "quiet_hour_adjustment"
    TRUSTED_ADMIN_EXCEPTION = "trusted_admin_exception"
    VERIFIED_BUSINESS_ADJUSTMENT = "verified_business_adjustment"


# --------------------------------------------------------------------------- #
# Score adjustments
# --------------------------------------------------------------------------- #


@dataclass
class ScoreAdjustment:
    """One numeric adjustment applied to the aggregated score.

    Distinct from an :class:`~src.schema.OverrideRecord`, which constrains the
    *action band*. An adjustment moves the score itself and is fully resolved
    inside this engine.

    Attributes
    ----------
    rule_id:
        The rule that motivated the adjustment.
    kind:
        Which category of rule handling produced it.
    before:
        Score before the adjustment.
    after:
        Score after the adjustment.
    reason:
        Human-readable justification for the explanation trace.
    binding:
        ``True`` when the adjustment actually moved the score.
    """

    rule_id: str
    kind: OverrideKind
    before: float
    after: float
    reason: str
    binding: bool = False

    def __post_init__(self) -> None:
        """Derive :attr:`binding` from whether the score actually moved."""
        self.rule_id = str(self.rule_id).strip()
        self.kind = OverrideKind(self.kind)
        self.before = float(self.before)
        self.after = float(self.after)
        self.binding = abs(self.after - self.before) > 1e-9

    @property
    def delta(self) -> float:
        """Signed change this adjustment produced."""
        return self.after - self.before

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "rule_id": self.rule_id,
            "kind": self.kind.value,
            "before": round(self.before, 3),
            "after": round(self.after, 3),
            "delta": round(self.delta, 3),
            "reason": self.reason,
            "binding": self.binding,
        }


# --------------------------------------------------------------------------- #
# Confidence breakdown
# --------------------------------------------------------------------------- #


@dataclass
class ConfidenceBreakdown:
    """Itemised inputs to the calibrated confidence figure.

    Kept as a separate object so the demo can show *why* a decision was
    confident, not just how confident it was.

    Attributes
    ----------
    coverage:
        How many components produced a non-zero contribution, normalised.
    agreement:
        How consistently the components pointed the same direction.
    rule_strength:
        Confidence of the strongest triggered rule, or a neutral value when no
        rules fired.
    evidence_availability:
        Whether historical interaction evidence was available.
    retrieval_completeness:
        How full the retrieval evidence pool was.
    component_confidence:
        Weighted mean of the individual components' self-reported confidence.
    final:
        The blended result, clamped to the configured range.
    """

    coverage: float = 0.0
    agreement: float = 0.0
    rule_strength: float = 0.0
    evidence_availability: float = 0.0
    retrieval_completeness: float = 0.0
    component_confidence: float = 0.0
    final: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "coverage": round(self.coverage, 4),
            "agreement": round(self.agreement, 4),
            "rule_strength": round(self.rule_strength, 4),
            "evidence_availability": round(self.evidence_availability, 4),
            "retrieval_completeness": round(self.retrieval_completeness, 4),
            "component_confidence": round(self.component_confidence, 4),
            "final": round(self.final, 4),
        }


# --------------------------------------------------------------------------- #
# Priority assessment
# --------------------------------------------------------------------------- #


@dataclass
class PriorityAssessment:
    """Fully explainable aggregation result for one message.

    Carries no routing action. The decision layer consumes this, resolves the
    :attr:`overrides` against each other, applies routing thresholds, and only
    then produces a band.

    Attributes
    ----------
    message_id:
        The message this assessment covers.
    raw_score:
        Sum of every component contribution, before clamping and adjustments.
    priority_score:
        Final score after adjustments, clamped to ``[0, 100]``.
    normalized_score:
        :attr:`priority_score` expressed in ``[0, 1]``.
    positive_score:
        Sum of positive component contributions.
    penalty_score:
        Sum of penalty component contributions (a negative number or zero).
    confidence:
        Calibrated confidence in ``[0, 1]``, never exactly ``1.0`` unless the
        situation is genuinely deterministic.
    confidence_breakdown:
        Itemised confidence inputs.
    recommended_priority:
        Descriptive :class:`PriorityLevel`, not a routing action.
    contributions:
        Every component's contribution, in component order.
    triggered_rules:
        Every rule that fired, carried through from the rule evaluation.
    overrides:
        Unresolved band constraints for the decision layer.
    adjustments:
        Numeric score adjustments this engine applied and resolved.
    suggested_message_type:
        Best-guess message type, carried through from the rule evaluation.
    evidence_message_ids:
        Deduplicated evidence ids gathered from components and rules.
    explanation:
        One-paragraph human-readable summary.
    metadata:
        Flat bundle for downstream explanation generation.
    """

    message_id: str
    raw_score: float = 0.0
    priority_score: float = 0.0
    normalized_score: float = 0.0
    positive_score: float = 0.0
    penalty_score: float = 0.0
    confidence: float = 0.5
    confidence_breakdown: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)
    recommended_priority: PriorityLevel = PriorityLevel.MINIMAL
    contributions: tuple[ScoreContribution, ...] = ()
    triggered_rules: tuple[TriggeredRule, ...] = ()
    overrides: tuple[OverrideRecord, ...] = ()
    adjustments: tuple[ScoreAdjustment, ...] = ()
    suggested_message_type: MessageType = MessageType.OTHER
    evidence_message_ids: tuple[str, ...] = ()
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- derived views ------------------------------------------------- #

    @property
    def binding_adjustments(self) -> tuple[ScoreAdjustment, ...]:
        """Only the adjustments that actually moved the score."""
        return tuple(adjustment for adjustment in self.adjustments if adjustment.binding)

    @property
    def firing_contributions(self) -> tuple[ScoreContribution, ...]:
        """Only the components that contributed a non-zero amount."""
        return tuple(contribution for contribution in self.contributions if contribution.fired)

    @property
    def has_hard_override(self) -> bool:
        """``True`` when a rule recorded a FORCE-effect constraint."""
        return any(override.effect is OverrideEffect.FORCE for override in self.overrides)

    @property
    def floors(self) -> tuple[OverrideRecord, ...]:
        """Constraints that set a lower bound on the eventual band."""
        return tuple(o for o in self.overrides if o.effect is OverrideEffect.FLOOR)

    @property
    def ceilings(self) -> tuple[OverrideRecord, ...]:
        """Constraints that set an upper bound on the eventual band."""
        return tuple(o for o in self.overrides if o.effect is OverrideEffect.CEILING)

    @property
    def forces(self) -> tuple[OverrideRecord, ...]:
        """Constraints that force a specific band outright."""
        return tuple(o for o in self.overrides if o.effect is OverrideEffect.FORCE)

    def contribution(self, name: str) -> ScoreContribution | None:
        """Return the contribution named ``name``, or ``None`` if absent."""
        for contribution in self.contributions:
            if contribution.name.lower() == name.lower():
                return contribution
        return None

    def top_contributions(self, n: int = 3) -> tuple[ScoreContribution, ...]:
        """Return the ``n`` components with the largest absolute contribution."""
        ranked = sorted(
            self.firing_contributions, key=lambda c: abs(c.points), reverse=True
        )
        return tuple(ranked[: max(0, n)])

    # ---- conversion ------------------------------------------------------ #

    def to_score_components(self) -> tuple[ScoreComponent, ...]:
        """Convert contributions to the frozen :class:`~src.schema.ScoreComponent`."""
        return tuple(
            contribution.to_score_component() for contribution in self.contributions
        )

    def to_schema_result(
        self,
        thresholds: tuple[float, float],
        band: Action,
    ) -> SchemaPriorityResult:
        """Build the frozen :class:`~src.schema.PriorityResult` for a chosen band.

        Called by the decision layer *after* it has resolved the overrides and
        applied routing thresholds. This engine never calls it itself, because
        doing so would require choosing a band.

        Parameters
        ----------
        thresholds:
            The ``(mute_digest_cut, digest_notify_cut)`` pair in use.
        band:
            The action band the decision layer selected.

        Returns
        -------
        src.schema.PriorityResult
        """
        return SchemaPriorityResult(
            message_id=self.message_id,
            components=self.to_score_components(),
            raw_score=self.raw_score,
            final_score=self.priority_score,
            band=band,
            thresholds=thresholds,
            overrides=self.overrides,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full assessment for the explanation trace."""
        return {
            "message_id": self.message_id,
            "raw_score": round(self.raw_score, 3),
            "priority_score": round(self.priority_score, 3),
            "normalized_score": round(self.normalized_score, 4),
            "positive_score": round(self.positive_score, 3),
            "penalty_score": round(self.penalty_score, 3),
            "confidence": round(self.confidence, 4),
            "confidence_breakdown": self.confidence_breakdown.to_dict(),
            "recommended_priority": self.recommended_priority.value,
            "contributions": [c.to_dict() for c in self.contributions],
            "triggered_rules": [r.to_dict() for r in self.triggered_rules],
            "overrides": [o.to_dict() for o in self.overrides],
            "adjustments": [a.to_dict() for a in self.adjustments],
            "suggested_message_type": self.suggested_message_type.value,
            "evidence_message_ids": list(self.evidence_message_ids),
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


#: Call-site alias. The frozen schema class remains available as
#: ``src.schema.PriorityResult``; this name refers to the action-free
#: aggregation result produced by this engine.
PriorityResult = PriorityAssessment


# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PriorityTuning:
    """Numeric constants governing aggregation and rule handling.

    Grouped into one frozen object so that every magic number in this engine
    is visible in a single place and can be swapped for a variant without
    editing logic.
    """

    #: Score floor imposed by an emergency escalation.
    emergency_floor_score: float = 82.0
    #: Score ceiling imposed by scam suppression.
    scam_ceiling_score: float = 8.0
    #: Score ceiling imposed by a spam finding without a scam-level match.
    spam_ceiling_score: float = 22.0
    #: Multiplicative damping applied during the user's quiet hours.
    quiet_hour_damping: float = 0.85
    #: Score below which quiet-hour damping is skipped (already low).
    quiet_hour_damping_min_score: float = 30.0
    #: Additive bonus for a message from a trusted group admin.
    trusted_admin_bonus: float = 4.0
    #: Additive bonus for a transactional message from a verified business.
    verified_business_bonus: float = 3.0
    #: Score floor imposed when a rule recorded a FLOOR toward notify.
    hard_floor_notify_score: float = 75.0
    #: Score floor imposed when a rule recorded a FLOOR toward digest.
    hard_floor_digest_score: float = 42.0
    #: Score ceiling imposed when a rule recorded a CEILING toward mute.
    hard_ceiling_mute_score: float = 18.0
    #: Score ceiling imposed when a rule recorded a CEILING toward digest.
    hard_ceiling_digest_score: float = 68.0
    #: Confidence assigned when a deterministic FORCE override is present.
    deterministic_confidence: float = 0.96
    #: Weight of each confidence factor in the blend. Must sum to 1.0.
    weight_coverage: float = 0.15
    weight_agreement: float = 0.20
    weight_rule_strength: float = 0.25
    weight_evidence: float = 0.15
    weight_retrieval: float = 0.10
    weight_component_confidence: float = 0.15


DEFAULT_TUNING = PriorityTuning()


# --------------------------------------------------------------------------- #
# Priority engine
# --------------------------------------------------------------------------- #


class PriorityEngine:
    """Aggregates component contributions and rule constraints into a score.

    Deterministic: the same inputs always produce the same assessment. No
    randomness, no wall-clock reads, no network access.

    Parameters
    ----------
    components:
        Pre-built scoring components. Built from the component registry when
        omitted, so newly registered components are picked up automatically.
    caps:
        Component cap budget, used only when ``components`` is omitted.
    config:
        Application configuration. Defaults to the process-wide singleton.
    tuning:
        Numeric tuning constants. Defaults to :data:`DEFAULT_TUNING`.
    """

    def __init__(
        self,
        components: Sequence[ScoringComponent] | None = None,
        caps: ComponentCaps | None = None,
        config: AppConfig | None = None,
        tuning: PriorityTuning | None = None,
    ) -> None:
        self._config = config or get_config()
        self._tuning = tuning or DEFAULT_TUNING
        self._caps = caps or ComponentCaps.from_scoring_config(self._config.scoring)
        self._components = (
            list(components)
            if components is not None
            else build_components(caps=self._caps, config=self._config)
        )
        logger.debug(
            "PriorityEngine initialised with %d component(s): %s",
            len(self._components),
            [component.name for component in self._components],
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def assess(self, inputs: ScoringInput) -> PriorityAssessment:
        """Produce a full :class:`PriorityAssessment` for one message.

        Parameters
        ----------
        inputs:
            Bundled scoring inputs, already carrying the rule evaluation.

        Returns
        -------
        PriorityAssessment
            Never raises: component failures are isolated inside
            :meth:`~src.scoring.components.ScoringComponent.score`, and rule
            handling degrades to no-adjustment on unexpected data.
        """
        contributions = score_all(inputs, components=self._components)
        return self.aggregate(inputs, contributions)

    def aggregate(
        self,
        inputs: ScoringInput,
        contributions: Sequence[ScoreContribution],
    ) -> PriorityAssessment:
        """Aggregate pre-computed contributions into an assessment.

        Separated from :meth:`assess` so a caller that has already scored the
        components (for example, to inspect them) does not pay for a second
        pass.

        Parameters
        ----------
        inputs:
            Bundled scoring inputs.
        contributions:
            One contribution per scoring component.

        Returns
        -------
        PriorityAssessment
        """
        rules = inputs.rules
        message_id = inputs.message.message_id

        positive_score, penalty_score, raw_score = self._split_scores(contributions)
        normalised_base = self._normalise(raw_score)

        score, adjustments = self._apply_rule_handling(
            normalised_base, inputs, rules
        )
        score = clamp(score, self._config.scoring.score_min, self._config.scoring.score_max)

        breakdown = self._compute_confidence(inputs, contributions, rules, adjustments)
        evidence = self._collect_evidence(contributions, rules)
        level = PriorityLevel.from_score(score)
        explanation = self._build_explanation(
            score, level, contributions, adjustments, rules
        )
        metadata = self._build_metadata(
            inputs, contributions, rules, adjustments, positive_score, penalty_score
        )

        assessment = PriorityAssessment(
            message_id=message_id,
            raw_score=raw_score,
            priority_score=score,
            normalized_score=clamp_unit(score / 100.0),
            positive_score=positive_score,
            penalty_score=penalty_score,
            confidence=breakdown.final,
            confidence_breakdown=breakdown,
            recommended_priority=level,
            contributions=tuple(contributions),
            triggered_rules=rules.triggered_rules,
            overrides=rules.overrides,
            adjustments=tuple(adjustments),
            suggested_message_type=rules.suggested_message_type,
            evidence_message_ids=evidence,
            explanation=explanation,
            metadata=metadata,
        )

        logger.debug(
            "PriorityEngine: message_id=%s score=%.1f level=%s confidence=%.2f "
            "adjustments=%d overrides=%d",
            message_id,
            score,
            level.value,
            breakdown.final,
            len(assessment.binding_adjustments),
            len(rules.overrides),
        )
        return assessment

    def assess_many(self, inputs_batch: Iterable[ScoringInput]) -> list[PriorityAssessment]:
        """Assess many messages, reusing the same component instances.

        Parameters
        ----------
        inputs_batch:
            Scoring inputs for each message.

        Returns
        -------
        list[PriorityAssessment]
            One assessment per input, in input order.
        """
        results = [self.assess(inputs) for inputs in inputs_batch]
        if results:
            mean_score = sum(result.priority_score for result in results) / len(results)
            logger.info(
                "PriorityEngine: assessed %d message(s), mean score %.1f.",
                len(results),
                mean_score,
            )
        return results

    # ------------------------------------------------------------------ #
    # Aggregation internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_scores(
        contributions: Sequence[ScoreContribution],
    ) -> tuple[float, float, float]:
        """Split contributions into positive, penalty, and combined totals.

        Parameters
        ----------
        contributions:
            Component contributions.

        Returns
        -------
        tuple
            ``(positive_score, penalty_score, raw_score)``. ``penalty_score``
            is negative or zero.
        """
        positive = sum(c.points for c in contributions if c.points > 0)
        penalty = sum(c.points for c in contributions if c.points < 0)
        return positive, penalty, positive + penalty

    def _normalise(self, raw_score: float) -> float:
        """Map the raw component sum onto the configured 0-100 score range.

        The component cap budget is arranged so positive caps sum to the
        configured maximum and penalties to the configured floor, so this is a
        clamp rather than a rescale. Kept as a named step so a future
        non-linear normalisation can be swapped in here alone.

        Parameters
        ----------
        raw_score:
            Sum of every component contribution.

        Returns
        -------
        float
            Score clamped into ``[score_min, score_max]``.
        """
        scoring = self._config.scoring
        return clamp(raw_score, scoring.score_min, scoring.score_max)

    # ------------------------------------------------------------------ #
    # Rule handling
    # ------------------------------------------------------------------ #

    def _apply_rule_handling(
        self,
        score: float,
        inputs: ScoringInput,
        rules: RuleEvaluation,
    ) -> tuple[float, list[ScoreAdjustment]]:
        """Apply every supported rule adjustment, in a fixed deterministic order.

        Order matters and is fixed: mandatory penalties, then contextual
        adjustments (quiet hours, trusted admin, verified business), then
        emergency escalation, then scam suppression, then generic hard
        floors and ceilings derived from the rule engine's override records.
        Scam suppression runs after emergency escalation so that a message
        which is both OTP-shaped and a KYC scam ends up suppressed.

        Parameters
        ----------
        score:
            Score after component aggregation and normalisation.
        inputs:
            Bundled scoring inputs.
        rules:
            The rule evaluation for this message.

        Returns
        -------
        tuple
            ``(adjusted_score, adjustments)``.
        """
        adjustments: list[ScoreAdjustment] = []

        score = self._apply_mandatory_penalties(score, rules, adjustments)
        score = self._apply_quiet_hours(score, inputs, rules, adjustments)
        score = self._apply_trusted_admin(score, inputs, rules, adjustments)
        score = self._apply_verified_business(score, inputs, rules, adjustments)
        score = self._apply_emergency_escalation(score, rules, adjustments)
        score = self._apply_scam_suppression(score, rules, adjustments)
        score = self._apply_hard_constraints(score, rules, adjustments)

        return score, adjustments

    def _apply_mandatory_penalties(
        self,
        score: float,
        rules: RuleEvaluation,
        adjustments: list[ScoreAdjustment],
    ) -> float:
        """Apply penalties that must reduce the score regardless of components.

        Covers blocked senders and reported senders: cases where the user has
        given an explicit negative signal that no amount of positive component
        evidence should be able to outvote.
        """
        tuning = self._tuning
        for rule in rules.triggered_rules:
            if rule.rule_id == "sys_blocked_sender":
                before = score
                score = min(score, 0.0)
                adjustments.append(
                    ScoreAdjustment(
                        rule_id=rule.rule_id,
                        kind=OverrideKind.MANDATORY_PENALTY,
                        before=before,
                        after=score,
                        reason="Sender is blocked; score floored to zero.",
                    )
                )
            elif rule.rule_id == "sys_reported_sender":
                before = score
                score = min(score, tuning.hard_ceiling_mute_score)
                adjustments.append(
                    ScoreAdjustment(
                        rule_id=rule.rule_id,
                        kind=OverrideKind.MANDATORY_PENALTY,
                        before=before,
                        after=score,
                        reason="Sender previously reported by this user; score capped.",
                    )
                )
        return score

    def _apply_quiet_hours(
        self,
        score: float,
        inputs: ScoringInput,
        rules: RuleEvaluation,
        adjustments: list[ScoreAdjustment],
    ) -> float:
        """Damp the score during the user's quiet hours.

        Multiplicative rather than a hard cap, and skipped for already-low
        scores, so quiet hours nudge borderline messages downward without
        flattening genuinely important ones. An emergency escalation applied
        afterwards can still lift the score back up.
        """
        if not any(rule.rule_id == "sys_quiet_hours" for rule in rules.triggered_rules):
            return score
        if score < self._tuning.quiet_hour_damping_min_score:
            return score

        before = score
        score = score * self._tuning.quiet_hour_damping
        adjustments.append(
            ScoreAdjustment(
                rule_id="sys_quiet_hours",
                kind=OverrideKind.QUIET_HOUR_ADJUSTMENT,
                before=before,
                after=score,
                reason=(
                    f"Quiet hours: score damped by "
                    f"{(1.0 - self._tuning.quiet_hour_damping) * 100:.0f}%."
                ),
            )
        )
        return score

    def _apply_trusted_admin(
        self,
        score: float,
        inputs: ScoringInput,
        rules: RuleEvaluation,
        adjustments: list[ScoreAdjustment],
    ) -> float:
        """Grant a small bonus to announcements from a trusted group admin."""
        if not any(rule.rule_id == "sys_trusted_admin" for rule in rules.triggered_rules):
            return score

        before = score
        score = score + self._tuning.trusted_admin_bonus
        adjustments.append(
            ScoreAdjustment(
                rule_id="sys_trusted_admin",
                kind=OverrideKind.TRUSTED_ADMIN_EXCEPTION,
                before=before,
                after=score,
                reason=(
                    f"Sender is a group admin: +{self._tuning.trusted_admin_bonus:.0f} bonus."
                ),
            )
        )
        return score

    def _apply_verified_business(
        self,
        score: float,
        inputs: ScoringInput,
        rules: RuleEvaluation,
        adjustments: list[ScoreAdjustment],
    ) -> float:
        """Grant a small bonus to transactional messages from verified accounts.

        Applied only when the message is transactional; a verified account
        sending marketing copy gets no benefit from its verification.
        """
        business = inputs.business
        if business is None or not business.is_verified:
            return score
        if not inputs.verdict.is_transactional:
            return score

        before = score
        score = score + self._tuning.verified_business_bonus
        adjustments.append(
            ScoreAdjustment(
                rule_id="business_verified_transactional",
                kind=OverrideKind.VERIFIED_BUSINESS_ADJUSTMENT,
                before=before,
                after=score,
                reason=(
                    f"Verified business sending a transactional update: "
                    f"+{self._tuning.verified_business_bonus:.0f} bonus."
                ),
            )
        )
        return score

    def _apply_emergency_escalation(
        self,
        score: float,
        rules: RuleEvaluation,
        adjustments: list[ScoreAdjustment],
    ) -> float:
        """Raise the score to an emergency floor when an emergency rule fired.

        Only escalates -- never lowers an already-higher score. Scam
        suppression runs afterwards and can still override this.
        """
        emergency_rules = rules.rules_in(RuleFamily.EMERGENCY)
        if not emergency_rules:
            return score

        strongest = max(emergency_rules, key=lambda rule: rule.confidence)
        if strongest.confidence < 0.6:
            return score

        floor = self._tuning.emergency_floor_score
        if score >= floor:
            return score

        before = score
        score = floor
        adjustments.append(
            ScoreAdjustment(
                rule_id=strongest.rule_id,
                kind=OverrideKind.EMERGENCY_ESCALATION,
                before=before,
                after=score,
                reason=(
                    f"Emergency rule {strongest.rule_id} "
                    f"(confidence {strongest.confidence:.2f}) escalated the score to {floor:.0f}."
                ),
            )
        )
        return score

    def _apply_scam_suppression(
        self,
        score: float,
        rules: RuleEvaluation,
        adjustments: list[ScoreAdjustment],
    ) -> float:
        """Suppress the score when a scam or spam rule fired.

        Runs after emergency escalation deliberately: a message that looks
        like an OTP but is actually a credential-phishing scam must end up
        suppressed, not escalated.
        """
        tuning = self._tuning

        scam_rules = rules.rules_in(RuleFamily.SCAM)
        if scam_rules:
            strongest = max(scam_rules, key=lambda rule: rule.confidence)
            before = score
            score = min(score, tuning.scam_ceiling_score)
            adjustments.append(
                ScoreAdjustment(
                    rule_id=strongest.rule_id,
                    kind=OverrideKind.SCAM_SUPPRESSION,
                    before=before,
                    after=score,
                    reason=(
                        f"Scam pattern {strongest.rule_id} "
                        f"(confidence {strongest.confidence:.2f}) suppressed the score "
                        f"to at most {tuning.scam_ceiling_score:.0f}."
                    ),
                )
            )
            return score

        spam_rules = [
            rule for rule in rules.rules_in(RuleFamily.SPAM) if rule.rule_id == "spam_detected"
        ]
        if spam_rules:
            strongest = max(spam_rules, key=lambda rule: rule.confidence)
            before = score
            score = min(score, tuning.spam_ceiling_score)
            adjustments.append(
                ScoreAdjustment(
                    rule_id=strongest.rule_id,
                    kind=OverrideKind.SCAM_SUPPRESSION,
                    before=before,
                    after=score,
                    reason=(
                        f"Spam pattern suppressed the score to at most "
                        f"{tuning.spam_ceiling_score:.0f}."
                    ),
                )
            )
        return score

    def _apply_hard_constraints(
        self,
        score: float,
        rules: RuleEvaluation,
        adjustments: list[ScoreAdjustment],
    ) -> float:
        """Translate the rule engine's band constraints into score bounds.

        The rule engine records floors, ceilings and forces in terms of an
        action band. This engine cannot choose a band, but it *can* move the
        score into a region consistent with each constraint, so the decision
        layer sees a score that already agrees with the rules. The
        :class:`~src.schema.OverrideRecord` objects themselves are passed
        through untouched and resolved downstream.

        FORCE constraints are applied last and unconditionally, since they
        outrank floors by the arbiter's documented conflict rule.
        """
        tuning = self._tuning

        floor_targets = {
            Action.NOTIFY: tuning.hard_floor_notify_score,
            Action.DIGEST: tuning.hard_floor_digest_score,
            Action.MUTE: self._config.scoring.score_min,
        }
        ceiling_targets = {
            Action.MUTE: tuning.hard_ceiling_mute_score,
            Action.DIGEST: tuning.hard_ceiling_digest_score,
            Action.NOTIFY: self._config.scoring.score_max,
        }

        for override in rules.overrides:
            if override.effect is OverrideEffect.FLOOR:
                target = floor_targets.get(override.bound)
                if target is None or score >= target:
                    continue
                before = score
                score = target
                adjustments.append(
                    ScoreAdjustment(
                        rule_id=override.rule_id,
                        kind=OverrideKind.SCORE_FLOOR,
                        before=before,
                        after=score,
                        reason=(
                            f"Rule {override.rule_id} floors the outcome at "
                            f"{override.bound.value}; score raised to {target:.0f}."
                        ),
                    )
                )
            elif override.effect is OverrideEffect.CEILING:
                target = ceiling_targets.get(override.bound)
                if target is None or score <= target:
                    continue
                before = score
                score = target
                adjustments.append(
                    ScoreAdjustment(
                        rule_id=override.rule_id,
                        kind=OverrideKind.SCORE_CEILING,
                        before=before,
                        after=score,
                        reason=(
                            f"Rule {override.rule_id} caps the outcome at "
                            f"{override.bound.value}; score lowered to {target:.0f}."
                        ),
                    )
                )

        for override in rules.overrides:
            if override.effect is not OverrideEffect.FORCE:
                continue
            target = {
                Action.MUTE: self._config.scoring.score_min,
                Action.DIGEST: tuning.hard_floor_digest_score,
                Action.NOTIFY: tuning.hard_floor_notify_score,
            }.get(override.bound)
            if target is None:
                continue
            before = score
            score = target
            adjustments.append(
                ScoreAdjustment(
                    rule_id=override.rule_id,
                    kind=OverrideKind.HARD_OVERRIDE,
                    before=before,
                    after=score,
                    reason=(
                        f"Rule {override.rule_id} forces {override.bound.value}; "
                        f"score set to {target:.0f}."
                    ),
                )
            )

        return score

    # ------------------------------------------------------------------ #
    # Confidence
    # ------------------------------------------------------------------ #

    def _compute_confidence(
        self,
        inputs: ScoringInput,
        contributions: Sequence[ScoreContribution],
        rules: RuleEvaluation,
        adjustments: Sequence[ScoreAdjustment],
    ) -> ConfidenceBreakdown:
        """Compute a calibrated confidence from six independent factors.

        Deliberately avoids returning ``1.0``: the only path to a very high
        figure is a deterministic FORCE override, and even that is capped at
        :attr:`PriorityTuning.deterministic_confidence`.

        Parameters
        ----------
        inputs:
            Bundled scoring inputs.
        contributions:
            Component contributions.
        rules:
            The rule evaluation.
        adjustments:
            Score adjustments already applied.

        Returns
        -------
        ConfidenceBreakdown
        """
        tuning = self._tuning
        confidence_config = self._config.confidence
        breakdown = ConfidenceBreakdown()

        # 1. Coverage: how many components had something to say.
        firing = [c for c in contributions if c.fired]
        breakdown.coverage = (
            clamp_unit(len(firing) / max(len(contributions), 1)) if contributions else 0.0
        )

        # 2. Agreement: do the components point the same direction?
        breakdown.agreement = self._component_agreement(firing)

        # 3. Rule strength: how confident was the strongest rule?
        if rules.triggered_rules:
            strongest = max(rules.triggered_rules, key=lambda rule: rule.confidence)
            breakdown.rule_strength = clamp_unit(strongest.confidence)
        else:
            breakdown.rule_strength = 0.45

        # 4. Evidence availability: is there real interaction history?
        events = inputs.sender_events.total_events
        breakdown.evidence_availability = clamp_unit(
            0.20 + 0.80 * min(events / 10.0, 1.0)
        )

        # 5. Retrieval completeness: how full was the evidence pool?
        max_candidates = max(self._config.retrieval.max_candidates, 1)
        pool_size = len(inputs.context.retrieval.candidates)
        breakdown.retrieval_completeness = clamp_unit(pool_size / max_candidates)

        # 6. The components' own self-reported confidence, weighted by how much
        #    each actually contributed, so a confident but silent component
        #    does not dominate.
        breakdown.component_confidence = self._weighted_component_confidence(contributions)

        blended = (
            tuning.weight_coverage * breakdown.coverage
            + tuning.weight_agreement * breakdown.agreement
            + tuning.weight_rule_strength * breakdown.rule_strength
            + tuning.weight_evidence * breakdown.evidence_availability
            + tuning.weight_retrieval * breakdown.retrieval_completeness
            + tuning.weight_component_confidence * breakdown.component_confidence
        )

        # A deterministic FORCE override is one of the few genuinely certain
        # situations, so it raises the ceiling -- but not to 1.0.
        has_force = any(
            adjustment.kind is OverrideKind.HARD_OVERRIDE and adjustment.binding
            for adjustment in adjustments
        )
        if has_force:
            blended = max(blended, tuning.deterministic_confidence)

        breakdown.final = clamp(
            blended,
            confidence_config.confidence_min,
            confidence_config.confidence_max,
        )
        return breakdown

    @staticmethod
    def _component_agreement(firing: Sequence[ScoreContribution]) -> float:
        """Measure how consistently the firing components point one direction.

        Returns ``1.0`` when every component agrees in sign and ``0.0`` when
        positive and penalty magnitudes exactly cancel. A single firing
        component returns a moderate value rather than perfect agreement,
        since one voice is not a consensus.

        Parameters
        ----------
        firing:
            Components that contributed a non-zero amount.

        Returns
        -------
        float
            Agreement in ``[0, 1]``.
        """
        if not firing:
            return 0.0
        if len(firing) == 1:
            return 0.55

        positive = sum(c.points for c in firing if c.points > 0)
        negative = abs(sum(c.points for c in firing if c.points < 0))
        total = positive + negative
        if total <= 0:
            return 0.0

        # 1.0 when all magnitude is on one side, 0.0 when perfectly split.
        return clamp_unit(abs(positive - negative) / total)

    @staticmethod
    def _weighted_component_confidence(
        contributions: Sequence[ScoreContribution],
    ) -> float:
        """Weight each component's self-reported confidence by its magnitude.

        A component that contributed nothing should not drag the mean up or
        down, so weights are proportional to absolute points contributed.

        Parameters
        ----------
        contributions:
            Component contributions.

        Returns
        -------
        float
            Weighted mean confidence in ``[0, 1]``, or a neutral ``0.5`` when
            nothing contributed.
        """
        weighted_sum = 0.0
        weight_total = 0.0
        for contribution in contributions:
            weight = abs(contribution.points)
            if weight <= 0.0:
                continue
            weighted_sum += contribution.confidence * weight
            weight_total += weight

        if weight_total <= 0.0:
            return 0.5
        return clamp_unit(weighted_sum / weight_total)

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_evidence(
        contributions: Sequence[ScoreContribution],
        rules: RuleEvaluation,
    ) -> tuple[str, ...]:
        """Gather deduplicated evidence ids from components and rules.

        Ordered by source strength: rule evidence first (rules are the more
        precise signal), then component evidence.

        Parameters
        ----------
        contributions:
            Component contributions.
        rules:
            The rule evaluation.

        Returns
        -------
        tuple[str, ...]
            Deduplicated, order-preserved evidence ids.
        """
        collected: list[str] = []
        for rule in sorted(rules.triggered_rules, key=lambda r: r.confidence, reverse=True):
            collected.extend(rule.evidence_message_ids)
        for contribution in sorted(
            contributions, key=lambda c: abs(c.points), reverse=True
        ):
            collected.extend(contribution.evidence_message_ids)
        return tuple(dict.fromkeys(mid for mid in collected if mid))

    def _build_explanation(
        self,
        score: float,
        level: PriorityLevel,
        contributions: Sequence[ScoreContribution],
        adjustments: Sequence[ScoreAdjustment],
        rules: RuleEvaluation,
    ) -> str:
        """Compose a one-paragraph human-readable summary.

        Deliberately phrased in terms of priority, never in terms of a routing
        action, since this engine does not choose one.

        Parameters
        ----------
        score:
            Final priority score.
        level:
            Descriptive priority level.
        contributions:
            Component contributions.
        adjustments:
            Score adjustments applied.
        rules:
            The rule evaluation.

        Returns
        -------
        str
        """
        parts: list[str] = [
            f"Priority score {score:.0f}/100 ({level.value})."
        ]

        firing = sorted(
            (c for c in contributions if c.fired), key=lambda c: abs(c.points), reverse=True
        )
        if firing:
            drivers = ", ".join(
                f"{c.name} {c.points:+.0f}" for c in firing[:3]
            )
            parts.append(f"Main drivers: {drivers}.")
        else:
            parts.append("No scoring component produced a signal.")

        binding = [adjustment for adjustment in adjustments if adjustment.binding]
        if binding:
            strongest = max(binding, key=lambda adjustment: abs(adjustment.delta))
            parts.append(strongest.reason)

        if rules.overrides:
            constraint_text = ", ".join(
                f"{override.rule_id} ({override.effect.value} {override.bound.value})"
                for override in rules.overrides[:2]
            )
            parts.append(f"Pending constraints for the decision layer: {constraint_text}.")

        return " ".join(parts)

    def _build_metadata(
        self,
        inputs: ScoringInput,
        contributions: Sequence[ScoreContribution],
        rules: RuleEvaluation,
        adjustments: Sequence[ScoreAdjustment],
        positive_score: float,
        penalty_score: float,
    ) -> dict[str, Any]:
        """Assemble the flat metadata bundle for downstream explanation generation.

        Parameters
        ----------
        inputs:
            Bundled scoring inputs.
        contributions:
            Component contributions.
        rules:
            The rule evaluation.
        adjustments:
            Score adjustments applied.
        positive_score:
            Sum of positive contributions.
        penalty_score:
            Sum of penalty contributions.

        Returns
        -------
        dict
        """
        metadata: dict[str, Any] = {
            "component_count": len(contributions),
            "firing_component_count": sum(1 for c in contributions if c.fired),
            "component_points": {
                c.name: round(c.points, 3) for c in contributions
            },
            "component_utilisation": {
                c.name: round(c.utilisation, 3) for c in contributions
            },
            "positive_score": round(positive_score, 3),
            "penalty_score": round(penalty_score, 3),
            "rule_count": len(rules.triggered_rules),
            "rule_families": [family.value for family in rules.families_triggered],
            "rule_total_weight": round(rules.total_weight, 3),
            "adjustment_count": len(adjustments),
            "binding_adjustment_count": sum(1 for a in adjustments if a.binding),
            "override_count": len(rules.overrides),
            "has_force_override": any(
                override.effect is OverrideEffect.FORCE for override in rules.overrides
            ),
            "floor_count": sum(
                1 for o in rules.overrides if o.effect is OverrideEffect.FLOOR
            ),
            "ceiling_count": sum(
                1 for o in rules.overrides if o.effect is OverrideEffect.CEILING
            ),
            "retrieval_pool_size": len(inputs.context.retrieval.candidates),
            "retrieval_used_lexical": inputs.context.retrieval.used_lexical,
            "sender_event_count": inputs.sender_events.total_events,
            "cap_budget": self._caps.to_dict(),
        }
        metadata.update(
            {
                f"rules_{key}": value
                for key, value in rules.metadata.items()
                if isinstance(value, (bool, int, float, str))
            }
        )
        return metadata


__all__ = [
    "DEFAULT_TUNING",
    "ConfidenceBreakdown",
    "OverrideKind",
    "PriorityAssessment",
    "PriorityEngine",
    "PriorityLevel",
    "PriorityResult",
    "PriorityTuning",
    "ScoreAdjustment",
]