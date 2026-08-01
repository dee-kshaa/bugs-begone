"""
Reusable scoring components for the priority engine.

Each component answers one narrow question about a message ("how close is this
sender?", "how time-critical is this?", "how much has the user engaged with
this source before?") and returns a :class:`ScoreContribution` carrying points,
a cap, a confidence, a human-readable explanation, and supporting evidence.

The priority engine (``src/scoring/priority.py``, not this module) sums these
contributions, applies thresholds, and produces a
:class:`~src.schema.PriorityResult`. This module produces *inputs* to that
aggregation and never decides a band or an action.

Extensibility
-------------
Every component subclasses :class:`ScoringComponent` and is registered in
:data:`COMPONENT_REGISTRY`. New components can be added with
:func:`register_component` without any change to downstream code: the priority
engine iterates the registry rather than naming components individually.

Cap budget
----------
The nine components here are sub-splits of the six caps declared in
:class:`~src.config.ScoringConfig`, arranged so the positive caps still sum to
100 and the penalty caps to -30. See :class:`ComponentCaps`.

Dependencies
------------
``src.config``, ``src.schema``, ``src.rules.engine``, ``src.retrieval.context``,
``src.features.content``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from src.config import AppConfig, ScoringConfig, get_config
from src.features.content import ContentVerdict, analyse_content
from src.retrieval.context import EventSummary, MessageContext
from src.rules.engine import RuleEvaluation, RuleFamily, TriggeredRule
from src.schema import (
    BusinessProfile,
    GroupProfile,
    Message,
    RelationshipCategory,
    RelationshipResult,
    ScoreComponent,
    UserProfile,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range ``[low, high]``."""
    return max(low, min(high, value))


def clamp_unit(value: float) -> float:
    """Clamp ``value`` into ``[0.0, 1.0]``."""
    return clamp(float(value), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Cap budget
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ComponentCaps:
    """Per-component point ceilings, derived from the frozen scoring config.

    The six caps in :class:`~src.config.ScoringConfig` are split into nine
    component caps here. Positive caps sum to 100 and penalty caps to -30, so
    the overall score envelope is unchanged from the frozen design.

    Attributes
    ----------
    relationship, group:
        Split of ``cap_relationship`` (30 by default) into the sender's own
        closeness and the surrounding group's characteristics.
    urgency:
        Full ``cap_urgency``.
    trust, business:
        Split of ``cap_trust + cap_user_preference`` (30 by default). Trust
        absorbs the user's explicit preference signals (pinned, muted,
        verified sender), business covers commercial relationship depth.
    history, engagement:
        Split of ``cap_historical_behaviour`` (15 by default) into the user's
        observed open/reply behaviour and their dismissal/abandonment pattern.
    safety, media:
        Split of ``floor_safety`` (-30 by default). Media is a small
        reliability penalty for untranscribable content; safety carries the
        spam/scam/duplicate/burst penalties.
    """

    relationship: float = 22.0
    group: float = 8.0
    urgency: float = 25.0
    trust: float = 16.0
    business: float = 14.0
    history: float = 8.0
    engagement: float = 7.0
    safety: float = -24.0
    media: float = -6.0

    @classmethod
    def from_scoring_config(cls, config: ScoringConfig) -> "ComponentCaps":
        """Derive component caps proportionally from a :class:`ScoringConfig`.

        Splitting ratios are fixed; only the totals move if the frozen config
        is retuned, so the envelope stays consistent with whatever the config
        declares.

        Parameters
        ----------
        config:
            The scoring configuration whose six caps are being subdivided.

        Returns
        -------
        ComponentCaps
        """
        relationship_total = config.cap_relationship
        trust_total = config.cap_trust + config.cap_user_preference
        history_total = config.cap_historical_behaviour
        penalty_total = config.floor_safety

        return cls(
            relationship=relationship_total * (22.0 / 30.0),
            group=relationship_total * (8.0 / 30.0),
            urgency=config.cap_urgency,
            trust=trust_total * (16.0 / 30.0),
            business=trust_total * (14.0 / 30.0),
            history=history_total * (8.0 / 15.0),
            engagement=history_total * (7.0 / 15.0),
            safety=penalty_total * (24.0 / 30.0),
            media=penalty_total * (6.0 / 30.0),
        )

    @property
    def positive_total(self) -> float:
        """Sum of every positive component cap."""
        return (
            self.relationship
            + self.group
            + self.urgency
            + self.trust
            + self.business
            + self.history
            + self.engagement
        )

    @property
    def penalty_total(self) -> float:
        """Sum of every penalty component floor (a negative number)."""
        return self.safety + self.media

    def to_dict(self) -> dict[str, float]:
        """Serialise the cap budget for logging or the explanation trace."""
        return {
            "relationship": round(self.relationship, 3),
            "group": round(self.group, 3),
            "urgency": round(self.urgency, 3),
            "trust": round(self.trust, 3),
            "business": round(self.business, 3),
            "history": round(self.history, 3),
            "engagement": round(self.engagement, 3),
            "safety": round(self.safety, 3),
            "media": round(self.media, 3),
            "positive_total": round(self.positive_total, 3),
            "penalty_total": round(self.penalty_total, 3),
        }


# --------------------------------------------------------------------------- #
# Common abstractions
# --------------------------------------------------------------------------- #


@dataclass
class ScoringInput:
    """Everything a scoring component may need for one message.

    Bundled into a single object so that adding a new component never requires
    changing the priority engine's call signature. Components read only the
    fields they care about and must tolerate the optional ones being ``None``.

    Attributes
    ----------
    message:
        The message being scored.
    context:
        Assembled :class:`~src.retrieval.context.MessageContext` -- profiles,
        event summaries, duplicate/report status, evidence pool.
    rules:
        :class:`~src.rules.engine.RuleEvaluation` for this message. Components
        may read triggered rules and their metadata rather than re-deriving
        the same signals.
    relationship:
        Fused relationship result for the sender, when available.
    content:
        Pre-computed content verdict. Derived from the message on first access
        when omitted.
    features:
        Optional merged feature dictionary from ``src.features.*``. Read
        defensively with :meth:`feature`.
    """

    message: Message
    context: MessageContext
    rules: RuleEvaluation
    relationship: RelationshipResult | None = None
    content: ContentVerdict | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive the content verdict if the caller did not supply one."""
        if self.content is None:
            self.content = analyse_content(
                self.message.content,
                mentions_user=bool(self.rules.metadata.get("mentions_user")),
                is_reply_to_user=bool(self.rules.metadata.get("is_reply_to_user")),
                message=self.message,
            )

    # ---- convenience accessors ----------------------------------------- #

    @property
    def verdict(self) -> ContentVerdict:
        """The content verdict, guaranteed non-``None`` after construction."""
        assert self.content is not None  # established in __post_init__
        return self.content

    @property
    def user(self) -> UserProfile | None:
        """The receiving user's profile, when available."""
        return self.context.user

    @property
    def group(self) -> GroupProfile | None:
        """The group's profile, when this is a group message."""
        return self.context.group

    @property
    def business(self) -> BusinessProfile | None:
        """The sending business account's profile, when applicable."""
        return self.context.business

    @property
    def sender_events(self) -> EventSummary:
        """The user's historical event summary for this sender."""
        return self.context.sender_event_summary

    def feature(self, key: str, default: Any = None) -> Any:
        """Read one entry from the optional merged feature dictionary.

        Parameters
        ----------
        key:
            Feature name, e.g. ``"time_is_bursting"``.
        default:
            Value returned when the key is absent.

        Returns
        -------
        Any
        """
        return self.features.get(key, default)

    def metadata(self, key: str, default: Any = None) -> Any:
        """Read one entry from the rule engine's metadata bundle.

        Preferred over recomputing a signal the rule engine already derived.

        Parameters
        ----------
        key:
            Metadata name, e.g. ``"mentions_user"``.
        default:
            Value returned when the key is absent.

        Returns
        -------
        Any
        """
        return self.rules.metadata.get(key, default)

    def rules_in(self, family: RuleFamily) -> tuple[TriggeredRule, ...]:
        """Return every triggered rule belonging to ``family``."""
        return self.rules.rules_in(family)

    def rule_weight_in(self, *families: RuleFamily) -> float:
        """Sum the suggested weights of every rule in the given families.

        Parameters
        ----------
        *families:
            One or more rule families to aggregate over.

        Returns
        -------
        float
            Signed total; ``0.0`` when no rule in those families fired.
        """
        selected = set(families)
        return sum(
            rule.weight for rule in self.rules.triggered_rules if rule.family in selected
        )

    def has_rule(self, rule_id: str) -> bool:
        """Return whether a rule with the given id fired."""
        return any(rule.rule_id == rule_id for rule in self.rules.triggered_rules)


@dataclass
class ScoreContribution:
    """One component's contribution to the priority score.

    Attributes
    ----------
    name:
        Component name, matching :attr:`ScoringComponent.name`.
    points:
        Signed contribution, clamped into the component's own range by
        :meth:`__post_init__`.
    cap:
        The component's ceiling (positive components) or floor (penalty
        components, where ``cap`` is negative).
    confidence:
        How certain this component is about its own contribution, in
        ``[0, 1]``. The priority engine may use this to discount a component
        rather than dropping it.
    explanation:
        One-sentence human-readable summary for the demo trace.
    reasons:
        Itemised sub-reasons, each already human-readable.
    evidence_message_ids:
        Message ids supporting this contribution, where applicable.
    """

    name: str
    points: float
    cap: float
    confidence: float = 1.0
    explanation: str = ""
    reasons: tuple[str, ...] = ()
    evidence_message_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Clamp points into the component's range and validate confidence."""
        self.name = str(self.name).strip()
        self.points = float(self.points)
        self.cap = float(self.cap)
        if self.cap >= 0:
            self.points = clamp(self.points, 0.0, self.cap)
        else:
            self.points = clamp(self.points, self.cap, 0.0)

        self.confidence = clamp_unit(self.confidence)
        self.reasons = tuple(str(reason) for reason in self.reasons)
        self.evidence_message_ids = tuple(str(mid) for mid in self.evidence_message_ids)

        if not self.explanation:
            self.explanation = self._default_explanation()

    def _default_explanation(self) -> str:
        """Build a fallback explanation when the component supplied none."""
        if not self.reasons:
            return f"{self.name}: no signal ({self.points:+.1f})."
        return f"{self.name}: {self.reasons[0]} ({self.points:+.1f})."

    @property
    def utilisation(self) -> float:
        """Fraction of the component's range actually used, in ``[0, 1]``."""
        if self.cap == 0:
            return 0.0
        return clamp_unit(abs(self.points) / abs(self.cap))

    @property
    def is_penalty(self) -> bool:
        """``True`` for components whose cap is negative."""
        return self.cap < 0

    @property
    def fired(self) -> bool:
        """``True`` when this component contributed anything at all."""
        return abs(self.points) > 1e-9

    def to_score_component(self) -> ScoreComponent:
        """Convert to the frozen :class:`~src.schema.ScoreComponent` type.

        The priority engine uses this to build a
        :class:`~src.schema.PriorityResult` without this module needing to
        know how that result is assembled.

        Returns
        -------
        ScoreComponent
        """
        return ScoreComponent(
            name=self.name,
            points=self.points,
            cap=self.cap,
            reasons=self.reasons,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary for the explanation trace."""
        return {
            "name": self.name,
            "points": round(self.points, 3),
            "cap": round(self.cap, 3),
            "utilisation": round(self.utilisation, 3),
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "reasons": list(self.reasons),
            "evidence_message_ids": list(self.evidence_message_ids),
        }


class ScoringComponent(ABC):
    """Base class every scoring component implements.

    Subclasses declare a :attr:`name` and a :attr:`cap`, then implement
    :meth:`compute`. The public :meth:`score` wrapper handles clamping and
    exception isolation, so a single misbehaving component degrades to a
    zero contribution rather than failing the whole message.

    Parameters
    ----------
    caps:
        Component cap budget. Defaults to the budget derived from the
        process-wide scoring configuration.
    config:
        Application configuration. Defaults to the process-wide singleton.
    """

    #: Stable component name, used in traces and in the registry.
    name: str = "component"

    def __init__(
        self,
        caps: ComponentCaps | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self._config = config or get_config()
        self._caps = caps or ComponentCaps.from_scoring_config(self._config.scoring)

    @property
    @abstractmethod
    def cap(self) -> float:
        """This component's point ceiling, or floor when negative."""

    @abstractmethod
    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Compute this component's contribution.

        Subclasses implement scoring logic here. Points need not be
        pre-clamped: :class:`ScoreContribution` clamps on construction.

        Parameters
        ----------
        inputs:
            The bundled scoring inputs for one message.

        Returns
        -------
        ScoreContribution
        """

    def score(self, inputs: ScoringInput) -> ScoreContribution:
        """Public entry point: compute the contribution, isolating failures.

        Parameters
        ----------
        inputs:
            The bundled scoring inputs for one message.

        Returns
        -------
        ScoreContribution
            A zero contribution with an explanatory reason if :meth:`compute`
            raised, so one broken component never blocks a message.
        """
        try:
            return self.compute(inputs)
        except Exception as error:  # noqa: BLE001 - isolation is the point
            logger.error(
                "Scoring component %r failed on message_id=%s (%s); contributing 0.",
                self.name,
                inputs.message.message_id,
                error,
            )
            return ScoreContribution(
                name=self.name,
                points=0.0,
                cap=self.cap,
                confidence=0.0,
                explanation=f"{self.name}: component error, contributed 0.",
                reasons=(f"error: {error}",),
            )

    def zero(self, reason: str, confidence: float = 0.5) -> ScoreContribution:
        """Build a zero contribution with an explanatory reason.

        Convenience for the common "no applicable signal" path.

        Parameters
        ----------
        reason:
            Why this component contributed nothing.
        confidence:
            Confidence in the null result.

        Returns
        -------
        ScoreContribution
        """
        return ScoreContribution(
            name=self.name,
            points=0.0,
            cap=self.cap,
            confidence=confidence,
            explanation=f"{self.name}: {reason}",
            reasons=(reason,),
        )


# --------------------------------------------------------------------------- #
# Relationship
# --------------------------------------------------------------------------- #

#: Base point value per relationship category, before confidence shrinkage.
#: Expressed as a fraction of the relationship cap so retuning the cap scales
#: every category proportionally.
_CATEGORY_WEIGHT: dict[RelationshipCategory, float] = {
    RelationshipCategory.FAMILY: 1.00,
    RelationshipCategory.CLOSE_FRIEND: 0.85,
    RelationshipCategory.OFFICE: 0.65,
    RelationshipCategory.COLLEGE: 0.55,
    RelationshipCategory.SOCIETY: 0.40,
    RelationshipCategory.BUSINESS: 0.25,
    RelationshipCategory.UNKNOWN: 0.15,
}


class RelationshipScore(ScoringComponent):
    """Scores how close the sender is to the user.

    Applies confidence shrinkage: a low-confidence Family classification is
    pulled toward a neutral prior rather than granted full Family points, so a
    single bad name match cannot cascade into a wrong high-priority routing.
    """

    name = "Relationship"

    @property
    def cap(self) -> float:
        """The relationship component's ceiling."""
        return self._caps.relationship

    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Score the sender's relationship closeness with shrinkage applied."""
        scoring_config = self._config.scoring
        relationship = inputs.relationship

        if relationship is not None:
            category = relationship.category
            confidence = relationship.confidence
            method = relationship.method
            evidence = relationship.evidence_message_ids
        else:
            category_name = inputs.metadata("relationship_category", "Unknown")
            category = RelationshipCategory.from_any(category_name)
            confidence = float(inputs.metadata("relationship_confidence", 0.0) or 0.0)
            method = "rule_metadata"
            evidence = ()

        if category is RelationshipCategory.UNKNOWN and confidence <= 0.0:
            return self.zero("sender relationship could not be established", confidence=0.3)

        reasons: list[str] = []
        full_points = self.cap * _CATEGORY_WEIGHT.get(category, 0.15)

        # Shrinkage toward the neutral prior, scaled into this component's cap.
        prior_fraction = scoring_config.relationship_prior_mean / max(
            scoring_config.cap_relationship, 1e-9
        )
        prior_points = self.cap * prior_fraction
        shrunk_points = prior_points + confidence * (full_points - prior_points)

        reasons.append(
            f"{category.value} (confidence {confidence:.2f}, method {method})"
        )
        if confidence < 0.75:
            reasons.append(
                f"shrunk from {full_points:.1f} to {shrunk_points:.1f} on low confidence"
            )

        points = shrunk_points

        # A 1:1 conversation carries more weight than the same person in a group.
        is_direct = bool(inputs.metadata("is_direct_message", not inputs.message.is_group_message))
        if is_direct:
            direct_fraction = scoring_config.direct_conversation_bonus / max(
                scoring_config.cap_relationship, 1e-9
            )
            bonus = self.cap * direct_fraction
            points += bonus
            reasons.append(f"direct 1:1 conversation (+{bonus:.1f})")

        explanation = (
            f"Sender classified as {category.value} at confidence {confidence:.2f}"
            f"{' in a direct chat' if is_direct else ' in a group'}."
        )

        return ScoreContribution(
            name=self.name,
            points=points,
            cap=self.cap,
            confidence=clamp_unit(0.4 + 0.6 * confidence),
            explanation=explanation,
            reasons=tuple(reasons),
            evidence_message_ids=evidence,
        )


# --------------------------------------------------------------------------- #
# Urgency
# --------------------------------------------------------------------------- #


class UrgencyScore(ScoringComponent):
    """Scores how time-critical the message is.

    Deliberately bimodal: a message with a real deadline that names the user
    lands near the cap, an ordinary message lands near zero. A bell-shaped
    urgency distribution would collapse every message into the middle band.
    """

    name = "Urgency"

    @property
    def cap(self) -> float:
        """The urgency component's ceiling."""
        return self._caps.urgency

    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Score time-criticality from content signals and emergency rules."""
        verdict = inputs.verdict
        reasons: list[str] = []
        evidence: list[str] = []

        # The content layer already produced a calibrated [0,1] urgency score.
        base_fraction = clamp_unit(verdict.urgency_score)
        points = self.cap * base_fraction
        if base_fraction > 0.0:
            reasons.append(f"content urgency {base_fraction:.2f}")

        if verdict.has_deadline:
            reasons.append("explicit deadline present")

        if inputs.metadata("mentions_user"):
            reasons.append("user is @-mentioned")
        if inputs.metadata("is_reply_to_user"):
            reasons.append("direct reply to the user")

        # Emergency-family rules (OTP, emergency language) dominate when present.
        emergency_rules = inputs.rules_in(RuleFamily.EMERGENCY)
        if emergency_rules:
            strongest = max(emergency_rules, key=lambda rule: rule.confidence)
            emergency_fraction = clamp_unit(0.75 + 0.25 * strongest.confidence)
            emergency_points = self.cap * emergency_fraction
            if emergency_points > points:
                points = emergency_points
            reasons.append(f"emergency rule fired: {strongest.rule_id}")
            evidence.extend(strongest.evidence_message_ids)

        if not reasons:
            return self.zero("no time-critical signals detected", confidence=0.6)

        confidence = 0.85 if emergency_rules else clamp_unit(0.45 + 0.4 * base_fraction)
        explanation = (
            f"Message reads as {'highly ' if points > self.cap * 0.7 else ''}"
            f"time-critical: {reasons[0]}."
        )

        return ScoreContribution(
            name=self.name,
            points=points,
            cap=self.cap,
            confidence=confidence,
            explanation=explanation,
            reasons=tuple(reasons),
            evidence_message_ids=tuple(dict.fromkeys(evidence)),
        )


# --------------------------------------------------------------------------- #
# Trust
# --------------------------------------------------------------------------- #


class TrustScore(ScoringComponent):
    """Scores how much the user has signalled they trust this sender.

    Absorbs the explicit user-preference signals (pinned, muted, blocked) as
    well as sender-recognition signals, since both answer the same question:
    does this user want to hear from this source?
    """

    name = "Trust"

    @property
    def cap(self) -> float:
        """The trust component's ceiling."""
        return self._caps.trust

    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Score sender trust from explicit preferences and recognition."""
        reasons: list[str] = []
        points = 0.0

        user = inputs.user
        sender_id = inputs.message.sender_id

        if user is not None:
            if sender_id in user.pinned_contacts:
                points += self.cap * 0.55
                reasons.append("sender is pinned or starred by the user")
            if sender_id in user.muted_contacts:
                points -= self.cap * 0.40
                reasons.append("sender is muted by the user")
            if sender_id in user.blocked_contacts:
                points -= self.cap * 0.80
                reasons.append("sender is blocked by the user")

            group_id = inputs.message.group_id
            if group_id and group_id in user.muted_groups:
                points -= self.cap * 0.30
                reasons.append("group is muted by the user")

        # Sender recognition: a known contact outranks an unrecognised number.
        relationship = inputs.relationship
        is_known = bool(
            (relationship is not None and relationship.category is not RelationshipCategory.UNKNOWN)
            or inputs.sender_events.total_events > 0
        )
        if is_known:
            points += self.cap * 0.30
            reasons.append("sender is recognised from prior history")
        else:
            reasons.append("sender is not recognised")

        # Verified business accounts carry institutional trust.
        business = inputs.business
        if business is not None and business.is_verified:
            points += self.cap * 0.25
            reasons.append("business account is verified")

        # Trust is inversely related to spam-likeness.
        spam_score = clamp_unit(inputs.verdict.spam_score)
        if spam_score > 0.0:
            penalty = self.cap * 0.50 * spam_score
            points -= penalty
            reasons.append(f"content spam signal {spam_score:.2f} reduces trust")

        if inputs.metadata("is_reported_recently"):
            points -= self.cap * 0.60
            reasons.append("user has previously reported this sender")

        if not reasons:
            return self.zero("no trust signals available", confidence=0.4)

        explanation = f"Trust assessment: {reasons[0]}."
        confidence = 0.75 if user is not None else 0.45

        return ScoreContribution(
            name=self.name,
            points=points,
            cap=self.cap,
            confidence=confidence,
            explanation=explanation,
            reasons=tuple(reasons),
        )


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


class HistoryScore(ScoringComponent):
    """Scores the user's observed engagement with this specific sender.

    This is the component grounded in real behavioural data rather than
    heuristics: if ``message_events.csv`` records that the user opens and
    replies to a source, that outranks any inference about who they are.
    """

    name = "History"

    #: Minimum recorded events before behavioural rates are considered stable.
    MIN_EVENTS_FOR_CONFIDENCE = 3

    @property
    def cap(self) -> float:
        """The history component's ceiling."""
        return self._caps.history

    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Score from the user's historical open and reply rates for this sender."""
        summary = inputs.sender_events
        if summary.total_events == 0:
            return self.zero("no recorded interaction history with this sender", confidence=0.3)

        reasons: list[str] = []

        open_rate = clamp_unit(summary.open_rate)
        reply_rate = clamp_unit(summary.reply_rate)

        # Replying is a much stronger signal of interest than merely opening.
        engagement_fraction = clamp_unit(0.35 * open_rate + 0.65 * reply_rate)
        points = self.cap * engagement_fraction

        reasons.append(f"open rate {open_rate:.0%} over {summary.total_events} event(s)")
        if reply_rate > 0.0:
            reasons.append(f"reply rate {reply_rate:.0%}")

        # Fall back to the user's global prior when this sender's sample is thin.
        user = inputs.user
        if summary.total_events < self.MIN_EVENTS_FOR_CONFIDENCE and user is not None:
            prior_fraction = clamp_unit(
                0.35 * user.overall_open_rate + 0.65 * user.overall_reply_rate
            )
            blended = (engagement_fraction + prior_fraction) / 2.0
            points = self.cap * blended
            reasons.append(
                f"thin sample: blended with the user's global prior ({prior_fraction:.2f})"
            )

        confidence = clamp_unit(
            0.35 + 0.65 * min(summary.total_events / 10.0, 1.0)
        )
        explanation = (
            f"User engages with this sender at {engagement_fraction:.0%} "
            f"across {summary.total_events} recorded event(s)."
        )

        return ScoreContribution(
            name=self.name,
            points=points,
            cap=self.cap,
            confidence=confidence,
            explanation=explanation,
            reasons=tuple(reasons),
        )


# --------------------------------------------------------------------------- #
# Business
# --------------------------------------------------------------------------- #


class BusinessScore(ScoringComponent):
    """Scores the depth of the user's commercial relationship with the sender.

    The distinction that matters is not "is this a business" but "does the
    user have a live relationship with it". A shipping update for this
    morning's order is wanted; the same account's weekend sale is not.
    """

    name = "Business"

    @property
    def cap(self) -> float:
        """The business component's ceiling."""
        return self._caps.business

    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Score the commercial relationship and the message's transactional fit."""
        message = inputs.message
        business = inputs.business
        is_business = bool(
            message.business_id or message.is_from_business or business is not None
        )
        if not is_business:
            return self.zero("not a business message", confidence=0.8)

        verdict = inputs.verdict
        reasons: list[str] = []
        points = 0.0

        if business is None:
            reasons.append("business account has no profile on record")
            confidence = 0.4
        else:
            if business.has_active_order:
                points += self.cap * 0.55
                age = business.last_order_age_days
                age_text = f"{age:.0f} day(s) ago" if age is not None else "recently"
                reasons.append(f"active order placed {age_text}")
            elif business.is_known_to_user:
                points += self.cap * 0.30
                reasons.append(
                    f"user has {business.user_txn_count} prior transaction(s) with this business"
                )
            else:
                reasons.append("user has no purchase history with this business")

            if business.is_verified:
                points += self.cap * 0.15
                reasons.append("account is verified")

            if business.user_open_rate > 0.0:
                points += self.cap * 0.15 * clamp_unit(business.user_open_rate)
                reasons.append(f"user opens {business.user_open_rate:.0%} of this account's messages")

            if business.promo_ratio >= 0.5:
                points -= self.cap * 0.25 * clamp_unit(business.promo_ratio)
                reasons.append(
                    f"{business.promo_ratio:.0%} of this account's traffic is promotional"
                )

            confidence = 0.75

        # A transactional message from a known account is what the user actually wants.
        if verdict.is_transactional and (business is None or business.is_known_to_user):
            points += self.cap * 0.20
            reasons.append("message is transactional rather than promotional")
        elif verdict.is_promotional:
            points -= self.cap * 0.30
            reasons.append("message is promotional")

        explanation = f"Business relationship: {reasons[0]}."

        return ScoreContribution(
            name=self.name,
            points=points,
            cap=self.cap,
            confidence=confidence,
            explanation=explanation,
            reasons=tuple(reasons),
        )


# --------------------------------------------------------------------------- #
# Group
# --------------------------------------------------------------------------- #


class GroupScore(ScoringComponent):
    """Scores how much of a group's traffic is actually meant for this user.

    A three-person group behaves like a direct chat; a two-hundred-person
    announcement channel does not. An explicit @-mention collapses that
    distinction entirely.
    """

    name = "Group"

    @property
    def cap(self) -> float:
        """The group component's ceiling."""
        return self._caps.group

    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Score group relevance from size, mention status, and read rate."""
        message = inputs.message
        if not message.is_group_message:
            return self.zero("not a group message", confidence=0.9)

        group = inputs.group
        reasons: list[str] = []
        points = 0.0

        # A mention means the message is unambiguously for this user.
        if inputs.metadata("mentions_user"):
            points += self.cap * 0.70
            reasons.append("user is @-mentioned in this group")

        if group is None:
            reasons.append("group metadata unavailable")
            return ScoreContribution(
                name=self.name,
                points=points,
                cap=self.cap,
                confidence=0.35,
                explanation="Group message with no profile on record.",
                reasons=tuple(reasons),
            )

        if group.size > 0:
            if group.size <= 5:
                points += self.cap * 0.35
                reasons.append(f"small group ({group.size} members) behaves like a direct chat")
            elif group.is_large:
                points -= self.cap * 0.25
                reasons.append(f"large group ({group.size} members) carries broad traffic")
            else:
                reasons.append(f"mid-sized group ({group.size} members)")

        if group.user_read_rate > 0.0:
            points += self.cap * 0.30 * clamp_unit(group.user_read_rate)
            reasons.append(f"user reads {group.user_read_rate:.0%} of this group's messages")

        if group.user_reply_rate >= 0.15:
            points += self.cap * 0.20
            reasons.append(f"user replies in this group {group.user_reply_rate:.0%} of the time")

        if group.messages_per_day >= 30.0:
            points -= self.cap * 0.20
            reasons.append(f"high volume ({group.messages_per_day:.0f} messages/day)")

        if group.is_muted:
            points -= self.cap * 0.40
            reasons.append("group is muted")

        if inputs.context.sender_is_group_admin:
            points += self.cap * 0.25
            reasons.append("sender is a group admin")

        explanation = f"Group relevance: {reasons[0] if reasons else 'no distinguishing signals'}."

        return ScoreContribution(
            name=self.name,
            points=points,
            cap=self.cap,
            confidence=0.70,
            explanation=explanation,
            reasons=tuple(reasons),
        )


# --------------------------------------------------------------------------- #
# Engagement
# --------------------------------------------------------------------------- #


class EngagementScore(ScoringComponent):
    """Scores conversational momentum and abandonment.

    Distinct from :class:`HistoryScore`, which measures aggregate rates. This
    component asks whether the conversation is *live right now* and whether
    the user has been actively dismissing this source.
    """

    name = "Engagement"

    @property
    def cap(self) -> float:
        """The engagement component's ceiling."""
        return self._caps.engagement

    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Score conversational momentum, penalising a dismissal pattern."""
        summary = inputs.sender_events
        reasons: list[str] = []
        points = 0.0

        # An active thread means the user is probably still looking at it.
        thread_active = bool(inputs.feature("time_thread_is_active", False))
        thread_gap = inputs.feature("time_thread_gap_hours")
        if thread_active:
            points += self.cap * 0.55
            gap_text = f"{thread_gap:.1f}h" if isinstance(thread_gap, (int, float)) else "recently"
            reasons.append(f"conversation is active (last message {gap_text} ago)")

        if inputs.message.reply_to_id:
            points += self.cap * 0.20
            reasons.append("message continues an existing thread")

        # A user who dismisses this source consistently is telling us something.
        dismissal_rate = clamp_unit(summary.dismissal_rate)
        if summary.total_events >= 3 and dismissal_rate >= 0.4:
            points -= self.cap * 0.70 * dismissal_rate
            reasons.append(f"user dismisses {dismissal_rate:.0%} of this sender's messages")

        # First contact from an unknown source has no momentum to speak of.
        if inputs.feature("time_is_first_contact", False):
            reasons.append("first contact from this sender")

        if summary.replies > 0:
            points += self.cap * 0.25 * clamp_unit(summary.replies / 5.0)
            reasons.append(f"{summary.replies} prior repl(y/ies) to this sender")

        if not reasons:
            return self.zero("no engagement signals available", confidence=0.4)

        confidence = clamp_unit(0.4 + 0.5 * min(summary.total_events / 8.0, 1.0))
        explanation = f"Engagement: {reasons[0]}."

        return ScoreContribution(
            name=self.name,
            points=points,
            cap=self.cap,
            confidence=confidence,
            explanation=explanation,
            reasons=tuple(reasons),
        )


# --------------------------------------------------------------------------- #
# Media (penalty)
# --------------------------------------------------------------------------- #


class MediaScore(ScoringComponent):
    """Penalty component for content that could not be reliably transcribed.

    An image whose OCR confidence is 0.2, or a voice note Whisper skipped,
    means every content-derived signal above is standing on sand. This
    component records that uncertainty as a small score penalty; the priority
    engine's confidence calculation handles it separately.
    """

    name = "Media"

    @property
    def cap(self) -> float:
        """The media component's floor (a negative number)."""
        return self._caps.media

    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Penalise low-quality or failed media transcription."""
        message = inputs.message
        if not message.has_media:
            return self.zero("text message; no transcription risk", confidence=0.95)

        reasons: list[str] = []
        quality = clamp_unit(message.media_quality)
        penalty_magnitude = abs(self.cap)
        points = 0.0

        has_extracted_text = bool(message.ocr_text.strip() or message.asr_text.strip())

        if not has_extracted_text:
            points -= penalty_magnitude * 0.80
            reasons.append(f"no text extracted from {message.media_type.value} content")
        elif quality < 0.4:
            points -= penalty_magnitude * 0.60
            reasons.append(f"low transcription quality ({quality:.2f})")
        elif quality < 0.7:
            points -= penalty_magnitude * 0.25
            reasons.append(f"moderate transcription quality ({quality:.2f})")
        else:
            reasons.append(f"transcription quality acceptable ({quality:.2f})")

        # Whisper-skipped clips are flagged by the rule engine's metadata.
        if inputs.metadata("asr_success") is False:
            points -= penalty_magnitude * 0.20
            reasons.append("voice transcription failed or was skipped")
        if inputs.metadata("ocr_success") is False:
            points -= penalty_magnitude * 0.20
            reasons.append("image OCR failed")

        explanation = f"Media reliability: {reasons[0]}."

        return ScoreContribution(
            name=self.name,
            points=points,
            cap=self.cap,
            confidence=0.80,
            explanation=explanation,
            reasons=tuple(reasons),
        )


# --------------------------------------------------------------------------- #
# Safety (penalty)
# --------------------------------------------------------------------------- #


class SafetyScore(ScoringComponent):
    """Penalty component for spam, scam, duplication and flooding.

    Reads the rule engine's spam/scam/promotions findings rather than
    re-deriving them, so a pattern is defined in exactly one place. Note that
    the rule engine also records hard *overrides* for these cases; this
    component contributes the gradient portion, the overrides handle the
    categorical portion.
    """

    name = "Safety"

    @property
    def cap(self) -> float:
        """The safety component's floor (a negative number)."""
        return self._caps.safety

    def compute(self, inputs: ScoringInput) -> ScoreContribution:
        """Apply penalties for spam, scam, duplication, flooding and cold promo."""
        reasons: list[str] = []
        evidence: list[str] = []
        penalty_magnitude = abs(self.cap)
        points = 0.0

        verdict = inputs.verdict

        # Scam is the heaviest penalty; the rule engine also forces a mute.
        scam_rules = inputs.rules_in(RuleFamily.SCAM)
        if scam_rules:
            points -= penalty_magnitude * 0.85
            strongest = max(scam_rules, key=lambda rule: rule.confidence)
            reasons.append(f"scam pattern detected ({strongest.rule_id})")
            evidence.extend(strongest.evidence_message_ids)
        elif verdict.is_spam:
            points -= penalty_magnitude * 0.55 * clamp_unit(verdict.spam_score)
            reasons.append(f"spam signal {verdict.spam_score:.2f}")

        # Repeated cold promotional traffic.
        promo_rules = inputs.rules_in(RuleFamily.PROMOTIONS)
        if any(rule.rule_id == "promotions_repeated_cold" for rule in promo_rules):
            points -= penalty_magnitude * 0.45
            reasons.append("repeated promotional traffic from an unfamiliar account")
        elif verdict.is_promotional:
            points -= penalty_magnitude * 0.20 * clamp_unit(verdict.promo_score)
            reasons.append(f"promotional content ({verdict.promo_score:.2f})")

        if inputs.metadata("is_duplicate"):
            duplicate_count = int(inputs.metadata("duplicate_count", 1) or 1)
            points -= penalty_magnitude * min(0.15 * duplicate_count, 0.40)
            reasons.append(f"duplicate seen {duplicate_count} time(s) recently")
            duplicate_ids = inputs.context.duplicate_evidence_ids
            evidence.extend(duplicate_ids)

        if inputs.metadata("has_forward_marker"):
            points -= penalty_magnitude * 0.20
            reasons.append("chain-forward markers present")

        if inputs.metadata("is_reported_recently"):
            report_count = int(inputs.metadata("report_count", 1) or 1)
            points -= penalty_magnitude * min(0.25 * report_count, 0.50)
            reasons.append(f"sender reported {report_count} time(s) by this user")

        if inputs.feature("time_is_bursting", False):
            burst_count = inputs.feature("time_burst_count", 0)
            points -= penalty_magnitude * 0.25
            reasons.append(f"sender is flooding ({burst_count} messages in the burst window)")

        if not reasons:
            return self.zero("no safety concerns detected", confidence=0.8)

        confidence = 0.85 if scam_rules else 0.70
        explanation = f"Safety penalty: {reasons[0]}."

        return ScoreContribution(
            name=self.name,
            points=points,
            cap=self.cap,
            confidence=confidence,
            explanation=explanation,
            reasons=tuple(reasons),
            evidence_message_ids=tuple(dict.fromkeys(evidence)),
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

#: Factory type for constructing a component with caps and configuration.
ComponentFactory = Callable[[ComponentCaps | None, AppConfig | None], ScoringComponent]

#: Registry of every available component, keyed by name. The priority engine
#: iterates this rather than naming components individually, so a new
#: component becomes active as soon as it is registered.
COMPONENT_REGISTRY: dict[str, type[ScoringComponent]] = {}


def register_component(component_class: type[ScoringComponent]) -> type[ScoringComponent]:
    """Register a scoring component class in :data:`COMPONENT_REGISTRY`.

    Usable as a decorator on new component classes. Downstream code needs no
    change: the priority engine builds whatever the registry contains.

    Parameters
    ----------
    component_class:
        A concrete :class:`ScoringComponent` subclass with a unique ``name``.

    Returns
    -------
    type[ScoringComponent]
        The same class, so this can be used as a decorator.

    Raises
    ------
    ValueError
        If a different class is already registered under the same name.
    """
    name = component_class.name
    existing = COMPONENT_REGISTRY.get(name)
    if existing is not None and existing is not component_class:
        raise ValueError(
            f"A different component is already registered under name {name!r}: {existing!r}"
        )
    COMPONENT_REGISTRY[name] = component_class
    logger.debug("Registered scoring component %r.", name)
    return component_class


for _component_class in (
    RelationshipScore,
    UrgencyScore,
    TrustScore,
    HistoryScore,
    BusinessScore,
    GroupScore,
    EngagementScore,
    MediaScore,
    SafetyScore,
):
    register_component(_component_class)


def build_components(
    caps: ComponentCaps | None = None,
    config: AppConfig | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> list[ScoringComponent]:
    """Instantiate the registered scoring components.

    Parameters
    ----------
    caps:
        Cap budget shared by every component. Defaults to the budget derived
        from the process-wide scoring configuration.
    config:
        Application configuration. Defaults to the process-wide singleton.
    include:
        When given, only components whose names appear here are built.
    exclude:
        Component names to skip. Applied after ``include``.

    Returns
    -------
    list[ScoringComponent]
        Instantiated components, positive-cap ones first then penalties, so
        traces read in a natural order.
    """
    resolved_config = config or get_config()
    resolved_caps = caps or ComponentCaps.from_scoring_config(resolved_config.scoring)

    include_set = {str(name) for name in include} if include is not None else None
    exclude_set = {str(name) for name in exclude} if exclude is not None else set()

    components: list[ScoringComponent] = []
    for name, component_class in COMPONENT_REGISTRY.items():
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        components.append(component_class(caps=resolved_caps, config=resolved_config))

    components.sort(key=lambda component: (component.cap < 0, component.name))
    logger.debug(
        "Built %d scoring component(s): %s",
        len(components),
        [component.name for component in components],
    )
    return components


def score_all(
    inputs: ScoringInput,
    components: Sequence[ScoringComponent] | None = None,
    caps: ComponentCaps | None = None,
    config: AppConfig | None = None,
) -> list[ScoreContribution]:
    """Run every component over one message and collect the contributions.

    Convenience wrapper so the priority engine can call one function rather
    than managing component construction itself. Component failures are
    already isolated inside :meth:`ScoringComponent.score`.

    Parameters
    ----------
    inputs:
        The bundled scoring inputs for one message.
    components:
        Pre-built components. Built from the registry when omitted.
    caps:
        Cap budget, used only when ``components`` is omitted.
    config:
        Application configuration, used only when ``components`` is omitted.

    Returns
    -------
    list[ScoreContribution]
        One contribution per component, in the components' own order.
    """
    active = list(components) if components is not None else build_components(caps, config)
    contributions = [component.score(inputs) for component in active]

    logger.debug(
        "Scored message_id=%s: %s",
        inputs.message.message_id,
        {contribution.name: round(contribution.points, 2) for contribution in contributions},
    )
    return contributions


__all__ = [
    "COMPONENT_REGISTRY",
    "BusinessScore",
    "ComponentCaps",
    "ComponentFactory",
    "EngagementScore",
    "GroupScore",
    "HistoryScore",
    "MediaScore",
    "RelationshipScore",
    "SafetyScore",
    "ScoreContribution",
    "ScoringComponent",
    "ScoringInput",
    "TrustScore",
    "UrgencyScore",
    "build_components",
    "clamp",
    "clamp_unit",
    "register_component",
    "score_all",
]