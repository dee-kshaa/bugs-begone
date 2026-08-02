"""
Historical evidence selection for the decision layer.

Selects which prior message ids should populate ``Decision.evidence_message_ids``.
Combines two sources: message ids that a triggered rule already cited (e.g. the
duplicate rule's matched message), and the ranked retrieval pool assembled by
:class:`~src.retrieval.context.ContextRetriever`. Rule-cited evidence is
preferred, since it is the evidence that directly caused a rule to fire.

Preference order, matching the frozen retrieval tiers
-------------------------------------------------------
1. Message ids cited by a triggered rule (most directly explanatory).
2. Same sender (retrieval tier ``relational``).
3. Same business (retrieval tier ``business``).
4. Same conversation/group (retrieval tier ``structural``).
5. Similar content (retrieval tier ``lexical``).

If nothing qualifies, an empty selection is returned and the caller (the
arbiter) is expected to render this as ``"none"`` in the reason text, per the
task's output contract -- this module returns an empty tuple rather than the
literal string, so downstream code decides how to render it.

Dependencies
------------
``src.schema``, ``src.retrieval.context``, ``src.rules.engine``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.retrieval.context import MessageContext
from src.rules.engine import RuleEvaluation
from src.schema import Message, RetrievalCandidate

logger = logging.getLogger(__name__)

#: Retrieval tier ordering, lower value sorts first. Matches "same sender,
#: same business, same group, similar content" from the task description.
_TIER_PRIORITY: dict[str, int] = {
    "relational": 0,  # same sender
    "business": 1,  # same business
    "structural": 2,  # same conversation / group
    "lexical": 3,  # similar content
}

#: Human-readable label per retrieval tier, used in the explanation.
_TIER_LABEL: dict[str, str] = {
    "relational": "same sender",
    "business": "same business account",
    "structural": "same conversation or group",
    "lexical": "similar content",
}


def clamp_unit(value: float) -> float:
    """Clamp ``value`` into ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class EvidenceItem:
    """One selected piece of evidence.

    Attributes
    ----------
    message_id:
        The historical message id.
    reason:
        Why this message was selected, e.g. ``"same sender"`` or
        ``"cited by rule sys_duplicate_message"``.
    confidence:
        Selection confidence in ``[0, 1]``, derived from the source rule's
        confidence or the retrieval candidate's pre-score.
    source:
        ``"rule"`` or a retrieval tier name.
    """

    message_id: str
    reason: str
    confidence: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "message_id": self.message_id,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }


@dataclass
class EvidenceSelection:
    """Result of selecting evidence for one message.

    Attributes
    ----------
    evidence_message_ids:
        The final, ordered, deduplicated evidence ids. Empty when nothing
        qualified.
    items:
        The full :class:`EvidenceItem` records backing each id, in the same
        order.
    confidence:
        Overall confidence in this evidence selection, in ``[0, 1]``. Used by
        :mod:`src.decision.confidence` as the "historical evidence quality"
        factor.
    source_breakdown:
        Count of selected items per source, for the explanation trace.
    """

    evidence_message_ids: tuple[str, ...] = ()
    items: tuple[EvidenceItem, ...] = ()
    confidence: float = 0.0
    source_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """``True`` when no evidence could be found."""
        return not self.evidence_message_ids

    def render_summary(self) -> str:
        """Return a short human-readable summary, e.g. ``"none"``.

        Used directly wherever the task's output contract calls for the
        literal word ``"none"`` when no evidence exists.
        """
        if self.is_empty:
            return "none"
        labels = sorted({item.reason for item in self.items})
        return "; ".join(labels)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary for the explanation trace."""
        return {
            "evidence_message_ids": list(self.evidence_message_ids),
            "items": [item.to_dict() for item in self.items],
            "confidence": round(self.confidence, 4),
            "source_breakdown": dict(self.source_breakdown),
            "summary": self.render_summary(),
        }


def _rule_sourced_items(
    message: Message,
    rules: RuleEvaluation,
    exclude: set[str],
) -> list[EvidenceItem]:
    """Extract deduplicated evidence items cited by triggered rules.

    Rules are processed in descending confidence order; when the same message
    id is cited by more than one rule, the higher-confidence citation wins.

    Parameters
    ----------
    message:
        The message being decided, whose own id is always excluded.
    rules:
        The rule evaluation for this message.
    exclude:
        Message ids to skip (typically just the current message's own id).

    Returns
    -------
    list of EvidenceItem
    """
    best: dict[str, EvidenceItem] = {}
    for rule in sorted(rules.triggered_rules, key=lambda r: r.confidence, reverse=True):
        for candidate_id in rule.evidence_message_ids:
            if candidate_id in exclude:
                continue
            existing = best.get(candidate_id)
            if existing is not None and existing.confidence >= rule.confidence:
                continue
            best[candidate_id] = EvidenceItem(
                message_id=candidate_id,
                reason=f"cited by rule {rule.rule_id}",
                confidence=rule.confidence,
                source="rule",
            )
    return list(best.values())


def _retrieval_sourced_items(
    candidates: tuple[RetrievalCandidate, ...],
    exclude: set[str],
) -> list[EvidenceItem]:
    """Convert retrieval candidates into evidence items, tier-ordered.

    Parameters
    ----------
    candidates:
        The evidence pool from :attr:`~src.retrieval.context.MessageContext.retrieval`.
    exclude:
        Message ids already selected from rules, so they are not duplicated.

    Returns
    -------
    list of EvidenceItem
        Sorted by tier priority, then by descending pre-score.
    """
    ordered = sorted(
        (c for c in candidates if c.message_id not in exclude),
        key=lambda c: (_TIER_PRIORITY.get(c.tier, len(_TIER_PRIORITY)), -c.pre_score),
    )
    return [
        EvidenceItem(
            message_id=candidate.message_id,
            reason=_TIER_LABEL.get(candidate.tier, candidate.tier),
            confidence=clamp_unit(candidate.pre_score),
            source=candidate.tier,
        )
        for candidate in ordered
    ]


def select_evidence(
    message: Message,
    context: MessageContext,
    rules: RuleEvaluation,
    max_evidence: int = 4,
) -> EvidenceSelection:
    """Select and rank historical evidence for one message's decision.

    Parameters
    ----------
    message:
        The message being decided. Its own id is never included as evidence.
    context:
        Assembled :class:`~src.retrieval.context.MessageContext`, carrying the
        ranked retrieval pool in :attr:`MessageContext.retrieval`.
    rules:
        The rule evaluation for this message; rule-cited message ids are
        preferred over generically retrieved ones.
    max_evidence:
        Maximum number of evidence ids to return.

    Returns
    -------
    EvidenceSelection
        Empty (``evidence_message_ids=()``) when nothing qualifies; callers
        should render this as ``"none"`` via :meth:`EvidenceSelection.render_summary`.
    """
    exclude = {message.message_id}

    rule_items = _rule_sourced_items(message, rules, exclude)
    already_selected = {item.message_id for item in rule_items}

    remaining_slots = max(max_evidence - len(rule_items), 0)
    retrieval_items: list[EvidenceItem] = []
    if remaining_slots > 0:
        retrieval_items = _retrieval_sourced_items(
            context.retrieval.candidates, exclude | already_selected
        )[:remaining_slots]

    items = (rule_items + retrieval_items)[:max_evidence]

    if not items:
        logger.debug("select_evidence: no evidence found for message_id=%s", message.message_id)
        return EvidenceSelection()

    source_breakdown: dict[str, int] = {}
    for item in items:
        source_breakdown[item.source] = source_breakdown.get(item.source, 0) + 1

    mean_confidence = sum(item.confidence for item in items) / len(items)
    # Fewer items than the cap slightly discounts overall confidence: a single
    # weak match is less trustworthy than a full, well-populated pool.
    completeness_factor = 0.5 + 0.5 * min(len(items) / max_evidence, 1.0)
    overall_confidence = clamp_unit(mean_confidence * completeness_factor)

    selection = EvidenceSelection(
        evidence_message_ids=tuple(item.message_id for item in items),
        items=tuple(items),
        confidence=overall_confidence,
        source_breakdown=source_breakdown,
    )
    logger.debug(
        "select_evidence: message_id=%s selected=%d confidence=%.2f breakdown=%s",
        message.message_id,
        len(items),
        overall_confidence,
        source_breakdown,
    )
    return selection


__all__ = [
    "EvidenceItem",
    "EvidenceSelection",
    "select_evidence",
]