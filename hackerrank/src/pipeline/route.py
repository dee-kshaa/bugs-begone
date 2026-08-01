"""
Routing orchestration: Message -> enrichment -> rules -> scoring -> Decision.

:class:`Router` is the pipeline's single entry point. It wires together
:class:`~src.pipeline.enrich.MessageEnricher`, :class:`~src.rules.engine.RuleEngine`,
:class:`~src.scoring.priority.PriorityEngine`, and
:class:`~src.decision.arbiter.Arbiter`, running each incoming
:class:`~src.schema.Message` through the full stack and yielding a
:class:`~src.schema.Decision`.

This module does not write CSVs and does not evaluate against labels -- both
are the responsibility of code outside the pipeline package. It also does not
compute the fused sender relationship: no relationship-engine module has been
built yet in this project, so :class:`Router` accepts an optional
``relationship_resolver`` callable and otherwise passes ``relationship=None``
into the rule engine, which already degrades gracefully to its group-name-hint
fallback.

Dependencies
------------
``src.config``, ``src.schema``, ``src.io.loaders``, ``src.pipeline.enrich``,
``src.rules.engine``, ``src.scoring.components``, ``src.scoring.priority``,
``src.decision.arbiter``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from src.config import AppConfig, get_config
from src.decision.arbiter import Arbiter
from src.io.loaders import DataRepository
from src.pipeline.enrich import EnrichmentResult, MessageEnricher, refine_message_type
from src.retrieval.context import MessageContext
from src.rules.engine import RuleEngine, RuleEvaluation
from src.schema import Decision, Message, RelationshipResult
from src.scoring.components import ScoringInput
from src.scoring.priority import PriorityAssessment, PriorityEngine

logger = logging.getLogger(__name__)

#: Signature for an optional external relationship resolver. Returns ``None``
#: when no fused relationship is available, in which case the rule engine
#: falls back to its group-name-hint inference.
RelationshipResolver = Callable[[Message, MessageContext], "RelationshipResult | None"]


@dataclass
class RoutedMessage:
    """The full intermediate state produced while routing one message.

    Returned by :meth:`Router.route_with_detail` for callers that want the
    enrichment, rule evaluation, and priority assessment alongside the final
    decision -- for example, a demo trace renderer. :meth:`Router.route`
    returns only the :class:`~src.schema.Decision` for the common case.

    Attributes
    ----------
    enrichment:
        The enrichment result for this message.
    rules:
        The rule evaluation, with :attr:`~src.rules.engine.RuleEvaluation.suggested_message_type`
        already refined by :func:`~src.pipeline.enrich.refine_message_type`.
    priority:
        The priority engine's assessment.
    decision:
        The final decision.
    """

    enrichment: EnrichmentResult
    rules: RuleEvaluation
    priority: PriorityAssessment
    decision: Decision


class Router:
    """Orchestrates the full message-to-decision pipeline.

    Construct once per batch run and reuse across every message: enrichment,
    rule evaluation, scoring, and arbitration components are all built once
    and hold no per-message state between calls.

    Parameters
    ----------
    repo:
        Fully loaded dataset repository.
    config:
        Application configuration; defaults to the process-wide singleton.
    enricher:
        Pre-built message enricher. Built from ``repo`` when omitted.
    rule_engine:
        Pre-built rule engine. Built from ``config`` when omitted.
    priority_engine:
        Pre-built priority engine. Built from ``config`` when omitted.
    arbiter:
        Pre-built arbiter. Built from ``config`` when omitted.
    relationship_resolver:
        Optional callable returning a fused
        :class:`~src.schema.RelationshipResult` for a message, or ``None``.
        When omitted, every message is routed with ``relationship=None`` and
        the rule engine's group-name-hint fallback applies.
    """

    def __init__(
        self,
        repo: DataRepository,
        config: AppConfig | None = None,
        enricher: MessageEnricher | None = None,
        rule_engine: RuleEngine | None = None,
        priority_engine: PriorityEngine | None = None,
        arbiter: Arbiter | None = None,
        relationship_resolver: RelationshipResolver | None = None,
    ) -> None:
        self._config = config or get_config()
        self._enricher = enricher or MessageEnricher(repo, self._config)
        self._rule_engine = rule_engine or RuleEngine(self._config)
        self._priority_engine = priority_engine or PriorityEngine(config=self._config)
        self._arbiter = arbiter or Arbiter(config=self._config)
        self._relationship_resolver = relationship_resolver

        self._processed = 0
        self._failed = 0

        logger.info(
            "Router initialised (relationship_resolver=%s).",
            "provided" if relationship_resolver else "none (group-hint fallback)",
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def route(self, messages: Iterable[Message]) -> Iterator[Decision]:
        """Route an iterable of messages, yielding a decision per message.

        Each message is processed independently; a failure on one message
        yields a :meth:`~src.schema.Decision.fallback` for that message and
        continues with the rest of the batch, so one bad row never aborts a
        run.

        Parameters
        ----------
        messages:
            Any iterable of :class:`~src.schema.Message`.

        Yields
        ------
        Decision
            One decision per input message, in input order.
        """
        for message in messages:
            yield self._safe_route_one(message)

        if self._processed:
            logger.info(
                "Router.route complete: %d processed, %d failed (%.1f%% success).",
                self._processed,
                self._failed,
                100.0 * (self._processed - self._failed) / self._processed,
            )

    def route_to_list(self, messages: Iterable[Message]) -> list[Decision]:
        """Route an iterable of messages, materialising the result as a list.

        Convenience wrapper around :meth:`route` for callers that do not need
        streaming behaviour.

        Parameters
        ----------
        messages:
            Any iterable of :class:`~src.schema.Message`.

        Returns
        -------
        list[Decision]
        """
        return list(self.route(messages))

    def route_one(self, message: Message) -> Decision:
        """Route a single message and return its decision.

        Parameters
        ----------
        message:
            The message to route.

        Returns
        -------
        Decision
            A :meth:`~src.schema.Decision.fallback` if processing raised.
        """
        return self._safe_route_one(message)

    def route_with_detail(self, messages: Iterable[Message]) -> Iterator[RoutedMessage]:
        """Route messages, yielding the full intermediate state for each.

        Intended for demo tooling and error analysis that needs the
        enrichment, rule evaluation, and priority assessment, not just the
        final decision. Failures are logged and skipped rather than yielding a
        partial :class:`RoutedMessage`, since there is no meaningful
        intermediate state to show for a message that never got a rule
        evaluation.

        Parameters
        ----------
        messages:
            Any iterable of :class:`~src.schema.Message`.

        Yields
        ------
        RoutedMessage
            One per successfully processed message; failed messages are
            omitted from this stream (they are still logged).
        """
        for message in messages:
            try:
                yield self._process_one(message)
                self._processed += 1
            except Exception as error:  # noqa: BLE001 - isolate one bad message
                self._processed += 1
                self._failed += 1
                logger.error(
                    "Router.route_with_detail: failed on message_id=%s (%s); skipping.",
                    getattr(message, "message_id", "<unknown>"),
                    error,
                )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _safe_route_one(self, message: Message) -> Decision:
        """Route one message, converting any exception into a fallback decision."""
        self._processed += 1
        try:
            return self._process_one(message).decision
        except Exception as error:  # noqa: BLE001 - one bad message must not stop the batch
            self._failed += 1
            message_id = getattr(message, "message_id", "<unknown>")
            logger.error(
                "Router: failed to route message_id=%s (%s); using fallback decision.",
                message_id,
                error,
            )
            return Decision.fallback(message_id, reason=f"Pipeline error: {error}")

    def _process_one(self, message: Message) -> RoutedMessage:
        """Run the full enrichment -> rules -> scoring -> decision pipeline.

        Parameters
        ----------
        message:
            The message to process.

        Returns
        -------
        RoutedMessage
            May raise; callers are responsible for exception handling (see
            :meth:`_safe_route_one` and :meth:`route_with_detail`).
        """
        enrichment = self._enricher.enrich(message)

        relationship = None
        if self._relationship_resolver is not None:
            relationship = self._relationship_resolver(message, enrichment.context)

        rules = self._rule_engine.evaluate(
            message,
            enrichment.context,
            relationship=relationship,
            content_verdict=enrichment.content_verdict,
            ocr_result=enrichment.ocr_result,
            asr_result=enrichment.asr_result,
        )

        # Narrow an over-eager REMINDER classification for casual conversation.
        # RuleEvaluation is a plain mutable dataclass, so this mutation
        # propagates to the priority engine and arbiter without touching
        # either of those frozen modules.
        rules.suggested_message_type = refine_message_type(
            message, enrichment.content_verdict, rules.suggested_message_type
        )

        scoring_input = ScoringInput(
            message=message,
            context=enrichment.context,
            rules=rules,
            relationship=relationship,
            content=enrichment.content_verdict,
        )
        priority = self._priority_engine.assess(scoring_input)

        decision = self._arbiter.decide(
            message,
            enrichment.context,
            rules,
            priority,
            ocr_result=enrichment.ocr_result,
            asr_result=enrichment.asr_result,
        )

        return RoutedMessage(
            enrichment=enrichment,
            rules=rules,
            priority=priority,
            decision=decision,
        )

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, int | float]:
        """Return a small diagnostic summary of everything routed so far.

        Returns
        -------
        dict
            ``processed``, ``failed``, and ``success_rate`` (as a fraction).
        """
        success_rate = (
            (self._processed - self._failed) / self._processed if self._processed else 0.0
        )
        return {
            "processed": self._processed,
            "failed": self._failed,
            "success_rate": round(success_rate, 4),
        }


def route_messages(
    messages: Iterable[Message],
    repo: DataRepository,
    config: AppConfig | None = None,
    relationship_resolver: RelationshipResolver | None = None,
) -> Iterator[Decision]:
    """Module-level convenience: build a default :class:`Router` and route.

    Prefer constructing :class:`Router` directly when routing more than one
    batch, so the enrichment indices are built once and reused.

    Parameters
    ----------
    messages:
        Any iterable of :class:`~src.schema.Message`.
    repo:
        Fully loaded dataset repository.
    config:
        Application configuration; defaults to the process-wide singleton.
    relationship_resolver:
        Optional fused-relationship resolver; see :class:`Router`.

    Yields
    ------
    Decision
        One decision per input message, in input order.
    """
    router = Router(repo, config=config, relationship_resolver=relationship_resolver)
    yield from router.route(messages)


__all__ = [
    "RelationshipResolver",
    "Router",
    "RoutedMessage",
    "route_messages",
]