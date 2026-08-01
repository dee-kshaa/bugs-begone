"""
Message enrichment: OCR, ASR, feature/content analysis, and retrieval context.

Turns a bare :class:`~src.schema.Message` into an :class:`EnrichmentResult`
carrying everything the rule engine, scoring engine, and arbiter need:
transcribed media text (written back onto the message itself), the assembled
:class:`~src.retrieval.context.MessageContext`, and a pre-computed
:class:`~src.features.content.ContentVerdict`.

Media handling is defensive throughout: a missing image or voice-note path, an
unavailable OCR/ASR backend, or a decode failure all degrade to an empty
transcription and a logged warning rather than raising, so one bad media file
never blocks the batch.

Dependencies
------------
``src.config``, ``src.schema``, ``src.io.loaders``, ``src.retrieval.context``,
``src.media.ocr``, ``src.media.asr``, ``src.media.cache``,
``src.features.content``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.config import AppConfig, get_config
from src.features.content import ContentVerdict, analyse_content
from src.io.loaders import DataRepository
from src.media.asr import AsrResult, transcribe_voice_note_cached
from src.media.cache import MediaCache, get_asr_cache, get_ocr_cache
from src.media.ocr import OcrResult, extract_text_from_image_cached
from src.retrieval.context import ContextRetriever, MessageContext
from src.schema import MediaType, Message, MessageType

logger = logging.getLogger(__name__)

#: Stricter reminder cue pattern used to refine the frozen content layer's
#: broader ``REMINDER_PATTERN``. A casual mention of "meeting" or "call at"
#: should not alone classify a message as a reminder; an explicit reminder
#: verb, RSVP request, or deadline phrase should.
STRICT_REMINDER_PATTERN = re.compile(
    r"\b(remind(?:er)?|don'?t forget|do not forget|rsvp|deadline|"
    r"due (?:today|tomorrow|date|by)|last date|submission (?:due|deadline)|"
    r"please confirm attendance|save the date)\b",
    re.IGNORECASE,
)


def refine_message_type(
    message: Message,
    verdict: ContentVerdict,
    suggested_type: MessageType,
) -> MessageType:
    """Downgrade an over-eager ``REMINDER`` classification to a personal message.

    The frozen content layer's ``REMINDER_PATTERN`` matches broad conversational
    words ("meeting", "call at", "class at"), so a casual message like
    "let's meet at 6" can surface as ``MessageType.REMINDER`` with no actual
    reminder intent. This function only ever narrows that one case; every
    other suggested type (including a correctly-detected reminder backed by an
    explicit cue) passes through unchanged.

    Parameters
    ----------
    message:
        The message being classified, used to pick the group-vs-personal
        fallback.
    verdict:
        The content verdict for this message.
    suggested_type:
        The type suggested upstream (typically
        :attr:`~src.rules.engine.RuleEvaluation.suggested_message_type`).

    Returns
    -------
    MessageType
        ``suggested_type`` unchanged unless it was ``REMINDER`` without a
        genuine reminder cue, in which case ``PERSONAL`` or ``GROUP_CHAT``.
    """
    if suggested_type is not MessageType.REMINDER:
        return suggested_type

    has_explicit_cue = verdict.has_deadline or bool(
        STRICT_REMINDER_PATTERN.search(message.content)
    )
    if has_explicit_cue:
        return MessageType.REMINDER

    refined = MessageType.GROUP_CHAT if message.is_group_message else MessageType.PERSONAL
    logger.debug(
        "refine_message_type: message_id=%s downgraded reminder -> %s "
        "(no explicit reminder cue found).",
        message.message_id,
        refined.value,
    )
    return refined


@dataclass
class EnrichmentResult:
    """Everything produced by enriching one message.

    Attributes
    ----------
    message:
        The same :class:`~src.schema.Message` instance passed in, mutated in
        place with any OCR/ASR transcription that was performed.
    context:
        Assembled :class:`~src.retrieval.context.MessageContext`.
    content_verdict:
        Pre-computed content analysis over the (now transcription-complete)
        unified message content.
    ocr_result:
        Raw OCR result, when this was an image message with a resolvable path.
    asr_result:
        Raw ASR result, when this was a voice message with a resolvable path.
    notes:
        Human-readable notes about degraded paths (missing media path, failed
        transcription, unavailable engine), for the explanation trace.
    """

    message: Message
    context: MessageContext
    content_verdict: ContentVerdict
    ocr_result: OcrResult | None = None
    asr_result: AsrResult | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the enrichment metadata (not the full message) for tracing."""
        return {
            "message_id": self.message.message_id,
            "content_verdict": self.content_verdict.to_dict(),
            "ocr_result": self.ocr_result.to_dict() if self.ocr_result else None,
            "asr_result": self.asr_result.to_dict() if self.asr_result else None,
            "notes": list(self.notes),
        }


class MediaEnricher:
    """Resolves and transcribes image/voice media for a message, with caching.

    Every failure mode -- missing path, missing engine, corrupt file, decode
    failure -- degrades to a logged note and an empty transcription rather
    than raising, so a batch run is never blocked by one bad media file.

    Parameters
    ----------
    config:
        Application configuration; defaults to the process-wide singleton.
    ocr_cache:
        OCR cache instance; defaults to the process-wide OCR cache.
    asr_cache:
        ASR cache instance; defaults to the process-wide ASR cache.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        ocr_cache: MediaCache | None = None,
        asr_cache: MediaCache | None = None,
    ) -> None:
        self._config = config or get_config()
        self._ocr_cache = ocr_cache or get_ocr_cache(self._config)
        self._asr_cache = asr_cache or get_asr_cache(self._config)

    def enrich_image(self, message: Message) -> tuple[OcrResult | None, tuple[str, ...]]:
        """Run OCR on an image message and write the result back onto it.

        Parameters
        ----------
        message:
            The message to enrich. Mutated in place on success: ``ocr_text``
            and ``ocr_confidence`` are populated.

        Returns
        -------
        tuple
            ``(OcrResult | None, notes)``. ``None`` when no path could be
            resolved.
        """
        notes: list[str] = []

        if message.media_path:
            path = message.media_path
        else:
            notes.append(
                f"no media_path set for image message {message.message_id}; skipping OCR"
            )
            logger.warning(
                "MediaEnricher: image message %s has image_id=%s but no media_path; "
                "skipping OCR.",
                message.message_id,
                message.image_id,
            )
            return None, tuple(notes)

        image_id = message.image_id or message.message_id
        result = extract_text_from_image_cached(
            image_id, path, config=self._config.media, cache=self._ocr_cache
        )

        if not result.success:
            notes.append(f"OCR failed: {result.error}")
            logger.info(
                "MediaEnricher: OCR failed for message_id=%s (%s).",
                message.message_id,
                result.error,
            )
            return result, tuple(notes)

        message.ocr_text = result.text
        message.ocr_confidence = result.confidence
        if not result.text.strip():
            notes.append("OCR succeeded but extracted no text")

        return result, tuple(notes)

    def enrich_voice(self, message: Message) -> tuple[AsrResult | None, tuple[str, ...]]:
        """Run ASR on a voice message and write the result back onto it.

        Parameters
        ----------
        message:
            The message to enrich. Mutated in place on success: ``asr_text``,
            ``asr_avg_logprob``, and ``voice_duration_sec`` are populated.

        Returns
        -------
        tuple
            ``(AsrResult | None, notes)``. ``None`` when no path could be
            resolved.
        """
        notes: list[str] = []

        if message.media_path:
            path = message.media_path
        else:
            notes.append(
                f"no media_path set for voice message {message.message_id}; skipping ASR"
            )
            logger.warning(
                "MediaEnricher: voice message %s has voice_note_id=%s but no media_path; "
                "skipping ASR.",
                message.message_id,
                message.voice_note_id,
            )
            return None, tuple(notes)

        voice_note_id = message.voice_note_id or message.message_id
        result = transcribe_voice_note_cached(
            voice_note_id, path, config=self._config.media, cache=self._asr_cache
        )

        if not result.success:
            reason = "skipped (too long)" if result.skipped_too_long else result.error
            notes.append(f"ASR failed: {reason}")
            logger.info(
                "MediaEnricher: ASR failed for message_id=%s (%s).",
                message.message_id,
                reason,
            )
            return result, tuple(notes)

        message.asr_text = result.text
        message.asr_avg_logprob = result.avg_logprob
        message.voice_duration_sec = result.duration_sec
        if not result.text.strip():
            notes.append("ASR succeeded but produced an empty transcript")

        return result, tuple(notes)

    def enrich(self, message: Message) -> tuple[OcrResult | None, AsrResult | None, tuple[str, ...]]:
        """Dispatch to the correct transcriber based on media type.

        Text messages and unsupported media types pass through untouched.

        Parameters
        ----------
        message:
            The message to enrich.

        Returns
        -------
        tuple
            ``(ocr_result, asr_result, notes)``. Both results are ``None`` for
            a text message.
        """
        if message.media_type is MediaType.IMAGE:
            ocr_result, notes = self.enrich_image(message)
            return ocr_result, None, notes
        if message.media_type is MediaType.VOICE:
            asr_result, notes = self.enrich_voice(message)
            return None, asr_result, notes
        return None, None, ()


class MessageEnricher:
    """Top-level enrichment: media transcription, context, and content analysis.

    This is the pipeline's single entry point for turning a bare message into
    everything downstream needs. Construct once per batch and reuse across
    every message, since :class:`~src.retrieval.context.ContextRetriever`
    builds its indices once from the full dataset.

    Parameters
    ----------
    repo:
        Fully loaded dataset repository.
    config:
        Application configuration; defaults to the process-wide singleton.
    retriever:
        Pre-built context retriever. Built from ``repo`` when omitted.
    media_enricher:
        Pre-built media enricher. Built from ``config`` when omitted.
    """

    def __init__(
        self,
        repo: DataRepository,
        config: AppConfig | None = None,
        retriever: ContextRetriever | None = None,
        media_enricher: MediaEnricher | None = None,
    ) -> None:
        self._config = config or get_config()
        self._retriever = retriever or ContextRetriever(repo, self._config)
        self._media_enricher = media_enricher or MediaEnricher(self._config)
        logger.info("MessageEnricher initialised. %s", self._retriever.stats())

    def enrich(self, message: Message) -> EnrichmentResult:
        """Enrich one message with media transcription, context, and content analysis.

        Parameters
        ----------
        message:
            The message to enrich. Its ``ocr_text``/``asr_text`` fields (and
            related confidence fields) are mutated in place when media
            transcription succeeds.

        Returns
        -------
        EnrichmentResult
        """
        ocr_result, asr_result, media_notes = self._media_enricher.enrich(message)

        context = self._retriever.gather(message)

        mentions_user = message.mentions_user(
            message.recipient_user_id or (context.user.user_id if context.user else None)
        )
        is_reply_to_user = self._is_reply_to_user(message, context)

        verdict = analyse_content(
            message.content,
            mentions_user=mentions_user,
            is_reply_to_user=is_reply_to_user,
            message=message,
        )

        return EnrichmentResult(
            message=message,
            context=context,
            content_verdict=verdict,
            ocr_result=ocr_result,
            asr_result=asr_result,
            notes=media_notes,
        )

    @staticmethod
    def _is_reply_to_user(message: Message, context: MessageContext) -> bool:
        """Return whether this message replies to something the recipient wrote.

        Mirrors the same check performed inside the rule engine, kept local
        here so enrichment can compute an accurate content verdict without
        depending on rule evaluation having already run.
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


__all__ = [
    "STRICT_REMINDER_PATTERN",
    "EnrichmentResult",
    "MediaEnricher",
    "MessageEnricher",
    "refine_message_type",
]