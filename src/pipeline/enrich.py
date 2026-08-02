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
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.config import AppConfig, get_config
from src.features.content import ContentVerdict, analyse_content
from src.io.loaders import DataRepository
from src.media.asr import AsrResult, transcribe_voice_note_cached
from src.media.cache import MediaCache, get_asr_cache, get_ocr_cache
from src.media.ocr import OcrResult, extract_text_from_image_cached
from src.retrieval.context import ContextRetriever, MessageContext
from src.schema import ConversationType, MediaType, Message, MessageType

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Adversarial text normalisation
# --------------------------------------------------------------------------- #
#
# Every downstream detector -- the content verdict, the official message-type
# classifier and the rule engine -- matches regular expressions against
# ``Message.content``. That makes all of them evadable by trivial obfuscation:
# a zero-width space inside an OTP, a Cyrillic "а" inside "аccount", a
# fullwidth codepoint, or "hxxp://paypa1-secure[.]com" all defeat patterns
# that would otherwise fire.
#
# Normalising once during enrichment fixes every detector at the same time,
# because they all read the same normalised text.

#: Codepoints that carry no visual weight but break regex matching.
_ZERO_WIDTH = dict.fromkeys(
    [
        0x200B,  # zero-width space
        0x200C,  # zero-width non-joiner
        0x200D,  # zero-width joiner
        0x2060,  # word joiner
        0xFEFF,  # BOM / zero-width no-break space
        0x00AD,  # soft hyphen
        0x180E,  # Mongolian vowel separator
    ]
)

#: Bidirectional control characters, used to visually reverse text.
_BIDI_CONTROLS = dict.fromkeys(
    [0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069]
)

#: Cyrillic and Greek codepoints that render identically to Latin letters.
#: Applied only inside mixed-script tokens, so genuinely non-Latin text is
#: left alone.
_HOMOGLYPHS: dict[str, str] = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0443": "y", "\u0445": "x", "\u0456": "i", "\u0458": "j", "\u04bb": "h",
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
    "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
    "\u0423": "Y", "\u0425": "X", "\u0406": "I",
    "\u03b1": "a", "\u03bf": "o", "\u03c1": "p", "\u03c5": "u", "\u0391": "A",
    "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H", "\u039f": "O",
    "\u03a1": "P", "\u03a4": "T", "\u03a7": "X",
}

#: Defanged / obfuscated URL forms, mapped back to their real shape so URL
#: and scam patterns can see them.
_URL_DEOBFUSCATION: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"h[xX]{2}ps?://", re.IGNORECASE), "http://"),
    (re.compile(r"\[\s*\.\s*\]"), "."),
    (re.compile(r"\(\s*\.\s*\)"), "."),
    (re.compile(r"\{\s*\.\s*\}"), "."),
    (re.compile(r"\s+dot\s+", re.IGNORECASE), "."),
    (re.compile(r"%2e", re.IGNORECASE), "."),
    (re.compile(r"\u2024"), "."),
    (re.compile(r"\[\s*:\s*\]"), ":"),
)

_LATIN_RE = re.compile(r"[A-Za-z]")
_NON_LATIN_RE = re.compile(r"[\u0370-\u03ff\u0400-\u04ff]")


def _fold_homoglyphs(text: str) -> str:
    """Map look-alike Cyrillic/Greek letters to Latin inside mixed-script words.

    Folding is applied per whitespace-delimited token and only when that token
    already contains Latin letters. A token written entirely in Cyrillic is
    left untouched, so legitimate non-Latin content is never mangled; a token
    like ``"Уour"`` or ``"раypal.com"``, which mixes scripts precisely to
    impersonate Latin text, is folded.

    Parameters
    ----------
    text:
        Text to fold.

    Returns
    -------
    str
    """
    if not _NON_LATIN_RE.search(text):
        return text

    folded_tokens: list[str] = []
    for token in text.split(" "):
        if _LATIN_RE.search(token) and _NON_LATIN_RE.search(token):
            folded_tokens.append("".join(_HOMOGLYPHS.get(ch, ch) for ch in token))
        else:
            folded_tokens.append(token)
    return " ".join(folded_tokens)


def normalise_adversarial_text(text: str) -> str:
    """Strip obfuscation that would otherwise defeat every downstream detector.

    Applies, in order: Unicode NFKC normalisation (which folds fullwidth and
    other compatibility forms), removal of zero-width and bidirectional
    control characters, mixed-script homoglyph folding, and de-obfuscation of
    defanged URLs.

    This is deliberately conservative -- it only removes characters that carry
    no legitimate meaning in this corpus, and only folds homoglyphs where the
    surrounding token is already Latin.

    Parameters
    ----------
    text:
        Raw message text.

    Returns
    -------
    str
        Normalised text, safe to run pattern matching against.
    """
    if not text:
        return text

    normalised = unicodedata.normalize("NFKC", text)
    normalised = normalised.translate(_ZERO_WIDTH).translate(_BIDI_CONTROLS)
    normalised = _fold_homoglyphs(normalised)
    for pattern, replacement in _URL_DEOBFUSCATION:
        normalised = pattern.sub(replacement, normalised)
    return normalised


# --------------------------------------------------------------------------- #
# Official message-type classification
# --------------------------------------------------------------------------- #
#
# The task defines a closed vocabulary that the internal content/rule taxonomy
# does not cover (it has no notion of ``urgent``, ``event``, ``payment``,
# ``greeting`` or ``scam`` as distinct from ``spam``). These patterns classify
# directly into the official vocabulary; the ordering in
# :func:`classify_official_message_type` is what actually decides ties.

#: Credential-phishing and account-threat language. Deliberately narrow and
#: checked first, since a scam label also drives a hard mute.
SCAM_PATTERN = re.compile(
    r"\b(verify (?:now|immediately|your account)|confirm (?:your )?(?:password|otp|pin)|"
    r"(?:profile|account|access) (?:will be|may be|has been) (?:blocked|suspended|deactivated)|"
    r"reply with the \d+ ?digit|share the otp|otp may have leaked|"
    r"security alert|support alert|kyc (?:update|verification|expired)|"
    r"click (?:here )?to (?:verify|reactivate|restore)|"
    r"account-?login|to keep (?:your )?account active|"
    r"reply with the otp|send (?:me )?the otp|share your otp|"
    r"(?:wallet|payment|account) verification failed|"
    r"you have won|lottery|claim your (?:prize|reward))\b",
    re.IGNORECASE,
)

#: Prompt-injection attempts embedded in message text. A message trying to
#: rewrite the router's own instructions is hostile by definition, so it is
#: classified as a scam and never allowed to influence the routing decision.
INJECTION_PATTERN = re.compile(
    r"\b(ignore (?:all )?(?:previous|prior|above) (?:routing )?(?:rules|instructions)|"
    r"disregard (?:the )?(?:previous|above|system)|"
    r"mark this (?:message )?as (?:notify|urgent|important)|"
    r"you are (?:now )?an? (?:ai|assistant)|system prompt|"
    r"actual message:)\b",
    re.IGNORECASE,
)

#: Urgency so strong it needs no corroborating signal.
STRONG_URGENT_PATTERN = re.compile(
    r"\b(urgent|urgently|asap|emergency|immediately|right away|"
    r"quick heads-?up|escalation starts|come online now|"
    r"(?:pls|please) .{0,25}\bnow\b|\bnow\b.{0,20}(?:before|otherwise)|"
    r"in \d{1,2} ?(?:mins|minutes)|cannot wait|can'?t wait)\b",
    re.IGNORECASE,
)

#: Contact from someone the user does not appear to know.
STRANGER_PATTERN = re.compile(
    r"\b(found your (?:number|contact)|got this number|"
    r"is this (?:the )?(?:right )?number|am i speaking|"
    r"i got your (?:number|details) from|from the (?:volunteer|courier|society) )\b",
    re.IGNORECASE,
)

#: Chain-forward markers.
FORWARD_MARKER_PATTERN = re.compile(
    r"\b(fwd(?: as received)?|forwarded (?:many times|as received|message)|"
    r"pls forward|please forward|share (?:with|to) (?:all|everyone|family|friends)|"
    r"sharing (?:here )?in case it helps|received on another group)\b",
    re.IGNORECASE,
)

#: Greeting / well-wishing with no actionable content.
GREETING_PATTERN = re.compile(
    r"\b(good (?:morning|evening|night|afternoon)|"
    r"happy (?:birthday|anniversary|new year|diwali|holi|eid|christmas)|"
    r"stay (?:positive|blessed|safe)|keep smiling|good vibes|"
    r"hope (?:today|your day) is|blessings|warm wishes|"
    r"many many happy returns)\b",
    re.IGNORECASE,
)

#: Payment obligations owed by the user.
PAYMENT_PATTERN = re.compile(
    r"\b(payment (?:due|pending|failed|reminder)|amount due|due amount|"
    r"outstanding (?:amount|balance|due)|pay (?:now|before|by)|"
    r"emi|installment|invoice (?:due|attached|generated)|bill (?:due|generated|payment)|"
    r"minimum (?:amount )?due|overdue|last date (?:for|to) pay|"
    r"recharge (?:due|expire)|renewal (?:due|charge)|autopay)\b",
    re.IGNORECASE,
)

#: Scheduled happenings: meetings, appointments, timings, RSVPs, form windows.
EVENT_PATTERN = re.compile(
    r"\b(meeting|appointment|scheduled|schedule|reschedul|"
    r"bus (?:is |will )?leav|pickup at|drop at|assembly|"
    r"form (?:is )?(?:open|close|closes|closing)|registration|register by|"
    r"rsvp|cultural (?:night|day)|function|ceremony|celebration|"
    r"class (?:at|on)|exam (?:on|at)|session (?:at|on)|"
    r"prescription|claim|checkup|check-up|consultation|"
    r"submit (?:the |your )?(?:form|entry)|slot|venue|agenda)\b",
    re.IGNORECASE,
)

#: Time pressure that demands attention now.
URGENT_PATTERN = re.compile(
    r"\b(urgent|urgently|asap|immediately|right (?:now|away)|emergency|"
    r"quick heads-?up|last-?minute|before eod|by eod|"
    r"(?:today|tonight) (?:itself|only)|in \d+ ?(?:mins|minutes|hours)|"
    r"can wait (?:maybe )?\d+|cannot wait|can'?t wait|"
    r"need (?:this|it) (?:today|now)|close (?:the|this) .{0,20}before|"
    r"pulled to \d|got pulled|blocked|breakdown|leak(?:age|ing)?|"
    r"no (?:water|power|supply))\b",
    re.IGNORECASE,
)

#: Selling, offers and marketing -- including peer-to-peer selling in groups.
PROMOTION_PATTERN = re.compile(
    r"\b(\d+ ?% ?off|flat \d+|discount|offer|sale|deal|coupon|promo code|"
    r"cashback|limited (?:time|period|stock)|hurry|expires? soon|"
    r"shop now|buy now|order now|tap below|click below|use code|"
    r"dm if interested|pickup (?:is )?near|price final|selling|for sale|"
    r"barely used|no damage|per person|all in, from|"
    r"t&c apply|terms apply|unsubscribe|reply stop|opt ?out|"
    r"welcome offer|first order|refer and earn)\b",
    re.IGNORECASE,
)

#: Business operational updates: orders, deliveries, feedback, advisories.
BUSINESS_UPDATE_PATTERN = re.compile(
    r"\b(order (?:ending|no|number|id|has been|is)|"
    r"(?:has been |is )?(?:packed|shipped|dispatched|delivered|out for delivery)|"
    r"delivery (?:details|code|partner|update)|tracking|awb|"
    r"local hub|reach(?:ing)? (?:you|the)|"
    r"feedback|rate your|your experience|review your|"
    r"advisory|never ask for otp|account or card|statement|"
    r"booking (?:confirmed|details)|ticket (?:confirmed|details)|"
    r"refund (?:initiated|processed)|update is (?:now )?available)\b",
    re.IGNORECASE,
)

#: Direct-address / conversational markers that indicate a personal exchange.
PERSONAL_PATTERN = re.compile(
    r"\b(can you call|give me a call|are you (?:free|coming|there)|"
    r"let me know|what(?:'s| is) up|how are you|"
    r"anyone (?:watching|coming|going|free)|shall we|"
    r"see you|talk later|thanks|thank you|congrats|congratulations)\b",
    re.IGNORECASE,
)

#: Stricter reminder cue pattern retained for backward compatibility with the
#: original conversational-reminder refinement.
STRICT_REMINDER_PATTERN = re.compile(
    r"\b(remind(?:er)?|don'?t forget|do not forget|rsvp|deadline|"
    r"due (?:today|tomorrow|date|by)|last date|submission (?:due|deadline)|"
    r"please confirm attendance|save the date)\b",
    re.IGNORECASE,
)


def _mentions_recipient(message: Message) -> bool:
    """Return whether the message @-mentions its recipient by id."""
    recipient = message.recipient_user_id
    if not recipient:
        return False
    if message.mentions_user(recipient):
        return True
    return f"@{recipient}".lower() in message.content.lower()


def classify_official_message_type(
    message: Message,
    verdict: ContentVerdict,
) -> MessageType:
    """Classify a message into the task's official ``message_type`` vocabulary.

    The official values are ``personal, urgent, event, payment,
    business_update, promotion, greeting, forward, spam, scam, unknown``. The
    internal content taxonomy has no equivalent for several of these, so this
    classifier works directly from the message text, media transcription,
    conversation type and content verdict.

    Ordering is the substance of the classifier: risk labels are checked
    before benign ones, and specific categories before general ones, so that
    a scam that mentions an order is labelled ``scam`` rather than
    ``business_update``.

    Parameters
    ----------
    message:
        The message being classified. Uses :attr:`~src.schema.Message.content`,
        so OCR and ASR text participate.
    verdict:
        The content verdict already computed for this message.

    Returns
    -------
    MessageType
        Always one of the eleven official values.
    """
    text = message.content
    is_business = message.conversation_type is ConversationType.BUSINESS or bool(
        message.business_id
    )

    # 1. Risk first: credential phishing and prompt injection outrank whatever
    #    else the message mimics. An injection attempt is hostile by design.
    if (
        SCAM_PATTERN.search(text)
        or INJECTION_PATTERN.search(text)
        or verdict.spam_score >= 0.75
    ):
        return MessageType.SCAM

    # 2. Greetings, checked before forwards: a good-morning chain message is
    #    labelled a greeting in the reference data even when it says it was
    #    forwarded.
    if GREETING_PATTERN.search(text):
        return MessageType.GREETING

    # 3. Chain forwards.
    if FORWARD_MARKER_PATTERN.search(text) or (message.is_forwarded and not is_business):
        return MessageType.FORWARD

    # 4. Money the user owes.
    if PAYMENT_PATTERN.search(text):
        return MessageType.PAYMENT

    # 5. Unambiguous time pressure, which outranks a scheduling reading: an
    #    infrastructure disruption needing action now is urgent, not an event.
    if STRONG_URGENT_PATTERN.search(text):
        return MessageType.URGENT

    # 6. Scheduled happenings. Checked before weaker urgency so that a timing
    #    change to a known event stays an event.
    if EVENT_PATTERN.search(text):
        return MessageType.EVENT

    # 7. Weaker urgency, which needs corroboration to fire.
    directed = _mentions_recipient(message) or bool(message.reply_to_id)
    if URGENT_PATTERN.search(text) and (
        directed or verdict.has_deadline or verdict.urgency_score >= 0.45
    ):
        return MessageType.URGENT

    # 8. Business operations, checked before promotion so that an advisory
    #    ending in "tap below" is not read as marketing.
    if is_business and BUSINESS_UPDATE_PATTERN.search(text):
        return MessageType.BUSINESS_UPDATE

    # 9. Selling and marketing, including peer-to-peer sales in groups.
    if PROMOTION_PATTERN.search(text) or verdict.is_promotional:
        return MessageType.PROMOTION

    # 10. Remaining business traffic.
    if is_business:
        if verdict.is_spam or not text.strip():
            return MessageType.SPAM
        return MessageType.BUSINESS_UPDATE

    if verdict.is_spam:
        return MessageType.SPAM

    # 11. A deadline without urgency language is still an event.
    if verdict.has_deadline or STRICT_REMINDER_PATTERN.search(text):
        return MessageType.EVENT

    # 12. Contact from someone the user does not know.
    if STRANGER_PATTERN.search(text):
        return MessageType.UNKNOWN

    # 13. Conversational traffic between people. A voice note that failed to
    #     transcribe is far more likely to be personal than genuinely
    #     unclassifiable, so it defaults here rather than to ``unknown``.
    if message.conversation_type in (ConversationType.DIRECT, ConversationType.GROUP):
        if text.strip() or message.media_type is MediaType.VOICE:
            return MessageType.PERSONAL

    return MessageType.UNKNOWN


def refine_message_type(
    message: Message,
    verdict: ContentVerdict,
    suggested_type: MessageType,
) -> MessageType:
    """Return the official-vocabulary type for a message.

    Replaces the upstream suggestion (built on the internal taxonomy) with a
    classification into the task's required vocabulary. The ``suggested_type``
    argument is retained so the call site in
    :mod:`src.pipeline.route` does not change, and is used only as a weak
    prior: an upstream OTP detection is respected as ``urgent`` when the
    classifier finds nothing more specific.

    Parameters
    ----------
    message:
        The message being classified.
    verdict:
        Its content verdict.
    suggested_type:
        The type suggested upstream by the rule engine.

    Returns
    -------
    MessageType
        One of the eleven official values.
    """
    official = classify_official_message_type(message, verdict)

    # An upstream OTP detection is a strong urgency signal the text patterns
    # can miss, but never overrides a scam finding.
    if (
        suggested_type is MessageType.OTP
        and official not in (MessageType.SCAM, MessageType.URGENT)
    ):
        official = MessageType.URGENT

    logger.debug(
        "refine_message_type: message_id=%s %s -> %s",
        message.message_id,
        suggested_type.value,
        official.value,
    )
    return official


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

        # Normalise before any detector sees the text. Applied to the typed
        # body and to both transcriptions, since OCR in particular can carry
        # fullwidth and look-alike codepoints straight out of a screenshot.
        message.message_text = normalise_adversarial_text(message.message_text)
        message.ocr_text = normalise_adversarial_text(message.ocr_text)
        message.asr_text = normalise_adversarial_text(message.asr_text)

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
    "normalise_adversarial_text",
    "EnrichmentResult",
    "MediaEnricher",
    "MessageEnricher",
    "refine_message_type",
]