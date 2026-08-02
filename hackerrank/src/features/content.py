"""
Content feature extraction: what the words themselves say.

Operates on :attr:`src.schema.Message.content`, which already unifies typed
text with OCR and ASR transcriptions, so an OTP screenshot and a typed OTP take
the same path.

Produces two things:

* a ``content_*`` feature dictionary for the scoring engine, and
* a :class:`ContentVerdict` carrying the deterministic detections (OTP, promo,
  spam) that the rules layer acts on *before* any LLM call.

Dependencies
------------
``src.schema``. Standard library otherwise -- no regex here is expensive enough
to need caching beyond module-level compilation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from src.schema import MediaType, Message, MessageType

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Compiled patterns
# --------------------------------------------------------------------------- #

#: Words that accompany a one-time password. Deliberately narrow: a false
#: positive here forces a notify and is expensive.
OTP_KEYWORD_PATTERN = re.compile(
    r"\b(otp|one[\s-]?time[\s-]?password|verification[\s-]?code|security[\s-]?code|"
    r"login[\s-]?code|auth(?:entication)?[\s-]?code|passcode|2fa|two[\s-]?factor|"
    r"confirmation[\s-]?code|access[\s-]?code)\b",
    re.IGNORECASE,
)

#: A bare 4-8 digit code, optionally hyphen or space separated.
OTP_CODE_PATTERN = re.compile(r"\b\d{3}[\s-]?\d{3,5}\b|\b\d{4,8}\b")

#: The standard "never share this" clause that accompanies real OTPs.
OTP_WARNING_PATTERN = re.compile(
    r"(do not share|do not share this|don'?t share|never share|do not disclose|"
    r"valid for \d+\s*min)",
    re.IGNORECASE,
)

#: Transactional business language.
TRANSACTIONAL_PATTERN = re.compile(
    r"\b(order|shipped|dispatched|out for delivery|delivered|tracking|awb|"
    r"invoice|receipt|payment|paid|refund|debited|credited|transaction|"
    r"booking|reservation|appointment|ticket|pnr|itinerary|check[\s-]?in|"
    r"due date|statement|bill|emi|installment)\b",
    re.IGNORECASE,
)

#: Promotional / marketing language.
PROMOTIONAL_PATTERN = re.compile(
    r"\b(sale|offer|discount|deal|coupon|promo|cashback|flat \d+%|\d+% off|"
    r"limited time|hurry|shop now|buy now|order now|click here|claim now|"
    r"exclusive|mega|bonanza|festive|clearance|lowest price|free delivery|"
    r"unsubscribe|opt[\s-]?out|t&c apply|terms apply)\b",
    re.IGNORECASE,
)

#: Outright spam and scam markers.
SPAM_PATTERN = re.compile(
    r"\b(congratulations you|you have won|lottery|jackpot|prize money|"
    r"lucky winner|claim your (?:prize|reward)|work from home|earn \d+|"
    r"guaranteed income|investment opportunity|crypto (?:doubling|profit)|"
    r"kyc (?:suspend|block|expire)|account (?:suspend|block)ed|"
    r"click this link to verify)\b",
    re.IGNORECASE,
)

#: Time-bound and urgency language, including common Indian-English usage.
URGENCY_PATTERN = re.compile(
    r"\b(urgent|urgently|asap|immediately|right now|right away|emergency|"
    r"critical|important|priority|deadline|last date|final reminder|"
    r"reminder|due today|expires? (?:today|soon|in)|before \d|by \d{1,2}\s*(?:am|pm)|"
    r"today|tonight|tomorrow|jaldi|abhi|turant|quickly|need this|waiting for)\b",
    re.IGNORECASE,
)

#: Explicit clock or date deadlines, e.g. "by 6pm", "before 10:30", "on 12 Aug".
DEADLINE_PATTERN = re.compile(
    r"\b(?:by|before|until|till|due)\s+"
    r"(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day|tomorrow|today|tonight|eod|eob)\b",
    re.IGNORECASE,
)

#: Direct requests aimed at the reader.
REQUEST_PATTERN = re.compile(
    r"\b(please|pls|plz|kindly|can you|could you|would you|need you to|"
    r"let me know|revert|confirm|approve|review|send me|share the|"
    r"call me|ping me|reply|respond)\b",
    re.IGNORECASE,
)

#: Question words, checked alongside a literal question mark.
QUESTION_WORD_PATTERN = re.compile(
    r"\b(what|when|where|who|whom|which|why|how|is it|are you|did you|"
    r"have you|can we|shall we|should i)\b",
    re.IGNORECASE,
)

#: Reminder / scheduling language.
REMINDER_PATTERN = re.compile(
    r"\b(reminder|remind|don'?t forget|scheduled|meeting|call at|standup|"
    r"class at|exam on|submission|deadline|rsvp|calendar|invite)\b",
    re.IGNORECASE,
)

#: Forward markers that survive into the body text.
FORWARD_PATTERN = re.compile(
    r"(forwarded (?:many times|message)|fwd:|^fw:|"
    r"share with (?:all|everyone)|forward to \d+ (?:people|groups))",
    re.IGNORECASE | re.MULTILINE,
)

#: URLs, used both for promo detection and for a mild spam signal.
URL_PATTERN = re.compile(r"https?://\S+|\bwww\.\S+|\b\S+\.(?:com|in|co|net|org|ly|me)/\S*")

#: Phone numbers, useful for spam and for business-template detection.
PHONE_PATTERN = re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b")

#: Currency amounts.
AMOUNT_PATTERN = re.compile(r"(?:rs\.?|inr|₹|\$)\s?\d[\d,]*(?:\.\d{1,2})?", re.IGNORECASE)

#: Runs of exclamation or question marks, an emphasis signal.
EMPHASIS_PATTERN = re.compile(r"[!?]{2,}")

#: Threshold at which a promotional score flips the boolean flag.
PROMO_FLAG_THRESHOLD = 0.35

#: Threshold at which a spam score flips the boolean flag.
SPAM_FLAG_THRESHOLD = 0.45


# --------------------------------------------------------------------------- #
# Verdict object
# --------------------------------------------------------------------------- #


@dataclass
class ContentVerdict:
    """Deterministic detections drawn from a message body.

    The rules layer reads :attr:`is_otp` and :attr:`is_spam` directly; the
    scoring layer reads the continuous scores.
    """

    is_otp: bool = False
    is_transactional: bool = False
    is_promotional: bool = False
    is_spam: bool = False
    is_reminder: bool = False
    is_question: bool = False
    is_request: bool = False
    has_deadline: bool = False
    has_url: bool = False
    has_amount: bool = False
    has_forward_marker: bool = False

    urgency_score: float = 0.0
    promo_score: float = 0.0
    spam_score: float = 0.0

    matched_terms: tuple[str, ...] = ()
    suggested_type: MessageType = MessageType.OTHER
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "is_otp": self.is_otp,
            "is_transactional": self.is_transactional,
            "is_promotional": self.is_promotional,
            "is_spam": self.is_spam,
            "is_reminder": self.is_reminder,
            "is_question": self.is_question,
            "is_request": self.is_request,
            "has_deadline": self.has_deadline,
            "has_url": self.has_url,
            "has_amount": self.has_amount,
            "has_forward_marker": self.has_forward_marker,
            "urgency_score": round(self.urgency_score, 4),
            "promo_score": round(self.promo_score, 4),
            "spam_score": round(self.spam_score, 4),
            "matched_terms": list(self.matched_terms),
            "suggested_type": self.suggested_type.value,
            "reasons": list(self.reasons),
        }

    def to_signals(self) -> list[str]:
        """Render as human-readable signal strings for the explanation trace."""
        signals: list[str] = []
        if self.is_otp:
            signals.append("content:otp_detected")
        if self.is_spam:
            signals.append(f"content:spam_score={self.spam_score:.2f}")
        if self.is_promotional:
            signals.append(f"content:promo_score={self.promo_score:.2f}")
        if self.is_transactional:
            signals.append("content:transactional")
        if self.urgency_score > 0.0:
            signals.append(f"content:urgency={self.urgency_score:.2f}")
        if self.has_deadline:
            signals.append("content:deadline_present")
        return signals


# --------------------------------------------------------------------------- #
# Primitive helpers
# --------------------------------------------------------------------------- #


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    """Return the distinct lower-cased matches of ``pattern`` in ``text``."""
    found: list[str] = []
    for match in pattern.finditer(text):
        value = match.group(0).strip().lower()
        if value and value not in found:
            found.append(value)
    return found


def caps_ratio(text: str) -> float:
    """Return the share of alphabetic characters that are upper case.

    Shouting is a weak urgency signal. Strings with fewer than eight letters
    return ``0.0`` so that a two-letter "OK" does not read as a scream.

    Parameters
    ----------
    text:
        Raw message body.

    Returns
    -------
    float
        Ratio in ``[0, 1]``.
    """
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 8:
        return 0.0
    upper = sum(1 for character in letters if character.isupper())
    return upper / len(letters)


def detect_otp(text: str) -> tuple[bool, list[str]]:
    """Detect a one-time password with high precision.

    Requires an OTP keyword *and* a digit code, or a digit code plus the
    standard do-not-share warning. Either alone is not enough: "code" appears in
    developer chat, and bare digits appear everywhere.

    Parameters
    ----------
    text:
        Unified message content.

    Returns
    -------
    tuple
        ``(is_otp, matched_terms)``.
    """
    codes = _matches(OTP_CODE_PATTERN, text)
    if not codes:
        return False, []

    keywords = _matches(OTP_KEYWORD_PATTERN, text)
    if keywords:
        return True, keywords + codes[:1]

    warnings = _matches(OTP_WARNING_PATTERN, text)
    if warnings:
        return True, warnings + codes[:1]

    return False, []


def score_urgency(
    text: str,
    mentions_user: bool = False,
    is_reply_to_user: bool = False,
) -> tuple[float, list[str]]:
    """Score how time-critical a message reads, in ``[0, 1]``.

    Deliberately bimodal: a message with a real deadline and a direct address
    lands near the top, an ordinary message lands near zero. A bell-shaped
    urgency score would collapse the whole priority distribution into digest.

    Parameters
    ----------
    text:
        Unified message content.
    mentions_user:
        Whether the recipient is @-mentioned.
    is_reply_to_user:
        Whether the message replies to something the recipient wrote.

    Returns
    -------
    tuple
        ``(score, reasons)``.
    """
    reasons: list[str] = []
    score = 0.0

    urgency_terms = _matches(URGENCY_PATTERN, text)
    if urgency_terms:
        score += min(0.15 * len(urgency_terms), 0.35)
        reasons.append(f"urgency_terms:{','.join(urgency_terms[:3])}")

    deadlines = _matches(DEADLINE_PATTERN, text)
    if deadlines:
        score += 0.30
        reasons.append(f"deadline:{deadlines[0]}")

    if mentions_user:
        score += 0.30
        reasons.append("mention_of_user")

    if is_reply_to_user:
        score += 0.20
        reasons.append("reply_to_user")

    if QUESTION_WORD_PATTERN.search(text) or "?" in text:
        score += 0.10
        reasons.append("question_directed")

    if REQUEST_PATTERN.search(text):
        score += 0.10
        reasons.append("explicit_request")

    ratio = caps_ratio(text)
    if ratio > 0.6:
        score += 0.08
        reasons.append(f"caps_ratio={ratio:.2f}")

    if EMPHASIS_PATTERN.search(text):
        score += 0.05
        reasons.append("emphasis_punctuation")

    return min(score, 1.0), reasons


def score_promotional(text: str) -> tuple[float, list[str]]:
    """Score how much a message reads as marketing, in ``[0, 1]``.

    Parameters
    ----------
    text:
        Unified message content.

    Returns
    -------
    tuple
        ``(score, reasons)``.
    """
    reasons: list[str] = []
    score = 0.0

    promo_terms = _matches(PROMOTIONAL_PATTERN, text)
    if promo_terms:
        score += min(0.20 * len(promo_terms), 0.60)
        reasons.append(f"promo_terms:{','.join(promo_terms[:3])}")

    if URL_PATTERN.search(text):
        score += 0.15
        reasons.append("contains_url")

    if re.search(r"\b\d{1,2}%\s*off\b", text, re.IGNORECASE):
        score += 0.20
        reasons.append("discount_percentage")

    if re.search(r"\b(unsubscribe|opt[\s-]?out)\b", text, re.IGNORECASE):
        score += 0.25
        reasons.append("unsubscribe_footer")

    return min(score, 1.0), reasons


def score_spam(text: str) -> tuple[float, list[str]]:
    """Score how much a message reads as a scam or chain forward.

    Parameters
    ----------
    text:
        Unified message content.

    Returns
    -------
    tuple
        ``(score, reasons)``.
    """
    reasons: list[str] = []
    score = 0.0

    spam_terms = _matches(SPAM_PATTERN, text)
    if spam_terms:
        score += min(0.35 * len(spam_terms), 0.75)
        reasons.append(f"spam_terms:{','.join(spam_terms[:3])}")

    if FORWARD_PATTERN.search(text):
        score += 0.20
        reasons.append("forward_marker")

    urls = URL_PATTERN.findall(text)
    if len(urls) >= 2:
        score += 0.15
        reasons.append(f"multiple_urls={len(urls)}")

    if PHONE_PATTERN.search(text) and URL_PATTERN.search(text):
        score += 0.10
        reasons.append("phone_and_url")

    return min(score, 1.0), reasons


def suggest_message_type(
    verdict: ContentVerdict,
    message: Message | None = None,
) -> MessageType:
    """Map deterministic detections onto a :class:`MessageType`.

    Ordering matters: OTP outranks everything, spam outranks promo, and media
    fall through to ``MEDIA_SHARE`` only when the text says nothing else.

    Parameters
    ----------
    verdict:
        Detections already computed for this body.
    message:
        Optional message object, used to disambiguate when the text is silent.

    Returns
    -------
    MessageType
    """
    if verdict.is_otp:
        return MessageType.OTP
    if verdict.is_spam:
        return MessageType.SPAM
    if verdict.has_forward_marker:
        return MessageType.FORWARD
    if verdict.is_transactional:
        return MessageType.TRANSACTIONAL
    if verdict.is_promotional:
        return MessageType.PROMOTIONAL
    if verdict.is_reminder:
        return MessageType.REMINDER

    if message is not None:
        if message.is_forwarded:
            return MessageType.FORWARD
        if message.media_type in (MediaType.IMAGE, MediaType.VOICE, MediaType.VIDEO):
            return MessageType.MEDIA_SHARE
        if message.is_group_message:
            return MessageType.GROUP_CHAT
        if message.business_id:
            return MessageType.TRANSACTIONAL
        return MessageType.PERSONAL

    return MessageType.OTHER


def analyse_content(
    text: str,
    mentions_user: bool = False,
    is_reply_to_user: bool = False,
    message: Message | None = None,
) -> ContentVerdict:
    """Run every content detector over one message body.

    Parameters
    ----------
    text:
        Unified content -- typed text plus OCR plus ASR.
    mentions_user:
        Whether the recipient is @-mentioned.
    is_reply_to_user:
        Whether this message replies to the recipient.
    message:
        Optional message object, used only to refine the suggested type.

    Returns
    -------
    ContentVerdict
    """
    body = text or ""
    verdict = ContentVerdict()

    if not body.strip():
        verdict.suggested_type = suggest_message_type(verdict, message)
        verdict.reasons = ("empty_content",)
        return verdict

    matched: list[str] = []
    reasons: list[str] = []

    verdict.is_otp, otp_terms = detect_otp(body)
    matched.extend(otp_terms)
    if verdict.is_otp:
        reasons.append("otp_pattern")

    transactional_terms = _matches(TRANSACTIONAL_PATTERN, body)
    verdict.is_transactional = bool(transactional_terms)
    matched.extend(transactional_terms[:3])

    verdict.promo_score, promo_reasons = score_promotional(body)
    verdict.is_promotional = verdict.promo_score >= PROMO_FLAG_THRESHOLD
    reasons.extend(promo_reasons)

    verdict.spam_score, spam_reasons = score_spam(body)
    verdict.is_spam = verdict.spam_score >= SPAM_FLAG_THRESHOLD
    reasons.extend(spam_reasons)

    verdict.urgency_score, urgency_reasons = score_urgency(
        body, mentions_user=mentions_user, is_reply_to_user=is_reply_to_user
    )
    reasons.extend(urgency_reasons)

    verdict.is_reminder = bool(REMINDER_PATTERN.search(body))
    verdict.is_question = bool("?" in body or QUESTION_WORD_PATTERN.search(body))
    verdict.is_request = bool(REQUEST_PATTERN.search(body))
    verdict.has_deadline = bool(DEADLINE_PATTERN.search(body))
    verdict.has_url = bool(URL_PATTERN.search(body))
    verdict.has_amount = bool(AMOUNT_PATTERN.search(body))
    verdict.has_forward_marker = bool(FORWARD_PATTERN.search(body))

    verdict.matched_terms = tuple(dict.fromkeys(matched))[:8]
    verdict.reasons = tuple(dict.fromkeys(reasons))[:8]
    verdict.suggested_type = suggest_message_type(verdict, message)

    return verdict


def content_features(
    message: Message,
    mentions_user: bool = False,
    is_reply_to_user: bool = False,
) -> dict[str, Any]:
    """Assemble the ``content_*`` feature block for one message.

    Parameters
    ----------
    message:
        The message to analyse. Uses :attr:`Message.content`, so OCR and ASR
        text are included automatically.
    mentions_user:
        Whether the recipient is @-mentioned.
    is_reply_to_user:
        Whether this message replies to the recipient.

    Returns
    -------
    dict
        Feature dictionary with the ``content_`` prefix, plus the raw
        :class:`ContentVerdict` under ``content_verdict`` for the trace.
    """
    body = message.content
    verdict = analyse_content(
        body,
        mentions_user=mentions_user,
        is_reply_to_user=is_reply_to_user,
        message=message,
    )

    words = len(body.split())
    return {
        "content_length_chars": len(body),
        "content_length_words": words,
        "content_is_empty": words == 0,
        "content_is_otp": verdict.is_otp,
        "content_is_transactional": verdict.is_transactional,
        "content_is_promotional": verdict.is_promotional,
        "content_is_spam": verdict.is_spam,
        "content_is_reminder": verdict.is_reminder,
        "content_is_question": verdict.is_question,
        "content_is_request": verdict.is_request,
        "content_has_deadline": verdict.has_deadline,
        "content_has_url": verdict.has_url,
        "content_has_amount": verdict.has_amount,
        "content_has_forward_marker": verdict.has_forward_marker,
        "content_urgency_score": verdict.urgency_score,
        "content_promo_score": verdict.promo_score,
        "content_spam_score": verdict.spam_score,
        "content_caps_ratio": caps_ratio(body),
        "content_media_quality": message.media_quality,
        "content_is_transcribed": message.has_media,
        "content_suggested_type": verdict.suggested_type.value,
        "content_matched_terms": list(verdict.matched_terms),
        "content_verdict": verdict,
    }


def batch_analyse(texts: Iterable[str]) -> list[ContentVerdict]:
    """Analyse many bodies at once, for offline profiling of a sender.

    Used by the business profiler to estimate what share of an account's
    traffic is promotional.

    Parameters
    ----------
    texts:
        Message bodies.

    Returns
    -------
    list of ContentVerdict
    """
    return [analyse_content(text) for text in texts]


__all__ = [
    "AMOUNT_PATTERN",
    "ContentVerdict",
    "DEADLINE_PATTERN",
    "EMPHASIS_PATTERN",
    "FORWARD_PATTERN",
    "OTP_CODE_PATTERN",
    "OTP_KEYWORD_PATTERN",
    "OTP_WARNING_PATTERN",
    "PHONE_PATTERN",
    "PROMOTIONAL_PATTERN",
    "PROMO_FLAG_THRESHOLD",
    "QUESTION_WORD_PATTERN",
    "REMINDER_PATTERN",
    "REQUEST_PATTERN",
    "SPAM_FLAG_THRESHOLD",
    "SPAM_PATTERN",
    "TRANSACTIONAL_PATTERN",
    "URGENCY_PATTERN",
    "URL_PATTERN",
    "analyse_content",
    "batch_analyse",
    "caps_ratio",
    "content_features",
    "detect_otp",
    "score_promotional",
    "score_spam",
    "score_urgency",
    "suggest_message_type",
]