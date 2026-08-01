"""
Voice-note ASR: turn a voice-note path into a transcript plus a confidence score.

Uses OpenAI's Whisper via the ``openai-whisper`` package, which additionally
requires the ``ffmpeg`` binary on ``PATH`` to decode most audio formats. Both
are optional at import time -- the model and the decoder are imported lazily
so the rest of the pipeline can be imported before either is installed. A
missing engine, an undecodable file, or a clip over the configured duration
cap degrades to a logged failure result rather than raising.

Every result is cached in :mod:`src.media.cache`, keyed by ``voice_note_id``
with the source file's content hash, so re-running Stage A after an
interruption never re-transcribes a clip it already processed. This matters
more here than for OCR: Whisper is by far the slowest step in the pipeline.

Dependencies
------------
``pandas``, ``src.config``, ``src.media.cache``. ``openai-whisper`` is
required only when transcription actually runs; import and decode failures
are caught and reported as a failed :class:`AsrResult`, not raised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig, MediaConfig, get_config
from src.media.cache import MediaCache, compute_file_hash, get_asr_cache

logger = logging.getLogger(__name__)

#: Column candidates for a voice note's file path, in preference order.
_PATH_COLUMNS = ("media_path", "audio_path", "path", "file_path")

#: Whisper's fixed internal sample rate, used to compute duration from raw audio.
_WHISPER_SAMPLE_RATE = 16_000

#: Average log-probability assigned when Whisper produced no segments at all;
#: matches the confidence floor used elsewhere in the project.
_NO_SEGMENT_LOGPROB = -1.0


@dataclass
class AsrResult:
    """Outcome of transcribing one voice note.

    Attributes
    ----------
    voice_note_id:
        Identifier of the source voice note.
    text:
        Transcript, empty string on failure.
    avg_logprob:
        Whisper's average token log-probability across segments; roughly a
        confidence proxy, typically in ``[-1.5, 0.0]``. ``0.0`` is impossible
        in practice, so it never collides with a real value used as a sentinel.
    duration_sec:
        Clip duration as measured from the decoded audio, ``0.0`` on failure.
    success:
        ``False`` when the engine could not run, the file was undecodable, or
        the clip exceeded the configured duration cap.
    error:
        Human-readable failure reason, ``None`` on success.
    language:
        Language Whisper detected or was told to use, ``None`` on failure.
    skipped_too_long:
        ``True`` specifically when the clip was skipped for exceeding
        :attr:`~src.config.MediaConfig.whisper_max_duration_sec`, distinct from
        other failures so batch reporting can separate the two causes.
    engine:
        Name of the ASR engine used, for the explanation trace.
    """

    voice_note_id: str
    text: str = ""
    avg_logprob: float = _NO_SEGMENT_LOGPROB
    duration_sec: float = 0.0
    success: bool = False
    error: str | None = None
    language: str | None = None
    skipped_too_long: bool = False
    engine: str = "whisper"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary, used as the cache payload."""
        return {
            "voice_note_id": self.voice_note_id,
            "text": self.text,
            "avg_logprob": round(self.avg_logprob, 4),
            "duration_sec": round(self.duration_sec, 3),
            "success": self.success,
            "error": self.error,
            "language": self.language,
            "skipped_too_long": self.skipped_too_long,
            "engine": self.engine,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AsrResult":
        """Rebuild an :class:`AsrResult` from a cached payload dictionary."""
        return cls(
            voice_note_id=str(payload.get("voice_note_id", "")),
            text=str(payload.get("text", "")),
            avg_logprob=float(payload.get("avg_logprob", _NO_SEGMENT_LOGPROB)),
            duration_sec=float(payload.get("duration_sec", 0.0)),
            success=bool(payload.get("success", False)),
            error=payload.get("error"),
            language=payload.get("language"),
            skipped_too_long=bool(payload.get("skipped_too_long", False)),
            engine=str(payload.get("engine", "whisper")),
        )


def _load_whisper_module() -> Any | None:
    """Import ``whisper`` lazily.

    Returns
    -------
    module or None
        The imported ``whisper`` module, or ``None`` if the package is not
        installed. Logged once at ERROR level per call site, not raised.
    """
    try:
        import whisper
    except ImportError as error:
        logger.error(
            "ASR backend unavailable (%s). Install with "
            "`pip install openai-whisper` and ensure `ffmpeg` is on PATH.",
            error,
        )
        return None
    return whisper


@lru_cache(maxsize=4)
def _load_whisper_model(model_name: str) -> Any:
    """Load and cache a Whisper model checkpoint by name.

    Cached at process scope via ``lru_cache`` so repeated calls across many
    voice notes reuse the same in-memory model instead of reloading weights
    per file, which would dominate Stage A's runtime.

    Parameters
    ----------
    model_name:
        Whisper checkpoint name, e.g. ``"base"``.

    Returns
    -------
    Any
        A loaded Whisper model object.

    Raises
    ------
    RuntimeError
        If the ``whisper`` package is unavailable. Callers should check
        :func:`_load_whisper_module` first to avoid this path.
    """
    whisper = _load_whisper_module()
    if whisper is None:
        raise RuntimeError("whisper package unavailable")
    logger.info("Loading Whisper model %r (first use; this may take a while)...", model_name)
    model = whisper.load_model(model_name)
    logger.info("Whisper model %r loaded.", model_name)
    return model


def _probe_duration(whisper_module: Any, path: Path) -> float | None:
    """Decode a clip's raw audio to measure its duration in seconds.

    Decoding once here (rather than letting ``model.transcribe`` do it blind)
    lets the duration cap be enforced *before* paying for a full forward pass
    on an oversized file.

    Parameters
    ----------
    whisper_module:
        The imported ``whisper`` module.
    path:
        Path to the audio file.

    Returns
    -------
    float or None
        Duration in seconds, or ``None`` if the file could not be decoded
        (corrupt audio, unsupported codec, missing ffmpeg).
    """
    try:
        audio = whisper_module.load_audio(str(path))
    except Exception as error:  # noqa: BLE001 - ffmpeg/codec failures vary widely
        logger.warning("ASR: could not decode audio for duration probe at %s (%s)", path, error)
        return None
    return len(audio) / _WHISPER_SAMPLE_RATE


def transcribe_voice_note(
    voice_note_id: str,
    path: Path | str,
    config: MediaConfig | None = None,
) -> AsrResult:
    """Transcribe one voice note and return text plus a confidence proxy.

    Handles four failure modes without raising: a missing ASR backend, a
    missing file, an undecodable/corrupt clip, and a clip exceeding the
    configured duration cap. Each yields an :class:`AsrResult` with
    ``success=False`` and a descriptive ``error``.

    Parameters
    ----------
    voice_note_id:
        Identifier carried through to the result, for joining back to
        ``voice_notes.csv``.
    path:
        Path to the audio file.
    config:
        Media configuration; defaults to the process-wide singleton's.

    Returns
    -------
    AsrResult
    """
    cfg = config or get_config().media
    audio_path = Path(path)

    if not audio_path.exists():
        logger.warning("ASR: voice note not found for %s at %s", voice_note_id, audio_path)
        return AsrResult(
            voice_note_id=voice_note_id, success=False, error=f"file not found: {audio_path}"
        )

    whisper_module = _load_whisper_module()
    if whisper_module is None:
        return AsrResult(voice_note_id=voice_note_id, success=False, error="asr backend unavailable")

    duration = _probe_duration(whisper_module, audio_path)
    if duration is None:
        return AsrResult(
            voice_note_id=voice_note_id,
            success=False,
            error="undecodable audio (corrupt file or missing ffmpeg)",
        )

    if duration > cfg.whisper_max_duration_sec:
        logger.info(
            "ASR: skipping %s (%.1fs exceeds cap of %.1fs)",
            voice_note_id,
            duration,
            cfg.whisper_max_duration_sec,
        )
        return AsrResult(
            voice_note_id=voice_note_id,
            duration_sec=duration,
            success=False,
            error=f"duration {duration:.1f}s exceeds cap {cfg.whisper_max_duration_sec:.1f}s",
            skipped_too_long=True,
        )

    try:
        model = _load_whisper_model(cfg.whisper_model)
    except RuntimeError as error:
        return AsrResult(voice_note_id=voice_note_id, duration_sec=duration, success=False, error=str(error))

    try:
        result = model.transcribe(
            str(audio_path),
            language=cfg.whisper_language,
            fp16=False,
        )
    except Exception as error:  # noqa: BLE001 - a single bad clip must not stop the batch
        logger.warning("ASR: Whisper failed on %s (%s)", audio_path, error)
        return AsrResult(
            voice_note_id=voice_note_id,
            duration_sec=duration,
            success=False,
            error=f"asr engine error: {error}",
        )

    text = str(result.get("text", "")).strip()
    segments = result.get("segments") or []

    if segments:
        weighted_sum = 0.0
        total_duration = 0.0
        for segment in segments:
            seg_duration = max(
                float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)), 0.0
            )
            seg_logprob = float(segment.get("avg_logprob", _NO_SEGMENT_LOGPROB))
            weighted_sum += seg_logprob * seg_duration
            total_duration += seg_duration
        avg_logprob = weighted_sum / total_duration if total_duration > 0 else _NO_SEGMENT_LOGPROB
    else:
        avg_logprob = _NO_SEGMENT_LOGPROB

    detected_language = result.get("language")

    asr_result = AsrResult(
        voice_note_id=voice_note_id,
        text=text,
        avg_logprob=round(avg_logprob, 4),
        duration_sec=duration,
        success=True,
        error=None,
        language=detected_language,
    )

    if not text:
        logger.info("ASR: empty transcript for %s (voice_note_id=%s)", audio_path, voice_note_id)
    elif avg_logprob < cfg.asr_min_avg_logprob:
        logger.info(
            "ASR: low-confidence transcript for %s (avg_logprob=%.2f < min=%.2f)",
            voice_note_id,
            avg_logprob,
            cfg.asr_min_avg_logprob,
        )

    return asr_result


def transcribe_voice_note_cached(
    voice_note_id: str,
    path: Path | str,
    config: MediaConfig | None = None,
    cache: MediaCache | None = None,
) -> AsrResult:
    """Cache-aware wrapper around :func:`transcribe_voice_note`.

    Parameters
    ----------
    voice_note_id:
        Cache key and result identifier.
    path:
        Path to the audio file.
    config:
        Media configuration; defaults to the process-wide singleton's.
    cache:
        Cache instance; defaults to the process-wide ASR cache.

    Returns
    -------
    AsrResult
        From cache when the file's content hash is unchanged, otherwise freshly
        computed and stored. This is the expensive path, so a cache hit here
        matters far more to total runtime than an OCR cache hit does.
    """
    active_cache = cache or get_asr_cache()
    audio_path = Path(path)
    content_hash = compute_file_hash(audio_path)

    cached_payload = active_cache.get(voice_note_id, content_hash=content_hash)
    if cached_payload is not None:
        logger.debug("ASR cache hit for voice_note_id=%s", voice_note_id)
        return AsrResult.from_dict(cached_payload)

    result = transcribe_voice_note(voice_note_id, audio_path, config=config)
    active_cache.set(voice_note_id, result.to_dict(), content_hash=content_hash)
    return result


def _resolve_voice_path(row: pd.Series, media_root: Path | None) -> Path | None:
    """Resolve one row of ``voice_notes.csv`` to a filesystem path.

    Tries every known path-column alias; when the value is a bare filename
    rather than a full path, joins it against ``media_root``.
    """
    for column in _PATH_COLUMNS:
        if column in row.index and pd.notna(row[column]):
            raw = str(row[column]).strip()
            if not raw:
                continue
            candidate = Path(raw)
            if candidate.is_absolute() or candidate.exists():
                return candidate
            if media_root is not None:
                return media_root / candidate
            return candidate
    return None


def process_voice_notes(
    voice_notes: pd.DataFrame,
    config: AppConfig | None = None,
    cache: MediaCache | None = None,
) -> pd.DataFrame:
    """Run ASR over every row of ``voice_notes.csv``, using and populating the cache.

    This is the Stage A entry point: call it once, persist the returned frame
    to ``data/cache/asr.parquet`` (already true, since results are written
    through :class:`~src.media.cache.MediaCache`), and join it back onto
    messages by ``voice_note_id`` afterwards. Given Whisper's cost, this is the
    step to kick off in the background first.

    Parameters
    ----------
    voice_notes:
        ``voice_notes.csv`` frame with a ``voice_note_id`` column and some
        path column.
    config:
        Application configuration; defaults to the process-wide singleton.
    cache:
        Cache instance; defaults to the process-wide ASR cache.

    Returns
    -------
    pandas.DataFrame
        Columns: ``voice_note_id``, ``text``, ``avg_logprob``, ``duration_sec``,
        ``success``, ``error``, ``language``, ``skipped_too_long``. Empty when
        ``voice_notes`` is empty or missing ``voice_note_id``.
    """
    if voice_notes.empty or "voice_note_id" not in voice_notes.columns:
        logger.warning("process_voice_notes: empty frame or missing voice_note_id column.")
        return pd.DataFrame(
            columns=[
                "voice_note_id",
                "text",
                "avg_logprob",
                "duration_sec",
                "success",
                "error",
                "language",
                "skipped_too_long",
            ]
        )

    cfg = config or get_config()
    active_cache = cache or get_asr_cache(cfg)
    media_root = cfg.paths.media_voice

    total = len(voice_notes)
    succeeded = 0
    failed = 0
    skipped = 0
    rows: list[dict[str, Any]] = []

    for index, row in voice_notes.iterrows():
        voice_note_id = str(row["voice_note_id"]).strip()
        if not voice_note_id:
            continue

        path = _resolve_voice_path(row, media_root)
        if path is None:
            logger.warning(
                "process_voice_notes: no resolvable path for voice_note_id=%s", voice_note_id
            )
            result = AsrResult(
                voice_note_id=voice_note_id, success=False, error="no path column found"
            )
        else:
            result = transcribe_voice_note_cached(
                voice_note_id, path, config=cfg.media, cache=active_cache
            )

        succeeded += int(result.success)
        failed += int(not result.success and not result.skipped_too_long)
        skipped += int(result.skipped_too_long)
        rows.append(result.to_dict())

        if (index + 1) % 20 == 0 or (index + 1) == total:
            logger.info(
                "ASR progress: %d/%d (ok=%d, failed=%d, skipped_too_long=%d)",
                index + 1,
                total,
                succeeded,
                failed,
                skipped,
            )

    active_cache.close()
    logger.info(
        "process_voice_notes complete: %d total, %d succeeded, %d failed, %d skipped (too long).",
        total,
        succeeded,
        failed,
        skipped,
    )
    return pd.DataFrame(rows)


__all__ = [
    "AsrResult",
    "process_voice_notes",
    "transcribe_voice_note",
    "transcribe_voice_note_cached",
]