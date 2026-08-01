"""
Image OCR: turn an image path into text plus a confidence score.

Uses Tesseract via ``pytesseract``. The Tesseract binary and the Python
package are both optional at import time -- this module imports them lazily so
the rest of the pipeline can be imported and exercised before either is
installed. A missing engine degrades to a logged failure result rather than an
ImportError.

Every result is cached in :mod:`src.media.cache`, keyed by ``image_id`` with
the source file's content hash, so re-running Stage A after an interruption
never re-OCRs an image it already processed.

Dependencies
------------
``pandas``, ``src.config``, ``src.media.cache``. ``pytesseract`` and
``Pillow`` are required only when OCR actually runs; import failures there are
caught and reported as a failed :class:`OcrResult`, not raised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig, MediaConfig, get_config
from src.media.cache import MediaCache, compute_file_hash, get_ocr_cache

logger = logging.getLogger(__name__)

#: Column candidates for an image's file path, in preference order.
_PATH_COLUMNS = ("media_path", "image_path", "path", "file_path")

#: Below this pixel width, an image is upscaled before OCR to help small
#: screenshots (a common source of OTP and receipt images).
_UPSCALE_THRESHOLD_PX = 800


@dataclass
class OcrResult:
    """Outcome of running OCR on one image.

    Attributes
    ----------
    image_id:
        Identifier of the source image.
    text:
        Extracted text, empty string on failure.
    confidence:
        Mean word confidence in ``[0, 1]``. ``0.0`` on failure or when no text
        was detected.
    success:
        ``False`` when the engine could not run or the file was unreadable.
    error:
        Human-readable failure reason, ``None`` on success.
    word_count:
        Number of words Tesseract reported a confidence for.
    engine:
        Name of the OCR engine used, for the explanation trace.
    """

    image_id: str
    text: str = ""
    confidence: float = 0.0
    success: bool = False
    error: str | None = None
    word_count: int = 0
    engine: str = "tesseract"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary, used as the cache payload."""
        return {
            "image_id": self.image_id,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "success": self.success,
            "error": self.error,
            "word_count": self.word_count,
            "engine": self.engine,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OcrResult":
        """Rebuild an :class:`OcrResult` from a cached payload dictionary."""
        return cls(
            image_id=str(payload.get("image_id", "")),
            text=str(payload.get("text", "")),
            confidence=float(payload.get("confidence", 0.0)),
            success=bool(payload.get("success", False)),
            error=payload.get("error"),
            word_count=int(payload.get("word_count", 0)),
            engine=str(payload.get("engine", "tesseract")),
        )


def _load_ocr_backend() -> tuple[Any, Any] | None:
    """Import ``pytesseract`` and ``PIL.Image`` lazily.

    Returns
    -------
    tuple or None
        ``(pytesseract, PIL.Image)`` on success, ``None`` if either import
        fails. A failure is logged once at ERROR level per call site, not
        raised, so batch processing can continue reporting failures for every
        remaining image instead of crashing Stage A.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        logger.error(
            "OCR backend unavailable (%s). Install with "
            "`pip install pytesseract pillow` and the Tesseract binary "
            "(e.g. `brew install tesseract`).",
            error,
        )
        return None
    return pytesseract, Image


def _prepare_image(image: Any, pil_image_module: Any) -> Any:
    """Normalise an opened PIL image for OCR: convert mode, upscale if small.

    Parameters
    ----------
    image:
        An opened ``PIL.Image.Image``.
    pil_image_module:
        The imported ``PIL.Image`` module, used for the resampling constant.

    Returns
    -------
    PIL.Image.Image
        RGB image, upscaled when narrower than :data:`_UPSCALE_THRESHOLD_PX`.
    """
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    width, height = image.size
    if width < _UPSCALE_THRESHOLD_PX and width > 0:
        scale = _UPSCALE_THRESHOLD_PX / width
        new_size = (int(width * scale), int(height * scale))
        resample = getattr(pil_image_module, "LANCZOS", pil_image_module.BICUBIC)
        image = image.resize(new_size, resample)

    return image


def extract_text_from_image(
    image_id: str,
    path: Path | str,
    config: MediaConfig | None = None,
) -> OcrResult:
    """Run OCR on one image and return text plus a confidence score.

    Handles four failure modes without raising: a missing OCR backend, a
    missing file, a corrupt or unreadable image, and a Tesseract runtime
    error. Each yields an :class:`OcrResult` with ``success=False`` and a
    descriptive ``error``.

    Parameters
    ----------
    image_id:
        Identifier carried through to the result, for joining back to
        ``images.csv``.
    path:
        Path to the image file.
    config:
        Media configuration; defaults to the process-wide singleton's.

    Returns
    -------
    OcrResult
    """
    cfg = config or get_config().media
    image_path = Path(path)

    if not image_path.exists():
        logger.warning("OCR: image not found for %s at %s", image_id, image_path)
        return OcrResult(image_id=image_id, success=False, error=f"file not found: {image_path}")

    backend = _load_ocr_backend()
    if backend is None:
        return OcrResult(image_id=image_id, success=False, error="ocr backend unavailable")
    pytesseract, pil_image_module = backend

    try:
        with pil_image_module.open(image_path) as raw_image:
            raw_image.verify()
    except Exception as error:  # noqa: BLE001 - any corruption must degrade, not crash
        logger.warning("OCR: corrupt or unreadable image %s (%s)", image_path, error)
        return OcrResult(image_id=image_id, success=False, error=f"corrupt image: {error}")

    try:
        with pil_image_module.open(image_path) as raw_image:
            prepared = _prepare_image(raw_image, pil_image_module)

            data = pytesseract.image_to_data(
                prepared,
                lang=cfg.ocr_languages,
                config=f"--psm {cfg.ocr_psm}",
                output_type=pytesseract.Output.DICT,
            )
    except pytesseract.TesseractNotFoundError as error:
        logger.error(
            "OCR: Tesseract binary not found (%s). Install it separately from the "
            "pytesseract Python package.",
            error,
        )
        return OcrResult(image_id=image_id, success=False, error="tesseract binary not found")
    except Exception as error:  # noqa: BLE001 - a single bad image must not stop the batch
        logger.warning("OCR: Tesseract failed on %s (%s)", image_path, error)
        return OcrResult(image_id=image_id, success=False, error=f"ocr engine error: {error}")

    words: list[str] = []
    confidences: list[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_value < 0:
            # Tesseract emits -1 for boxes with no recognised text.
            continue
        words.append(cleaned)
        confidences.append(conf_value)

    joined_text = " ".join(words).strip()
    mean_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

    result = OcrResult(
        image_id=image_id,
        text=joined_text,
        confidence=round(mean_confidence, 4),
        success=True,
        error=None,
        word_count=len(words),
    )

    if not joined_text:
        logger.info("OCR: no text detected in %s (image_id=%s)", image_path, image_id)
    elif mean_confidence < cfg.ocr_min_confidence:
        logger.info(
            "OCR: low-confidence result for %s (conf=%.2f < min=%.2f)",
            image_id,
            mean_confidence,
            cfg.ocr_min_confidence,
        )

    return result


def extract_text_from_image_cached(
    image_id: str,
    path: Path | str,
    config: MediaConfig | None = None,
    cache: MediaCache | None = None,
) -> OcrResult:
    """Cache-aware wrapper around :func:`extract_text_from_image`.

    Parameters
    ----------
    image_id:
        Cache key and result identifier.
    path:
        Path to the image file.
    config:
        Media configuration; defaults to the process-wide singleton's.
    cache:
        Cache instance; defaults to the process-wide OCR cache.

    Returns
    -------
    OcrResult
        From cache when the file's content hash is unchanged, otherwise freshly
        computed and stored.
    """
    active_cache = cache or get_ocr_cache()
    image_path = Path(path)
    content_hash = compute_file_hash(image_path)

    cached_payload = active_cache.get(image_id, content_hash=content_hash)
    if cached_payload is not None:
        logger.debug("OCR cache hit for image_id=%s", image_id)
        return OcrResult.from_dict(cached_payload)

    result = extract_text_from_image(image_id, image_path, config=config)
    active_cache.set(image_id, result.to_dict(), content_hash=content_hash)
    return result


def _resolve_image_path(row: pd.Series, media_root: Path | None) -> Path | None:
    """Resolve one row of ``images.csv`` to a filesystem path.

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


def process_images(
    images: pd.DataFrame,
    config: AppConfig | None = None,
    cache: MediaCache | None = None,
) -> pd.DataFrame:
    """Run OCR over every row of ``images.csv``, using and populating the cache.

    This is the Stage A entry point: call it once, persist the returned frame
    to ``data/cache/ocr.parquet`` (already true, since results are written
    through :class:`~src.media.cache.MediaCache`), and join it back onto
    messages by ``image_id`` afterwards.

    Parameters
    ----------
    images:
        ``images.csv`` frame with an ``image_id`` column and some path column.
    config:
        Application configuration; defaults to the process-wide singleton.
    cache:
        Cache instance; defaults to the process-wide OCR cache.

    Returns
    -------
    pandas.DataFrame
        Columns: ``image_id``, ``text``, ``confidence``, ``success``, ``error``,
        ``word_count``. Empty when ``images`` is empty or missing ``image_id``.
    """
    if images.empty or "image_id" not in images.columns:
        logger.warning("process_images: empty images frame or missing image_id column.")
        return pd.DataFrame(
            columns=["image_id", "text", "confidence", "success", "error", "word_count"]
        )

    cfg = config or get_config()
    active_cache = cache or get_ocr_cache(cfg)
    media_root = cfg.paths.media_images

    total = len(images)
    succeeded = 0
    failed = 0
    rows: list[dict[str, Any]] = []

    for index, row in images.iterrows():
        image_id = str(row["image_id"]).strip()
        if not image_id:
            continue

        path = _resolve_image_path(row, media_root)
        if path is None:
            logger.warning("process_images: no resolvable path for image_id=%s", image_id)
            result = OcrResult(image_id=image_id, success=False, error="no path column found")
        else:
            result = extract_text_from_image_cached(
                image_id, path, config=cfg.media, cache=active_cache
            )

        succeeded += int(result.success)
        failed += int(not result.success)
        rows.append(result.to_dict())

        if (index + 1) % 50 == 0 or (index + 1) == total:
            logger.info(
                "OCR progress: %d/%d (ok=%d, failed=%d)", index + 1, total, succeeded, failed
            )

    active_cache.close()
    logger.info(
        "process_images complete: %d total, %d succeeded, %d failed.", total, succeeded, failed
    )
    return pd.DataFrame(rows)


__all__ = [
    "OcrResult",
    "extract_text_from_image",
    "extract_text_from_image_cached",
    "process_images",
]