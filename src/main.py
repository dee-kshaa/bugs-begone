"""
Application entrypoint: load datasets, route every message, write output.csv.

Usage
-----
    python -m src.main
    python -m src.main --data-dir /path/to/csvs --output outputs/output.csv
    python -m src.main --limit 100 --strict

Must be run with the repository root (the directory containing ``src/``) as
the current working directory, since ``-m`` resolves ``src.main`` against
``sys.path[0]``.

This script performs no evaluation and writes no file other than the single
output CSV. Its only job is: load -> build the router -> route every message
in ``messages.csv`` -> write the result.

Message construction
---------------------
No module in the frozen architecture converts a raw ``messages.csv`` row into
a :class:`src.schema.Message` instance -- ``src/io/loaders.py`` returns
DataFrames, and ``src/pipeline/route.py`` expects ``Message`` objects. That
conversion is entrypoint wiring rather than pipeline logic, so it lives here
as a private helper rather than as a new module.

Dependencies
------------
``pandas``, ``src.config``, ``src.schema``, ``src.io.loaders``,
``src.io.writers``, ``src.pipeline.route``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config import AppConfig, configure_logging, get_config
from src.io.loaders import DataRepository, load_all
from src.io.writers import OutputValidationError, write_output_csv
from src.media.cache import get_asr_cache, get_ocr_cache
from src.pipeline.route import Router
from src.schema import ConversationType, MediaType, Message

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Message construction from the raw messages frame
# --------------------------------------------------------------------------- #


def _clean_id(value: Any) -> str | None:
    """Return a stripped string id, or ``None`` for a missing/blank value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _clean_text(value: Any) -> str:
    """Return a stripped string, or ``""`` for a missing value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _clean_bool(value: Any) -> bool:
    """Parse a loosely-typed truthy cell into a Python bool."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _resolve_media_type(row: pd.Series) -> MediaType:
    """Resolve a row's media type, preferring an explicit column when present."""
    if "media_type" in row.index and _clean_text(row.get("media_type")):
        return MediaType.from_any(row.get("media_type"))
    if _clean_id(row.get("image_id")):
        return MediaType.IMAGE
    if _clean_id(row.get("voice_note_id")):
        return MediaType.VOICE
    return MediaType.TEXT


def _resolve_recipient_id(row: pd.Series) -> str | None:
    """Resolve the receiving user's id.

    The official dataset names this column ``user_id``; some variants use
    ``recipient_user_id``. ``user_id`` cannot be aliased globally because it
    means something different in ``users.csv`` and ``group_members.csv``, so
    the fallback is resolved here at the entrypoint instead.
    """
    return _clean_id(row.get("recipient_user_id")) or _clean_id(row.get("user_id"))


def _resolve_sender_id(row: pd.Series) -> str | None:
    """Resolve the sending party's id.

    ``sender_user_id`` is aliased to ``sender_id`` by the loader. Business
    messages in the official dataset carry no sender at all, so the business
    account itself is treated as the sender.
    """
    sender_id = _clean_id(row.get("sender_id"))
    if sender_id:
        return sender_id
    return _clean_id(row.get("business_id"))


def _resolve_media_ids(row: pd.Series) -> tuple[str | None, str | None]:
    """Split the official ``media_id`` column into image and voice-note ids.

    The dataset carries a single ``media_id`` disambiguated by ``media_type``;
    the schema keeps them as separate fields.

    Returns
    -------
    tuple
        ``(image_id, voice_note_id)``, at most one of which is set.
    """
    image_id = _clean_id(row.get("image_id"))
    voice_note_id = _clean_id(row.get("voice_note_id"))
    if image_id or voice_note_id:
        return image_id, voice_note_id

    media_id = _clean_id(row.get("media_id"))
    if not media_id:
        return None, None

    media_type = _clean_text(row.get("media_type")).lower()
    if media_type == "image":
        return media_id, None
    if media_type in {"voice", "audio"}:
        return None, media_id
    return None, None


def _resolve_forwarded(row: pd.Series) -> bool:
    """Resolve the forwarding flag from either a boolean or a forward count."""
    count = row.get("forwarded_count")
    if count is not None and not (isinstance(count, float) and pd.isna(count)):
        try:
            return float(count) > 0
        except (TypeError, ValueError):
            pass
    return _clean_bool(row.get("is_forwarded"))


def _resolve_conversation_type(row: pd.Series) -> ConversationType:
    """Resolve a row's conversation type from explicit or inferable signals."""
    if "conversation_type" in row.index and _clean_text(row.get("conversation_type")):
        return ConversationType.from_any(row.get("conversation_type"))
    if _clean_id(row.get("group_id")):
        return ConversationType.GROUP
    if _clean_id(row.get("business_id")):
        return ConversationType.BUSINESS
    return ConversationType.DIRECT


def _resolve_mentions(row: pd.Series) -> tuple[str, ...]:
    """Parse a pipe/comma-separated mentions cell into a tuple of user ids."""
    raw = row.get("mentions")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    separator = "|" if "|" in text else ","
    return tuple(part.strip() for part in text.split(separator) if part.strip())


def build_media_path_index(repo: DataRepository) -> tuple[dict[str, str], dict[str, str]]:
    """Build ``{image_id -> media_path}`` and ``{voice_note_id -> media_path}`` lookups.

    Media file paths live in ``images.csv`` and ``voice_notes.csv``, keyed by
    ``image_id`` / ``voice_note_id`` -- not in ``messages.csv``. Without this
    join, :attr:`src.schema.Message.media_path` is never populated and
    :class:`src.pipeline.enrich.MediaEnricher` skips every image and voice
    note, silently disabling the whole OCR/ASR subsystem.

    Parameters
    ----------
    repo:
        Loaded dataset repository.

    Returns
    -------
    tuple
        ``(image_paths, voice_paths)``. Either may be empty when the
        corresponding CSV is absent or lacks a ``media_path`` column.
    """
    image_paths: dict[str, str] = {}
    voice_paths: dict[str, str] = {}

    images = repo.images
    if not images.empty and {"image_id", "media_path"}.issubset(images.columns):
        for _, row in images.iterrows():
            key = _clean_id(row.get("image_id"))
            value = _clean_text(row.get("media_path"))
            if key and value:
                image_paths[key] = value

    voice_notes = repo.voice_notes
    if not voice_notes.empty and {"voice_note_id", "media_path"}.issubset(voice_notes.columns):
        for _, row in voice_notes.iterrows():
            key = _clean_id(row.get("voice_note_id"))
            value = _clean_text(row.get("media_path"))
            if key and value:
                voice_paths[key] = value

    logger.info(
        "build_media_path_index: %d image path(s), %d voice path(s).",
        len(image_paths),
        len(voice_paths),
    )
    return image_paths, voice_paths


def _resolve_media_path(
    row: pd.Series,
    config: AppConfig,
    image_id: str | None = None,
    voice_note_id: str | None = None,
    image_paths: dict[str, str] | None = None,
    voice_paths: dict[str, str] | None = None,
) -> str | None:
    """Resolve a row's media file to an on-disk path.

    Resolution order:

    1. A ``media_path`` inlined on the message row itself.
    2. The ``images.csv`` / ``voice_notes.csv`` lookup, whose paths in the
       official dataset are relative to the dataset directory
       (``media/images/img_001.jpg``, ``media/audio/vn_001.mp3``).
    3. The configured media roots, for datasets that store bare filenames.

    Parameters
    ----------
    row:
        The message row.
    config:
        Application configuration, supplying the dataset and media roots.
    image_id, voice_note_id:
        Already-resolved media ids from :func:`_resolve_media_ids`.
    image_paths, voice_paths:
        Lookups from :func:`build_media_path_index`.

    Returns
    -------
    str or None
        An existing path when one can be found, the best candidate otherwise,
        or ``None`` when the row references no media.
    """
    raw = row.get("media_path")
    text = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw).strip()

    if not text and image_id and image_paths:
        text = image_paths.get(image_id, "")
    if not text and voice_note_id and voice_paths:
        text = voice_paths.get(voice_note_id, "")

    if not text:
        return None

    candidate = Path(text)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)

    # Official dataset: paths are relative to the dataset directory.
    dataset_relative = config.paths.raw / candidate
    if dataset_relative.exists():
        return str(dataset_relative)

    # Fallback: bare filenames stored under the configured media roots.
    if image_id:
        root_relative = config.paths.media_images / candidate.name
        if root_relative.exists():
            return str(root_relative)
        return str(dataset_relative)
    if voice_note_id:
        root_relative = config.paths.media_voice / candidate.name
        if root_relative.exists():
            return str(root_relative)
        return str(dataset_relative)

    return str(dataset_relative)


def row_to_message(
    row: pd.Series,
    config: AppConfig,
    image_paths: dict[str, str] | None = None,
    voice_paths: dict[str, str] | None = None,
) -> Message | None:
    """Convert one normalised ``messages.csv`` row into a :class:`~src.schema.Message`.

    Handles the official HackerRank column naming, which differs from the
    canonical names used internally:

    * ``user_id`` is the *recipient*, not the sender.
    * ``sender_user_id`` is the sender, and is empty for business messages,
      where the business account itself is treated as the sender.
    * ``media_id`` is a single column disambiguated by ``media_type``.
    * ``forwarded_count`` is an integer rather than a boolean flag.

    Parameters
    ----------
    row:
        A row from the DataFrame returned by ``src.io.loaders.load_all``, with
        columns already normalised to canonical names.
    config:
        Application configuration, used to resolve relative media paths.
    image_paths:
        ``{image_id -> media_path}`` from :func:`build_media_path_index`.
        Optional, so the historical two-argument call signature still works.
    voice_paths:
        ``{voice_note_id -> media_path}`` from :func:`build_media_path_index`.

    Returns
    -------
    Message or None
        ``None`` when the row is missing a required field (``message_id``, a
        resolvable sender, or a parseable timestamp), so the caller can skip
        it and continue rather than aborting the whole load.
    """
    message_id = _clean_id(row.get("message_id"))
    sender_id = _resolve_sender_id(row)

    if not message_id or not sender_id:
        logger.warning(
            "row_to_message: skipping row with missing message_id/sender (message_id=%r)",
            message_id,
        )
        return None

    timestamp = pd.to_datetime(row.get("timestamp"), errors="coerce")
    if pd.isna(timestamp):
        logger.warning("row_to_message: skipping message_id=%s (unparseable timestamp)", message_id)
        return None

    image_id, voice_note_id = _resolve_media_ids(row)

    try:
        return Message(
            message_id=message_id,
            sender_id=sender_id,
            timestamp=timestamp.to_pydatetime(),
            recipient_user_id=_resolve_recipient_id(row),
            conversation_id=_clean_id(row.get("conversation_id")),
            group_id=_clean_id(row.get("group_id")),
            business_id=_clean_id(row.get("business_id")),
            reply_to_id=_clean_id(row.get("reply_to_id")),
            media_type=_resolve_media_type(row),
            conversation_type=_resolve_conversation_type(row),
            message_text=_clean_text(row.get("message_text")),
            image_id=image_id,
            voice_note_id=voice_note_id,
            media_path=_resolve_media_path(
                row,
                config,
                image_id,
                voice_note_id,
                image_paths,
                voice_paths,
            ),
            mentions=_resolve_mentions(row),
            is_forwarded=_resolve_forwarded(row),
            is_from_business=bool(_clean_id(row.get("business_id"))),
        )
    except (ValueError, TypeError) as error:
        logger.warning("row_to_message: skipping message_id=%s (%s)", message_id, error)
        return None


def iter_messages(
    repo: DataRepository, config: AppConfig, limit: int | None = None
) -> Iterator[Message]:
    """Yield :class:`~src.schema.Message` objects from ``repo.messages``, in file order.

    Parameters
    ----------
    repo:
        Loaded dataset repository.
    config:
        Application configuration.
    limit:
        When given, stop after yielding this many messages.

    Yields
    ------
    Message
        Rows that fail conversion are skipped with a logged warning rather
        than raising.
    """
    frame = repo.messages
    if frame.empty:
        logger.error("iter_messages: messages.csv is empty or failed to load; nothing to route.")
        return

    image_paths, voice_paths = build_media_path_index(repo)

    yielded = 0
    skipped = 0
    duplicates = 0
    seen_ids: set[str] = set()
    for _, row in frame.iterrows():
        message = row_to_message(row, config, image_paths, voice_paths)
        if message is None:
            skipped += 1
            continue
        # A repeated message_id in the input would otherwise produce a
        # repeated output row, which fails output validation and aborts the
        # entire run with no file written at all. One malformed input row
        # must not cost every other prediction, so later copies are dropped
        # and the first occurrence wins.
        if message.message_id in seen_ids:
            duplicates += 1
            logger.warning(
                "iter_messages: duplicate message_id=%s in input; keeping the first row.",
                message.message_id,
            )
            continue
        seen_ids.add(message.message_id)
        yield message
        yielded += 1
        if limit is not None and yielded >= limit:
            break

    logger.info(
        "iter_messages: yielded %d message(s), skipped %d invalid row(s), "
        "dropped %d duplicate id(s).",
        yielded,
        skipped,
        duplicates,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv:
        Argument list; defaults to ``sys.argv[1:]`` via ``argparse``.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Route every message in messages.csv and write output.csv."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the raw dataset directory (defaults to data/raw or WA_ROUTER_DATA_DIR).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (defaults to outputs/output.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only route the first N messages (for a quick smoke test).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing/malformed input files instead of degrading gracefully.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip decisions that fail output validation instead of aborting the write.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full load -> route -> write pipeline.

    Parameters
    ----------
    argv:
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` on a fatal error.
    """
    args = parse_args(argv)
    config = get_config()
    configure_logging(args.log_level, log_file=config.paths.logs / "main.log")

    if args.data_dir:
        import os

        os.environ["WA_ROUTER_DATA_DIR"] = args.data_dir
        config = AppConfig()  # rebuild so the override takes effect

    output_path = Path(args.output) if args.output else config.paths.outputs / "output.csv"

    start = time.monotonic()
    logger.info("Loading datasets...")
    try:
        repo = load_all(config, strict=args.strict)
    except Exception as error:  # noqa: BLE001 - top-level entrypoint boundary
        logger.error("Fatal: failed to load datasets (%s)", error)
        return 1

    if repo.messages.empty:
        logger.error("Fatal: messages.csv is empty or missing; nothing to route.")
        return 1

    logger.info("Building router...")
    try:
        router = Router(repo, config=config)
    except Exception as error:  # noqa: BLE001
        logger.error("Fatal: failed to build the router (%s)", error)
        return 1

    logger.info("Routing messages...")
    messages = iter_messages(repo, config, limit=args.limit)
    try:
        decisions = router.route_to_list(messages)
    finally:
        # MessageEnricher writes OCR/ASR results through the process-wide
        # MediaCache singletons, which only auto-flush every N writes
        # (see src/media/cache.py DEFAULT_FLUSH_EVERY). Without an explicit
        # flush here, any run touching fewer than that many images/voice
        # notes -- or whose count isn't an exact multiple -- silently loses
        # that transcription work on process exit, defeating the cache's
        # resumability guarantee. Flushing in `finally` ensures partial work
        # is persisted even if routing raises partway through.
        try:
            get_ocr_cache(config).close()
            get_asr_cache(config).close()
        except Exception as cache_error:  # noqa: BLE001 - never mask the real error
            logger.warning("Failed to flush media caches on exit: %s", cache_error)

    if not decisions:
        logger.error("Fatal: no decisions were produced.")
        return 1

    logger.info("Writing %s...", output_path)
    try:
        row_count = write_output_csv(
            decisions, output_path, validate=True, skip_invalid=args.skip_invalid
        )
    except OutputValidationError as error:
        logger.error("Fatal: output validation failed (%s)", error)
        return 1
    except OSError as error:
        logger.error("Fatal: could not write %s (%s)", output_path, error)
        return 1

    elapsed = time.monotonic() - start
    stats = router.stats()
    logger.info(
        "Done: %d row(s) written to %s in %.1fs (processed=%d, failed=%d, success_rate=%.1f%%).",
        row_count,
        output_path,
        elapsed,
        stats["processed"],
        stats["failed"],
        stats["success_rate"] * 100.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "build_media_path_index",
    "iter_messages",
    "main",
    "parse_args",
    "row_to_message",
]