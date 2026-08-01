"""
Reusable parquet-backed caching for expensive, idempotent media operations.

OCR and ASR are the two costly Stage A steps: OCR shells out to Tesseract per
image, ASR runs a Whisper forward pass per voice note. Both are deterministic
given the same file, so a content-hash-keyed cache lets Stage A be interrupted
and resumed without recomputation and lets the same image or clip be reused
across dataset reruns.

Design
------
* One :class:`MediaCache` instance wraps one parquet file.
* Keys are caller-supplied ids (``image_id`` / ``voice_note_id``); each entry
  also stores a content hash of the source file, so a changed file on disk
  invalidates its own cache entry automatically without a separate pass.
* Payloads are arbitrary JSON-serialisable dicts, kept generic so OCR and ASR
  share one implementation despite different result shapes.
* Writes are buffered in memory and flushed to disk periodically and on
  ``flush()``/``close()``, using an atomic tmp-file-then-rename so a crash
  mid-write never corrupts the cache.

Dependencies
------------
``pandas``, ``pyarrow`` (via pandas' parquet engine), ``src.config``.
Standard library otherwise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig, get_config

logger = logging.getLogger(__name__)

#: Number of chunks read at a time when hashing a file, to bound memory use.
_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB

#: Default number of writes buffered before an automatic flush.
DEFAULT_FLUSH_EVERY = 25


def compute_file_hash(path: Path) -> str | None:
    """Compute a stable SHA-256 hash of a file's contents.

    Used as the cache-invalidation key: if a file at a given id is replaced on
    disk (e.g. a corrupted image gets re-exported), its hash changes and the
    stale cache entry is treated as a miss automatically.

    Parameters
    ----------
    path:
        File to hash.

    Returns
    -------
    str or None
        Hex digest, or ``None`` if the file could not be read.
    """
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as error:
        logger.warning("compute_file_hash: could not read %s (%s)", path, error)
        return None
    return digest.hexdigest()


class MediaCache:
    """A generic, parquet-backed, content-hash-invalidated key-value cache.

    Thread-safe for concurrent ``get``/``set`` calls from a single process via
    an internal lock; not safe for concurrent processes writing the same file.

    Parameters
    ----------
    cache_path:
        Parquet file backing this cache. Parent directories are created on
        first flush.
    flush_every:
        Number of ``set`` calls buffered before an automatic flush to disk.
        Lower values trade write throughput for crash resilience.
    """

    def __init__(self, cache_path: Path, flush_every: int = DEFAULT_FLUSH_EVERY) -> None:
        self.cache_path = Path(cache_path)
        self.flush_every = max(1, flush_every)

        self._store: dict[str, dict[str, Any]] = {}
        self._dirty_count = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        self._load()

    # ---- persistence --------------------------------------------------------- #

    def _load(self) -> None:
        """Load existing cache entries from disk, if the parquet file exists."""
        if not self.cache_path.exists():
            logger.info("MediaCache: no existing cache at %s (starting empty).", self.cache_path)
            return

        try:
            frame = pd.read_parquet(self.cache_path)
        except Exception as error:  # noqa: BLE001 - a corrupt cache must not crash Stage A
            logger.error(
                "MediaCache: failed to read %s (%s); starting from an empty cache.",
                self.cache_path,
                error,
            )
            return

        required = {"key", "content_hash", "payload_json", "updated_at"}
        if not required.issubset(frame.columns):
            logger.error(
                "MediaCache: %s is missing expected columns %s; starting from an empty cache.",
                self.cache_path,
                required - set(frame.columns),
            )
            return

        loaded = 0
        for _, row in frame.iterrows():
            key = str(row["key"])
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning("MediaCache: unparseable payload for key %r; skipping.", key)
                continue
            self._store[key] = {
                "content_hash": row.get("content_hash"),
                "payload": payload,
                "updated_at": row.get("updated_at"),
            }
            loaded += 1

        logger.info("MediaCache: loaded %d entr(y/ies) from %s.", loaded, self.cache_path)

    def flush(self) -> None:
        """Write all buffered entries to the parquet file atomically.

        Uses a temp file in the same directory plus ``os.replace`` so a crash
        mid-write leaves the previous cache file intact rather than a
        half-written parquet file.
        """
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Flush implementation; caller must already hold ``self._lock``."""
        if not self._store:
            self._dirty_count = 0
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "key": key,
                "content_hash": record.get("content_hash"),
                "payload_json": json.dumps(record["payload"], default=str),
                "updated_at": record.get("updated_at"),
            }
            for key, record in self._store.items()
        ]
        frame = pd.DataFrame(rows)

        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.cache_path.parent), prefix=f".{self.cache_path.name}.", suffix=".tmp"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            frame.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, self.cache_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        finally:
            self._dirty_count = 0

        logger.debug("MediaCache: flushed %d entr(y/ies) to %s.", len(self._store), self.cache_path)

    def close(self) -> None:
        """Flush any buffered writes. Call at the end of a Stage A run."""
        self.flush()
        logger.info(
            "MediaCache[%s]: closed. hits=%d misses=%d entries=%d",
            self.cache_path.name,
            self._hits,
            self._misses,
            len(self._store),
        )

    # ---- key-value API --------------------------------------------------------- #

    def get(self, key: str, content_hash: str | None = None) -> dict[str, Any] | None:
        """Look up a cached payload.

        Parameters
        ----------
        key:
            Cache key, typically ``image_id`` or ``voice_note_id``.
        content_hash:
            When given, a stored entry whose hash does not match is treated as
            a miss (the underlying file changed since it was cached).

        Returns
        -------
        dict or None
            The cached payload, or ``None`` on a miss.
        """
        with self._lock:
            record = self._store.get(str(key))
            if record is None:
                self._misses += 1
                return None
            if content_hash is not None and record.get("content_hash") != content_hash:
                self._misses += 1
                return None
            self._hits += 1
            return record["payload"]

    def set(self, key: str, payload: dict[str, Any], content_hash: str | None = None) -> None:
        """Store a payload, buffering the write and flushing periodically.

        Parameters
        ----------
        key:
            Cache key, typically ``image_id`` or ``voice_note_id``.
        payload:
            JSON-serialisable result to cache.
        content_hash:
            Hash of the source file, used for future invalidation checks.
        """
        with self._lock:
            self._store[str(key)] = {
                "content_hash": content_hash,
                "payload": payload,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._dirty_count += 1
            should_flush = self._dirty_count >= self.flush_every

        if should_flush:
            self.flush()

    def invalidate(self, key: str) -> bool:
        """Remove one entry from the cache and flush immediately.

        Parameters
        ----------
        key:
            Cache key to remove.

        Returns
        -------
        bool
            ``True`` if the key was present and removed.
        """
        with self._lock:
            existed = self._store.pop(str(key), None) is not None
            if existed:
                self._flush_locked()
        return existed

    def clear(self) -> None:
        """Remove every entry and delete the backing parquet file."""
        with self._lock:
            self._store.clear()
            self._dirty_count = 0
            if self.cache_path.exists():
                self.cache_path.unlink()
        logger.info("MediaCache: cleared %s.", self.cache_path)

    def __len__(self) -> int:
        """Return the number of entries currently held (including unflushed)."""
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        """Return whether ``key`` is present, ignoring content-hash validity."""
        return str(key) in self._store

    @property
    def hit_rate(self) -> float:
        """Return the cumulative hit rate for this cache instance, in ``[0, 1]``."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        """Return a small diagnostic summary, useful for a Stage A log line."""
        return {
            "path": str(self.cache_path),
            "entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }


# --------------------------------------------------------------------------- #
# Factories tied to project configuration
# --------------------------------------------------------------------------- #

#: Process-wide cache instances, created lazily so importing this module never
#: touches the filesystem.
_ocr_cache_instance: MediaCache | None = None
_asr_cache_instance: MediaCache | None = None
_cache_lock = threading.Lock()


def get_ocr_cache(config: AppConfig | None = None) -> MediaCache:
    """Return the process-wide OCR cache, creating it on first use.

    Parameters
    ----------
    config:
        Application configuration; defaults to the process-wide singleton.

    Returns
    -------
    MediaCache
        Backed by ``config.paths.ocr_cache``.
    """
    global _ocr_cache_instance
    cfg = config or get_config()
    with _cache_lock:
        if _ocr_cache_instance is None:
            _ocr_cache_instance = MediaCache(cfg.paths.ocr_cache)
        return _ocr_cache_instance


def get_asr_cache(config: AppConfig | None = None) -> MediaCache:
    """Return the process-wide ASR cache, creating it on first use.

    Parameters
    ----------
    config:
        Application configuration; defaults to the process-wide singleton.

    Returns
    -------
    MediaCache
        Backed by ``config.paths.asr_cache``.
    """
    global _asr_cache_instance
    cfg = config or get_config()
    with _cache_lock:
        if _asr_cache_instance is None:
            _asr_cache_instance = MediaCache(cfg.paths.asr_cache)
        return _asr_cache_instance


def reset_caches_for_testing() -> None:
    """Drop the process-wide cache singletons without touching disk.

    Intended for test suites that need a clean in-memory state between cases;
    does not delete any parquet file.
    """
    global _ocr_cache_instance, _asr_cache_instance
    with _cache_lock:
        _ocr_cache_instance = None
        _asr_cache_instance = None


__all__ = [
    "DEFAULT_FLUSH_EVERY",
    "MediaCache",
    "compute_file_hash",
    "get_asr_cache",
    "get_ocr_cache",
    "reset_caches_for_testing",
]