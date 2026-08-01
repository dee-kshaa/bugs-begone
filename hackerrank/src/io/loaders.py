"""
Dataset loading, column normalisation and schema validation.

Downstream modules never open a CSV. They call :func:`load_all` once and then
read DataFrames off the returned :class:`DataRepository`.

What this module guarantees
---------------------------
* Column names are lower-cased, stripped, and mapped through
  ``config.COLUMN_ALIASES`` to canonical names.
* Every ``*_id`` column is a clean string, so joins never fail because one file
  wrote ``1001`` and another wrote ``"1001.0"``.
* Declared datetime columns are real ``datetime64`` values.
* A missing optional file yields an empty, correctly-columned DataFrame plus a
  warning instead of a crash.

Dependencies
------------
``pandas``, ``src.config``. Optional files are tolerated by design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config import (
    COLUMN_ALIASES,
    DATASETS,
    AppConfig,
    DatasetSpec,
    get_config,
)

logger = logging.getLogger(__name__)


class SchemaValidationError(RuntimeError):
    """Raised when a required column is missing and strict mode is enabled."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a required dataset file is absent from the raw directory."""


# --------------------------------------------------------------------------- #
# Column normalisation helpers
# --------------------------------------------------------------------------- #


def normalise_column_name(name: str) -> str:
    """Return the canonical form of a single column name.

    Lower-cases, strips whitespace, converts spaces and hyphens to
    underscores, collapses repeated underscores, then applies
    ``config.COLUMN_ALIASES``.

    Parameters
    ----------
    name:
        Raw column name straight from the CSV header.

    Returns
    -------
    str
        Canonical column name.
    """
    cleaned = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return COLUMN_ALIASES.get(cleaned, cleaned)


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename every column of ``frame`` to its canonical form.

    Collisions (two source columns mapping to the same canonical name) keep the
    first occurrence and leave the later column under its original name, because
    silently dropping data is worse than a noisy log line.

    Parameters
    ----------
    frame:
        DataFrame straight from :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        A renamed copy; the input is not mutated.
    """
    renamed: dict[str, str] = {}
    seen: set[str] = set()
    for column in frame.columns:
        canonical = normalise_column_name(column)
        if canonical in seen:
            logger.warning(
                "Column collision after normalisation: %r -> %r already exists; "
                "keeping the first and leaving %r under its original name.",
                column,
                canonical,
                column,
            )
            continue
        renamed[column] = canonical
        seen.add(canonical)
    return frame.rename(columns=renamed)


def coerce_id_column(series: pd.Series) -> pd.Series:
    """Coerce an identifier column to clean, join-safe strings.

    Handles the common pandas artefact where integer ids read from a column
    containing nulls become floats and stringify as ``"1001.0"``. Nulls become
    empty strings so that ``.str`` operations downstream never hit ``NaN``.

    Parameters
    ----------
    series:
        Raw identifier column.

    Returns
    -------
    pandas.Series
        Object-dtype series of stripped strings.
    """
    if pd.api.types.is_float_dtype(series):
        # Only integral floats can be safely rendered without a decimal part.
        as_string = series.map(
            lambda value: ""
            if pd.isna(value)
            else (str(int(value)) if float(value).is_integer() else str(value))
        )
    else:
        as_string = series.astype("string").fillna("")

    return (
        as_string.astype(str)
        .str.strip()
        .replace({"nan": "", "NaN": "", "None": "", "<NA>": "", "null": ""})
    )


def coerce_datetime_column(series: pd.Series) -> pd.Series:
    """Parse a column into ``datetime64``, tolerating epoch numbers or text.

    Numeric columns are treated as Unix epochs: seconds when the magnitude looks
    like seconds, milliseconds when it is large enough to imply them. Everything
    else goes through :func:`pandas.to_datetime` with ``errors="coerce"`` so a
    handful of malformed rows never aborts the load.

    Parameters
    ----------
    series:
        Raw timestamp column.

    Returns
    -------
    pandas.Series
        ``datetime64[ns]`` series; unparseable entries become ``NaT``.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    if pd.api.types.is_numeric_dtype(series):
        finite = series.dropna()
        unit = "s"
        if not finite.empty and float(finite.abs().max()) > 1e11:
            unit = "ms"
        return pd.to_datetime(series, unit=unit, errors="coerce")

    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        # ``format="mixed"`` is unavailable or unhappy; fall back to inference.
        return pd.to_datetime(series, errors="coerce")


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #


@dataclass
class DataRepository:
    """Container for every loaded dataset.

    Attributes are named after :attr:`DatasetSpec.key`, so downstream code
    writes ``repo.messages`` rather than knowing any path.
    """

    messages: pd.DataFrame = field(default_factory=pd.DataFrame)
    users: pd.DataFrame = field(default_factory=pd.DataFrame)
    groups: pd.DataFrame = field(default_factory=pd.DataFrame)
    group_members: pd.DataFrame = field(default_factory=pd.DataFrame)
    business_accounts: pd.DataFrame = field(default_factory=pd.DataFrame)
    user_business_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    message_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    message_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    images: pd.DataFrame = field(default_factory=pd.DataFrame)
    voice_notes: pd.DataFrame = field(default_factory=pd.DataFrame)

    #: ``key -> True`` when the file existed and parsed.
    loaded: dict[str, bool] = field(default_factory=dict)
    #: Non-fatal problems accumulated during loading.
    warnings: list[str] = field(default_factory=list)
    #: Source directory the frames came from.
    source_dir: Path | None = None

    # ---- access helpers ---------------------------------------------------- #

    def get(self, key: str) -> pd.DataFrame:
        """Return the DataFrame registered under ``key``.

        Parameters
        ----------
        key:
            One of the dataset keys declared in ``config.DATASETS``.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        KeyError
            If ``key`` is not a known dataset attribute.
        """
        if not hasattr(self, key):
            raise KeyError(f"Unknown dataset {key!r}. Known: {sorted(self.keys())}")
        value = getattr(self, key)
        if not isinstance(value, pd.DataFrame):
            raise KeyError(f"{key!r} is not a dataset attribute")
        return value

    def has(self, key: str) -> bool:
        """Return whether ``key`` loaded successfully and is non-empty."""
        return bool(self.loaded.get(key, False)) and not self.get(key).empty

    def keys(self) -> tuple[str, ...]:
        """Return the names of all registered datasets, in load order."""
        return tuple(spec.key for spec in DATASETS)

    def frames(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """Iterate over ``(key, frame)`` pairs for every dataset."""
        for key in self.keys():
            yield key, self.get(key)

    def has_column(self, key: str, column: str) -> bool:
        """Return whether dataset ``key`` contains ``column``."""
        return column in self.get(key).columns

    def columns(self, key: str) -> list[str]:
        """Return the column names of dataset ``key``."""
        return list(self.get(key).columns)

    def require(self, key: str) -> pd.DataFrame:
        """Return dataset ``key``, raising if it is missing or empty.

        Use this in stages that genuinely cannot proceed without the data.

        Raises
        ------
        DatasetNotFoundError
            If the dataset failed to load or contains no rows.
        """
        frame = self.get(key)
        if not self.loaded.get(key, False):
            raise DatasetNotFoundError(f"Dataset {key!r} was never loaded.")
        if frame.empty:
            raise DatasetNotFoundError(f"Dataset {key!r} loaded but is empty.")
        return frame

    # ---- reporting --------------------------------------------------------- #

    def summary(self) -> pd.DataFrame:
        """Return a one-row-per-dataset overview for logging or a notebook."""
        rows: list[dict[str, Any]] = []
        for key, frame in self.frames():
            rows.append(
                {
                    "dataset": key,
                    "loaded": self.loaded.get(key, False),
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "column_names": ", ".join(map(str, list(frame.columns)[:12])),
                }
            )
        return pd.DataFrame(rows)

    def log_summary(self) -> None:
        """Emit the dataset overview through the standard logger."""
        logger.info("Dataset summary (source=%s):", self.source_dir)
        for key, frame in self.frames():
            status = "ok " if self.loaded.get(key) else "MISS"
            logger.info(
                "  [%s] %-22s rows=%-8d cols=%-3d  %s",
                status,
                key,
                len(frame),
                len(frame.columns),
                ", ".join(map(str, list(frame.columns)[:10])),
            )
        if self.warnings:
            logger.warning("%d loader warning(s) recorded:", len(self.warnings))
            for warning in self.warnings:
                logger.warning("  - %s", warning)


# --------------------------------------------------------------------------- #
# Loading internals
# --------------------------------------------------------------------------- #


def _empty_frame(spec: DatasetSpec) -> pd.DataFrame:
    """Build an empty DataFrame carrying ``spec``'s declared columns.

    Downstream code can then do ``frame["message_id"]`` without a guard, even
    when the underlying file was absent.
    """
    columns = list(dict.fromkeys(spec.required_columns + spec.id_columns))
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one CSV defensively.

    Falls back through UTF-8 with BOM handling and then latin-1, because
    hackathon datasets are frequently exported from spreadsheets.

    Parameters
    ----------
    path:
        Absolute path to the CSV file.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    UnicodeDecodeError
        If no known encoding could decode the file.
    """
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                keep_default_na=True,
                na_values=["", "NA", "N/A", "null", "NULL", "None"],
                low_memory=False,
            )
        except UnicodeDecodeError as error:
            last_error = error
            logger.debug("Encoding %s failed for %s, trying next.", encoding, path.name)

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Could not decode {path} with utf-8-sig, utf-8 or latin-1 ({last_error})",
    )


def _validate_required_columns(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    strict: bool,
    warnings: list[str],
) -> None:
    """Check that ``spec.required_columns`` are present.

    In strict mode a missing column raises :class:`SchemaValidationError`;
    otherwise it is logged, recorded in ``warnings``, and the column is created
    empty so downstream code does not have to guard every access.

    Parameters
    ----------
    frame:
        Already column-normalised DataFrame. Mutated in place when columns are
        backfilled.
    spec:
        Dataset description carrying the required column list.
    strict:
        Whether a missing column is fatal.
    warnings:
        Mutable list that non-fatal problems are appended to.

    Raises
    ------
    SchemaValidationError
        If a required column is missing and ``strict`` is ``True``.
    """
    missing = [column for column in spec.required_columns if column not in frame.columns]
    if not missing:
        return

    message = (
        f"{spec.filename}: missing required column(s) {missing}. "
        f"Present columns: {list(frame.columns)}"
    )
    if strict:
        raise SchemaValidationError(message)

    logger.error("%s", message)
    logger.error(
        "  -> Add an entry to config.COLUMN_ALIASES if these exist under another name."
    )
    warnings.append(message)
    for column in missing:
        frame[column] = pd.Series([pd.NA] * len(frame), dtype="object")


def _apply_coercions(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Coerce declared id and datetime columns to their canonical dtypes.

    Parameters
    ----------
    frame:
        Column-normalised DataFrame.
    spec:
        Dataset description naming the id and datetime columns.

    Returns
    -------
    pandas.DataFrame
        The same frame, coerced in place and returned for chaining.
    """
    for column in spec.id_columns:
        if column in frame.columns:
            frame[column] = coerce_id_column(frame[column])

    for column in spec.datetime_columns:
        if column not in frame.columns:
            continue
        before_null = int(frame[column].isna().sum())
        frame[column] = coerce_datetime_column(frame[column])
        after_null = int(frame[column].isna().sum())
        unparsed = after_null - before_null
        if unparsed > 0:
            logger.warning(
                "%s: %d value(s) in %r could not be parsed as datetime.",
                spec.filename,
                unparsed,
                column,
            )

    return frame


# --------------------------------------------------------------------------- #
# Public loading API
# --------------------------------------------------------------------------- #


def load_dataset(
    spec: DatasetSpec,
    raw_dir: Path,
    strict: bool,
    warnings: list[str],
) -> tuple[pd.DataFrame, bool]:
    """Load, normalise and validate one dataset.

    Parameters
    ----------
    spec:
        Declarative description of the file.
    raw_dir:
        Directory holding the raw CSVs.
    strict:
        When ``True``, missing required files and columns raise.
    warnings:
        Mutable list that non-fatal problems are appended to.

    Returns
    -------
    tuple
        ``(frame, was_loaded)``. ``was_loaded`` is ``False`` when the file was
        absent and an empty placeholder was substituted.

    Raises
    ------
    DatasetNotFoundError
        If a required file is missing and ``strict`` is ``True``.
    SchemaValidationError
        If a required column is missing and ``strict`` is ``True``.
    """
    path = raw_dir / spec.filename

    if not path.exists():
        message = f"{spec.filename} not found in {raw_dir}"
        if spec.required_file and strict:
            raise DatasetNotFoundError(message)
        log = logger.error if spec.required_file else logger.warning
        log("%s -- substituting an empty frame.", message)
        warnings.append(message)
        return _empty_frame(spec), False

    logger.debug("Reading %s", path)
    frame = _read_csv(path)
    original_columns = list(frame.columns)

    frame = normalise_columns(frame)
    renamed = {
        before: after
        for before, after in zip(original_columns, list(frame.columns))
        if before != after
    }
    if renamed:
        logger.info(
            "%s: normalised %d column name(s): %s", spec.filename, len(renamed), renamed
        )

    _validate_required_columns(frame, spec, strict, warnings)
    frame = _apply_coercions(frame, spec)

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        logger.warning(
            "%s: %d fully duplicated row(s) present (not dropped).",
            spec.filename,
            duplicate_rows,
        )

    logger.info(
        "Loaded %-24s rows=%-8d cols=%d", spec.filename, len(frame), len(frame.columns)
    )
    return frame, True


def load_all(config: AppConfig | None = None, strict: bool | None = None) -> DataRepository:
    """Load every configured dataset into a :class:`DataRepository`.

    This is the single entry point downstream modules should use. It creates the
    project directories, reads all ten CSVs, normalises their schemas, and logs a
    summary table.

    Parameters
    ----------
    config:
        Application configuration. Defaults to the process-wide singleton
        returned by :func:`src.config.get_config`.
    strict:
        Override for ``config.strict_schema``. When ``True``, missing required
        files or columns raise instead of warning.

    Returns
    -------
    DataRepository
        Populated repository. Always returned in non-strict mode, even if some
        files were absent.

    Raises
    ------
    DatasetNotFoundError
        In strict mode, when the raw directory or a required file is missing.
    SchemaValidationError
        In strict mode, when a required column is missing.
    """
    cfg = config or get_config()
    is_strict = cfg.strict_schema if strict is None else strict
    raw_dir = cfg.paths.raw

    cfg.paths.ensure()
    logger.info("Loading datasets from %s (strict=%s)", raw_dir, is_strict)

    if not raw_dir.exists():
        message = f"Raw data directory does not exist: {raw_dir}"
        if is_strict:
            raise DatasetNotFoundError(message)
        logger.error("%s", message)

    repository = DataRepository(source_dir=raw_dir)

    for spec in DATASETS:
        frame, was_loaded = load_dataset(spec, raw_dir, is_strict, repository.warnings)
        setattr(repository, spec.key, frame)
        repository.loaded[spec.key] = was_loaded

    repository.log_summary()

    missing_required = [
        spec.key
        for spec in DATASETS
        if spec.required_file and not repository.loaded.get(spec.key, False)
    ]
    if missing_required:
        logger.error(
            "Required dataset(s) missing: %s. Downstream stages will degrade.",
            missing_required,
        )

    return repository


__all__ = [
    "DataRepository",
    "DatasetNotFoundError",
    "SchemaValidationError",
    "coerce_datetime_column",
    "coerce_id_column",
    "load_all",
    "load_dataset",
    "normalise_column_name",
    "normalise_columns",
]