"""
Output writing: Decision objects -> output.csv.

Writes the final submission file in the exact schema required:
``message_id, action, message_type, reason, confidence, evidence_message_ids``.

Deliberately narrow in scope: this module only serialises already-computed
:class:`~src.schema.Decision` objects to disk. It performs no scoring, no
enrichment, and no evaluation.

Dependencies
------------
``pandas``, ``src.schema``. Standard library otherwise.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.schema import Action, Decision, MessageType

logger = logging.getLogger(__name__)

#: Exact column order required by the output schema. Never reorder these:
#: downstream graders match on position and header name.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

#: Literal written when a decision has no evidence, per the task's contract.
NO_EVIDENCE_LITERAL = "none"

#: Decimal places for the confidence column. Fixed so every row formats
#: identically regardless of how the value was computed upstream.
CONFIDENCE_DECIMALS = 4


class OutputValidationError(ValueError):
    """Raised when a decision or the assembled output fails schema validation."""


def format_confidence(confidence: float) -> str:
    """Format a confidence value with a fixed, deterministic decimal precision.

    Parameters
    ----------
    confidence:
        Confidence in ``[0, 1]``.

    Returns
    -------
    str
        Fixed-precision decimal string, e.g. ``"0.8421"``.

    Raises
    ------
    OutputValidationError
        If ``confidence`` is not a finite number in ``[0, 1]``.
    """
    try:
        value = float(confidence)
    except (TypeError, ValueError) as error:
        raise OutputValidationError(f"confidence is not numeric: {confidence!r}") from error

    if not (0.0 <= value <= 1.0) or value != value:  # value != value catches NaN
        raise OutputValidationError(f"confidence out of range [0,1]: {value!r}")

    return f"{value:.{CONFIDENCE_DECIMALS}f}"


def format_evidence(evidence_message_ids: Sequence[str]) -> str:
    """Render an evidence id list as the pipe-joined CSV field.

    Parameters
    ----------
    evidence_message_ids:
        Ordered evidence message ids. May be empty.

    Returns
    -------
    str
        Pipe-joined ids, or :data:`NO_EVIDENCE_LITERAL` when empty.
    """
    cleaned = [str(item).strip() for item in evidence_message_ids if str(item).strip()]
    if not cleaned:
        return NO_EVIDENCE_LITERAL
    return "|".join(cleaned)


def decision_to_row(decision: Decision) -> dict[str, str]:
    """Convert one :class:`~src.schema.Decision` into an output row.

    Deliberately does not call :meth:`~src.schema.Decision.to_submission_row`
    for the evidence field, since that frozen method renders an empty list as
    ``""`` rather than the ``"none"`` literal this output schema requires.

    Parameters
    ----------
    decision:
        A validated decision, as produced by
        :class:`~src.decision.arbiter.Arbiter`.

    Returns
    -------
    dict
        Keys exactly matching :data:`OUTPUT_COLUMNS`, all string-valued.

    Raises
    ------
    OutputValidationError
        If any required field is missing or malformed.
    """
    if not decision.message_id:
        raise OutputValidationError("Decision.message_id is empty")
    if not decision.reason:
        raise OutputValidationError(f"Decision.reason is empty for {decision.message_id}")

    action = Action.from_any(decision.action)
    message_type = MessageType.from_any(decision.message_type)

    return {
        "message_id": decision.message_id,
        "action": action.value,
        "message_type": message_type.value,
        "reason": decision.reason,
        "confidence": format_confidence(decision.confidence),
        "evidence_message_ids": format_evidence(decision.evidence_message_ids),
    }


def validate_rows(rows: Sequence[dict[str, str]]) -> None:
    """Validate a full set of output rows before writing.

    Checks column completeness, non-empty required fields, and duplicate
    ``message_id`` values.

    Parameters
    ----------
    rows:
        Rows already produced by :func:`decision_to_row`.

    Raises
    ------
    OutputValidationError
        On any schema violation. The error message names every problem found,
        not just the first, so a single validation pass surfaces everything.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()

    for index, row in enumerate(rows):
        missing = [column for column in OUTPUT_COLUMNS if column not in row]
        if missing:
            problems.append(f"row {index}: missing column(s) {missing}")
            continue

        message_id = row["message_id"]
        if not message_id:
            problems.append(f"row {index}: empty message_id")
        elif message_id in seen_ids:
            problems.append(f"row {index}: duplicate message_id {message_id!r}")
        else:
            seen_ids.add(message_id)

        if row["action"] not in {a.value for a in Action}:
            problems.append(f"row {index} ({message_id}): invalid action {row['action']!r}")
        if row["message_type"] not in {t.value for t in MessageType}:
            problems.append(
                f"row {index} ({message_id}): invalid message_type {row['message_type']!r}"
            )
        if not row["reason"]:
            problems.append(f"row {index} ({message_id}): empty reason")

        try:
            confidence_value = float(row["confidence"])
        except (TypeError, ValueError):
            problems.append(f"row {index} ({message_id}): unparseable confidence")
        else:
            if not 0.0 <= confidence_value <= 1.0:
                problems.append(f"row {index} ({message_id}): confidence out of range")

        if not row["evidence_message_ids"]:
            problems.append(f"row {index} ({message_id}): empty evidence field (should be 'none')")

    if problems:
        preview = "; ".join(problems[:10])
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise OutputValidationError(f"Output validation failed: {preview}{more}")


def write_output_csv(
    decisions: Iterable[Decision],
    output_path: Path | str,
    validate: bool = True,
    skip_invalid: bool = False,
) -> int:
    """Write decisions to ``output.csv`` in the exact required schema.

    Row order is preserved exactly as ``decisions`` is iterated: no sorting,
    grouping, or reordering happens anywhere in this function, so output order
    always matches input message order.

    Parameters
    ----------
    decisions:
        Decisions to write, in the order they should appear in the file.
    output_path:
        Destination path. Parent directories are created if needed.
    validate:
        When ``True`` (default), run :func:`validate_rows` before writing and
        raise on any problem.
    skip_invalid:
        When ``True``, a decision that fails row-level conversion
        (:func:`decision_to_row`) is logged and skipped rather than aborting
        the whole write. Has no effect on the batch-level
        :func:`validate_rows` check, which still runs (and can still raise)
        over whatever rows were successfully converted.

    Returns
    -------
    int
        Number of rows written.

    Raises
    ------
    OutputValidationError
        If validation fails, or if a row fails conversion and
        ``skip_invalid`` is ``False``.
    OSError
        If the file cannot be written.
    """
    output_path = Path(output_path)
    rows: list[dict[str, str]] = []
    skipped = 0

    for decision in decisions:
        try:
            rows.append(decision_to_row(decision))
        except OutputValidationError as error:
            if skip_invalid:
                skipped += 1
                logger.error(
                    "write_output_csv: skipping invalid decision (%s): %s",
                    getattr(decision, "message_id", "<unknown>"),
                    error,
                )
                continue
            raise

    if validate:
        validate_rows(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(OUTPUT_COLUMNS),
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "write_output_csv: wrote %d row(s) to %s%s.",
        len(rows),
        output_path,
        f" ({skipped} skipped)" if skipped else "",
    )
    return len(rows)


def read_output_csv_for_check(path: Path | str) -> list[dict[str, Any]]:
    """Read back a written output CSV, for a post-write sanity check.

    Not used for evaluation -- purely a structural round-trip check that the
    file written matches the expected column schema.

    Parameters
    ----------
    path:
        Path to a previously written output CSV.

    Returns
    -------
    list[dict]
        One dict per row, in file order.

    Raises
    ------
    OutputValidationError
        If the file's header does not exactly match :data:`OUTPUT_COLUMNS`.
    """
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != OUTPUT_COLUMNS:
            raise OutputValidationError(
                f"{csv_path}: header {header} does not match expected {OUTPUT_COLUMNS}"
            )
        return list(reader)


__all__ = [
    "CONFIDENCE_DECIMALS",
    "NO_EVIDENCE_LITERAL",
    "OUTPUT_COLUMNS",
    "OutputValidationError",
    "decision_to_row",
    "format_confidence",
    "format_evidence",
    "read_output_csv_for_check",
    "validate_rows",
    "write_output_csv",
]