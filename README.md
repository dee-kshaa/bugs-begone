# AI-Powered WhatsApp Message Notification Router

For every incoming message, predicts an `action` (`notify` / `digest` / `mute`), a
`message_type`, a human-readable `reason`, a calibrated `confidence`, and supporting
`evidence_message_ids`.

Deterministic and fully explainable: every decision carries a trace showing which rules
fired, what each scoring component contributed, and which overrides applied.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# macOS only -- needed for OCR and voice transcription
brew install tesseract ffmpeg

# 2. Place the ten competition CSVs in data/raw/
#    (see "Dataset layout" below)

# 3. Run
python -m src.main
```

Output is written to `outputs/output.csv`.

Must be run from the project root (the directory containing `src/`), since `-m`
resolves `src.main` against the current working directory.

### Options

| Flag | Purpose |
|---|---|
| `--limit N` | Route only the first N messages (quick smoke test) |
| `--data-dir PATH` | Read CSVs from somewhere other than `data/raw` |
| `--output PATH` | Write the CSV somewhere other than `outputs/output.csv` |
| `--strict` | Fail loudly on a missing/malformed input file instead of degrading |
| `--skip-invalid` | Skip decisions that fail output validation instead of aborting |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` (default `INFO`) |

Example smoke test:

```bash
python -m src.main --limit 20
```

---

## Dataset layout

The ten CSVs go **flat** in `data/raw/`:

```
data/raw/users.csv
data/raw/groups.csv
data/raw/group_members.csv
data/raw/business_accounts.csv
data/raw/user_business_history.csv
data/raw/messages.csv
data/raw/message_history.csv
data/raw/message_events.csv
data/raw/images.csv
data/raw/voice_notes.csv
```

Media files go in:

```
data/media/images/    <- filenames referenced by images.csv media_path
data/media/voice/     <- filenames referenced by voice_notes.csv media_path
```

Bare filenames in those CSVs are resolved against these roots; absolute paths are
used as-is.

### Column name flexibility

`src/io/loaders.py` normalises column names through `config.COLUMN_ALIASES`, so common
variants (`msg_id` → `message_id`, `from_user` → `sender_id`, `sent_at` → `timestamp`, …)
are handled automatically. If the dataset uses a name that isn't covered, add one entry
to that dict -- nothing else needs to change.

---

## Output schema

`outputs/output.csv`:

```
message_id,action,message_type,reason,confidence,evidence_message_ids
```

- `action` -- one of `notify`, `digest`, `mute`
- `message_type` -- closed vocabulary (`otp`, `personal`, `group_chat`, `transactional`,
  `promotional`, `reminder`, `media_share`, `forward`, `spam`, `work`, `other`)
- `reason` -- one deterministic sentence, template-generated (no LLM)
- `confidence` -- fixed 4-decimal value in `[0.05, 0.97]`
- `evidence_message_ids` -- pipe-joined ids, or the literal `none`

Row order always matches input message order.

---

## Architecture

Two stages, so that most messages never touch an expensive code path.

### Stage A -- offline enrichment (run once, resumable)

```
raw CSVs ─┬─> joins, profiles (sender affinity, group, business)
          ├─> images  -> OCR    -> data/cache/ocr.parquet
          └─> voice   -> Whisper -> data/cache/asr.parquet
```

Both media caches are content-hash keyed, so re-running after an interruption never
re-transcribes work already done.

### Stage B -- per-message routing

```
Message
  -> enrichment  (OCR / ASR text folded into unified content, retrieval context built)
  -> RuleEngine      (13 rule families -> triggered rules, weights, overrides)
  -> PriorityEngine  (9 scoring components -> 0-100 priority score)
  -> Arbiter         (thresholds + override resolution -> final Decision)
```

### Module map

| Path | Responsibility |
|---|---|
| `src/config.py` | Every path, threshold, cap and constant |
| `src/schema.py` | Typed domain objects shared across all stages |
| `src/io/loaders.py` | Load + normalise the ten CSVs into a `DataRepository` |
| `src/io/writers.py` | Serialise `Decision` objects to the output schema |
| `src/features/` | Relationship, content, group, business, temporal signals |
| `src/retrieval/` | Three-tier evidence retrieval (pandas only, no embeddings) |
| `src/media/` | OCR, ASR, and the parquet-backed content-hash cache |
| `src/rules/engine.py` | Deterministic rule families -> `RuleEvaluation` |
| `src/scoring/` | Nine scoring components -> aggregated `PriorityAssessment` |
| `src/decision/` | Band resolution, confidence calibration, evidence selection |
| `src/pipeline/` | Enrichment and routing orchestration |
| `src/main.py` | CLI entrypoint: load -> route -> write |

### Priority score

Nine components sum into a 0-100 score. Positive caps total 100, penalties total -30:

| Component | Cap | Component | Cap |
|---|---|---|---|
| Relationship | 22 | Business | 14 |
| Urgency | 25 | History | 8 |
| Trust | 16 | Engagement | 7 |
| Group | 8 | Safety | -24 |
| | | Media | -6 |

Thresholds map the score to a band: `>= 71` notify, `>= 38` digest, else mute.

### Rules vs. scoring

The score handles **gradient** cases. Rules handle **categorical** ones via floor /
ceiling / force constraints, so an additive score can never silently drop an OTP or
deliver a scam. Ordering is fixed and deterministic: mandatory penalties, contextual
adjustments, emergency escalation, scam suppression, then hard constraints. Scam
suppression runs *after* emergency escalation, so a message that looks like an OTP but
is actually credential phishing ends up suppressed rather than escalated.

---

## Graceful degradation

Nothing in the pipeline hard-fails on incomplete input:

- A missing optional CSV yields an empty frame plus a warning.
- A missing OCR/ASR engine, a corrupt image, or a clip over the duration cap yields a
  logged failure result -- the message still routes on its remaining signals.
- A message that raises anywhere in the pipeline gets a low-confidence `digest`
  fallback rather than aborting the batch.
- Evidence is never empty; it falls back to `none`.

---

## Requirements

- Python 3.11+
- `pandas`, `pyarrow` (required)
- `pytesseract` + the `tesseract` binary (optional -- image OCR)
- `openai-whisper` + `ffmpeg` (optional -- voice transcription)
