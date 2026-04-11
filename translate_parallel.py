"""
Parallel Moroccan Darija Translation Pipeline
==============================================
Uses 2 API keys in parallel, each responsible for half the dataset.
Worker 0 → rows [0,       63_500)
Worker 1 → rows [63_500,  127_000)

Each worker has its own:
  - rate limiter  (30 RPM / 1500 RPD)
  - checkpoint    (checkpoint_worker0.json / checkpoint_worker1.json)
  - output file   (darija_worker0.jsonl   / darija_worker1.jsonl)

After both finish, run:  python merge.py
"""

import os
import re
import sys
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from datetime import date

from google import genai
from google.genai import types
from datasets import load_dataset

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_NAME = "gemma-4-31b-it"
BATCH_SIZE = 50
MAX_RPM = 28  # stay 2 below 30 RPM hard limit
MAX_RPD = 1480  # stay 20 below 1500 RPD hard limit

SYSTEM_PROMPT = """You are a professional translator specializing in Moroccan Darija (الدارجة المغربية).
Your task is to translate English sentences into natural, colloquial Moroccan Darija written in Arabic script.
Rules:
- Use authentic Moroccan Darija vocabulary and expressions, NOT Modern Standard Arabic.
- Write ONLY in Arabic script (no romanization).
- Keep proper nouns (names, places) as-is or transliterate them naturally.
- Preserve the tone and register of the original text.
- Return ONLY a JSON array of translated strings, in the same order as input, nothing else.
"""

TRANSLATION_PROMPT_TEMPLATE = """Translate the following {n} English sentences to Moroccan Darija.
Return ONLY a valid JSON array with exactly {n} strings. No explanation, no markdown, no extra text.

English sentences:
{sentences}
"""


# ──────────────────────────────────────────────
# Per-worker logger (avoids interleaved output)
# ──────────────────────────────────────────────
def make_logger(worker_id: int) -> logging.Logger:
    name = f"worker{worker_id}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s [W{worker_id}] [%(levelname)s] %(message)s")
    fh = logging.FileHandler(f"translation_worker{worker_id}.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────
def load_checkpoint(worker_id: int) -> dict:
    path = Path(f"checkpoint_worker{worker_id}.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"last_index": -1, "requests_today": 0, "day": str(date.today())}


def save_checkpoint(worker_id: int, ck: dict):
    with open(f"checkpoint_worker{worker_id}.json", "w") as f:
        json.dump(ck, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# Rate limiter
# ──────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm: int, rpd: int, logger: logging.Logger):
        self.rpm = rpm
        self.rpd = rpd
        self.log = logger
        self._min_calls: list[float] = []
        self._day_calls: list[float] = []

    def _prune(self):
        now = time.monotonic()
        self._min_calls = [t for t in self._min_calls if now - t < 60]
        self._day_calls = [t for t in self._day_calls if now - t < 86400]

    def wait_if_needed(self):
        while True:
            self._prune()
            if len(self._day_calls) >= self.rpd:
                oldest = min(self._day_calls)
                wait = 86400 - (time.monotonic() - oldest) + 5
                self.log.warning(f"Daily limit hit. Sleeping {wait / 3600:.2f}h …")
                time.sleep(wait)
                self._prune()
                continue
            if len(self._min_calls) >= self.rpm:
                oldest = min(self._min_calls)
                wait = 60 - (time.monotonic() - oldest) + 1
                self.log.debug(f"RPM limit. Sleeping {wait:.1f}s …")
                time.sleep(wait)
                self._prune()
                continue
            break

    def record(self):
        now = time.monotonic()
        self._min_calls.append(now)
        self._day_calls.append(now)


# ──────────────────────────────────────────────
# Translation
# ──────────────────────────────────────────────
def translate_batch(
    client: genai.Client,
    model_name: str,
    sentences: list[str],
    rate_limiter: RateLimiter,
    logger: logging.Logger,
    retries: int = 4,
) -> list[str]:
    prompt = TRANSLATION_PROMPT_TEMPLATE.format(
        n=len(sentences),
        sentences="\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences)),
    )

    # Build contents with system instruction
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    # Build generation config
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=4096,
        response_mime_type="application/json",
        system_instruction=SYSTEM_PROMPT,
    )

    for attempt in range(retries):
        rate_limiter.wait_if_needed()
        raw = None
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            )
            rate_limiter.record()

            # Guard: response.text is None when the model has no candidates
            # (e.g. blocked by safety filters or empty finish reason)
            if response.text is None:
                finish = "unknown"
                try:
                    finish = response.candidates[0].finish_reason
                except Exception:
                    pass
                raise ValueError(f"Empty response from model (finish_reason={finish})")

            raw = response.text.strip()

            # Strip markdown code fences: ```json ... ``` or ``` ... ```
            fence_match = re.match(
                r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", raw, re.DOTALL
            )
            if fence_match:
                raw = fence_match.group(1).strip()

            translations: list[str] = json.loads(raw)
            if len(translations) != len(sentences):
                raise ValueError(f"Expected {len(sentences)} got {len(translations)}")
            return translations

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Parse error attempt {attempt + 1}: {e} "
                f"| raw={repr(raw[:200]) if raw else repr(raw)}"
            )
        except Exception as e:
            logger.warning(f"API error attempt {attempt + 1}: {e}")

        backoff = 2**attempt * 5
        logger.info(f"Backoff {backoff}s …")
        time.sleep(backoff)

    # Fallback: one by one
    logger.error("Batch failed. Falling back to single-sentence mode.")
    results = []
    for s in sentences:
        try:
            rate_limiter.wait_if_needed()
            single_prompt = f"Translate this English sentence to Moroccan Darija (Arabic script only, no explanation):\n\n{s}"
            single_contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=single_prompt)],
                ),
            ]
            single_config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4096,
                system_instruction=SYSTEM_PROMPT,
            )
            r = client.models.generate_content(
                model=model_name,
                contents=single_contents,
                config=single_config,
            )
            rate_limiter.record()
            results.append(r.text.strip())
        except Exception as ex:
            logger.error(f"Single translation failed: {ex}")
            results.append("")
    return results


# ──────────────────────────────────────────────
# Worker function (runs in its own thread)
# ──────────────────────────────────────────────
def run_worker(
    worker_id: int,
    api_key: str,
    ds,
    row_start: int,
    row_end: int,
    batch_size: int,
    dry_run: bool,
):
    logger = make_logger(worker_id)
    logger.info(f"Worker {worker_id} starting. Rows {row_start:,} → {row_end:,}")

    client = genai.Client(api_key=api_key)
    rate_limiter = RateLimiter(rpm=MAX_RPM, rpd=MAX_RPD, logger=logger)

    ck = load_checkpoint(worker_id)
    today = str(date.today())
    if ck["day"] != today:
        logger.info("New day — resetting daily counter.")
        ck["requests_today"] = 0
        ck["day"] = today
        save_checkpoint(worker_id, ck)

    # Resume from last saved position (or start of this worker's range)
    start = ck["last_index"] + 1 if ck["last_index"] >= row_start else row_start

    if start >= row_end:
        logger.info("Worker already finished. Nothing to do.")
        return

    logger.info(f"Resuming from row {start:,}")

    output_path = Path(f"darija_worker{worker_id}.jsonl")
    batch_count = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        i = start
        while i < row_end:
            batch_end = min(i + batch_size, row_end)
            batch_rows = ds[i:batch_end]

            en_sentences = [t["en"] for t in batch_rows["translation"]]
            fr_sentences = [t["fr"] for t in batch_rows["translation"]]

            pct = (batch_end - row_start) / (row_end - row_start) * 100
            logger.info(
                f"Batch {batch_count + 1} | rows {i}–{batch_end - 1} | "
                f"progress {pct:.1f}% | req_today={ck['requests_today']}"
            )

            if dry_run:
                darija_translations = [f"[DRY] {s[:25]}" for s in en_sentences]
            else:
                darija_translations = translate_batch(
                    client, MODEL_NAME, en_sentences, rate_limiter, logger
                )

            for j, (en, fr, da) in enumerate(
                zip(en_sentences, fr_sentences, darija_translations)
            ):
                out_f.write(
                    json.dumps(
                        {"id": i + j, "en": en, "fr": fr, "darija": da},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            out_f.flush()

            batch_count += 1
            ck["last_index"] = batch_end - 1
            ck["requests_today"] += 1
            ck["day"] = str(date.today())
            save_checkpoint(worker_id, ck)

            i = batch_end

            if dry_run and batch_count >= 2:
                logger.info("Dry run done.")
                break

    logger.info(f"Worker {worker_id} finished. Output → {output_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Parallel Darija translation with 2 API keys"
    )
    parser.add_argument(
        "--key0",
        default=os.environ.get("GOOGLE_API_KEY_0", ""),
        help="API key for worker 0 (first half)",
    )
    parser.add_argument(
        "--key1",
        default=os.environ.get("GOOGLE_API_KEY_1", ""),
        help="API key for worker 1 (second half)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--dry-run", action="store_true", help="Test with 2 fake batches per worker"
    )
    parser.add_argument(
        "--worker",
        type=int,
        choices=[0, 1],
        default=None,
        help="Run only one worker (useful when running on 2 machines)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total number of examples to process (for testing)",
    )
    args = parser.parse_args()

    if not args.dry_run:
        if not args.key0 or not args.key1:
            raise SystemExit(
                "Provide both --key0 and --key1, or set "
                "GOOGLE_API_KEY_0 and GOOGLE_API_KEY_1 env vars.\n"
                "Get keys at: https://aistudio.google.com/apikey"
            )

    print("Loading dataset …")
    ds = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")
    total = len(ds)

    # Apply limit if specified
    if args.limit:
        total = min(args.limit, total)
        print(f"Limited to first {total:,} examples (for testing)")

    midpoint = total // 2
    print(f"Total rows: {total:,}  |  split at {midpoint:,}")

    splits = [
        (0, (0, midpoint), args.key0 or "dummy"),
        (1, (midpoint, total), args.key1 or "dummy"),
    ]

    if args.worker is not None:
        # Single-worker mode (e.g. two terminals / two machines)
        wid, (r_start, r_end), key = splits[args.worker]
        run_worker(wid, key, ds, r_start, r_end, args.batch_size, args.dry_run)
    else:
        # Parallel mode — both workers in separate threads
        threads = []
        for wid, (r_start, r_end), key in splits:
            t = threading.Thread(
                target=run_worker,
                args=(wid, key, ds, r_start, r_end, args.batch_size, args.dry_run),
                daemon=True,
                name=f"worker-{wid}",
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    print("\nAll workers done. Run `python merge.py` to combine outputs.")


if __name__ == "__main__":
    main()
