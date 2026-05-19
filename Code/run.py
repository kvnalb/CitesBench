import os
import time
import math
import random
from pathlib import Path
from evaluate_pdfs import (
    list_pdfs,
    extract_pdf_text,
    load_prompt,
    format_prompt,
    call_model,
    parse_json_output,
    save_evaluation,
)


# Configuration (env overrides)
# Provider: 'openai' or 'dllm'
PROVIDER = os.getenv('LLM_PROVIDER', 'openai').strip().lower()
API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
MODEL = os.getenv('MODEL', 'gpt-5').strip()
# D-LLM base URL (only used when PROVIDER='dllm')
DLLM_BASE_URL = os.getenv('DLLM_BASE_URL', '').strip()
PROMPT_PATH = os.getenv('PROMPT_PATH', '../prompts/evaluation_prompt.txt').strip()
INPUT_DIR = os.getenv('INPUT_DIR', '../rawdata').strip()
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '../processed/evaluation').strip()

# Safety limits and pacing
TPM_LIMIT = int(os.getenv('TPM_LIMIT', '30000'))  # tokens per minute limit
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '5'))
INITIAL_BACKOFF_S = float(os.getenv('INITIAL_BACKOFF_S', '5'))

# Truncation to avoid oversized requests (approx 4 chars per token)
MAX_CONTENT_CHARS = int(os.getenv('MAX_CONTENT_CHARS', '300000'))


def approx_token_count(text: str) -> int:
    return max(1, int(len(text) / 4))


def pace_for_tpm(prompt_text: str) -> None:
    tokens = approx_token_count(prompt_text)
    seconds_needed = (tokens / max(TPM_LIMIT, 1)) * 60.0
    if seconds_needed > 0:
        # minimal sleep to respect TPM; add small jitter
        time.sleep(seconds_needed + 0.25 + random.random() * 0.25)


def safe_call_model(api_key: str, model: str, prompt: str, base_url: str = None) -> str:
    backoff = INITIAL_BACKOFF_S
    attempt = 0
    while True:
        try:
            pace_for_tpm(prompt)
            return call_model(api_key, model, prompt, base_url=base_url)
        except Exception as e:
            msg = str(e)
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            # Exponential backoff on rate limits or large requests
            wait_s = backoff * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"Rate/size error ({attempt}/{MAX_RETRIES}). Sleeping {wait_s:.1f}s. Error: {msg[:200]}")
            time.sleep(wait_s)


def main() -> int:
    if not API_KEY:
        print("Missing OPENAI_API_KEY. Set env var or edit run.py.")
        return 2

    # Determine base_url based on provider
    if PROVIDER == 'dllm':
        if not DLLM_BASE_URL:
            print("Missing DLLM_BASE_URL. Set env var for D-LLM provider.")
            return 2
        base_url = DLLM_BASE_URL
        print(f"Using D-LLM provider at: {base_url}")
    elif PROVIDER == 'openai':
        base_url = None
        print("Using OpenAI provider")
    else:
        print(f"Unknown provider: {PROVIDER}. Use 'openai' or 'dllm'.")
        return 2

    pdfs = list_pdfs(INPUT_DIR)
    print("PDFs found:", len(pdfs))

    template = load_prompt(PROMPT_PATH)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for pdf in pdfs:
        pdf_stem = pdf.stem
        prompt_stem = Path(PROMPT_PATH).stem
        out_file = out_dir / f"{pdf_stem}_{prompt_stem}.json"
        if out_file.exists():
            print(f"Skipping (exists): {out_file}")
            skipped += 1
            continue

        paper = extract_pdf_text(str(pdf))
        content = (paper.get('abstract', '') + "\n\n" + paper.get('text', '')).strip()

        if len(content) > MAX_CONTENT_CHARS:
            print(f"Truncating {pdf.name}: {len(content)} -> {MAX_CONTENT_CHARS} chars")
            try:
                with open("../truncated.txt", "a", encoding="utf-8") as tf:
                    tf.write(f"{pdf.name}\n")
            except Exception as e:
                print(f"Could not record truncation for {pdf.name}: {e}")
            content = content[:MAX_CONTENT_CHARS] + "\n\n[Truncated due to size]"

        prompt = format_prompt(template, paper_title=paper.get('title', ''), paper_content=content)

        try:
            print(f"Sending to API ({MODEL}): {pdf.name}")
            raw = safe_call_model(API_KEY, MODEL, prompt, base_url=base_url)
            data = parse_json_output(raw)
            out = save_evaluation(data, str(pdf), PROMPT_PATH)
            print('Saved:', out)
            processed += 1
        except Exception as e:
            print(f"Failed: {pdf} — {e}")
            # do not abort; continue to next to enable resume
            continue

    print(f"Done. Processed={processed}, Skipped={skipped}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
