"""
02_download_fulltext.py

Download PDFs from OpenReview and extract full text for selected papers.

Reads paper IDs from abstracts.jsonl, downloads each PDF via curl,
extracts text with PyMuPDF (fitz), and saves as .txt files.

Prerequisites:
    pip install PyMuPDF

Input:  rawdata/ICLR2025/foundation_or_frontier_models_including_LLMs/abstracts.jsonl
Output: rawdata/ICLR2025/foundation_or_frontier_models_including_LLMs/fulltext/{paper_id}.txt
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF not installed. Run: pip install PyMuPDF")

# --- Config ---
BASE_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "rawdata", "ICLR2025", "foundation_or_frontier_models_including_LLMs",
)
ABSTRACTS_FILE = os.path.join(BASE_DIR, "abstracts.jsonl")
FULLTEXT_DIR = os.path.join(BASE_DIR, "fulltext")

DELAY_SECONDS = 1.5          # polite rate limit between requests
RETRY_DELAY_SECONDS = 5.0    # extra delay on 429/403
MAX_RETRIES = 2
CURL_TIMEOUT = 30


def download_pdf(paper_id: str, dest_path: str) -> str:
    """Download a PDF from OpenReview. Returns HTTP status code."""
    url = f"https://openreview.net/pdf?id={paper_id}"
    result = subprocess.run(
        [
            "curl", "-sL",
            "-o", dest_path,
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "-H", f"Referer: https://openreview.net/forum?id={paper_id}",
            "--max-time", str(CURL_TIMEOUT),
            "-w", "%{http_code}",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=CURL_TIMEOUT + 10,
    )
    return result.stdout.strip()


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyMuPDF and clean line numbers."""
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    # Remove review-format line numbers (e.g. 001\n, 042\n)
    text = re.sub(r"^\d{3}\n", "", text, flags=re.MULTILINE)
    return text


def is_valid_pdf(path: str) -> bool:
    """Check that the file exists, is large enough, and starts with %PDF."""
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        return False
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def main():
    # Load paper list
    with open(ABSTRACTS_FILE) as f:
        papers = [json.loads(line) for line in f]

    os.makedirs(FULLTEXT_DIR, exist_ok=True)

    success, failed, skipped = 0, 0, 0
    failed_papers = []

    for i, paper in enumerate(papers):
        pid = paper["paper_id"]
        txt_path = os.path.join(FULLTEXT_DIR, f"{pid}.txt")

        # Skip if already downloaded
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 100:
            skipped += 1
            continue

        # Try downloading with retries
        downloaded = False
        for attempt in range(MAX_RETRIES):
            pdf_tmp = tempfile.mktemp(suffix=".pdf")
            try:
                http_code = download_pdf(pid, pdf_tmp)

                if is_valid_pdf(pdf_tmp):
                    text = extract_text(pdf_tmp)
                    with open(txt_path, "w") as f:
                        f.write(text)
                    success += 1
                    downloaded = True
                    break
                else:
                    if http_code in ("403", "429"):
                        time.sleep(RETRY_DELAY_SECONDS)
            except Exception:
                pass
            finally:
                if os.path.exists(pdf_tmp):
                    os.unlink(pdf_tmp)

        if not downloaded:
            failed += 1
            failed_papers.append((pid, paper["title"]))

        # Progress
        if i % 50 == 0:
            total_done = success + skipped
            print(f"  [{i}/{len(papers)}] downloaded={success} skipped={skipped} failed={failed}")

        time.sleep(DELAY_SECONDS)

    # --- Summary ---
    total_files = len([f for f in os.listdir(FULLTEXT_DIR) if f.endswith(".txt")])
    print(f"\nDone! Success: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total text files: {total_files} / {len(papers)}")

    if failed_papers:
        print(f"\nFailed papers ({len(failed_papers)}):")
        for pid, title in failed_papers:
            print(f"  {pid}: {title[:70]}")


if __name__ == "__main__":
    main()
