"""
Download ICLR PDFs from OpenReview, keyed by forum id.

This is the input side of the like-for-like question (issue #9). The 2018-2020 reviews
were generated from OpenReview PDFs run through Docling; ReviewArena's OCR text is a
different artifact. Fetching the PDFs lets us extract 2025 text the same way the
archive extracted its own, which removes the confound instead of measuring it.

Layout mirrors the archive's deliberately:
    archive:  .../rawdata/Design/OpenReview/<group>/pdf/{forum_id}.pdf
    here:     data/pdf_{year}/{forum_id}.pdf

Anonymous access is blocked — openreview.net returns ChallengeRequiredError 403 for
both the PDF endpoint and api2. Credentials are required and live in .env as
OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD.

Resumable by design: a paper whose PDF already exists on disk is skipped, so a killed
run restarts for free. Failures are appended to a log rather than raised, because one
withdrawn or embargoed paper must not end a 3,703-paper download.

OpenReview throttles the PDF endpoint to ~26 requests per rolling hour per account.
That is the binding constraint, not bandwidth: a request takes 0.7s, but the full 3,703
papers is ~142 hours. openreview-py makes this worse on its own — it auto-retries 429s,
and each retry consumes quota — so this script paces requests itself and treats a 429
as "stop and wait for the stated reset", never as something to retry through.

Consequence: this is a long, resumable trickle, not a download. Run it under tmux/nohup
and expect days for a full year. Fetch only the papers you actually need (the 2020
overlap set is ~2,180 and is what the like-for-like check requires). Ask OpenReview for
elevated research access before attempting a whole corpus.

~7.5MB per paper, so 2025 would be ~28GB. `data/` is gitignored, so none of this lands
in the repo.

Outputs:
  data/pdf_{year}/{forum_id}.pdf
  outputs/openreview_pdf_fetch_{year}.log   one line per paper: ok/skip/fail

Run: python src/fetch/fetch_openreview_pdfs.py --year 2020 --limit 300   # control set
     python src/fetch/fetch_openreview_pdfs.py --year 2025               # full, slow
"""
import os
import sys
import time
import argparse
import threading

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build.build_slim_2025_papers import load_year

load_dotenv()

BASEURL = "https://api2.openreview.net"
PDF_DIR = "data/pdf_{year}"
LOG = "outputs/openreview_pdf_fetch_{year}.log"
SAMPLE_2025 = "outputs/samples/slim_2025_papers.csv"

# observed server limit: 26 per rolling hour. 25 leaves headroom for the odd retry.
DEFAULT_PER_HOUR = 25
COOLDOWN_SECONDS = 65 * 60   # past the hourly reset, with slack for clock skew

_lock = threading.Lock()


class RateLimiter:
    """One request per interval, serialised. Deliberately not a token bucket — a burst
    is exactly what trips OpenReview's limit, and there is nothing to gain from one
    when the sustainable rate is 25/hour."""

    def __init__(self, per_hour):
        self.interval = 3600.0 / max(per_hour, 1)
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            gap = self.interval - (time.time() - self.last)
            if gap > 0:
                time.sleep(gap)
            self.last = time.time()


def client():
    import openreview
    u = os.environ.get("OPENREVIEW_USERNAME")
    p = os.environ.get("OPENREVIEW_PASSWORD")
    if not (u and p):
        sys.exit("ERROR: OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD not set in .env")
    return openreview.api.OpenReviewClient(baseurl=BASEURL, username=u, password=p)


def paper_ids(year):
    """The frozen 2025 population if we have one, else every paper for that year."""
    if year == 2025 and os.path.exists(SAMPLE_2025):
        d = pd.read_csv(SAMPLE_2025).sort_values("run_order")
        return d.paper_id.astype(str).tolist()
    return load_year(year).forum_id.astype(str).tolist()


def fetch_one(c, pid, out_dir, flog, limiter):
    path = os.path.join(out_dir, f"{pid}.pdf")
    # a non-trivial file on disk means a completed download; 0-byte files are retried
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return "skip"
    limiter.wait()
    try:
        blob = c.get_pdf(pid)
        if not blob.startswith(b"%PDF"):
            raise ValueError(f"not a PDF (starts {blob[:8]!r})")
        tmp = path + ".part"          # write-then-rename so a kill can't leave a
        with open(tmp, "wb") as f:    # truncated file that a rerun would treat as done
            f.write(blob)
        os.replace(tmp, path)
        status = f"ok {len(blob)}"
    except Exception as e:
        status = f"fail {type(e).__name__}: {str(e)[:120]}"
        # A 429 means the quota for this hour is gone. Don't retry into it and don't
        # give up: this is meant to run for days, so sleep past the reset and carry on.
        # The paper is left undone and gets picked up on the next pass.
        if "RateLimit" in type(e).__name__ or "429" in str(e) or "Too many requests" in str(e):
            with _lock:
                flog.write(f"{pid}\trate_limited\n")
                flog.flush()
            print(f"  rate limited on {pid}; sleeping {COOLDOWN_SECONDS/60:.0f}min",
                  flush=True)
            time.sleep(COOLDOWN_SECONDS)
            return "fail"
    with _lock:
        flog.write(f"{pid}\t{status}\n")
        flog.flush()
    return status.split()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--per-hour", type=int, default=DEFAULT_PER_HOUR,
                    help=f"requests/hour (server limit is 26; default {DEFAULT_PER_HOUR})")
    ap.add_argument("--forever", action="store_true",
                    help="keep passing over the list until every PDF is fetched")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N papers (0 = all); use to fetch a control subset")
    args = ap.parse_args()
    # serial by construction: at 25/hour, concurrency buys nothing and risks bursts
    args.workers = 1

    out_dir = PDF_DIR.format(year=args.year)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    ids = paper_ids(args.year)
    if args.limit:
        ids = ids[:args.limit]
    args.passes = 200 if args.forever else 1
    c = client()
    limiter = RateLimiter(args.per_hour)
    todo = [p for p in ids
            if not (os.path.exists(os.path.join(out_dir, f"{p}.pdf"))
                    and os.path.getsize(os.path.join(out_dir, f"{p}.pdf")) > 1024)]
    print(f"{args.year}: {len(ids)} papers, {len(todo)} still to fetch -> {out_dir}\n"
          f"rate {args.per_hour}/hour -> ~{len(todo) / max(args.per_hour, 1):.1f}h remaining",
          flush=True)

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with open(LOG.format(year=args.year), "a") as flog:
        for pass_no in range(1, args.passes + 1):
            pending = [p for p in ids
                       if not (os.path.exists(os.path.join(out_dir, f"{p}.pdf"))
                               and os.path.getsize(os.path.join(out_dir, f"{p}.pdf")) > 1024)]
            if not pending:
                print(f"all {len(ids)} PDFs present — nothing left", flush=True)
                break
            if pass_no > 1:
                print(f"\n--- pass {pass_no}: {len(pending)} still missing ---", flush=True)
            for i, pid in enumerate(pending, 1):
                counts[fetch_one(c, pid, out_dir, flog, limiter)] += 1
                if i % 10 == 0 or i == len(pending):
                    el = time.time() - t0
                    gb = sum(os.path.getsize(os.path.join(out_dir, x))
                             for x in os.listdir(out_dir) if x.endswith(".pdf")) / 1e9
                    print(f"[{i}/{len(pending)}] ok={counts['ok']} skip={counts['skip']} "
                          f"fail={counts['fail']}  {gb:.1f}GB  {el/3600:.1f}h "
                          f"eta {(len(pending)-i)/max(args.per_hour,1):.0f}h", flush=True)

    print(f"\ndone: {counts} in {(time.time()-t0)/60:.1f}min", flush=True)
    if counts["fail"]:
        print(f"failures listed in {LOG.format(year=args.year)}", flush=True)


if __name__ == "__main__":
    main()
