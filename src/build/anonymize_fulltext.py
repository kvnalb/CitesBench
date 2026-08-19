"""
Strip the decision out of a paper's front matter, before it reaches a model.

The extracted text hands the answer over twice (see outputs/audits/, #41):

  1. the running header — "Published as a conference paper at ICLR 2018" against
     "Under review as a conference paper at ICLR 2018". Predicts acceptance 99.5-99.9%
     of the time from one line, and it is printed on EVERY PAGE: median 17 occurrences
     per accepted paper, up to 95. Removing only the first one leaves the label
     repeated down the whole document.
  2. the author block — a camera-ready names its authors and affiliations, a
     double-blind submission says "Anonymous authors". 1,474 of 1,517 accepted papers
     carry real names; 2,923 of 2,980 rejected ones are anonymous. Also hands over
     institution and seniority.

Both live in the same region: line 1 through ABSTRACT. So rather than trying to tell
a wrapped title apart from an author list, this replaces the whole region with a
synthesized blind-submission block and keeps everything from ABSTRACT onward. Every
paper ends up identical in form, and the title comes from eval_table rather than from
parsing, so there is nothing to get wrong per paper.

NO COPIES ARE WRITTEN. This is a pure function the runner applies before the call.
Materializing a second corpus would add ~250MB and a staleness problem — a derived
file that can drift from its source is the failure mode this repo keeps hitting — and
buys nothing, because the runner's trace file already records the verbatim messages
sent for every call. What the model saw is recoverable from the traces.

Refusing is safer than guessing. A paper whose ABSTRACT cannot be located, or whose
extraction is garbage, returns None and is dropped from the run. 4,487 of 4,497
papers locate ABSTRACT on the strict pattern; the loose one, which tolerates the
extractor's letter-spaced headings ("A B S T R AC T"), recovers most of the rest.
Dropping a handful is better than sending a paper whose header we failed to remove.

Self-check: python src/build/anonymize_fulltext.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The heading must be the WHOLE line. Matching a prefix instead let the title of
# "Abstract Diagrammatic Reasoning with Multiplex Graph Networks" pass as the abstract
# heading: the cut landed above the author block, and Duo Wang, Mateja Jamnik, Pietro
# Lio and a cl.cam.ac.uk address survived into the anonymized text. The acceptance
# test caught it; the prefix version had looked fine on 4,496 of 4,497 papers.
# Letter-spacing is tolerated ("A B S T R AC T") because the extractor produces it.
ABSTRACT = re.compile(r"^\s*(?:\d+[.\s]*)?a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*[:.]?\s*$", re.I)
SCAN_LINES = 80

# Mojibake check. One paper's text is a font-encoding failure end to end
# ("lM/2` `2pB2r b +QM72`2M+2" is "Under review as a conference"), which no front
# matter surgery can rescue — the body is unusable too.
COMMON = re.compile(r"\b(the|and|of|we|is|to|in|that|for|with)\b", re.I)

# The running header is printed on EVERY page, so it survives front-matter surgery:
# median 17 occurrences per accepted paper, max 95, and 97.5% of accepted papers carry
# more than one. Cutting only the first one leaves the label repeated throughout the
# body. Every occurrence goes.
RUNNING_HEADER = re.compile(
    r"^.*\b(published|under\s+review|accepted)\s+as\s+an?\s+"
    r"(conference|workshop)\s+(paper|contribution)\s+at\s+iclr\b.*$",
    re.I | re.M)

BLIND = ("Under review as a conference paper at ICLR {year}\n"
         "{title}\n"
         "Anonymous authors\n"
         "Paper under double-blind review\n")


def looks_extractable(text, sample=4000):
    """Cheap sanity check: real English prose hits common words constantly."""
    head = text[:sample]
    return len(COMMON.findall(head)) >= 10


def find_abstract(lines):
    """Index of the ABSTRACT line among non-empty lines, or None."""
    seen = 0
    for i, l in enumerate(lines):
        if not l.strip():
            continue
        seen += 1
        if ABSTRACT.match(l):
            return i
        if seen >= SCAN_LINES:
            break
    return None


def anonymize(text, title, year):
    """Front matter replaced by a blind-submission block, or None if unsafe.

    None means "do not send this paper" — never "send it unchanged". The whole point
    is that a paper we cannot anonymize is a paper that would leak its own answer.
    """
    if not text or not looks_extractable(text):
        return None
    lines = text.split("\n")
    i = find_abstract(lines)
    if i is None:
        return None
    body = "\n".join(lines[i:])
    # strip the per-page header wherever it recurs, then collapse the blank lines that
    # leaves behind so the page breaks do not become obvious gaps
    body = RUNNING_HEADER.sub("", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return BLIND.format(year=int(year), title=str(title).strip()) + body


def demo():
    import glob
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "audit"))
    from audit_pdf_headers import classify, author_block, head_verdict

    et = pd.read_csv("outputs/eval_table.csv", low_memory=False)
    et = et[et.year.isin([2018, 2019, 2020])]
    idx = {}
    for d in ("data/OpenReview/rdd_bandwidth_2018_2020__gemma4_dedicated_stage1/fulltext",
              "data/OpenReview/full_2018_2020_remaining/fulltext"):
        for p in glob.glob(os.path.join(d, "*.txt")):
            pid = os.path.basename(p)[:-4]
            if os.path.getsize(p) > idx.get(pid, ("", -1))[1]:
                idx[pid] = (p, os.path.getsize(p))

    ok = dropped = 0
    leaks = []
    drops = []
    for r in et.itertuples():
        e = idx.get(r.paper_id)
        if not e:
            continue
        with open(e[0], errors="replace") as f:
            raw = f.read()
        out = anonymize(raw, r.title, r.year)
        if out is None:
            dropped += 1
            drops.append(r.paper_id)
            continue
        ok += 1
        # ACCEPTANCE TEST: the audit's own classifiers must now see a blind submission
        lines = [l.strip() for l in out.split("\n") if l.strip()][:40]
        _, header, _ = head_verdict(lines)
        block, _ = author_block(lines)
        if header != "under_review" or block != "anonymous":
            leaks.append((r.paper_id, header, block))

    print(f"anonymized {ok:,} papers, dropped {dropped}")
    if drops:
        print(f"  dropped: {', '.join(drops)}")
    assert not leaks, f"{len(leaks)} papers still leak: {leaks[:5]}"
    assert dropped <= 15, f"too many drops ({dropped}) — the ABSTRACT rule regressed"
    assert ok > 4400, ok
    print("ok — every anonymized paper classifies as under_review + anonymous")


def residual_note():
    """What this does NOT remove, stated so it is not mistaken for solved.

    Author names can recur in the body as self-citations ("... and Debbie Marr.
    Accelerating deep convolutional ..."). Those are not stripped: references
    legitimately carry names, removing them would damage the paper, and an anonymous
    submission cites its own prior work too, so a name in the bibliography is a much
    weaker decision signal than a header that says "Published".
    """


if __name__ == "__main__":
    demo()
