"""
Probe: can OpenReview give us author affiliation AT SUBMISSION, for rejects too?

WHY. Our institution data comes from OpenAlex. It resolves an institution for 68.0%
of accepted papers but 25.1% of rejected ones (a 43 pp gap), and the flags use each
author's 2026 institution, which acceptance may have changed. OpenReview holds the
submission itself, and ICLR de-anonymizes rejected papers after decisions, so
whatever it has should cover both arms alike and be dated at submission.

WHAT IT MEASURES, per author on each sampled submission:
  email route    authorids that are email addresses -> the domain as submitted
  profile route  authorids that are profile IDs (~Name1) -> the profile history entry
                 whose dates cover the submission year
Email-based profile lookup is refused by the API (it requires profile IDs), so an
author listed only by email cannot be linked to a profile.

SAMPLE. 10 accepted + 10 rejected per year, 2018-2020, seeded. Read-only API calls,
no model calls. One row per author, appended as each paper returns; a restart skips
papers already in the output.

Run: python src/fetch/probe_openreview_affiliations.py
"""
import argparse
import os
import sys

import pandas as pd
from dotenv import dotenv_values

os.makedirs("outputs/audits", exist_ok=True)

EVAL_TABLE = "outputs/eval_table.csv"
OUT_CSV = "outputs/audits/openreview_affiliation_probe.csv"
OUT_MD = "outputs/audits/openreview_affiliation_probe.md"
SEED = 7
PER_CELL = 10   # papers per year x decision

# Free-mail domains say nothing about institution; count them as unknown.
FREE_MAIL = {"gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com",
             "163.com", "126.com", "qq.com", "foxmail.com", "live.com", "icloud.com",
             "mail.com", "protonmail.com", "yandex.ru", "sina.com", "aol.com", "me.com",
             "gmx.de", "naver.com"}

COLS = ["paper_id", "year", "accepted", "position", "authorid", "kind",
        "email_domain", "profile_found", "hist_domain", "hist_institution", "hist_covers_year"]


def clients():
    import openreview
    env = dotenv_values(".env")
    u = env.get("OPENREVIEW_USERNAME") or os.environ.get("OPENREVIEW_USERNAME")
    p = env.get("OPENREVIEW_PASSWORD") or os.environ.get("OPENREVIEW_PASSWORD")
    if not (u and p):
        sys.exit("ERROR: OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD not set in .env")
    # One login, shared: the login endpoint allows only 3 requests per short window.
    v2 = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net",
                                         username=u, password=p)
    v1 = openreview.Client(baseurl="https://api.openreview.net", token=v2.token)
    return v1, v2


def sample():
    et = pd.read_csv(EVAL_TABLE, usecols=["paper_id", "year", "decision"])
    et["accepted"] = et.decision.str.startswith("Accept", na=False)
    return (et.groupby(["year", "accepted"], group_keys=False)
              .apply(lambda g: g.sample(PER_CELL, random_state=SEED))
              .reset_index(drop=True))


def history_at(profile, year):
    """The history entry whose [start, end] covers `year`; open ends count as covering."""
    for h in profile.content.get("history") or []:
        start, end = h.get("start"), h.get("end")
        if (start is None or start <= year) and (end is None or end >= year):
            inst = h.get("institution") or {}
            return inst.get("domain"), inst.get("name"), True
    return None, None, False


def probe_paper(v1, v2, row):
    note = v1.get_note(row.paper_id)
    ids = note.content.get("authorids") or []
    tilde = [a for a in ids if a.startswith("~")]
    profiles = {}
    if tilde:
        found = v2.search_profiles(ids=tilde)
        profiles = {p.id: p for p in found}
        # a profile can be found under a merged id; index its other names too
        for p in found:
            for n in p.content.get("names") or []:
                if n.get("username"):
                    profiles.setdefault(n["username"], p)
    out = []
    for i, a in enumerate(ids):
        rec = dict(paper_id=row.paper_id, year=row.year, accepted=row.accepted,
                   position=i + 1, authorid=a, kind="profile" if a.startswith("~") else
                   "email" if "@" in a else "other", email_domain=None,
                   profile_found=False, hist_domain=None, hist_institution=None,
                   hist_covers_year=False)
        if rec["kind"] == "email":
            rec["email_domain"] = a.rsplit("@", 1)[1].lower()
        elif rec["kind"] == "profile" and a in profiles:
            rec["profile_found"] = True
            d, n, ok = history_at(profiles[a], row.year)
            rec.update(hist_domain=d, hist_institution=n, hist_covers_year=ok)
        out.append(rec)
    return out


def run():
    v1, v2 = clients()
    s = sample()
    done = set(pd.read_csv(OUT_CSV).paper_id) if os.path.exists(OUT_CSV) else set()
    todo = s[~s.paper_id.isin(done)]
    print(f"{len(done)} papers done, {len(todo)} to probe")
    for row in todo.itertuples():
        try:
            rows = probe_paper(v1, v2, row)
        except Exception as e:                       # one bad note must not stop the probe
            print(f"  {row.paper_id}: {type(e).__name__}: {str(e)[:120]}")
            continue
        pd.DataFrame(rows, columns=COLS).to_csv(
            OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)


def report():
    a = pd.read_csv(OUT_CSV)
    a["free_mail"] = a.email_domain.isin(FREE_MAIL)
    a["affil_at_submission"] = (a.email_domain.notna() & ~a.free_mail) | a.hist_covers_year
    p = a.groupby(["paper_id", "year", "accepted"]).agg(
        any_author=("affil_at_submission", "any"),
        first_author=("affil_at_submission", "first"),
        all_authors=("affil_at_submission", "all")).reset_index()

    lines = ["# OpenReview affiliation probe", "",
             f"{p.paper_id.nunique()} submissions ({PER_CELL} accepted + {PER_CELL} rejected per "
             f"year, seed {SEED}), {len(a)} author slots. Generated by "
             "`python src/fetch/probe_openreview_affiliations.py`.", "",
             "Affiliation at submission = an institutional email domain as submitted "
             "(free-mail such as gmail.com counts as unknown), or a profile history entry "
             "covering the submission year. With 30 papers per arm, each share is uncertain "
             "by about +/-8 pp.", "",
             f"Free-mail share of author slots: accepted {a[a.accepted].free_mail.mean():.0%}, "
             f"rejected {a[~a.accepted].free_mail.mean():.0%}.", "",
             "## Author IDs by kind", "",
             "| year | email | profile id | other |", "|---|---:|---:|---:|"]
    for y, g in a.groupby("year"):
        k = g.kind.value_counts(normalize=True)
        lines.append(f"| {y} | {k.get('email', 0):.0%} | {k.get('profile', 0):.0%} | "
                     f"{k.get('other', 0):.0%} |")
    lines += ["", "## Paper-level coverage, by decision", "",
              "| measure | accepted | rejected | gap (pp) |", "|---|---:|---:|---:|"]
    for m in ["any_author", "first_author", "all_authors"]:
        acc, rej = p[p.accepted][m].mean(), p[~p.accepted][m].mean()
        lines.append(f"| {m.replace('_', ' ')} | {acc:.0%} | {rej:.0%} | {100 * (acc - rej):+.0f} |")
    prof = a[a.kind == "profile"]
    if len(prof):
        lines += ["", f"Profile-ID authors: {len(prof)}; profile found {prof.profile_found.mean():.0%}; "
                  f"history entry covering the submission year {prof.hist_covers_year.mean():.0%}."]
    lines += ["", "## Most common email domains", "", "| domain | author slots |", "|---|---:|"]
    for d, n in a.email_domain.value_counts().head(12).items():
        lines.append(f"| {d} | {n} |")
    open(OUT_MD, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    if not ap.parse_args().report_only:
        run()
    report()
