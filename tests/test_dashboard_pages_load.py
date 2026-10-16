"""Every dashboard page must load without raising.

The Provenance page shipped broken for some time: its sys.path bootstrap had one
dirname too many, pointing at src/ when provenance_tab.py is in src/app/. The main
dashboard was fine, so the app looked healthy — the failure only appeared when
someone clicked that page in the sidebar, which no check exercised.

Streamlit does not add a page's parent directory to sys.path in multipage mode, so
testing a page file in isolation is not the same as testing it in the app. This drives
the real multipage app and switches pages the way a user does.

Run: python tests/test_dashboard_pages_load.py
"""
import io
import json
import os
import sys
import contextlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ["pages/1_Provenance.py", "pages/2_Committee_Pipeline_Xray.py",
         "pages/3_Era_Comparison.py", "pages/4_Venue_Premium_RDD.py", "pages/5_Leakage.py"]


def main():
    os.chdir(REPO)                      # pages read data/ and outputs/ by relative path
    from streamlit.testing.v1 import AppTest

    failures = []
    at = AppTest.from_file(os.path.join(REPO, "src/app/dashboard.py"), default_timeout=120)
    buf = io.StringIO()                 # streamlit is noisy; keep the report readable
    with contextlib.redirect_stderr(buf):
        at.run()
    print(f"{'dashboard.py':38s} exceptions={len(at.exception)}")
    failures += [("dashboard.py", e.value) for e in at.exception]

    if not at.exception:
        # Citation-negative counts must never include papers with unknown outcomes.
        import pandas as pd
        pool = pd.read_csv(os.path.join(REPO, "outputs/eval_table.csv"))
        known = pool.dropna(subset=["s2_citations"])
        accepted = set(pool.loc[pool["decision"].str.startswith("Accept", na=False), "paper_id"])
        ideal = set()
        for _, year_pool in pool.groupby("year"):
            n = int(year_pool["decision"].str.startswith("Accept", na=False).sum())
            ideal.update(year_pool.dropna(subset=["s2_citations"]).nlargest(n, "s2_citations")["paper_id"])
        expected_fp = len((accepted & set(known["paper_id"])) - ideal)
        spec = json.loads(at.get("plotly_chart")[1].proto.spec)
        annotations = [a["text"] for a in spec["layout"]["annotations"]]
        assert any(f"FP<br>{expected_fp:,}" in text for text in annotations), annotations
        assert any(f"{len(known):,} of {len(pool):,}" in c.value for c in at.caption)
        with contextlib.redirect_stderr(buf):
            next(c for c in at.sidebar.checkbox if c.label == "Show drawdown from ideal").check().run()
        assert not at.exception
        assert any("lower = better, 0% = ideal" in m.value for m in at.markdown)
        with contextlib.redirect_stderr(buf):
            next(c for c in at.sidebar.checkbox if c.label == "Impute 0 citations for unmatched papers").check().run()
        assert not at.exception
        assert any("Unmatched citations are imputed as zero" in c.value for c in at.sidebar.caption)
        assert any(f"{len(pool):,} of {len(pool):,}" in c.value for c in at.caption)
        print("dashboard controls and citation population: OK")

    for page in PAGES:
        with contextlib.redirect_stderr(buf):
            at.switch_page(page).run()
        print(f"{page:38s} exceptions={len(at.exception)}")
        failures += [(page, e.value) for e in at.exception]

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for page, err in failures:
            print(f"  {page}: {str(err)[:300]}")
        sys.exit(1)
    print(f"\nOK — dashboard and {len(PAGES)} pages all load")


if __name__ == "__main__":
    main()
