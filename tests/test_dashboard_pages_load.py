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
import os
import sys
import contextlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ["pages/1_Provenance.py", "pages/2_Committee_Pipeline_Xray.py"]


def main():
    os.chdir(REPO)                      # pages read data/ and outputs/ by relative path
    from streamlit.testing.v1 import AppTest

    failures = []
    at = AppTest.from_file("src/app/dashboard.py", default_timeout=120)
    buf = io.StringIO()                 # streamlit is noisy; keep the report readable
    with contextlib.redirect_stderr(buf):
        at.run()
    print(f"{'dashboard.py':38s} exceptions={len(at.exception)}")
    failures += [("dashboard.py", e.value) for e in at.exception]

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
