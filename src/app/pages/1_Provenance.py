"""Data provenance page — was Section 6 of the main dashboard.

Streamlit auto-discovers src/pages/*.py when you run `streamlit run src/app/dashboard.py`,
so this shows up as a sidebar nav entry with no wiring.
"""
import os
import sys
import streamlit as st

# Both are needed and neither is optional: src/app/ so `import provenance_tab`
# resolves, and src/ so provenance_tab's own `from audit import import_graph` does.
# Streamlit adds neither in multipage mode, so a missing path here fails only when
# somebody clicks the page — not at startup.
_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_APP, os.path.dirname(_APP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import provenance_tab

st.set_page_config(page_title="CitesBench — Data Provenance", layout="wide")
provenance_tab.render()
