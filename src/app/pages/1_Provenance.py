"""Data provenance page — was Section 6 of the main dashboard.

Streamlit auto-discovers src/pages/*.py when you run `streamlit run src/app/dashboard.py`,
so this shows up as a sidebar nav entry with no wiring.
"""
import os
import sys
import streamlit as st

# src/app/ — where provenance_tab.py lives. Three dirnames would reach src/, which is
# one level too high; Streamlit does not add the parent page directory to sys.path in
# multipage mode, so the import failed the moment anyone clicked this page.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import provenance_tab

st.set_page_config(page_title="CitesBench — Data Provenance", layout="wide")
provenance_tab.render()
