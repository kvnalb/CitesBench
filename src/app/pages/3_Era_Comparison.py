"""Era comparison page.

Streamlit auto-discovers src/app/pages/*.py when you run
`streamlit run src/app/dashboard.py`, so this needs no wiring.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import era_comparison_tab

st.set_page_config(page_title="CitesBench — Era Comparison", layout="wide")
era_comparison_tab.render()
