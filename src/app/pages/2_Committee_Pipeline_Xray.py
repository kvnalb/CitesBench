"""Committee Pipeline X-Ray page.

Streamlit auto-discovers src/app/pages/*.py when you run
`streamlit run src/app/dashboard.py`, so this needs no wiring.
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_xray

st.set_page_config(page_title="CitesBench — Committee Pipeline X-Ray", layout="wide")
pipeline_xray.render()
