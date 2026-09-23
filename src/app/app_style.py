"""Colours, CSS and repo paths shared by the dashboard and its pages.

Streamlit runs each page as its own script, so a page cannot import these from
dashboard.py without running the whole dashboard.
"""
import os

import streamlit as st

COLORS = {
    "Human (AC decisions)":                  "#2563EB",
    "Human (score top-N)":                   "#D97706",
    "Human (disagreement-adjusted)":         "#0D9488",
    "LLM Committee (Gemma)":                 "#DC2626",
    "LLM Decision Head":                     "#7C3AED",
}
RANDOM_COLOR = "#94A3B8"
IDEAL_COLOR  = "#1E293B"
BG, BORDER, TEXT, SUBTEXT = "#F8FAFC", "#E2E8F0", "#0F172A", "#64748B"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _abs(rel):
    return os.path.join(REPO, rel)


EVAL_TABLE = _abs("outputs/eval_table.csv")
CITATIONS_CSV = _abs("outputs/citations.csv")


def apply_css():
    st.markdown(f"""
<style>
  [data-testid="stAppViewContainer"] {{ background: {BG}; }}
  [data-testid="stSidebar"] {{ background: white; border-right: 1px solid {BORDER}; }}
  .section-header {{ font-size:13px; font-weight:700; letter-spacing:.8px;
                     text-transform:uppercase; color:{SUBTEXT}; margin:0 0 4px 0; }}
  .explainer {{ font-size:13px; color:{SUBTEXT}; margin-bottom:14px; line-height:1.5; }}
</style>
""", unsafe_allow_html=True)
