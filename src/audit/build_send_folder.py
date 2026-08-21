"""
Assemble the Dropbox deliverable: outputs/send/{figures,tables,RDD}/ (issue-free copy).

WHY A SCRIPT AND NOT A MANUAL COPY. The exhibits are regenerated often. A folder
copied by hand goes stale silently, and a stale figure in a shared folder is worse
than a missing one. This rebuilds the whole tree from scratch every run, so the
folder is either current or absent.

WHY THE FILES ARE RENUMBERED. Dropbox sorts alphabetically, which scrambles reading
order. Each copy is prefixed 01_, 02_, ... in the order below. The repo keeps its
own names so MANIFEST.md and every script stay consistent; the numbering exists
only in the copy.

Every entry names an exhibit that must already exist. A missing file is an error,
not a warning: a deliverable with a hole in it should not assemble.

Run: python src/audit/build_send_folder.py
"""
import os
import shutil
import sys

SRC = "outputs/figures"
DEST = "outputs/send"

# folder -> ordered list of (stem, what it is). Stems are basenames in outputs/figures.
TREE = {
    "figures": [
        ("fig1_design", "The selection-function framing"),
        ("fig2_headline", "Selection quality by regime"),
        ("fig3_citation_decile", "Selection rate by true citation decile"),
        ("heterogeneity_cuts", "Selection probability by author and institution"),
        ("fig3_score_quintile", "Selection rate by human score quintile"),
        ("fig4_contrast_scatter", "Paired bootstrap draws vs the area chairs"),
    ],
    "tables": [
        ("sample_stats", "Sample characteristics"),
        ("heterogeneity_cuts_table", "Selection probability by author and institution"),
        ("sample_coverage", "Data availability by source"),
        ("table1_sample", "Sample and citation coverage by year and decision"),
    ],
    "RDD": [
        ("venue_premium_binscatter", "Discontinuity at the cutoff"),
        ("rdd_b_first_stage", "First stage by year"),
        ("venue_premium_rdd", "The acceptance premium: full specification table"),
        ("table2_regression", "Regime comparison on log citations"),
        ("fig5_venue_adjusted", "Regime comparison after removing a premium of size tau"),
        ("rdd_a_support", "Support of the running variable"),
        ("rdd_c_stability", "Premium across bandwidths"),
        ("rdd_d_rating_by_decision", "Rating distribution by decision"),
        ("rdd_e_observability", "Outcome observability along the score axis"),
        ("rdd_f_preprint_timing", "Preprint timing relative to the decision"),
        ("venue_premium_balance", "Covariate balance at the cutoff"),
        ("venue_premium_by_year", "Premium by year"),
        ("venue_premium_speccurve", "Premium across 60 specifications"),
    ],
}

# PDF for the paper, PNG for pasting into a message, CSV/TEX where they exist.
WANT = (".pdf", ".png", ".csv", ".tex")


def build():
    if os.path.isdir(DEST):
        shutil.rmtree(DEST)          # rebuild whole, so a renamed exhibit cannot linger

    missing, n = [], 0
    for folder, items in TREE.items():
        out = os.path.join(DEST, folder)
        os.makedirs(out, exist_ok=True)
        for i, (stem, what) in enumerate(items, 1):
            got = [e for e in WANT if os.path.exists(os.path.join(SRC, stem + e))]
            if ".pdf" not in got:
                missing.append(f"{folder}/{stem}.pdf")
                continue
            for e in got:
                shutil.copy2(os.path.join(SRC, stem + e),
                             os.path.join(out, f"{i:02d}_{stem}{e}"))
                n += 1
            print(f"  {folder}/{i:02d}_{stem}  [{' '.join(got)}]  {what}")

    assert not missing, f"exhibits absent, rerun their scripts first: {missing}"
    print(f"\n{n} files in {DEST}/ across {len(TREE)} folders")
    return n


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
