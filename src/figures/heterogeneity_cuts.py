"""
Heterogeneity: who each regime picks, cut by author and institution attributes.

THE QUESTION. Human area chairs may favour senior authors and famous labs. Does an
LLM reproduce that preference, reduce it, or increase it? This is the mechanism
question behind the headline numbers, and it is what the cuts below measure.

WHAT IS PLOTTED. Selection PROBABILITY, not membership in one slate. A regime with
coarse scores does not pick one admitted set; it is indifferent across many. The
single call must break ties for 77% of its picks. So each paper gets its probability
of selection over spec.N_SHUFFLE tie orderings, and a subgroup rate is the mean of
those probabilities. A single ordering would draw a line that is mostly row order.

WHY DIFFERENTIAL COVERAGE DOES NOT INVALIDATE THIS. Institution data exists for
39.5% of papers, and accepted papers are almost three times as likely to have it: a
42.9 point gap. That gap is fatal for one kind of claim and harmless for another.

  NOT SUPPORTED: "X% of accepted papers come from top institutions." The labelled
  subsample is not representative of the pool, so any level read off it is wrong.

  SUPPORTED: "the council favours top institutions more (or less) than the area
  chairs do." All three regimes choose from the SAME full pool of 4,567 papers. We
  compute each paper's selection probability first, then read the rates off the same
  labelled papers for every regime. Missing labels change which papers we can look
  at. They do not change how the three regimes treat the papers we can look at.

So the regime CONTRAST inside a subgroup is identified. The subgroup LEVEL is not.
Every table reports the contrast, and every level carries its coverage.

Papers with no value for a cut are not dropped silently. They appear as a "no data"
row in the table, with their own rates, so the reader sees how large that group is
and how the regimes treat it.

QUARTILES ARE WITHIN YEAR. Citations and h-index both grow with time, so a pooled
quartile would sort mostly by year.

Run: python src/figures/heterogeneity_cuts.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs  # noqa: E402

MASTER = "outputs/paper_master.parquet"

OUT_CSV = "outputs/figures/heterogeneity_cuts.csv"
OUT_TEX = "outputs/figures/heterogeneity_cuts.tex"
OUT_PDF = "outputs/figures/heterogeneity_cuts.pdf"
OUT_PNG = "outputs/figures/heterogeneity_cuts.png"

SERIES = spec.HEADLINE
LABEL = {"human_ac": "Area chairs", "llm_council": "Council",
         "llm_single": "Single call"}

# (key, column, kind, panel title, tick labels). kind "q" = within-year quartile,
# "b" = binary flag, "bin" = fixed bins.
CUTS = [
    ("senior", "s2_max_h_index", "q", "Team seniority (max h-index)",
     ["Q1 low", "Q2", "Q3", "Q4 high"]),
    ("first_h", "s2_first_author_h_index", "q", "First-author h-index",
     ["Q1 low", "Q2", "Q3", "Q4 high"]),
    ("team", "n_s2_authors", "bin", "Team size", ["1-2", "3", "4", "5-6", "7+"]),
    ("top_inst", "top_institution_flag", "b", "Top institution on team",
     ["No", "Yes"]),
    ("industry", "industry_flag", "b", "Industry affiliation on team", ["No", "Yes"]),
    ("us", "us_team_flag", "b", "US affiliation on team", ["No", "Yes"]),
]
TEAM_BINS = [0, 2, 3, 4, 6, np.inf]


def selection_probability(et):
    """Each paper's probability of selection under each regime, over tie orderings.

    Computed on the FULL pool before any cut is applied. This is what makes the
    regime contrast inside a subgroup interpretable: the regimes never see the cut.
    """
    for r in SERIES:
        prob = pd.Series(0.0, index=et.paper_id)
        n_ord = 0
        for yr in spec.YEARS:
            p = et[et.year == yr]
            for k, sel in enumerate(spec.select_with_ties(p, r, spec.n_for(et, yr))):
                prob.loc[sel] += 1.0
            n_ord = k + 1
        et[r.key] = (prob / n_ord).to_numpy()
    return et


def apply_cuts(et):
    for key, col, kind, _, ticks in CUTS:
        s = pd.to_numeric(et[col], errors="coerce")
        if kind == "q":
            out = pd.Series(np.nan, index=et.index)
            for yr in spec.YEARS:
                m = et.year == yr
                # duplicates="drop" because h-index ties can collapse a boundary
                out.loc[m] = pd.qcut(s[m], 4, labels=False,
                                     duplicates="drop") + 1
            et[key] = out
        elif kind == "bin":
            et[key] = pd.cut(s, TEAM_BINS, labels=False) + 1
        else:
            et[key] = s + 1        # 0/1 -> level 1/2
    return et


def collect(et):
    rows = []
    for key, col, kind, title, ticks in CUTS:
        cov = et[col].notna().mean()
        acc_cov = et.loc[et.accepted, col].notna().mean()
        rej_cov = et.loc[~et.accepted, col].notna().mean()
        levels = list(range(1, len(ticks) + 1))
        for lv in levels + [np.nan]:
            d = et[et[key].isna()] if pd.isna(lv) else et[et[key] == lv]
            if not len(d):
                continue
            rec = {"cut": key, "cut_label": title, "column": col,
                   "level": "no data" if pd.isna(lv) else ticks[int(lv) - 1],
                   "level_index": np.nan if pd.isna(lv) else int(lv),
                   "n": len(d), "accept_rate": d.accepted.mean(),
                   "coverage": cov, "coverage_differential_pp":
                       100 * (acc_cov - rej_cov)}
            for r in SERIES:
                rec[r.key] = d[r.key].mean()
            rec["council_minus_ac"] = rec["llm_council"] - rec["human_ac"]
            rec["single_minus_ac"] = rec["llm_single"] - rec["human_ac"]
            rows.append(rec)
    return pd.DataFrame(rows)


def render(res):
    os.makedirs("outputs/figures", exist_ok=True)
    fs.apply(nrows=2, ncols=3)
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.4))
    for ax, (key, _, _, title, ticks) in zip(axes.ravel(), CUTS):
        sub = res[(res.cut == key) & res.level_index.notna()].sort_values("level_index")
        x = np.arange(len(sub))
        w = 0.26
        for i, r in enumerate(SERIES):
            ax.bar(x + (i - 1) * w, sub[r.key], width=w, color=r.color,
                   zorder=3, label=LABEL[r.key])
        # the pool-wide accept rate: the line every regime is drawing n from
        ax.axhline(res.accept_rate.iloc[0] * 0 + 1526 / 4567, color=fs.MUTED,
                   ls=(0, (4, 3)), lw=1.0, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(sub.level.tolist(), fontsize="small")
        ax.set_xlabel(title)
        # no bar exceeds 55%; a 0-100 axis would spend half the panel on air
        ax.set_ylim(0, 0.6)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        fs.clean(ax)
    axes[0, 0].set_ylabel("P(selected)")
    axes[1, 0].set_ylabel("P(selected)")
    axes[0, 0].legend(frameon=False, fontsize=5.6, loc="upper left")
    fs.frame(fig, top_in=0.08, bottom_in=0.52, left=0.085, right=0.99,
             wspace=0.32, hspace=0.58)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)


def table(res):
    body, rules = [], []
    for key, _, _, title, _ in CUTS:
        sub = res[res.cut == key]
        cov, diff = sub.coverage.iloc[0], sub.coverage_differential_pp.iloc[0]
        rules.append(len(body))
        body.append([f"{title}  ({100 * cov:.1f}% covered, "
                     f"{diff:+.1f} pp gap)".replace("-", "−")] + [""] * 6)
        for _, w in sub.iterrows():
            body.append([f"   {w.level}", f"{w.n:,}",
                         f"{100 * w.human_ac:.1f}", f"{100 * w.llm_council:.1f}",
                         f"{100 * w.llm_single:.1f}",
                         f"{100 * w.council_minus_ac:+.1f}".replace("-", "−"),
                         f"{100 * w.single_minus_ac:+.1f}".replace("-", "−")])
    fig = fs.table(
        header=[[("", 0, 0), ("N", 1, 1), ("P(selected), %", 2, 4),
                 ("Difference vs AC, pp", 5, 6)],
                ["", "", "AC", "Council", "Single", "Council", "Single"]],
        body=body, align="lrrrrrr",
        colw=[3.00, 0.70, 0.66, 0.86, 0.76, 0.90, 0.80],
        rules=tuple(rules[1:]),
        note="Mean selection probability over tie orderings.")
    fig.savefig(OUT_PDF.replace(".pdf", "_table.pdf"))
    fig.savefig(OUT_PNG.replace(".png", "_table.png"), dpi=220)
    plt.close(fig)

    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& N & AC & Council & Single & Council $-$ AC & Single $-$ AC \\",
         r"\midrule"]
    for i, (key, _, _, title, _) in enumerate(CUTS):
        sub = res[res.cut == key]
        if i:
            L.append(r"\addlinespace")
        L.append(r"\multicolumn{7}{l}{\textit{%s}\quad(%.1f\%% covered, %s pp gap)} \\"
                 % (title, 100 * sub.coverage.iloc[0],
                    f"{sub.coverage_differential_pp.iloc[0]:+.1f}".replace("-", "$-$")))
        for _, w in sub.iterrows():
            L.append(r"\quad %s & %s & %.1f & %.1f & %.1f & %s & %s \\" % (
                w.level, f"{w.n:,}", 100 * w.human_ac, 100 * w.llm_council,
                100 * w.llm_single,
                f"{100 * w.council_minus_ac:+.1f}".replace("-", "$-$"),
                f"{100 * w.single_minus_ac:+.1f}".replace("-", "$-$")))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(OUT_TEX, "w").write("\n".join(L) + "\n")


def build():
    d = pd.read_parquet(MASTER)
    d = d[d.year.isin(spec.YEARS)].copy()
    assert d.paper_id.is_unique, "paper_master is not one row per paper"
    # spec owns the pool definition; paper_master must agree with it
    et = spec.read_eval_table()
    assert set(d.paper_id) == set(et.paper_id), "paper_master and eval_table disagree"
    d["accepted"] = d.decision.str.lower().str.contains("accept")

    d = collect(apply_cuts(selection_probability(d)))
    d.to_csv(OUT_CSV, index=False)
    render(d)
    table(d)

    show = d[["cut", "level", "n", "accept_rate", "human_ac", "llm_council",
              "llm_single", "council_minus_ac"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {OUT_PDF} / {OUT_PNG} / {OUT_TEX} / {OUT_CSV}")
    return d


def demo():
    res = build()

    # Every regime draws the same total n, so every cut's n-weighted mean selection
    # probability must equal the pool accept rate. This is the arithmetic check that
    # the probabilities were built on the full pool and not on a subgroup.
    base = 1526 / 4567
    for key, _, _, _, _ in CUTS:
        sub = res[res.cut == key]
        for r in SERIES:
            got = float((sub[r.key] * sub.n).sum() / sub.n.sum())
            assert abs(got - base) < 1e-6, \
                f"{key}/{r.key}: weighted rate {got:.6f}, pool rate {base:.6f}"

    # The AC regime is the historical decision, so its selection probability must
    # equal the accept rate exactly. If it does not, select_with_ties is not
    # reproducing the decisions it is given.
    for _, w in res.iterrows():
        assert abs(w.human_ac - w.accept_rate) < 1e-12, \
            f"{w.cut}/{w.level}: AC rate {w.human_ac} != accept rate {w.accept_rate}"

    # A cut with no signal would leave every regime flat. The seniority cut must
    # separate, or the exhibit says nothing.
    sen = res[(res.cut == "senior") & res.level_index.notna()].sort_values("level_index")
    assert sen.human_ac.iloc[-1] > sen.human_ac.iloc[0], \
        "area chairs should select senior teams more often"

    # The "no data" group must be reported, not dropped.
    for key, col, _, _, _ in CUTS:
        if res[res.column == col].coverage.iloc[0] < 0.999:
            assert (res[(res.cut == key)].level == "no data").any(), \
                f"{key}: missing group not reported"

    sen_gap = sen.council_minus_ac.iloc[-1] - sen.council_minus_ac.iloc[0]
    print(f"\nok — {len(res)} rows over {len(CUTS)} cuts; "
          f"area chairs pick top-seniority teams {sen.human_ac.iloc[-1]:.1%} "
          f"vs bottom {sen.human_ac.iloc[0]:.1%}; "
          f"council−AC spread across seniority {sen_gap:+.1%}")


if __name__ == "__main__":
    demo()
