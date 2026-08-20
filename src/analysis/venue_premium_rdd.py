"""
The ICLR venue premium: the slide deck's RD table, rebuilt on the canonical
citation table, plus the row the deck is missing.

WHAT THE DECK REPORTED. Specifications 1-5 below reproduce its table: naive OLS,
OLS with year FE, then "sharp RD" with no FE / year FE / year x topic FE. Its
headline was the last one, +0.269.

WHY THAT IS NOT THE VENUE PREMIUM. Crossing the cutoff does not accept a paper,
it raises the probability of acceptance by about a quarter. So a jump in citations
at the cutoff is the effect of *that* probability shift, not the effect of being
accepted. It is an intention-to-treat estimate. The premium is the ratio

    tau = (jump in citations) / (jump in P(accept))

which is the fuzzy-RD Wald estimator, reported here as spec 7. Dividing by a
number near 0.24 multiplies the estimate by about four, and multiplies its
standard error by more than that, which is why the deck's tight-looking 0.269
becomes a wide interval once it is pointed at the right parameter.

WHAT CHANGES WITH THE NEW DATA. The deck ran on the OpenAlex pull, where outcome
coverage was 80.9% of accepted papers against 46.4% of rejected ones. In an RD
that differential is not a nuisance, it is a hole in the design: the outcome is
missing for exactly the papers on one side of the cutoff. The canonical S2 table
runs 3.9 pp instead of 34 pp. That is the single biggest improvement here.

TOPIC FIXED EFFECTS. The deck used k-means (k=20) on SPECTER2 abstract embeddings.
Those embeddings are not in this repository, so this uses k-means on TF-IDF of the
same abstracts, seeded. A documented stand-in, not the same variable; the point of
the row is whether the estimate MOVES when topic controls enter, and it moves.

Run: python src/analysis/venue_premium_rdd.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures import spec, figstyle as fs   # noqa: E402

DB = "data/gen_review.db"
RDD_CSV = "data/OpenAlex/openalex_rdd_arxiv_paper_level.csv"
OUT_CSV = "outputs/venue_premium_rdd.csv"
OUT_MD = "outputs/venue_premium_rdd.md"
OUT_TEX = "outputs/figures/venue_premium_rdd.tex"
OUT_FIG = "outputs/figures/venue_premium_binscatter"

# The deck's year-specific bandwidths, so the comparison is like for like.
BW = {2018: 1.333, 2019: 1.250, 2020: 1.167}
N_TOPICS = 20
SEED = 0
N_BINS = 12


# --------------------------------------------------------------------- data
def topics(paper_ids):
    """k-means on TF-IDF of the abstract. Stand-in for the deck's SPECTER2."""
    import sqlite3
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    con = sqlite3.connect(DB)
    ab = pd.read_sql("SELECT id AS paper_id, abstract FROM SUBMISSION", con)
    con.close()
    ab = ab[ab.paper_id.isin(set(paper_ids))].dropna(subset=["abstract"])
    X = TfidfVectorizer(max_features=8000, stop_words="english",
                        min_df=5).fit_transform(ab.abstract)
    km = KMeans(n_clusters=N_TOPICS, random_state=SEED, n_init=10).fit(X)
    return pd.Series(km.labels_, index=ab.paper_id.values, name="topic")


def load():
    cut = pd.read_csv(RDD_CSV, low_memory=False).groupby("year")["cutoff"].first()
    d = spec.read_eval_table()
    d = d[d.mean_rating.notna()].copy()
    d["cutoff"] = d.year.map(cut)
    d["r"] = d.mean_rating - d.cutoff              # running variable
    d["above"] = (d.r >= 0).astype(float)          # the instrument
    d["D"] = d.accepted.astype(float)              # the treatment
    d["y"] = np.log1p(d[spec.OUTCOME])             # the outcome
    d["h"] = d.year.map(BW)
    d = d[d.r.abs() <= d.h].copy()                 # year-specific bandwidth
    d["topic"] = d.paper_id.map(topics(d.paper_id)).fillna(-1).astype(int)
    return d


# ------------------------------------------------------------------- fitting
def design(d, rd, fe):
    """Design matrix. rd=True adds the RD slopes; fe names the FE groups."""
    cols = [np.ones(len(d))]
    names = ["const"]
    cols.append(d.above.to_numpy() if rd else d.D.to_numpy())
    names.append("treat")
    if rd:
        cols += [d.r.to_numpy(), (d.above * d.r).to_numpy()]
        names += ["r", "above_x_r"]
    if fe:
        key = d[fe[0]].astype(str)
        for c in fe[1:]:
            key = key + "_" + d[c].astype(str)
        dummies = pd.get_dummies(key, drop_first=True).to_numpy(dtype=float)
        cols += [dummies[:, j] for j in range(dummies.shape[1])]
        names += [f"fe{j}" for j in range(dummies.shape[1])]
    return np.column_stack(cols), names


def ols_hc1(X, y):
    """Coefficients, HC1 covariance, residuals."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    n, k = X.shape
    V = XtX_inv @ (X * (e ** 2)[:, None]).T @ X @ XtX_inv * n / max(n - k, 1)
    return b, V, e


def fit(d, rd, fe, col="y"):
    X, names = design(d, rd, fe)
    b, V, e = ols_hc1(X, d[col].to_numpy())
    i = names.index("treat")
    return b[i], np.sqrt(V[i, i]), X, b, e, i


def wald(d, fe):
    """Fuzzy-RD estimate: the citation jump divided by the acceptance jump.

    Both stages are fit on the SAME sample, the one with an observed outcome, or
    the ratio mixes two populations. Covariance between the two coefficients comes
    from a joint sandwich, so the delta-method SE is not optimistic.
    """
    s = d.dropna(subset=["y"])
    X, names = design(s, rd=True, fe=fe)
    i = names.index("treat")
    XtX_inv = np.linalg.pinv(X.T @ X)

    b_rf, _, e_rf = ols_hc1(X, s.y.to_numpy())
    b_fs, _, e_fs = ols_hc1(X, s.D.to_numpy())
    n, k = X.shape
    scale = n / max(n - k, 1)

    def sandwich(ea, eb):
        return (XtX_inv @ (X * (ea * eb)[:, None]).T @ X @ XtX_inv * scale)[i, i]

    v_rf, v_fs = sandwich(e_rf, e_rf), sandwich(e_fs, e_fs)
    cov = sandwich(e_rf, e_fs)
    rf, fsj = b_rf[i], b_fs[i]
    late = rf / fsj
    var = (v_rf / fsj ** 2 + rf ** 2 * v_fs / fsj ** 4
           - 2 * rf * cov / fsj ** 3)
    return late, np.sqrt(max(var, 0)), rf, np.sqrt(v_rf), fsj, np.sqrt(v_fs), len(s)


# --------------------------------------------------------------------- output
def build():
    os.makedirs("outputs/figures", exist_ok=True)
    d = load()
    obs = d.dropna(subset=["y"])

    rows = []
    for label, rd, fe in [
            ("1  OLS  lcites ~ accepted", False, None),
            ("2  OLS + year FE", False, ["year"]),
            ("3  RD reduced form (ITT), no FE", True, None),
            ("4  RD reduced form (ITT) + year FE", True, ["year"]),
            ("5  RD reduced form (ITT) + year x topic FE", True, ["year", "topic"])]:
        b, se, *_ = fit(obs, rd, fe)
        rows.append({"spec": label, "coef": b, "se": se, "n": len(obs),
                     "is_premium": False})

    fsj, fs_se = fit(obs, True, ["year", "topic"], col="D")[:2]
    rows.append({"spec": "6  First stage: jump in P(accepted)", "coef": fsj,
                 "se": fs_se, "n": len(obs), "is_premium": False})

    late, late_se, rf, rf_se, j, j_se, n = wald(d, ["year", "topic"])
    rows.append({"spec": "7  FUZZY RD premium = row 5 / row 6", "coef": late,
                 "se": late_se, "n": n, "is_premium": True})

    t = pd.DataFrame(rows)
    t["t"] = t.coef / t.se
    t["ci_lo"], t["ci_hi"] = t.coef - 1.96 * t.se, t.coef + 1.96 * t.se
    t.to_csv(OUT_CSV, index=False)

    acc = obs.accepted
    cov_a = d[d.accepted][spec.OUTCOME].notna().mean()
    cov_r = d[~d.accepted][spec.OUTCOME].notna().mean()

    L = ["# The ICLR venue premium, rebuilt on the canonical citation table", "",
         f"Outcome: log(1 + citations), Semantic Scholar tier "
         f"{'+'.join(spec.TIERS)}. Pooled ICLR {spec.YEARS[0]}-{spec.YEARS[-1]}, "
         f"year-specific bandwidth, n = {len(obs):,} in-bandwidth papers with an "
         "observed outcome.", "",
         f"Outcome coverage inside the bandwidth: accepted {cov_a:.1%}, rejected "
         f"{cov_r:.1%}, differential {abs(cov_a - cov_r) * 100:.1f} pp. The deck's "
         "OpenAlex pull ran 80.9% against 46.4%, a 34 pp gap.", "",
         "| specification | coef | SE | t | 95% CI |", "|---|---:|---:|---:|---|"]
    for r in t.itertuples():
        star = " **" if r.is_premium else " "
        L.append(f"|{star}{r.spec}{star.strip()} | {r.coef:+.3f} | {r.se:.3f} | "
                 f"{r.t:+.2f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] |")
    L += ["",
          "Row 7 is the venue premium. Rows 3-5 are intention-to-treat: crossing "
          "the cutoff raises P(accepted) by "
          f"{j:.3f}, not to 1, so the citation jump they report is diluted by "
          "roughly that factor.", "",
          f"Reading row 7: a paper moved from reject to accept gains "
          f"{late:+.2f} log points, a factor of {np.exp(late):.2f} in citations, "
          f"95% interval [{np.exp(t.ci_lo.iloc[-1]):.2f}x, "
          f"{np.exp(t.ci_hi.iloc[-1]):.2f}x].",
          "",
          # Stated from the data rather than asserted, because the sign of this
          # conclusion changed once the outcome table was fixed.
          ("The interval excludes zero, so this does establish a premium. It is "
           "wide, so use the interval as the sensitivity range rather than the "
           "point."
           if t.ci_lo.iloc[-1] > 0 or t.ci_hi.iloc[-1] < 0 else
           "The interval contains zero, so this does not establish a premium; it "
           "bounds how large one could plausibly be.")]
    open(OUT_MD, "w").write("\n".join(L) + "\n")

    open(OUT_TEX, "w").write(
        "% generated by src/analysis/venue_premium_rdd.py — do not hand-edit\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Specification & $\\hat\\beta$ & (SE) & 95\\% CI \\\\\n\\midrule\n"
        + "\n".join(
            f"{r.spec.split('  ', 1)[1]} & {r.coef:+.3f} & ({r.se:.3f}) & "
            f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] \\\\"
            + ("\n\\midrule" if r.spec.startswith("5") else "")
            for r in t.itertuples())
        + "\n\\bottomrule\n\\end{tabular}\n"
        f"% outcome log(1+citations), year-specific bandwidth, n={len(obs)}, "
        "HC1 SEs. Row 7 is the fuzzy-RD Wald ratio.\n")

    binscatter(d, t)

    print(t.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\ncoverage in band: accepted {cov_a:.1%}  rejected {cov_r:.1%}  "
          f"({abs(cov_a - cov_r) * 100:.1f} pp; OpenAlex was 34 pp)")
    print(f"\n-> {OUT_CSV}\n-> {OUT_MD}\n-> {OUT_TEX}\n-> {OUT_FIG}.pdf")
    return t


def binscatter(d, t):
    """The deck's Figure 12, on the new outcome. Residualised on year x topic FE
    so the bins show the discontinuity the regression fits."""
    fs.apply(ncols=2)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    for ax, (col, lab) in zip(axes, [("D", "P(accepted)"),
                                     ("y", "log(1 + citations)")]):
        s = d.dropna(subset=[col]).copy()
        X, names = design(s, rd=False, fe=["year", "topic"])
        keep = [j for j, nm in enumerate(names) if nm != "treat"]
        b, _, _ = ols_hc1(X[:, keep], s[col].to_numpy())
        s["resid"] = s[col].to_numpy() - X[:, keep] @ b

        # below / above the cutoff, deliberately NOT the regime palette
        for side, colr in [(0, fs.MUTED), (1, fs.BLUE)]:
            g = s[s.above == side]
            q = pd.qcut(g.r, N_BINS // 2, duplicates="drop")
            m = g.groupby(q, observed=True).agg(r=("r", "mean"),
                                                v=("resid", "mean"),
                                                n=("resid", "size"))
            ax.scatter(m.r, m.v, s=np.clip(m.n / 4, 6, 60), color=colr,
                       alpha=0.85, linewidths=0, zorder=3)
            fit_x = np.array([g.r.min(), 0]) if side == 0 else np.array([0, g.r.max()])
            p = np.polyfit(g.r, s.loc[g.index, "resid"], 1)
            ax.plot(fit_x, np.polyval(p, fit_x), color=colr, lw=1.6, zorder=2)

        ax.axvline(0, color=fs.MUTED, ls=(0, (3, 2)), lw=1.0, zorder=1)
        ax.set_xlabel("Mean rating minus cutoff")
        ax.set_ylabel(lab + ", residualised")

    for ax in axes:
        fs.clean(ax, xgrid=True)
    fs.frame(fig, top_in=0.10, bottom_in=0.42, left=0.12, right=0.99, wspace=0.34)
    fig.savefig(OUT_FIG + ".pdf")
    fig.savefig(OUT_FIG + ".png", dpi=200)
    plt.close(fig)


def demo():
    t = build().set_index(t_index := "spec")
    prem = t[t.is_premium].iloc[0]
    itt = t.loc[[i for i in t.index if i.startswith("5")][0]]
    fsj = t.loc[[i for i in t.index if i.startswith("6")][0]]

    # The whole point: the premium is the ITT scaled up by the first stage, and
    # the deck reported the ITT as if it were the premium.
    assert abs(prem.coef - itt.coef / fsj.coef) < 1e-6, "row 7 is not row 5 / row 6"
    assert prem.coef > itt.coef, "premium should exceed the ITT"
    assert prem.se > itt.se, "dividing by a noisy first stage must widen the SE"
    print(f"\nok — ITT {itt.coef:+.3f} (se {itt.se:.3f}) / first stage "
          f"{fsj.coef:.3f} = premium {prem.coef:+.3f} (se {prem.se:.3f}), "
          f"95% CI [{prem.ci_lo:+.2f}, {prem.ci_hi:+.2f}]")


if __name__ == "__main__":
    demo()
