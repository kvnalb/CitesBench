library(data.table)
library(ggplot2)

# ── paths ──────────────────────────────────────────────────────────────────
this_dir <- tryCatch(dirname(sys.frame(1)$ofile),
                     error = function(e) getwd())
find_repo_root <- function(start_dir) {
  current <- normalizePath(start_dir, winslash = "/", mustWork = TRUE)
  repeat {
    if (dir.exists(file.path(current, "Code")) && dir.exists(file.path(current, "Report"))) {
      return(current)
    }
    parent <- dirname(current)
    if (identical(parent, current)) {
      stop(sprintf("Could not locate repo root from %s", start_dir))
    }
    current <- parent
  }
}
root <- find_repo_root(this_dir)
plotdir <- file.path(root, "OutputNew/Results/RDD/Plots")
tabdir  <- file.path(root, "OutputNew/Results/RDD/Tables")
dir.create(plotdir, recursive = TRUE, showWarnings = FALSE)
dir.create(tabdir,  recursive = TRUE, showWarnings = FALSE)

# ── load embeddings ───────────────────────────────────────────────────────
emb <- fread(file.path(root, "OutputNew/Empirics/embeddings",
                       "abstracts_specter2_2018_2023.csv"))
emb_cols <- grep("^emb_", names(emb), value = TRUE)
cat("Embeddings:", nrow(emb), "papers,", length(emb_cols), "dimensions\n")

# ── load RDD sample with citations ───────────────────────────────────────
rdd <- fread(file.path(root, "OutputNew/Design/iclr_local_rdd",
  "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv"),
  select = c("paper_id", "year", "mean_rating", "score_centered", "cutoff",
             "bandwidth", "accepted", "fe_group",
             "has_arxiv_match", "openalex_matched",
             "openalex_cited_by_count"))
setnames(rdd, "openalex_cited_by_count", "cites")

# merge embeddings onto RDD sample
dt <- merge(rdd, emb, by = c("paper_id", "year"))
cat("Merged:", nrow(dt), "RDD papers with embeddings\n")

# papers with known citations (training set for k-NN)
dt_known <- dt[openalex_matched == TRUE & !is.na(cites)]
cat("Known citations:", nrow(dt_known), "\n\n")

# ── year-normalize citations ──────────────────────────────────────────────
# Older papers have more time to accumulate cites. We predict year-normalized
# citations, then map back.
dt_known[, cites_yearmed := median(cites), by = year]
dt_known[, cites_norm := cites / (cites_yearmed + 1)]
dt_known[, lcites := log1p(cites)]
dt_known[, lcites_yearmed := median(lcites), by = year]

cat("── Median cites by year ──\n")
print(dt_known[, .(n = .N, med_cites = median(cites), mean_cites = round(mean(cites), 1)),
               by = year][order(year)])

# ── k-NN prediction (leave-one-out, year-aware) ──────────────────────────
# For each paper, find k nearest neighbors among papers from same or earlier
# years with known citations. Predict log(1+cites) as distance-weighted average.

K <- 20
cat("\n── Running", K, "-NN prediction (leave-one-out) ──\n")

emb_mat <- as.matrix(dt_known[, ..emb_cols])
# L2-normalize for cosine similarity
emb_norm <- emb_mat / sqrt(rowSums(emb_mat^2))

n <- nrow(dt_known)
pred_lcites <- rep(NA_real_, n)
years_vec <- dt_known$year
lcites_vec <- dt_known$lcites

for (i in seq_len(n)) {
  # eligible neighbors: same or earlier year, not self
  eligible <- which(years_vec <= years_vec[i] & seq_len(n) != i)
  if (length(eligible) < K) {
    eligible <- which(seq_len(n) != i)
  }

  # cosine similarity
  sims <- as.numeric(emb_norm[eligible, , drop = FALSE] %*% emb_norm[i, ])
  top_k <- eligible[order(sims, decreasing = TRUE)[1:min(K, length(eligible))]]
  top_sims <- sims[order(sims, decreasing = TRUE)[1:min(K, length(eligible))]]

  # distance-weighted average of log(1+cites)
  wts <- pmax(top_sims, 0)
  if (sum(wts) > 0) {
    pred_lcites[i] <- sum(wts * lcites_vec[top_k]) / sum(wts)
  } else {
    pred_lcites[i] <- mean(lcites_vec[top_k])
  }

  if (i %% 2000 == 0) cat("  ", i, "/", n, "\n")
}

dt_known[, pred_lcites := pred_lcites]
dt_known[, pred_cites := expm1(pred_lcites)]

# ── evaluate prediction quality ──────────────────────────────────────────
cat("\n── Prediction quality (leave-one-out) ──\n")
cat("Correlation (log):  ", round(cor(dt_known$lcites, dt_known$pred_lcites), 3), "\n")
cat("Correlation (level):", round(cor(dt_known$cites, dt_known$pred_cites), 3), "\n")

# by year
cat("\n── Correlation by year ──\n")
print(dt_known[, .(
  n = .N,
  cor_log   = round(cor(lcites, pred_lcites), 3),
  cor_level = round(cor(cites, pred_cites), 3)
), by = year][order(year)])

# ── RDD validation: is predicted citation potential smooth at the cutoff? ─
cat("\n── RDD validation: predicted cites at cutoff ──\n")
library(fixest)

dt_known[, kern_wt := pmax(0, 1 - abs(score_centered) / bandwidth)]
dt_known[, above := as.integer(score_centered >= 0)]

# actual cites
m_actual <- fepois(cites ~ above + score_centered | year,
                   data = dt_known, weights = ~kern_wt)
# predicted cites (should be smooth if surrogate captures intrinsic quality)
m_pred <- feols(pred_lcites ~ above + score_centered | year,
                data = dt_known, weights = ~kern_wt)

cat("\n  Actual cites (Poisson):\n")
print(etable(m_actual, keep = "above"))
cat("\n  Predicted log-cites (OLS):\n")
print(etable(m_pred, keep = "above"))

# ── plot: actual vs predicted ─────────────────────────────────────────────
p1 <- ggplot(dt_known, aes(x = pred_lcites, y = lcites)) +
  geom_point(alpha = 0.1, size = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  facet_wrap(~year) +
  labs(title = "Predicted vs actual log(1 + cites)",
       x = "Predicted (k-NN, SPECTER2)", y = "Actual") +
  theme_minimal(base_size = 12)
ggsave(file.path(plotdir, "fig_knn_pred_vs_actual.png"),
       p1, width = 10, height = 8, dpi = 200)

# ── plot: predicted cites across RDD cutoff ───────────────────────────────
dt_known[, score_bin := round(score_centered * 4) / 4]
binned <- dt_known[, .(mean_pred = mean(pred_lcites),
                       mean_actual = mean(lcites),
                       n = .N), by = score_bin]

p2 <- ggplot(binned, aes(x = score_bin)) +
  geom_point(aes(y = mean_actual, color = "Actual"), size = 2) +
  geom_point(aes(y = mean_pred, color = "Predicted"), size = 2, shape = 17) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Actual vs predicted citations at RDD cutoff (pooled)",
       x = "Centered review score", y = "Mean log(1 + cites)",
       color = NULL) +
  theme_minimal(base_size = 12)
ggsave(file.path(plotdir, "fig_knn_rdd_validation.png"),
       p2, width = 8, height = 5, dpi = 200)

# ── save predictions ─────────────────────────────────────────────────────
fwrite(dt_known[, .(paper_id, year, score_centered, accepted, cites,
                    lcites, pred_lcites, pred_cites)],
       file.path(tabdir, "knn_citation_predictions.csv"))

cat("\nDone. Outputs in", plotdir, "and", tabdir, "\n")
