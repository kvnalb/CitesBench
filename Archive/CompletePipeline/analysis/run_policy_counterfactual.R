#!/usr/bin/env Rscript
# ==============================================================================
# Fuzzy RDD policy counterfactual
#   "What if the LLM committee had been the tiebreaker for borderline papers?"
#
# Design
#   1. Work on the RDD 2018-2020 sample.  Define a "borderline band" as papers
#      whose human mean_rating is within ±delta of the year-specific cutoff.
#   2. Baseline policy (P0)     = actual conference decision (fuzzy RDD, observed).
#      Rule policy  (P_human)   = accept iff mean_rating >= cutoff
#                                 (sharp human rule).
#      Counterfactual (P_llm)   = accept iff (mean_rating >= cutoff AND outside band)
#                                 OR  (inside band AND llm_rating >= llm_cutoff).
#      llm_cutoff is calibrated so that the within-band acceptance rate under
#      P_llm matches the within-band acceptance rate under P_human (volume-matched
#      tiebreaker -- isolates who is picked, not how many).
#   3. Flips: for each policy pair, count how many papers move accept <-> reject.
#   4. Citation impact: fit a model on observed citations,
#        log(1+cites) ~ accepted + mean_rating | year_topic
#      and use it to predict log(1+cites) under each counterfactual by toggling
#      the "accepted" indicator while holding mean_rating and year_topic fixed.
#   5. Sensitivity: loop over delta in {0.25, 0.5, 0.75}.
#
# Outputs (in Playground/fuzzy_rdd_llm_tiebreaker/):
#   summary.csv              flip counts + citation deltas per band width
#   flips_delta_0.50.csv     paper-level flip records at the main band width
#   fig_flips_by_band.png    # flips and citation change vs delta
#   fig_band_scatter.png     LLM vs human rating inside band, marking flips
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
  library(ggplot2)
  library(grid)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
this_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg)) else getwd()
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
cat("Root:", root, "\n")

OUT_DIR  <- file.path(root, "OutputNew/Playground/fuzzy_rdd_llm_tiebreaker")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

DATA_CSV <- file.path(root, "OutputNew/Design/iclr_local_rdd",
                      "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv")
EMB_CSV  <- file.path(root, "OutputNew/Empirics/embeddings",
                      "abstracts_specter2_2018_2023.csv")
LLM_CSV  <- file.path(root, "OutputNew/Empirics",
                      "human_vs_llm_committee_scores_20260421",
                      "human_vs_llm_committee_scores.csv")

YEARS     <- c(2018, 2019, 2020)
N_TOPICS  <- 20
SEED      <- 42
DELTAS    <- c(0.25, 0.5, 0.75)    # band half-widths in rating-units
MAIN_D    <- 0.5                   # primary delta for detailed flip table

# ── 1. load RDD sample, SPECTER2 topics, LLM scores ----------------------
dt <- fread(DATA_CSV,
  select = c("paper_id","year","mean_rating","score_centered","cutoff",
             "bandwidth","accepted","openalex_cited_by_count",
             "in_year_specific_rdd_sample"))
dt <- dt[in_year_specific_rdd_sample == TRUE & year %in% YEARS &
         !is.na(score_centered) & !is.na(mean_rating) & !is.na(accepted)]
dt[, year := as.integer(year)]
dt[, has_cites := !is.na(openalex_cited_by_count)]
dt[, lcites := ifelse(has_cites, log1p(openalex_cited_by_count), NA_real_)]

emb <- fread(EMB_CSV)
emb_cols <- grep("^emb_", names(emb), value = TRUE)
dt <- merge(dt, emb[, c("paper_id", emb_cols), with = FALSE],
            by = "paper_id", all.x = TRUE)
emb_mat <- as.matrix(dt[, ..emb_cols])
emb_mat <- emb_mat / sqrt(rowSums(emb_mat^2))
set.seed(SEED)
km <- kmeans(emb_mat, centers = N_TOPICS, nstart = 5, iter.max = 50)
dt[, topic := factor(km$cluster)]
dt[, year_topic := factor(paste0(year, "::", as.integer(topic)))]
dt[, (emb_cols) := NULL]

llm <- fread(LLM_CSV, select = c("paper_id","llm_rating"))
dt <- merge(dt, llm, by = "paper_id", all.x = TRUE)

cat(sprintf("\nRDD 2018-2020: n=%d   w/ cites=%d   w/ LLM=%d\n",
            nrow(dt), sum(dt$has_cites), sum(!is.na(dt$llm_rating))))

# ── 2. citation models (supply-side of counterfactual) -------------------
train <- dt[has_cites == TRUE & !is.na(topic)]
train[, yt_n := .N, by = year_topic]; train <- train[yt_n > 1]
train[, year_topic := droplevels(year_topic)]
train[, cites := openalex_cited_by_count]

# PRIMARY: Poisson on raw cites (canonical for citation counts)
m_pois <- fepois(cites ~ accepted + mean_rating | year_topic, data = train)
# SECONDARY: OLS on levels (LPM-style on raw cites)
m_ols  <- feols(cites  ~ accepted + mean_rating | year_topic, data = train,
                fixef.rm = "none")
# ROBUSTNESS: OLS on log(1 + cites)
m_log  <- feols(lcites ~ accepted + mean_rating | year_topic, data = train,
                fixef.rm = "none")

cat(sprintf("\nCitation models (n = %d):\n", nrow(train)))
cat(sprintf("  Poisson  cites ~ accepted + mean_rating | year_topic\n"))
cat(sprintf("    beta(accepted)    = %+.3f  (se %.3f)   => %+.0f%% cites\n",
            coef(m_pois)["accepted"], se(m_pois)["accepted"],
            100 * (exp(coef(m_pois)["accepted"]) - 1)))
cat(sprintf("    beta(mean_rating) = %+.3f  (se %.3f)\n",
            coef(m_pois)["mean_rating"], se(m_pois)["mean_rating"]))
cat(sprintf("  OLS      cites ~ accepted + mean_rating | year_topic\n"))
cat(sprintf("    beta(accepted)    = %+.2f citations (se %.2f)\n",
            coef(m_ols)["accepted"], se(m_ols)["accepted"]))
cat(sprintf("  OLS   lcites ~ accepted + mean_rating | year_topic\n"))
cat(sprintf("    beta(accepted)    = %+.3f log-cites (se %.3f)\n",
            coef(m_log)["accepted"], se(m_log)["accepted"]))

# per-paper baseline fitted values under the observed acceptance decision
dt_pred <- copy(dt[!is.na(topic) & year_topic %in% unique(train$year_topic)
                   & !is.na(llm_rating)])
dt_pred[, cites_hat_pois  := as.numeric(predict(m_pois, newdata = dt_pred,
                                                type = "response"))]
dt_pred[, cites_hat_ols   := as.numeric(predict(m_ols,  newdata = dt_pred))]
dt_pred[, lcites_hat_base := as.numeric(predict(m_log,  newdata = dt_pred))]

# ── 3. policy machinery --------------------------------------------------
# P_human: accept iff mean_rating >= cutoff   (score_centered >= 0)
dt_pred[, d_human := as.integer(score_centered >= 0)]

# llm_cutoff_fn: within a band, return the llm_rating threshold that matches
# the within-band acceptance RATE of the human rule.
compute_llm_cutoff_in_band <- function(d_band, target_rate) {
  if (nrow(d_band) == 0 || target_rate == 0) return(Inf)
  if (target_rate >= 1) return(-Inf)
  # accept the top target_rate fraction by llm_rating; ties broken by mean_rating
  k <- max(1L, round(target_rate * nrow(d_band)))
  sorted <- d_band[order(-llm_rating, -mean_rating)]
  sorted[k, llm_rating]
}

run_policy <- function(d, delta) {
  d <- copy(d)
  d[, in_band := abs(score_centered) <= delta]

  band_target <- mean(d[in_band == TRUE, d_human])
  cutoff_llm  <- compute_llm_cutoff_in_band(d[in_band == TRUE], band_target)

  # P_llm: human rule outside band; LLM rule inside band (volume-matched)
  d[, d_llm := d_human]
  d[in_band == TRUE,
    d_llm := as.integer(llm_rating > cutoff_llm |
                        (llm_rating == cutoff_llm &
                         rank(-mean_rating, ties.method = "first") <=
                         max(1, round(band_target * .N)) -
                         sum(llm_rating > cutoff_llm)))]

  # enforce exact volume match within band (rank-based fallback in case ties)
  band_n  <- sum(d$in_band)
  band_k  <- round(band_target * band_n)
  if (band_n > 0 && sum(d[in_band == TRUE, d_llm]) != band_k) {
    ord <- order(-d[in_band == TRUE, llm_rating],
                 -d[in_band == TRUE, mean_rating])
    d[in_band == TRUE, d_llm := 0L]
    rows <- which(d$in_band == TRUE)[ord[seq_len(band_k)]]
    d[rows, d_llm := 1L]
  }

  # Citation impact:  under volume-matched flip, beta_acc * (d_llm - d_human)
  # sums to zero by construction (same number accepted).  The real signal is
  # the quality of who gets picked -- i.e. lcites_hat_base on flipped-in
  # vs flipped-out papers.  We evaluate each policy by the TOTAL predicted
  # log(1+cites) over its accepted set, using lcites_hat_base (the paper's
  # "quality" fitted value given observed mean_rating + year_topic FE,
  # conditional on the observed acceptance).
  d[, flip_type := ifelse(d_human == d_llm, "none",
                    ifelse(d_llm == 1, "flip_to_accept", "flip_to_reject"))]

  attr(d, "cutoff_llm")  <- cutoff_llm
  attr(d, "band_target") <- band_target
  attr(d, "band_n")      <- band_n
  attr(d, "band_k")      <- band_k
  d
}

# ── 4. loop over deltas, build summary ----------------------------------
rows <- list()
main_flips <- NULL
for (dl in DELTAS) {
  d <- run_policy(dt_pred, dl)

  band <- d[in_band == TRUE]
  n_band   <- nrow(band)
  n_acc_h  <- sum(band$d_human == 1)
  n_acc_l  <- sum(band$d_llm   == 1)
  n_flip_a <- sum(band$flip_type == "flip_to_accept")
  n_flip_r <- sum(band$flip_type == "flip_to_reject")

  # quality = predicted citations under the baseline (observed-acceptance) model
  metric_summary <- function(yvar) {
    sum_h <- sum(band[d_human == 1, get(yvar)])
    sum_l <- sum(band[d_llm   == 1, get(yvar)])
    q_in  <- if (n_flip_a > 0) mean(band[flip_type == "flip_to_accept", get(yvar)]) else NA
    q_out <- if (n_flip_r > 0) mean(band[flip_type == "flip_to_reject", get(yvar)]) else NA
    per_flip <- if (n_flip_a > 0) (sum_l - sum_h) / n_flip_a else NA
    list(sum_h = sum_h, sum_l = sum_l, delta = sum_l - sum_h,
         q_in = q_in, q_out = q_out, per_flip = per_flip)
  }
  m_p <- metric_summary("cites_hat_pois")
  m_o <- metric_summary("cites_hat_ols")
  m_l <- metric_summary("lcites_hat_base")

  # also report on the subset with OBSERVED cites (no modelling)
  band_obs <- band[has_cites == TRUE]
  obs_h <- sum(band_obs[d_human == 1, openalex_cited_by_count])
  obs_l <- sum(band_obs[d_llm   == 1, openalex_cited_by_count])
  n_band_obs <- nrow(band_obs)
  n_flip_a_obs <- sum(band_obs$flip_type == "flip_to_accept")
  obs_q_in  <- if (n_flip_a_obs > 0)
    mean(band_obs[flip_type == "flip_to_accept", openalex_cited_by_count]) else NA
  obs_q_out <- if (sum(band_obs$flip_type == "flip_to_reject") > 0)
    mean(band_obs[flip_type == "flip_to_reject", openalex_cited_by_count]) else NA

  rows[[as.character(dl)]] <- data.table(
    delta = dl,
    n_band = n_band,
    n_band_w_cites = n_band_obs,
    band_accept_rate = n_acc_h / max(1, n_band),
    n_flip_each_way = n_flip_a,
    flip_share = n_flip_a / max(1, n_band),
    llm_cutoff_in_band = attr(d, "cutoff_llm"),
    # Poisson citations (main):
    pois_cites_human = m_p$sum_h,
    pois_cites_llm   = m_p$sum_l,
    pois_delta_total = m_p$delta,
    pois_delta_per_flip = m_p$per_flip,
    pois_pct_change = 100 * m_p$delta / max(1e-9, m_p$sum_h),
    # OLS on raw cites:
    ols_cites_human = m_o$sum_h,
    ols_cites_llm   = m_o$sum_l,
    ols_delta_total = m_o$delta,
    ols_delta_per_flip = m_o$per_flip,
    # Log-cites (robustness):
    log_delta_total = m_l$delta,
    log_delta_per_flip = m_l$per_flip,
    # Raw observed citations (no model, only on OpenAlex-matched subset):
    obs_cites_human = obs_h,
    obs_cites_llm   = obs_l,
    obs_delta_total = obs_l - obs_h,
    obs_quality_in  = obs_q_in,
    obs_quality_out = obs_q_out
  )

  if (abs(dl - MAIN_D) < 1e-9) main_flips <- d
}
summary_dt <- rbindlist(rows)
print(summary_dt, row.names = FALSE)
fwrite(summary_dt, file.path(OUT_DIR, "summary.csv"))

# paper-level flip records at the main band width
if (!is.null(main_flips)) {
  flips_main <- main_flips[flip_type != "none",
    .(paper_id, year, cutoff, mean_rating, score_centered, llm_rating,
      accepted, d_human, d_llm, flip_type,
      has_cites, openalex_cited_by_count,
      cites_hat_pois, cites_hat_ols, lcites_hat_base)]
  fwrite(flips_main, file.path(OUT_DIR, sprintf("flips_delta_%.2f.csv", MAIN_D)))
  cat(sprintf("\nMain flips file: %d rows (delta=%.2f)\n",
              nrow(flips_main), MAIN_D))
}

# ── 5. plots --------------------------------------------------------------
p_flips <- ggplot(summary_dt, aes(x = factor(delta))) +
  geom_col(aes(y = n_flip_each_way), fill = "#2CA02C", width = 0.4,
           position = position_nudge(x = -0.22)) +
  geom_col(aes(y = n_flip_each_way), fill = "#D65F5F", width = 0.4,
           position = position_nudge(x =  0.22)) +
  geom_text(aes(y = n_flip_each_way,
                label = paste0("+", n_flip_each_way)),
            position = position_nudge(x = -0.22), vjust = -0.3, size = 3.3) +
  geom_text(aes(y = n_flip_each_way,
                label = paste0("-", n_flip_each_way)),
            position = position_nudge(x =  0.22), vjust = -0.3, size = 3.3) +
  labs(title = "Decisions flipped by LLM tiebreaker (volume-matched within band)",
       subtitle = "Green = flipped to ACCEPT   |   Red = flipped to REJECT (same count by construction)",
       x = "Band half-width (rating units around cutoff)",
       y = "# papers flipped each way") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())

p_dcite <- ggplot(summary_dt,
                  aes(x = factor(delta), y = pois_delta_per_flip)) +
  geom_hline(yintercept = 0, colour = "grey60", linewidth = 0.4) +
  geom_col(fill = "#4878CF", width = 0.55) +
  geom_text(aes(label = sprintf("%+.1f cites  (%+.0f%%)", pois_delta_per_flip, pois_pct_change),
                vjust = ifelse(pois_delta_per_flip >= 0, -0.3, 1.2)),
            size = 3.3) +
  labs(title = "Citation gain per flipped accept -- LLM vs human tiebreaker",
       subtitle = "Poisson predicted cites, accepted set inside the band.  Bar = mean cites of LLM-picked new accept minus human-picked new accept.",
       x = "Band half-width (rating units around cutoff)",
       y = "Predicted citations gained per swapped-in accept") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())

gf <- ggplotGrob(p_flips); gc <- ggplotGrob(p_dcite)
save_stack <- function(path, w, h, device_fn) {
  device_fn(path, width = w, height = h)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(
    2, 1, heights = unit.c(unit(1, "null"), unit(1, "null")))))
  pushViewport(viewport(layout.pos.row = 1)); grid.draw(gf); upViewport()
  pushViewport(viewport(layout.pos.row = 2)); grid.draw(gc); upViewport(2)
  dev.off()
}
save_stack(file.path(OUT_DIR, "fig_flips_by_band.pdf"),
           10, 9, function(f, ...) pdf(f, ...))
save_stack(file.path(OUT_DIR, "fig_flips_by_band.png"),
           10, 9, function(f, ...) png(f, ..., units = "in", res = 300))

# Scatter within main band: LLM vs human rating, color by flip type
if (!is.null(main_flips)) {
  dband <- main_flips[in_band == TRUE]
  ft_pal <- c("none" = "grey70",
              "flip_to_accept" = "#2CA02C",
              "flip_to_reject" = "#D65F5F")
  llm_cut <- attr(run_policy(dt_pred, MAIN_D), "cutoff_llm")

  p_scat <- ggplot(dband, aes(x = mean_rating, y = llm_rating,
                              colour = flip_type)) +
    geom_vline(aes(xintercept = cutoff), linetype = "dashed",
               colour = "grey40") +
    geom_hline(yintercept = llm_cut, linetype = "dashed",
               colour = "#4878CF") +
    geom_point(alpha = 0.6, size = 1.6) +
    scale_colour_manual(values = ft_pal, name = "") +
    facet_wrap(~ year, nrow = 1) +
    labs(title = sprintf(
          "Borderline papers at delta = %.2f:  human mean rating vs LLM committee rating",
          MAIN_D),
         subtitle = sprintf(
          "Blue line = volume-matched LLM cutoff within band (= %.2f).  Dashed grey = year-specific human cutoff.",
          llm_cut),
         x = "Human mean rating",
         y = "LLM committee rating") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold"),
          strip.text = element_text(face = "bold"),
          legend.position = "bottom",
          panel.grid.minor = element_blank())
  ggsave(file.path(OUT_DIR, "fig_band_scatter.pdf"), p_scat,
         width = 12, height = 4.5)
  ggsave(file.path(OUT_DIR, "fig_band_scatter.png"), p_scat,
         width = 12, height = 4.5, dpi = 300)
}

cat("\nSaved outputs to:", OUT_DIR, "\n")
