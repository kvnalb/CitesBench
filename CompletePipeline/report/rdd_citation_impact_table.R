#!/usr/bin/env Rscript
# Impact of acceptance on citations — pooled sharp RDD.
# Multiple specs for a presentation table.

suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
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

DATA_CSV <- file.path(root, "OutputNew/Design/iclr_local_rdd",
                      "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv")
EMB_CSV  <- file.path(root, "OutputNew/Empirics/embeddings",
                      "abstracts_specter2_2018_2023.csv")

YEARS   <- c(2018, 2019, 2020)
N_TOPIC <- 20
SEED    <- 42

dt <- fread(DATA_CSV,
  select = c("paper_id","year","mean_rating","score_centered","cutoff",
             "bandwidth","accepted","openalex_cited_by_count",
             "in_year_specific_rdd_sample"))
dt <- dt[in_year_specific_rdd_sample == TRUE & year %in% YEARS &
         !is.na(score_centered) & !is.na(mean_rating) & !is.na(accepted)]
dt[, year := as.integer(year)]
dt[, D := as.integer(score_centered >= 0)]
dt[, has_cites := !is.na(openalex_cited_by_count)]
dt[, lcites := ifelse(has_cites, log1p(openalex_cited_by_count), NA_real_)]

# SPECTER2 topic clusters
emb <- fread(EMB_CSV)
emb_cols <- grep("^emb_", names(emb), value = TRUE)
dt <- merge(dt, emb[, c("paper_id", emb_cols), with = FALSE],
            by = "paper_id", all.x = TRUE)
has_emb <- !is.na(dt[[emb_cols[1]]])
emb_mat <- as.matrix(dt[has_emb, ..emb_cols])
emb_mat <- emb_mat / sqrt(rowSums(emb_mat^2))
set.seed(SEED)
km <- kmeans(emb_mat, centers = N_TOPIC, nstart = 5, iter.max = 50)
dt[has_emb, topic := factor(km$cluster)]
dt[, year_topic := factor(paste0(year, "::", as.integer(topic)))]
dt[, (emb_cols) := NULL]

d <- dt[has_cites == TRUE]
cat("Pooled sample (with citations): n =", nrow(d), "\n\n")

report <- function(m, label) {
  tau <- coef(m)["D"]
  se  <- se(m)["D"]
  p   <- pvalue(m)["D"]
  cat(sprintf("%-40s  tau = %+.3f  se = %.3f  p = %.3f  n = %d\n",
              label, tau, se, p, nobs(m)))
}

# Linear OLS — naive (no RDD)
m_ols <- feols(lcites ~ accepted, data = d)
report(m_ols, "OLS  lcites ~ accepted")

# Sharp RDD — no controls
m1 <- feols(lcites ~ D + score_centered + D:score_centered, data = d)
report(m1, "Sharp RDD  (no FE)")

# + Year FE
m2 <- feols(lcites ~ D + score_centered + D:score_centered | year, data = d)
report(m2, "Sharp RDD  + year FE")

# + Year x Topic FE  (preferred)
d2 <- d[!is.na(topic)]
d2[, yt_size := .N, by = year_topic]
d2 <- d2[yt_size > 1]
d2[, year_topic := droplevels(year_topic)]
m3 <- feols(lcites ~ D + score_centered + D:score_centered | year_topic, data = d2)
report(m3, "Sharp RDD  + year x topic FE")

# First stage-ish: accept jump at cutoff
m_fs <- feols(accepted ~ D + score_centered + D:score_centered | year_topic, data = d2)
report(m_fs, "First stage  (accepted)")

cat("\n")
