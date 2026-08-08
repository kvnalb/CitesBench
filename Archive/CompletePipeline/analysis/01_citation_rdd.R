library(data.table)
library(ggplot2)
library(fixest)
library(rdrobust)
library(rddensity)

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
indir   <- file.path(root, "OutputNew/Design/iclr_local_rdd")
plotdir <- file.path(root, "OutputNew/Results/RDD/Plots")
tabdir  <- file.path(root, "OutputNew/Results/RDD/Tables")
dir.create(plotdir, recursive = TRUE, showWarnings = FALSE)
dir.create(tabdir,  recursive = TRUE, showWarnings = FALSE)

# ── load ───────────────────────────────────────────────────────────────────
dt <- fread(file.path(indir,
  "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv"))

cat("Loaded:", nrow(dt), "rows\n")

# ── keep essential columns, clean up ──────────────────────────────────────
dt <- dt[, .(paper_id, title, year, primary_area,
             mean_rating, score_centered, cutoff, bandwidth,
             accepted, decision_group, fe_group,
             n_reviews, std_rating, mean_confidence,
             has_arxiv_match, openalex_matched,
             cites = openalex_cited_by_count)]

cat("Citation coverage:", dt[, mean(openalex_matched, na.rm = TRUE)], "\n")
cat("Conditional on match, mean cites:", dt[openalex_matched == TRUE, mean(cites, na.rm = TRUE)], "\n")
cat("Conditional on match, median cites:", dt[openalex_matched == TRUE, median(cites, na.rm = TRUE)], "\n")

# ── outcome transformations ───────────────────────────────────────────────
dt[, `:=`(
  above   = as.integer(score_centered >= 0),
  lcites  = log1p(cites),
  acites  = asinh(cites)
)]

# ── load embeddings → topic clusters ──────────────────────────────────────
emb_file <- file.path(root, "OutputNew/Empirics/embeddings",
                      "abstracts_specter2_2018_2023.csv")
if (file.exists(emb_file)) {
  emb <- fread(emb_file)
  emb_cols <- grep("^emb_", names(emb), value = TRUE)
  dt <- merge(dt, emb[, c("paper_id", emb_cols), with = FALSE],
              by = "paper_id", all.x = TRUE)
  has_emb <- !is.na(dt[[emb_cols[1]]])
  cat("Papers with embeddings:", sum(has_emb), "/", nrow(dt), "\n")

  N_TOPICS <- 20
  emb_mat <- as.matrix(dt[has_emb, ..emb_cols])
  emb_mat <- emb_mat / sqrt(rowSums(emb_mat^2))
  set.seed(42)
  km <- kmeans(emb_mat, centers = N_TOPICS, nstart = 5, iter.max = 50)
  dt[has_emb, topic := factor(km$cluster)]
  dt[has_emb, year_topic := paste0(year, "::", topic)]
  dt[, (emb_cols) := NULL]
  cat("Clustered into", N_TOPICS, "topics\n")
} else {
  cat("No embeddings file found — skipping topic FE\n")
}

# ── bandwidth: use year-specific bandwidths already in the data ───────────
# The sample was already trimmed to within these bandwidths (h ≈ 1.12–1.33).
# Median year-specific bandwidth:
h_pool <- dt[, median(unique(bandwidth))]
cat("Pooled bandwidth (median of year-specific):", h_pool, "\n\n")

# ── common rdrobust options ───────────────────────────────────────────────
# masspoints = "adjust" handles the discrete running variable properly
rd_opts <- list(c = 0, h = h_pool, masspoints = "adjust")

rdrun <- function(y, x, ...) {
  do.call(rdrobust, c(list(y = y, x = x), rd_opts, list(...)))
}

# ══════════════════════════════════════════════════════════════════════════
# 1. MISSINGNESS BALANCE — is citation coverage smooth at the cutoff?
# ══════════════════════════════════════════════════════════════════════════
cat("── Missingness RDD ─────────────────────────────────────\n")
miss_rd <- rdrun(as.numeric(dt$openalex_matched), dt$score_centered)
summary(miss_rd)

# ══════════════════════════════════════════════════════════════════════════
# 2. McCRARY DENSITY TEST — manipulation of running variable?
# ══════════════════════════════════════════════════════════════════════════
cat("── McCrary density test ────────────────────────────────\n")
dens <- rddensity(X = dt$score_centered, c = 0)
summary(dens)

# ══════════════════════════════════════════════════════════════════════════
# 3. FIRST STAGE — acceptance discontinuity
# ══════════════════════════════════════════════════════════════════════════
cat("── First stage: acceptance ─────────────────────────────\n")
fs_rd <- rdrun(dt$accepted, dt$score_centered)
summary(fs_rd)

# ══════════════════════════════════════════════════════════════════════════
# 4. REDUCED FORM — citations on running variable (matched sample)
# ══════════════════════════════════════════════════════════════════════════
dm <- dt[openalex_matched == TRUE & year <= 2023]
cat("── Reduced form sample:", nrow(dm), "papers with citations (2018-2023) ──\n\n")

cat("── RF: log(1+cites) ───────────────────────────────────\n")
rf_log <- rdrun(dm$lcites, dm$score_centered)
summary(rf_log)

cat("── RF: asinh(cites) ───────────────────────────────────\n")
rf_ihs <- rdrun(dm$acites, dm$score_centered)
summary(rf_ihs)

cat("── RF: raw cites ──────────────────────────────────────\n")
rf_raw <- rdrun(dm$cites, dm$score_centered)
summary(rf_raw)

# ══════════════════════════════════════════════════════════════════════════
# 5. FUZZY RDD — acceptance as treatment, citations as outcome
# ══════════════════════════════════════════════════════════════════════════
cat("── Fuzzy RDD: log(1+cites) ────────────────────────────\n")
fz_log <- rdrun(dm$lcites, dm$score_centered, fuzzy = dm$accepted)
summary(fz_log)

cat("── Fuzzy RDD: asinh(cites) ────────────────────────────\n")
fz_ihs <- rdrun(dm$acites, dm$score_centered, fuzzy = dm$accepted)
summary(fz_ihs)

# ══════════════════════════════════════════════════════════════════════════
# 6. PARAMETRIC LOCAL LINEAR with fixest — year and year×area FE
# ══════════════════════════════════════════════════════════════════════════
# triangular kernel weights
dm[, kern_wt := pmax(0, 1 - abs(score_centered) / h_pool)]

cat("── Parametric local linear (fixest) ───────────────────\n\n")

for (yvar in c("lcites", "acites", "cites")) {
  # Basic RDD: common slope
  fml_base <- as.formula(paste(yvar, "~ above + score_centered | year"))
  # With interaction: separate slopes
  fml_int  <- as.formula(paste(yvar, "~ above * score_centered | year"))
  # Year x Area FE with interaction
  fml_ya   <- as.formula(paste(yvar, "~ above * score_centered | fe_group"))

  mb <- feols(fml_base, data = dm, weights = ~kern_wt)
  m1 <- feols(fml_int,  data = dm, weights = ~kern_wt)
  m2 <- feols(fml_ya,   data = dm, weights = ~kern_wt)

  cat("── OLS outcome:", yvar, "──\n")
  print(etable(mb, m1, m2,
               headers = c("Basic RDD", "Diff slopes", "Diff slopes + YxA FE"),
               keep = "above"))
  cat("\n")
}

# Poisson — natural model for citation counts
cat("── Poisson ────────────────────────────────────────────\n\n")
pb <- fepois(cites ~ above + score_centered            | year,     data = dm, weights = ~kern_wt)
p1 <- fepois(cites ~ above * score_centered            | year,     data = dm, weights = ~kern_wt)
p2 <- fepois(cites ~ above * score_centered            | fe_group, data = dm, weights = ~kern_wt)

if ("topic" %in% names(dm)) {
  dm_t <- dm[!is.na(topic)]
  p3 <- fepois(cites ~ above + score_centered | year + topic,  data = dm_t, weights = ~kern_wt)
  p4 <- fepois(cites ~ above + score_centered | year_topic,    data = dm_t, weights = ~kern_wt)

  print(etable(pb, p1, p2, p3, p4,
               headers = c("Basic RDD", "Diff slopes", "Diff slopes + YxA FE",
                            "Year + Topic FE", "Year x Topic FE"),
               keep = "above"))
} else {
  print(etable(pb, p1, p2,
               headers = c("Basic RDD", "Diff slopes", "Diff slopes + YxA FE"),
               keep = "above"))
}

# ══════════════════════════════════════════════════════════════════════════
# 7. RDD PLOT
# ══════════════════════════════════════════════════════════════════════════
rdp <- rdplot(y = dm$lcites, x = dm$score_centered, c = 0,
              title = "Citation RDD: log(1 + cites)",
              x.label = "Centered review score",
              y.label = "log(1 + cited-by count)")
ggsave(file.path(plotdir, "fig_citation_rdd_log.png"),
       rdp$rdplot, width = 8, height = 5, dpi = 200)

rdp2 <- rdplot(y = dm$acites, x = dm$score_centered, c = 0,
               title = "Citation RDD: asinh(cites)",
               x.label = "Centered review score",
               y.label = "asinh(cited-by count)")
ggsave(file.path(plotdir, "fig_citation_rdd_ihs.png"),
       rdp2$rdplot, width = 8, height = 5, dpi = 200)

# ══════════════════════════════════════════════════════════════════════════
# 8. COLLECT RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════
collect <- function(tag, obj) {
  data.table(
    spec    = tag,
    coef    = obj$coef[1],
    se      = obj$se[1],
    pval    = obj$pv[1],
    ci_lo   = obj$ci[1, 1],
    ci_hi   = obj$ci[1, 2],
    bw_l    = obj$bws[1, 1],
    bw_r    = obj$bws[1, 2],
    n_left  = obj$N_h[1],
    n_right = obj$N_h[2]
  )
}

results <- rbindlist(list(
  collect("first_stage_accept",  fs_rd),
  collect("missingness",         miss_rd),
  collect("rf_log_cites",        rf_log),
  collect("rf_ihs_cites",        rf_ihs),
  collect("rf_raw_cites",        rf_raw),
  collect("fuzzy_log_cites",     fz_log),
  collect("fuzzy_ihs_cites",     fz_ihs)
))

print(results)
fwrite(results, file.path(tabdir, "rdd_results.csv"))
cat("\nResults written to", file.path(tabdir, "rdd_results.csv"), "\n")
