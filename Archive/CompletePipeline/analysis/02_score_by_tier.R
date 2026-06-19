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

# ── load RDD sample + OpenReview venue ────────────────────────────────────
rdd <- fread(file.path(root, "OutputNew/Design/iclr_local_rdd",
  "rdd_sample_year_specific_bandwidth.csv"),
  select = c("paper_id", "year", "mean_rating", "cutoff", "score_centered",
             "accepted", "decision", "n_reviews", "std_rating"))

or <- fread(file.path(root, "OutputNew/rawdata/Design/OpenReview",
  "openreview_yearly_submissions.csv"),
  select = c("paper_id", "openreview_venue"))

dt <- merge(rdd, or, by = "paper_id", all.x = TRUE)

# ── parse venue into tier ─────────────────────────────────────────────────
dt[, tier := fcase(
  grepl("Oral|oral",           openreview_venue), "Oral",
  grepl("Spotlight|spotlight", openreview_venue), "Spotlight",
  grepl("notable top 5%",      openreview_venue), "Oral",
  grepl("notable top 25%",     openreview_venue), "Spotlight",
  grepl("poster|Poster",       openreview_venue), "Poster",
  accepted == 0,                                  "Rejected",
  default = NA_character_
)]
dt[, tier := factor(tier, levels = c("Rejected", "Poster", "Spotlight", "Oral"))]

cat("── Tier coverage by year ──\n")
print(dt[, .N, by = .(year, tier)][order(year, tier)])

# ── score summary by tier ─────────────────────────────────────────────────
cat("\n── Mean rating by tier ──\n")
tab <- dt[!is.na(tier), .(
  n       = .N,
  mean    = round(mean(mean_rating), 2),
  sd      = round(sd(mean_rating), 2),
  p10     = round(quantile(mean_rating, 0.10), 2),
  median  = round(median(mean_rating), 2),
  p90     = round(quantile(mean_rating, 0.90), 2)
), by = tier][order(tier)]
print(tab)
fwrite(tab, file.path(tabdir, "score_by_tier.csv"))

cat("\n── Mean rating by tier × year ──\n")
tab_yr <- dt[!is.na(tier), .(
  n    = .N,
  mean = round(mean(mean_rating), 2),
  sd   = round(sd(mean_rating), 2)
), by = .(year, tier)][order(year, tier)]
print(tab_yr)
fwrite(tab_yr, file.path(tabdir, "score_by_tier_year.csv"))

# ── plot: score distribution by tier ──────────────────────────────────────
p1 <- ggplot(dt[!is.na(tier)], aes(x = mean_rating, fill = tier)) +
  geom_histogram(binwidth = 0.25, position = "identity", alpha = 0.5) +
  facet_wrap(~year, scales = "free_y") +
  labs(title = "Mean reviewer rating by acceptance tier",
       x = "Mean reviewer rating", y = "Count", fill = "Tier") +
  theme_minimal(base_size = 12)
ggsave(file.path(plotdir, "fig_score_by_tier_hist.png"),
       p1, width = 12, height = 8, dpi = 200)

# ── plot: box plot by tier per year ───────────────────────────────────────
p2 <- ggplot(dt[!is.na(tier)], aes(x = tier, y = mean_rating, fill = tier)) +
  geom_boxplot(outlier.size = 0.5) +
  facet_wrap(~year) +
  labs(title = "Score distributions by acceptance tier",
       x = NULL, y = "Mean reviewer rating") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
ggsave(file.path(plotdir, "fig_score_by_tier_box.png"),
       p2, width = 12, height = 8, dpi = 200)

cat("\nPlots saved to", plotdir, "\n")
