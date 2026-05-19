library(data.table)
library(ggplot2)
library(fixest)

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

# ── load RDD sample (2018-2023) ───────────────────────────────────────────
dt <- fread(file.path(root, "OutputNew/Design/iclr_local_rdd",
  "rdd_sample_year_specific_bandwidth.csv"),
  select = c("paper_id", "year", "mean_rating", "std_rating",
             "min_rating", "max_rating", "n_reviews",
             "mean_confidence", "score_centered", "cutoff",
             "bandwidth", "accepted"))
dt <- dt[year <= 2023]
dt[, above := as.integer(score_centered >= 0)]
dt[, range_rating := max_rating - min_rating]

h_pool <- dt[, median(unique(bandwidth))]
dt[, kern_wt := pmax(0, 1 - abs(score_centered) / h_pool)]

cat("Sample:", nrow(dt), "papers (2018-2023 RDD sample)\n\n")

# ── disagreement summary ─────────────────────────────────────────────────
cat("── Disagreement measures ──\n")
print(dt[, .(mean_std = round(mean(std_rating), 2),
             med_std  = round(median(std_rating), 2),
             mean_range = round(mean(range_rating), 2),
             med_range  = round(median(range_rating), 2),
             mean_nrev  = round(mean(n_reviews), 1)), by = year][order(year)])

# ── disagreement by score bin ─────────────────────────────────────────────
dt[, score_bin := round(score_centered * 4) / 4]
binned <- dt[, .(mean_std   = mean(std_rating),
                 mean_range = mean(range_rating),
                 mean_conf  = mean(mean_confidence, na.rm = TRUE),
                 n = .N), by = score_bin]

p1 <- ggplot(binned, aes(x = score_bin, y = mean_range)) +
  geom_point(aes(size = n), alpha = 0.7) +
  geom_smooth(data = dt, aes(x = score_centered, y = range_rating),
              method = "loess", se = TRUE, color = "red", linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Reviewer disagreement across RDD cutoff",
       x = "Centered review score", y = "Range of reviewer ratings",
       size = "Papers") +
  theme_minimal(base_size = 12)
ggsave(file.path(plotdir, "fig_disagreement_at_cutoff.png"),
       p1, width = 8, height = 5, dpi = 200)

# ── RDD on disagreement: is it smooth at the cutoff? ─────────────────────
cat("\n── RDD on disagreement (covariate balance test) ──\n\n")

m_std   <- feols(std_rating    ~ above + score_centered | year,
                 data = dt, weights = ~kern_wt)
m_range <- feols(range_rating  ~ above + score_centered | year,
                 data = dt, weights = ~kern_wt)
m_conf  <- feols(mean_confidence ~ above + score_centered | year,
                 data = dt[!is.na(mean_confidence)], weights = ~kern_wt)
m_nrev  <- feols(n_reviews     ~ above + score_centered | year,
                 data = dt, weights = ~kern_wt)

print(etable(m_std, m_range, m_conf, m_nrev,
             headers = c("SD rating", "Range rating", "Confidence", "N reviews"),
             keep = "above"))

# ── disagreement by year, above vs below ──────────────────────────────────
cat("\n── Mean disagreement: above vs below cutoff by year ──\n")
print(dt[, .(
  below_std = round(mean(std_rating[above == 0]), 2),
  above_std = round(mean(std_rating[above == 1]), 2),
  diff_std  = round(mean(std_rating[above == 1]) - mean(std_rating[above == 0]), 2),
  below_range = round(mean(range_rating[above == 0]), 2),
  above_range = round(mean(range_rating[above == 1]), 2)
), by = year][order(year)])

fwrite(binned, file.path(tabdir, "disagreement_by_score_bin.csv"))

# ── full panel plot: all 2018-23 papers, by year ─────────────────────────
full <- fread(file.path(root, "OutputNew/Design/iclr_local_rdd",
  "paper_level_all_years.csv"),
  select = c("paper_id", "year", "mean_rating", "std_rating",
             "min_rating", "max_rating", "n_reviews", "accepted"))
full <- full[year <= 2023]
full[, range_rating := max_rating - min_rating]

# year-specific cutoffs
cuts <- dt[, .(cutoff = cutoff[1], bandwidth = bandwidth[1]), by = year]

p3 <- ggplot(full, aes(x = mean_rating, y = range_rating)) +
  geom_point(alpha = 0.08, size = 0.5) +
  geom_smooth(method = "loess", se = TRUE, color = "red", linewidth = 0.8) +
  geom_vline(data = cuts, aes(xintercept = cutoff),
             linetype = "dashed", color = "blue", linewidth = 0.6) +
  geom_rect(data = cuts,
            aes(xmin = cutoff - bandwidth, xmax = cutoff + bandwidth,
                ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "blue", alpha = 0.05) +
  facet_wrap(~year, scales = "free_x") +
  labs(title = "Reviewer disagreement vs mean rating (all 2018-23 papers)",
       subtitle = "Blue dashed = acceptance cutoff; shaded = RDD bandwidth",
       x = "Mean reviewer rating", y = "Range of reviewer ratings") +
  theme_minimal(base_size = 12)
ggsave(file.path(plotdir, "fig_disagreement_full_panel.png"),
       p3, width = 12, height = 8, dpi = 200)

cat("\nDone.\n")
