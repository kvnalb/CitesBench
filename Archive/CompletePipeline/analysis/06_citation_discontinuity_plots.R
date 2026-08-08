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
dir.create(plotdir, recursive = TRUE, showWarnings = FALSE)

# ── load 2018-2023 papers with citations ──────────────────────────────────
dt <- fread(file.path(root, "OutputNew/Design/iclr_local_rdd",
  "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv"),
  select = c("paper_id", "year", "mean_rating", "score_centered",
             "cutoff", "bandwidth", "accepted",
             "openalex_matched", "openalex_cited_by_count"))
dt <- dt[year <= 2023 & openalex_matched == TRUE]
setnames(dt, "openalex_cited_by_count", "cites")

cat("Sample:", nrow(dt), "papers\n")

# ── binned means ──────────────────────────────────────────────────────────
bin_width <- 0.25
dt[, score_bin := round(score_centered / bin_width) * bin_width]

# ── FIGURE 1: pooled ─────────────────────────────────────────────────────
pooled <- dt[, .(mean_cites = mean(cites),
                 se = sd(cites) / sqrt(.N),
                 n = .N), by = score_bin]

p1 <- ggplot(pooled, aes(x = score_bin, y = mean_cites)) +
  geom_point(aes(size = n), alpha = 0.7) +
  geom_errorbar(aes(ymin = mean_cites - 1.96 * se,
                    ymax = mean_cites + 1.96 * se),
                width = 0.05, alpha = 0.4) +
  geom_smooth(data = dt[score_centered < 0],
              aes(x = score_centered, y = cites),
              method = "lm", se = TRUE, color = "red", linewidth = 0.8) +
  geom_smooth(data = dt[score_centered >= 0],
              aes(x = score_centered, y = cites),
              method = "lm", se = TRUE, color = "red", linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Citation discontinuity at acceptance cutoff (2018-2023)",
       x = "Centered review score",
       y = "Mean citations",
       size = "Papers") +
  theme_minimal(base_size = 13) +
  theme(legend.position = c(0.15, 0.85))
ggsave(file.path(plotdir, "fig_citation_discontinuity_pooled.png"),
       p1, width = 8, height = 5.5, dpi = 200)

# ── FIGURE 2: by year ────────────────────────────────────────────────────
by_year <- dt[, .(mean_cites = mean(cites),
                  se = sd(cites) / sqrt(.N),
                  n = .N), by = .(year, score_bin)]

p2 <- ggplot(by_year, aes(x = score_bin, y = mean_cites)) +
  geom_point(aes(size = n), alpha = 0.7) +
  geom_errorbar(aes(ymin = mean_cites - 1.96 * se,
                    ymax = mean_cites + 1.96 * se),
                width = 0.05, alpha = 0.3) +
  geom_smooth(data = dt[score_centered < 0],
              aes(x = score_centered, y = cites),
              method = "lm", se = TRUE, color = "red", linewidth = 0.8) +
  geom_smooth(data = dt[score_centered >= 0],
              aes(x = score_centered, y = cites),
              method = "lm", se = TRUE, color = "red", linewidth = 0.8) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  facet_wrap(~year, scales = "free_y") +
  labs(title = "Citation discontinuity by year",
       x = "Centered review score",
       y = "Mean citations",
       size = "Papers") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")
ggsave(file.path(plotdir, "fig_citation_discontinuity_by_year.png"),
       p2, width = 12, height = 8, dpi = 200)

cat("Done.\n")
