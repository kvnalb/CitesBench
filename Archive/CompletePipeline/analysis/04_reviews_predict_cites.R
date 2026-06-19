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

# ── load RDD sample with citations (2018-2023) ───────────────────────────
dt <- fread(file.path(root, "OutputNew/Design/iclr_local_rdd",
  "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv"),
  select = c("paper_id", "year", "mean_rating", "median_rating",
             "std_rating", "min_rating", "max_rating",
             "n_reviews", "mean_confidence", "mean_binocular",
             "score_centered", "accepted", "fe_group",
             "openalex_matched", "openalex_cited_by_count"))

dt <- dt[year <= 2023 & openalex_matched == TRUE]
setnames(dt, "openalex_cited_by_count", "cites")
dt[, lcites := log1p(cites)]

cat("Sample:", nrow(dt), "papers (2018-2023 with citations)\n\n")

# ── load embeddings and cluster into topics ───────────────────────────────
emb <- fread(file.path(root, "OutputNew/Empirics/embeddings",
                       "abstracts_specter2_2018_2023.csv"))
emb_cols <- grep("^emb_", names(emb), value = TRUE)

# merge embeddings
dt <- merge(dt, emb[, c("paper_id", emb_cols), with = FALSE],
            by = "paper_id", all.x = TRUE)
has_emb <- !is.na(dt[[emb_cols[1]]])
cat("Papers with embeddings:", sum(has_emb), "/", nrow(dt), "\n")

# k-means clustering on L2-normalized embeddings
N_TOPICS <- 20
emb_mat <- as.matrix(dt[has_emb, ..emb_cols])
emb_mat <- emb_mat / sqrt(rowSums(emb_mat^2))

set.seed(42)
km <- kmeans(emb_mat, centers = N_TOPICS, nstart = 5, iter.max = 50)
dt[has_emb, topic := factor(km$cluster)]
dt[has_emb, year_topic := paste0(year, "::", topic)]

cat("Clustered into", N_TOPICS, "topics\n")
cat("── Topic sizes ──\n")
print(dt[has_emb, .N, by = topic][order(-N)])

# drop embedding columns to save memory
dt[, (emb_cols) := NULL]

# ── summary ───────────────────────────────────────────────────────────────
cat("── Cites by year ──\n")
print(dt[, .(n = .N, med = median(cites), mean = round(mean(cites), 1),
             p90 = quantile(cites, 0.9)), by = year][order(year)])

# ── OLS: reviewer scores → log(1+cites) ──────────────────────────────────
cat("\n── OLS: log(1+cites) ──\n")
m1 <- feols(lcites ~ mean_rating, data = dt)
m2 <- feols(lcites ~ mean_rating | year, data = dt)
m3 <- feols(lcites ~ mean_rating + std_rating + mean_confidence + n_reviews | year, data = dt)
m4 <- feols(lcites ~ mean_rating + std_rating + mean_confidence + n_reviews | fe_group, data = dt)

dt_t <- dt[!is.na(topic) & !is.na(mean_confidence)]
m5 <- feols(lcites ~ mean_rating + std_rating + mean_confidence + n_reviews | year + topic, data = dt_t)
m6 <- feols(lcites ~ mean_rating + std_rating + mean_confidence + n_reviews | year_topic, data = dt_t)

print(etable(m1, m2, m3, m5, m6,
             headers = c("Bivariate", "Year FE", "+ Controls",
                         "Year + Topic FE", "Year x Topic FE")))

# ── Poisson ───────────────────────────────────────────────────────────────
cat("\n── Poisson: cites ──\n")
p1 <- fepois(cites ~ mean_rating, data = dt)
p2 <- fepois(cites ~ mean_rating | year, data = dt)
p3 <- fepois(cites ~ mean_rating + std_rating + mean_confidence + n_reviews | year, data = dt)
p4 <- fepois(cites ~ mean_rating + std_rating + mean_confidence + n_reviews | fe_group, data = dt)

p5 <- fepois(cites ~ mean_rating + std_rating + mean_confidence + n_reviews | year + topic, data = dt_t)
p6 <- fepois(cites ~ mean_rating + std_rating + mean_confidence + n_reviews | year_topic, data = dt_t)

print(etable(p1, p2, p3, p5, p6,
             headers = c("Bivariate", "Year FE", "+ Controls",
                         "Year + Topic FE", "Year x Topic FE")))

# ── within-year R² from reviewer scores ──────────────────────────────────
cat("\n── Within-year R² (OLS on log cites) ──\n")
r2_list <- list()
for (yr in sort(unique(dt$year))) {
  d <- dt[year == yr & !is.na(mean_confidence)]
  if (nrow(d) < 20) next
  m <- lm(lcites ~ mean_rating + std_rating + mean_confidence, data = d)
  r2_list[[as.character(yr)]] <- data.table(
    year = yr, n = nrow(d),
    r2 = round(summary(m)$r.squared, 3),
    cor_rating = round(cor(d$lcites, d$mean_rating), 3)
  )
}
r2_yr <- rbindlist(r2_list)
print(r2_yr)
fwrite(r2_yr, file.path(tabdir, "review_score_r2_by_year.csv"))

# ── plot: mean_rating vs log cites by year ────────────────────────────────
p <- ggplot(dt, aes(x = mean_rating, y = lcites)) +
  geom_point(alpha = 0.15, size = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 0.8) +
  facet_wrap(~year, scales = "free") +
  labs(title = "Reviewer rating vs citations (2018-2023)",
       x = "Mean reviewer rating", y = "log(1 + cites)") +
  theme_minimal(base_size = 12)
ggsave(file.path(plotdir, "fig_rating_vs_cites.png"),
       p, width = 10, height = 8, dpi = 200)

cat("\nDone.\n")
