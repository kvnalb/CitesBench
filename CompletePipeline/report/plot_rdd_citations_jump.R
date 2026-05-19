#!/usr/bin/env Rscript
# RDD-style plot: do citations jump at the acceptance cutoff?
# Running variable: score_centered = mean_rating - year-specific cutoff.
# One panel per year (2018, 2019, 2020). Binscatter + local-linear fit on
# each side of the cutoff.  Dashed vertical line at 0 marks the threshold.

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

DATA_CSV <- file.path(root, "OutputNew/Design/iclr_local_rdd",
                      "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv")
EMB_CSV  <- file.path(root, "OutputNew/Empirics/embeddings",
                      "abstracts_specter2_2018_2023.csv")
PLOT_DIR <- file.path(root, "OutputNew/Report/RDD_Coarse/plots")

YEARS <- c(2018, 2019, 2020)
BIN_W <- 0.1   # binscatter width for running variable
N_TOPICS <- 20
SEED <- 42

# ── load -------------------------------------------------------------------
dt <- fread(DATA_CSV,
  select = c("paper_id","year","mean_rating","score_centered","cutoff",
             "bandwidth","accepted","openalex_cited_by_count",
             "in_year_specific_rdd_sample"))
dt <- dt[in_year_specific_rdd_sample == TRUE & year %in% YEARS &
         !is.na(score_centered) & !is.na(mean_rating) & !is.na(accepted)]
dt[, year := as.integer(year)]

# mark citation availability
dt[, has_cites := !is.na(openalex_cited_by_count)]
dt[has_cites == TRUE, cites  := openalex_cited_by_count]
dt[has_cites == TRUE, lcites := log1p(openalex_cited_by_count)]

cat("Papers per year (total / w/ cites):\n")
print(dt[, .(total = .N, w_cites = sum(has_cites)), by = year][order(year)])

meta <- dt[, .(cutoff = first(cutoff), bandwidth = first(bandwidth),
               n_total = .N, n_acc = sum(accepted == 1),
               n_cites = sum(has_cites),
               n_acc_cites = sum(accepted == 1 & has_cites),
               n_rej_cites = sum(accepted == 0 & has_cites)),
           by = year][order(year)]
print(meta)

# ── binscatter data -------------------------------------------------------
dc <- dt[has_cites == TRUE]
dc[, bin := floor(score_centered / BIN_W) * BIN_W + BIN_W / 2]
bin_stats <- dc[, .(x = mean(score_centered),
                    lcites_mean = mean(lcites),
                    lcites_se   = sd(lcites) / sqrt(.N),
                    n = .N,
                    side = ifelse(mean(score_centered) >= 0, "Accepted", "Rejected")),
                by = .(year, bin)]
bin_stats[, year := factor(year)]
bin_stats[, side := factor(side, levels = c("Rejected","Accepted"))]

# side-of-cutoff label on raw data for smoother
dc[, side := factor(ifelse(score_centered >= 0, "Accepted","Rejected"),
                    levels = c("Rejected","Accepted"))]
dc[, year := factor(year)]

# ── per-year RDD discontinuity: canonical sharp-RDD reduced form ----------
#    lcites ~ D + r + D:r   where D = 1{r >= 0}, r = score_centered
jump_rows <- list()
for (y in YEARS) {
  d <- dc[year == as.character(y)]
  d[, D := as.integer(score_centered >= 0)]
  m <- feols(lcites ~ D + score_centered + D:score_centered, data = d)
  jump_rows[[as.character(y)]] <- data.table(
    year = factor(y),
    jump = coef(m)["D"],
    jump_se = se(m)["D"],
    pval = pvalue(m)["D"],
    n = nobs(m)
  )
}
jumps <- rbindlist(jump_rows)
print(jumps)

# meta label text for each year panel
meta[, year := factor(year)]
facet_labels <- merge(meta, jumps, by = "year")
facet_labels[, label_text := sprintf(
  "cutoff = %.2f\nbandwidth = %.2f\nn total = %d  (acc %d / rej %d)\nn w/ cites = %d  (acc %d / rej %d)\njump in log(1+cites) = %+.2f (se %.2f)",
  cutoff, bandwidth,
  n_total, n_acc, n_total - n_acc,
  n_cites, n_acc_cites, n_rej_cites,
  jump, jump_se)]

# ── plot -------------------------------------------------------------------
side_pal <- c("Rejected" = "#D65F5F", "Accepted" = "#2CA02C")

p <- ggplot(bin_stats, aes(x = x, y = lcites_mean)) +
  geom_vline(xintercept = 0, linetype = "dashed",
             colour = "grey40", linewidth = 0.5) +
  geom_smooth(data = dc,
              aes(x = score_centered, y = lcites,
                  colour = side, group = side),
              method = "lm", se = TRUE, formula = y ~ x,
              linewidth = 0.9, alpha = 0.15) +
  geom_errorbar(aes(ymin = lcites_mean - lcites_se,
                    ymax = lcites_mean + lcites_se,
                    colour = side),
                width = 0, linewidth = 0.35, alpha = 0.6) +
  geom_point(aes(size = n, colour = side), alpha = 0.85) +
  geom_text(data = facet_labels,
            aes(x = -Inf, y = Inf, label = label_text),
            hjust = -0.03, vjust = 1.15, size = 3.0,
            family = "mono", lineheight = 0.95,
            inherit.aes = FALSE) +
  scale_colour_manual(values = side_pal, name = "Side of cutoff") +
  scale_size_area(max_size = 5, guide = "none") +
  facet_wrap(~ year, nrow = 1, scales = "free_x") +
  labs(
    title = "RDD: Citations Discontinuity at Acceptance Cutoff",
    subtitle = sprintf(
      "ICLR %s  |  Running var = mean rating - cutoff  |  Binscatter (bin=%.1f) + local-linear fit either side",
      paste(YEARS, collapse = "-"), BIN_W),
    x = "Running variable:  mean reviewer rating  -  year-specific cutoff",
    y = "log(1 + citations)  (bin mean)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

# ── pooled panel with Year x Topic FEs ------------------------------------
emb <- fread(EMB_CSV)
emb_cols <- grep("^emb_", names(emb), value = TRUE)
dc2 <- merge(dc, emb[, c("paper_id", emb_cols), with = FALSE],
             by = "paper_id", all.x = TRUE)
has_emb <- !is.na(dc2[[emb_cols[1]]])
cat("pooled: papers w/ cites & SPECTER2 =", sum(has_emb), "/", nrow(dc2), "\n")

emb_mat <- as.matrix(dc2[has_emb, ..emb_cols])
emb_mat <- emb_mat / sqrt(rowSums(emb_mat^2))
set.seed(SEED)
km <- kmeans(emb_mat, centers = N_TOPICS, nstart = 5, iter.max = 50)
dc2[has_emb, topic := factor(km$cluster)]
dc2[, year_topic := factor(paste0(year, "::", as.integer(topic)))]
dc2[, (emb_cols) := NULL]

dc2p <- dc2[!is.na(topic)]
dc2p[, yt_size := .N, by = year_topic]
dc2p <- dc2p[yt_size > 1]
dc2p[, year_topic := droplevels(year_topic)]

# residualize y on year_topic FE (FWL demean)
m_y <- feols(lcites ~ 1 | year_topic, data = dc2p)
dc2p[, lcites_resid := as.numeric(resid(m_y))]
dc2p[, side := factor(ifelse(score_centered >= 0, "Accepted","Rejected"),
                      levels = c("Rejected","Accepted"))]

# RDD estimate with Year x Topic FE:
#   lcites = τ·D + β₁·r + β₂·D·r + year_topic FE
dc2p[, D := as.integer(score_centered >= 0)]
m_rdd <- feols(lcites ~ D + score_centered + D:score_centered | year_topic,
               data = dc2p)
jump_fe <- coef(m_rdd)["D"]
se_fe   <- se(m_rdd)["D"]
p_fe    <- pvalue(m_rdd)["D"]
cat(sprintf("\nPooled RDD with Year*Topic FE: jump = %+.3f (se %.3f, p = %.3g, n = %d)\n",
            jump_fe, se_fe, p_fe, nobs(m_rdd)))

# bin on raw score_centered, mean of residualized lcites per bin
dc2p[, bin := floor(score_centered / BIN_W) * BIN_W + BIN_W / 2]
bin_pooled <- dc2p[, .(x = mean(score_centered),
                       y = mean(lcites_resid),
                       se = sd(lcites_resid) / sqrt(.N),
                       n = .N), by = bin]
bin_pooled[, side := factor(ifelse(x >= 0, "Accepted","Rejected"),
                            levels = c("Rejected","Accepted"))]

pooled_label <- sprintf(
  "Pooled 2018-2020  |  Year x Topic FE (k=%d SPECTER2 topics)\nn = %d  |  discontinuity in log(1+cites) = %+.2f  (se %.2f, p = %.3f)",
  N_TOPICS, nobs(m_rdd), jump_fe, se_fe, p_fe)

p_bot <- ggplot(bin_pooled, aes(x = x, y = y)) +
  geom_vline(xintercept = 0, linetype = "dashed",
             colour = "grey40", linewidth = 0.5) +
  geom_hline(yintercept = 0, linetype = "dotted",
             colour = "grey60", linewidth = 0.3) +
  geom_smooth(data = dc2p,
              aes(x = score_centered, y = lcites_resid,
                  colour = side, group = side),
              method = "lm", se = TRUE, formula = y ~ x,
              linewidth = 0.9, alpha = 0.15) +
  geom_errorbar(aes(ymin = y - se, ymax = y + se, colour = side),
                width = 0, linewidth = 0.35, alpha = 0.6) +
  geom_point(aes(size = n, colour = side), alpha = 0.85) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.02, vjust = 1.3,
           label = pooled_label, size = 3.1, family = "mono",
           lineheight = 0.95) +
  scale_colour_manual(values = side_pal, name = "Side of cutoff") +
  scale_size_area(max_size = 5, guide = "none") +
  labs(
    title = "Pooled RDD with Year x Topic Fixed Effects",
    x = "Running variable:  mean reviewer rating  -  year-specific cutoff",
    y = "log(1 + citations)  (residualized on Year x Topic FE)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

# ── combine top (3-panel) and bottom (pooled) via grid viewports ---------
top_grob <- ggplotGrob(p)
bot_grob <- ggplotGrob(p_bot)

save_combined <- function(path, width, height, device_fn) {
  device_fn(path, width = width, height = height)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(
    2, 1, heights = unit.c(unit(1, "null"), unit(1, "null")))))
  pushViewport(viewport(layout.pos.row = 1))
  grid.draw(top_grob)
  upViewport()
  pushViewport(viewport(layout.pos.row = 2))
  grid.draw(bot_grob)
  upViewport(2)
  dev.off()
}

save_combined(file.path(PLOT_DIR, "rdd_citations_jump.pdf"),
              14, 11, function(f, ...) pdf(f, ...))
save_combined(file.path(PLOT_DIR, "rdd_citations_jump.png"),
              14, 11, function(f, ...) png(f, ..., units = "in", res = 300))

cat("\nSaved:", file.path(PLOT_DIR, "rdd_citations_jump.png"), "\n")
