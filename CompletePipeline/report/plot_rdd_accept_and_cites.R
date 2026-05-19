#!/usr/bin/env Rscript
# Two-panel pooled RDD (ICLR 2018-2020), canonical sharp-RDD reduced form
# with Year x Topic fixed effects:
#   (a) Outcome = accepted (0/1)
#   (b) Outcome = log(1 + citations)
# Running variable: r = mean_rating - year-specific cutoff  (D = 1{r >= 0})
#   Y = tau * D + beta1 * r + beta2 * D * r + year_topic FE

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
BIN_W <- 0.1
N_TOPICS <- 20
SEED <- 42

# ── load ------------------------------------------------------------------
dt <- fread(DATA_CSV,
  select = c("paper_id","year","mean_rating","score_centered","cutoff",
             "bandwidth","accepted","openalex_cited_by_count",
             "in_year_specific_rdd_sample"))
dt <- dt[in_year_specific_rdd_sample == TRUE & year %in% YEARS &
         !is.na(score_centered) & !is.na(mean_rating) & !is.na(accepted)]
dt[, year := as.integer(year)]
dt[, has_cites := !is.na(openalex_cited_by_count)]
dt[, lcites := ifelse(has_cites, log1p(openalex_cited_by_count), NA_real_)]
dt[, D := as.integer(score_centered >= 0)]

# ── SPECTER2 topics -------------------------------------------------------
emb <- fread(EMB_CSV)
emb_cols <- grep("^emb_", names(emb), value = TRUE)
dt <- merge(dt, emb[, c("paper_id", emb_cols), with = FALSE],
            by = "paper_id", all.x = TRUE)
has_emb <- !is.na(dt[[emb_cols[1]]])
emb_mat <- as.matrix(dt[has_emb, ..emb_cols])
emb_mat <- emb_mat / sqrt(rowSums(emb_mat^2))
set.seed(SEED)
km <- kmeans(emb_mat, centers = N_TOPICS, nstart = 5, iter.max = 50)
dt[has_emb, topic := factor(km$cluster)]
dt[, year_topic := factor(paste0(year, "::", as.integer(topic)))]
dt[, (emb_cols) := NULL]

cat("Pooled 2018-2020 RDD sample:", nrow(dt), "papers\n")
cat("  with cites:", sum(dt$has_cites), "\n")
cat("  with topics:", sum(!is.na(dt$topic)), "\n")

# ── helper: run RDD, residualize, binscatter, build panel frame ----------
build_rdd_panel <- function(d, yvar, panel_letter, outcome_label) {
  d <- copy(d[!is.na(get(yvar)) & !is.na(topic)])
  d[, yt_size := .N, by = year_topic]
  d <- d[yt_size > 1]
  d[, year_topic := droplevels(year_topic)]

  # canonical sharp RDD: tau = coef on D
  f <- as.formula(sprintf("%s ~ D + score_centered + D:score_centered | year_topic", yvar))
  m <- feols(f, data = d)
  tau <- coef(m)["D"];  se_tau <- se(m)["D"];  p_tau <- pvalue(m)["D"]

  # residualize y on year_topic FE for binscatter y-axis
  m_fe <- feols(as.formula(sprintf("%s ~ 1 | year_topic", yvar)), data = d)
  d[, y_resid := as.numeric(resid(m_fe))]

  d[, bin := floor(score_centered / BIN_W) * BIN_W + BIN_W / 2]
  bin <- d[, .(x = mean(score_centered),
               y = mean(y_resid),
               se = sd(y_resid) / sqrt(.N),
               n = .N), by = bin]
  bin[, side := factor(ifelse(x >= 0, "Accepted","Rejected"),
                       levels = c("Rejected","Accepted"))]

  d[, side := factor(ifelse(score_centered >= 0, "Accepted","Rejected"),
                     levels = c("Rejected","Accepted"))]

  label <- sprintf("%s  |  %s\nn = %d   Year x Topic FE  (k=%d)\ntau (jump at cutoff) = %+.3f  (se %.3f, p = %.3f)",
                   panel_letter, outcome_label, nobs(m), N_TOPICS,
                   tau, se_tau, p_tau)

  list(bin = bin, raw = d,
       stats = list(tau = tau, se = se_tau, pval = p_tau, n = nobs(m),
                    label = label, ylab = outcome_label,
                    panel = paste0(panel_letter, "  ", outcome_label)))
}

accept_panel <- build_rdd_panel(dt,         "accepted", "(a)",
                                "Accepted (0/1)")
cites_panel  <- build_rdd_panel(dt[has_cites==TRUE], "lcites", "(b)",
                                "log(1 + citations)")

cat(sprintf("\n(a) accepted jump     : %+.3f (se %.3f, p = %.3f, n = %d)\n",
            accept_panel$stats$tau, accept_panel$stats$se,
            accept_panel$stats$pval, accept_panel$stats$n))
cat(sprintf("(b) log(1+cites) jump : %+.3f (se %.3f, p = %.3f, n = %d)\n",
            cites_panel$stats$tau,  cites_panel$stats$se,
            cites_panel$stats$pval, cites_panel$stats$n))

# ── plot each panel -------------------------------------------------------
side_pal <- c("Rejected" = "#D65F5F", "Accepted" = "#2CA02C")

make_panel <- function(pack) {
  ggplot(pack$bin, aes(x = x, y = y)) +
    geom_vline(xintercept = 0, linetype = "dashed",
               colour = "grey40", linewidth = 0.5) +
    geom_hline(yintercept = 0, linetype = "dotted",
               colour = "grey60", linewidth = 0.3) +
    geom_smooth(data = pack$raw,
                aes(x = score_centered, y = y_resid,
                    colour = side, group = side),
                method = "lm", se = TRUE, formula = y ~ x,
                linewidth = 0.9, alpha = 0.15) +
    geom_errorbar(aes(ymin = y - se, ymax = y + se, colour = side),
                  width = 0, linewidth = 0.35, alpha = 0.6) +
    geom_point(aes(size = n, colour = side), alpha = 0.85) +
    annotate("text", x = -Inf, y = Inf, hjust = -0.02, vjust = 1.3,
             label = pack$stats$label, size = 3.1, family = "mono",
             lineheight = 0.95) +
    scale_colour_manual(values = side_pal, name = "Side of cutoff") +
    scale_size_area(max_size = 5, guide = "none") +
    labs(
      x = "Running variable:  mean reviewer rating - year-specific cutoff",
      y = sprintf("%s  (residualized on Year x Topic FE)", pack$stats$ylab),
      title = pack$stats$panel
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 11),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )
}

p_a <- make_panel(accept_panel)
p_b <- make_panel(cites_panel)

# ── save two-panel layout via grid viewports ------------------------------
ga <- ggplotGrob(p_a)
gb <- ggplotGrob(p_b)

save_combined <- function(path, width, height, device_fn) {
  device_fn(path, width = width, height = height)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(
    2, 2,
    heights = unit.c(unit(0.35, "inches"), unit(1, "null")),
    widths  = unit.c(unit(1, "null"),     unit(1, "null")))))
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 1:2))
  grid.text(sprintf(
    "Canonical Sharp RDD  |  ICLR 2018-2020, pooled  |  Y = tau*D + b1*r + b2*D*r + (year x topic) FE"),
    gp = gpar(fontface = "bold", fontsize = 12))
  upViewport()
  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 1))
  grid.draw(ga)
  upViewport()
  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = 2))
  grid.draw(gb)
  upViewport()
  dev.off()
}

save_combined(file.path(PLOT_DIR, "rdd_accept_and_cites.pdf"),
              14, 6.5, function(f, ...) pdf(f, ...))
save_combined(file.path(PLOT_DIR, "rdd_accept_and_cites.png"),
              14, 6.5, function(f, ...) png(f, ..., units = "in", res = 300))

cat("\nSaved:", file.path(PLOT_DIR, "rdd_accept_and_cites.png"), "\n")
