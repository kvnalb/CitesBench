#!/usr/bin/env Rscript
# 2x2 FWL residualized scatter of mean reviewer rating vs log(1+cites) for
# ICLR 2018-2020 papers in the RDD sample:
#   Top row    = accepted; Bottom row = rejected
#   Left column  = Year FE; Right column = Year x Topic FE
# Topics: k-means (k=20, seed=42) on L2-normalized SPECTER2 abstract embeddings

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

DATA_CSV <- file.path(root, "OutputNew/Design/iclr_local_rdd",
                      "rdd_sample_year_specific_bandwidth_with_openalex_citations.csv")
EMB_CSV  <- file.path(root, "OutputNew/Empirics/embeddings",
                      "abstracts_specter2_2018_2023.csv")
PLOT_DIR <- file.path(root, "OutputNew/Report/RDD_Coarse/plots")
dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

YEARS <- c(2018, 2019, 2020)
N_TOPICS <- 20
SEED <- 42

# ── load ------------------------------------------------------------------
dt_all <- fread(DATA_CSV,
  select = c("paper_id", "year", "mean_rating", "accepted",
             "openalex_cited_by_count", "in_year_specific_rdd_sample"))
dt_all <- dt_all[in_year_specific_rdd_sample == TRUE &
                 year %in% YEARS &
                 !is.na(mean_rating) &
                 !is.na(accepted)]

n_sample_acc <- nrow(dt_all[accepted == 1])
n_sample_rej <- nrow(dt_all[accepted == 0])

dt <- dt_all[!is.na(openalex_cited_by_count)]
setnames(dt, "openalex_cited_by_count", "cites")
dt[, lcites := log1p(cites)]
dt[, year := as.integer(year)]

cat(sprintf("Total RDD sample 2018-2020:  Accepted=%d, Rejected=%d\n",
            n_sample_acc, n_sample_rej))
cat("w/ OpenAlex cites:\n")
print(dt[, .N, by = accepted])

# ── embeddings + k-means topics (one global clustering) -------------------
emb <- fread(EMB_CSV)
emb_cols <- grep("^emb_", names(emb), value = TRUE)
dt <- merge(dt, emb[, c("paper_id", emb_cols), with = FALSE],
            by = "paper_id", all.x = TRUE)
has_emb <- !is.na(dt[[emb_cols[1]]])
cat("w/ SPECTER2 embeddings:", sum(has_emb), "/", nrow(dt), "\n")

emb_mat <- as.matrix(dt[has_emb, ..emb_cols])
emb_mat <- emb_mat / sqrt(rowSums(emb_mat^2))
set.seed(SEED)
km <- kmeans(emb_mat, centers = N_TOPICS, nstart = 5, iter.max = 50)
dt[has_emb, topic := factor(km$cluster)]
dt[, year_topic := factor(paste0(year, "::", as.integer(topic)))]
dt[, (emb_cols) := NULL]

# ── helper: build one panel's data ----------------------------------------
build_panel <- function(sub, fe_col, panel_label, model_col) {
  d <- copy(sub)
  if (fe_col == "year_topic") {
    d <- d[!is.na(topic)]
    d[, yt_size := .N, by = year_topic]
    d <- d[yt_size > 1]
    d[, year_topic := droplevels(year_topic)]
  }
  f_y  <- as.formula(paste("lcites ~ 1 |", fe_col))
  f_r  <- as.formula(paste("mean_rating ~ 1 |", fe_col))
  f_yr <- as.formula(paste("lcites ~ mean_rating |", fe_col))
  m_y  <- feols(f_y,  data = d)
  m_r  <- feols(f_r,  data = d)
  m_yr <- feols(f_yr, data = d, fixef.rm = "none")

  slope <- coef(m_yr)["mean_rating"]
  se_b  <- se(m_yr)["mean_rating"]
  pval  <- pvalue(m_yr)["mean_rating"]
  r_resid <- cor(resid(m_r), resid(m_y))
  r2w <- fitstat(m_yr, "war2", simplify = TRUE)

  list(
    frame = data.table(
      panel = panel_label,
      year = factor(d$year),
      x = resid(m_r),
      y = resid(m_y)
    ),
    stats = data.table(
      panel = panel_label,
      label = sprintf("n = %d\nslope = %+.3f (se %.3f)\nPearson r = %+.3f",
                      nobs(m_yr), slope, se_b, r_resid)
    ),
    reg = list(model = m_yr, col = model_col, slope = slope, se = se_b,
               pval = pval, n = nobs(m_yr), r2w = r2w)
  )
}

accepted <- dt[accepted == 1]
rejected <- dt[accepted == 0]

n_acc_cites <- nrow(accepted)
n_rej_cites <- nrow(rejected)

cat(sprintf("\nAccepted: total=%d, w/ cites=%d\n", n_sample_acc, n_acc_cites))
cat(sprintf("Rejected: total=%d, w/ cites=%d\n", n_sample_rej, n_rej_cites))

panels <- list(
  build_panel(accepted, "year",       "(a)  Accepted | Year FE",        "(a)"),
  build_panel(accepted, "year_topic", "(b)  Accepted | Year x Topic FE","(b)"),
  build_panel(rejected, "year",       "(c)  Rejected | Year FE",        "(c)"),
  build_panel(rejected, "year_topic", "(d)  Rejected | Year x Topic FE","(d)")
)

plot_df  <- rbindlist(lapply(panels, `[[`, "frame"))
stats_df <- rbindlist(lapply(panels, `[[`, "stats"))

panel_order <- c(
  "(a)  Accepted | Year FE",
  "(b)  Accepted | Year x Topic FE",
  "(c)  Rejected | Year FE",
  "(d)  Rejected | Year x Topic FE"
)
plot_df[,  panel := factor(panel, levels = panel_order)]
stats_df[, panel := factor(panel, levels = panel_order)]

for (p in panels) cat(p$stats$panel, ":", p$stats$label, "\n\n")

year_pal <- c("2018" = "#4878CF", "2019" = "#6ACC65", "2020" = "#D65F5F")

p <- ggplot(plot_df, aes(x = x, y = y)) +
  geom_hline(yintercept = 0, linewidth = 0.3, linetype = "dashed", colour = "grey60") +
  geom_vline(xintercept = 0, linewidth = 0.3, linetype = "dashed", colour = "grey60") +
  geom_point(aes(colour = year), alpha = 0.5, size = 1.1) +
  geom_smooth(method = "lm", se = FALSE, colour = "black",
              linewidth = 0.9, formula = y ~ x) +
  geom_text(data = stats_df,
            aes(x = -Inf, y = Inf, label = label),
            hjust = -0.05, vjust = 1.2, size = 3.1,
            family = "mono", lineheight = 0.95,
            inherit.aes = FALSE) +
  scale_colour_manual(values = year_pal, name = "Year") +
  facet_wrap(~ panel, nrow = 2, ncol = 2, scales = "free") +
  labs(
    title = "Rating vs. Citations -- FWL Residualized  (ICLR 2018-2020, RDD sample)",
    subtitle = sprintf(
      "Accepted n=%d (w/ OpenAlex cites=%d)   |   Rejected n=%d (w/ OpenAlex cites=%d)   |   Topics: k-means k=%d on SPECTER2",
      n_sample_acc, n_acc_cites, n_sample_rej, n_rej_cites, N_TOPICS),
    x = "Mean human reviewer rating  (residualized on FE)",
    y = "log(1 + citations)  (residualized on FE)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold", size = 10),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

# ── build regression table as text ----------------------------------------
regs <- lapply(panels, `[[`, "reg")

sig_stars <- function(pv) {
  if (is.na(pv)) return("")
  if (pv < 0.001) "***" else if (pv < 0.01) "**" else
  if (pv < 0.05)  "*"   else if (pv < 0.1)  "."  else ""
}

col_w  <- 11
lbl_w  <- 24
fmt_row <- function(label, vals) {
  paste0(formatC(label, width = lbl_w, flag = "-"),
         paste(sapply(vals, function(v) formatC(v, width = col_w)),
               collapse = ""))
}

header1 <- fmt_row("", sapply(regs, function(r) r$col))
header2 <- fmt_row("Sample:", c("Acc.","Acc.","Rej.","Rej."))
header3 <- fmt_row("Fixed effects:", c("Year","Year*Top","Year","Year*Top"))
sep     <- paste0(strrep("-", lbl_w + 4 * col_w))

coef_row <- fmt_row("Mean rating  (beta)",
  sapply(regs, function(r) sprintf("%+.3f%s", r$slope, sig_stars(r$pval))))
se_row   <- fmt_row("   (SE)",
  sapply(regs, function(r) sprintf("(%.3f)", r$se)))
n_row    <- fmt_row("Observations",
  sapply(regs, function(r) formatC(r$n, big.mark = ",")))
r2_row   <- fmt_row("Within R2",
  sapply(regs, function(r) sprintf("%.4f", r$r2w)))

table_lines <- c(
  "Dep. var: log(1 + citations)",
  "",
  header1, header2, header3,
  sep,
  coef_row, se_row,
  sep,
  n_row, r2_row,
  sep,
  "Signif: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1"
)
table_text <- paste(table_lines, collapse = "\n")
cat("\n", table_text, "\n", sep = "")

# ── combined figure: plot on the left, table on the right ----------------
plot_grob  <- ggplotGrob(p)
title_grob <- textGrob("Regression Table (4 models)",
                       gp = gpar(fontface = "bold", fontsize = 12))
table_grob <- textGrob(table_text,
                       x = unit(0.02, "npc"), y = unit(0.5, "npc"),
                       just = c("left", "centre"),
                       gp = gpar(fontfamily = "mono", fontsize = 9))

save_combined <- function(path, width, height, device_fn) {
  device_fn(path, width = width, height = height)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(
    1, 2, widths = unit.c(unit(1, "null"), unit(6, "inches")))))
  pushViewport(viewport(layout.pos.col = 1))
  grid.draw(plot_grob)
  upViewport()
  pushViewport(viewport(layout.pos.col = 2,
                        layout = grid.layout(2, 1,
                          heights = unit.c(unit(0.35, "inches"),
                                           unit(1, "null")))))
  pushViewport(viewport(layout.pos.row = 1))
  grid.draw(title_grob)
  upViewport()
  pushViewport(viewport(layout.pos.row = 2))
  grid.draw(table_grob)
  upViewport(2)
  upViewport()
  dev.off()
}

save_combined(file.path(PLOT_DIR, "rating_vs_citations_topicFE.pdf"),
              18, 9, function(f, ...) pdf(f, ...))
save_combined(file.path(PLOT_DIR, "rating_vs_citations_topicFE.png"),
              18, 9, function(f, ...) png(f, ..., units = "in", res = 300))

cat("\nSaved:", file.path(PLOT_DIR, "rating_vs_citations_topicFE.png"), "\n")
