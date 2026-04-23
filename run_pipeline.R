# One-pass pipeline runner for burnable_breadbasket.
# Run from RStudio (repo root):
#   source("run_pipeline.R")
#
# Design notes:
# - Keeps full study area workflows from Python scripts.
# - Reduces model training rows AFTER table build via stratified sampling.
# - Uses one entrypoint so it can be called later from GitHub Actions.

options(stringsAsFactors = FALSE)

cfg <- list(
  # Core execution
  python = "python3",
  seed = 42L,
  run_ingest = TRUE,
  run_build_table = TRUE,
  run_train = TRUE,
  run_risk_grid = TRUE,
  run_forecast = TRUE,
  area_shapefile = "data/demo_counties.shp",
  use_shapefile_bbox = TRUE,

  # Ingest settings
  start_year = 2004L,
  end_year = 2023L,
  include_firms = TRUE,
  firms_sources = "VIIRS_SNPP_SP,VIIRS_NOAA20_SP",
  firms_map_key = Sys.getenv("FIRMS_MAP_KEY", ""),

  # Build table settings
  season = "spring_fall",
  control_mode = "hard_spatial",
  controls_per_positive = 1L,   # lower for speed
  hard_negative_min_km = 14,    # lower for speed/robustness
  weather_sample_km = 8,
  join_weather = TRUE,

  # Downsample settings (full-area preserved; fewer training plots)
  downsample_table = TRUE,
  sampled_model_table = "analysis_output/model_table_sampled.csv",
  max_sample_rows = 8000L,
  positive_fraction = 0.50,     # target class mix in sampled table

  # Train settings
  test_split = "year",
  test_start_year = 2021L,
  epochs = 1800L,
  learning_rate = 0.05,
  l2 = 0.0005,

  # Forecast settings
  forecast_start = "2026-04-22",
  forecast_end = "2026-04-29",
  forecast_weather_sample_km = 6
)

run_cmd <- function(bin, args, label) {
  cat(sprintf("\n[%s] %s %s\n", label, bin, paste(args, collapse = " ")))
  status <- system2(bin, args = args)
  if (!identical(status, 0L)) {
    stop(sprintf("Step failed: %s (exit=%s)", label, status), call. = FALSE)
  }
}

must_exist <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Expected file not found: %s", path), call. = FALSE)
  }
}

derive_bbox <- function(cfg) {
  if (!isTRUE(cfg$use_shapefile_bbox)) return(NULL)
  shp <- cfg$area_shapefile
  must_exist(shp)
  if (!requireNamespace("sf", quietly = TRUE)) {
    stop("Package 'sf' is required to derive bbox from shapefile.", call. = FALSE)
  }
  geom <- sf::st_read(shp, quiet = TRUE)
  geom <- sf::st_transform(geom, 4326)
  bb <- sf::st_bbox(geom)
  c(unname(bb["xmin"]), unname(bb["ymin"]), unname(bb["xmax"]), unname(bb["ymax"]))
}

thin_model_table <- function(input_csv, output_csv, max_rows, positive_fraction = 0.5, seed = 42L) {
  cat(sprintf("\n[downsample] Reading %s\n", input_csv))
  d <- read.csv(input_csv, check.names = FALSE)
  if (!("label" %in% names(d))) {
    stop("model_table.csv missing 'label' column", call. = FALSE)
  }
  if (nrow(d) <= max_rows) {
    cat(sprintf("[downsample] Table already <= max_rows (%d <= %d), copying.\n", nrow(d), max_rows))
    write.csv(d, output_csv, row.names = FALSE)
    return(invisible(output_csv))
  }

  if (!("year" %in% names(d))) d$year <- 0L
  d$label <- as.integer(d$label)
  d$year <- as.integer(d$year)

  set.seed(seed)

  pos_idx <- which(d$label == 1L)
  neg_idx <- which(d$label == 0L)
  if (length(pos_idx) == 0 || length(neg_idx) == 0) {
    stop("Downsample failed: need both positive and negative rows", call. = FALSE)
  }

  target_pos <- max(1L, min(length(pos_idx), as.integer(round(max_rows * positive_fraction))))
  target_neg <- max(1L, min(length(neg_idx), max_rows - target_pos))

  sample_stratified <- function(indices, target_n) {
    rows <- d[indices, , drop = FALSE]
    g <- split(seq_len(nrow(rows)), rows$year)
    weights <- vapply(g, length, integer(1))
    alloc <- floor(target_n * (weights / sum(weights)))
    remainder <- target_n - sum(alloc)
    if (remainder > 0) {
      ord <- order(weights, decreasing = TRUE)
      take <- ord[seq_len(min(remainder, length(ord)))]
      alloc[take] <- alloc[take] + 1L
    }
    out <- integer(0)
    i <- 1L
    for (nm in names(g)) {
      idx <- g[[nm]]
      n_take <- min(length(idx), alloc[i])
      if (n_take > 0) out <- c(out, sample(idx, n_take))
      i <- i + 1L
    }
    rownames(rows)[out]
  }

  pos_rows <- sample_stratified(pos_idx, target_pos)
  neg_rows <- sample_stratified(neg_idx, target_neg)
  keep <- c(pos_rows, neg_rows)
  out <- d[keep, , drop = FALSE]

  if (nrow(out) > max_rows) {
    out <- out[sample(seq_len(nrow(out)), max_rows), , drop = FALSE]
  }

  cat(sprintf("[downsample] Wrote %s rows to %s\n", nrow(out), output_csv))
  write.csv(out, output_csv, row.names = FALSE)
  invisible(output_csv)
}

dir.create("analysis_output", recursive = TRUE, showWarnings = FALSE)
dir.create("docs/data", recursive = TRUE, showWarnings = FALSE)

bbox_vals <- derive_bbox(cfg)
if (!is.null(bbox_vals)) {
  cat(sprintf(
    "[area] Using shapefile bbox from %s: %.5f, %.5f, %.5f, %.5f\n",
    cfg$area_shapefile, bbox_vals[1], bbox_vals[2], bbox_vals[3], bbox_vals[4]
  ))
}

if (isTRUE(cfg$run_ingest)) {
  ingest_args <- c(
    "scripts/ingest_historical_fires.py",
    "--start-year", cfg$start_year,
    "--end-year", cfg$end_year
  )
  if (!is.null(bbox_vals)) {
    ingest_args <- c(ingest_args, "--bbox", bbox_vals)
  }
  if (isTRUE(cfg$include_firms)) {
    if (!nzchar(cfg$firms_map_key)) {
      stop("FIRMS_MAP_KEY is empty but include_firms=TRUE", call. = FALSE)
    }
    ingest_args <- c(
      ingest_args,
      "--include-firms",
      "--firms-map-key", cfg$firms_map_key,
      "--firms-sources", cfg$firms_sources
    )
  }
  run_cmd(cfg$python, ingest_args, "ingest")
  must_exist("docs/data/historical_fires.geojson")
}

if (isTRUE(cfg$run_build_table)) {
  bt_args <- c(
    "scripts/build_model_table.py",
    "--input-geojson", "docs/data/historical_fires.geojson",
    "--output-dir", "analysis_output",
    "--start-year", cfg$start_year,
    "--end-year", cfg$end_year,
    "--season", cfg$season,
    "--control-mode", cfg$control_mode,
    "--controls-per-positive", cfg$controls_per_positive,
    "--hard-negative-min-km", cfg$hard_negative_min_km,
    "--weather-sample-km", cfg$weather_sample_km,
    "--seed", cfg$seed
  )
  if (!is.null(bbox_vals)) {
    bt_args <- c(bt_args, "--bbox", bbox_vals)
  }
  if (isTRUE(cfg$join_weather)) bt_args <- c(bt_args, "--join-weather")
  run_cmd(cfg$python, bt_args, "build_model_table")
  must_exist("analysis_output/model_table.csv")
}

train_input <- "analysis_output/model_table.csv"
if (isTRUE(cfg$downsample_table)) {
  must_exist(train_input)
  thin_model_table(
    input_csv = train_input,
    output_csv = cfg$sampled_model_table,
    max_rows = cfg$max_sample_rows,
    positive_fraction = cfg$positive_fraction,
    seed = cfg$seed
  )
  train_input <- cfg$sampled_model_table
}

if (isTRUE(cfg$run_train)) {
  tr_args <- c(
    "scripts/train_baseline_model.py",
    "--input-csv", train_input,
    "--output-dir", "analysis_output",
    "--test-split", cfg$test_split,
    "--test-start-year", cfg$test_start_year,
    "--epochs", cfg$epochs,
    "--learning-rate", cfg$learning_rate,
    "--l2", cfg$l2
  )
  if (!is.null(bbox_vals)) {
    tr_args <- c(tr_args, "--norm-bbox", bbox_vals)
  }
  run_cmd(cfg$python, tr_args, "train")
  must_exist("analysis_output/model_metrics.json")
}

if (isTRUE(cfg$run_risk_grid)) {
  rg_args <- c("scripts/generate_risk_grid.py")
  if (!is.null(bbox_vals)) {
    rg_args <- c(rg_args, "--bbox", bbox_vals)
  }
  run_cmd(cfg$python, rg_args, "risk_grid")
  must_exist("docs/data/risk_chunks/index.geojson")
}

if (isTRUE(cfg$run_forecast)) {
  fc_args <- c(
    "scripts/generate_next_week_forecast.py",
    "--start-date", cfg$forecast_start,
    "--end-date", cfg$forecast_end,
    "--weather-sample-km", cfg$forecast_weather_sample_km
  )
  if (!is.null(bbox_vals)) {
    fc_args <- c(fc_args, "--bbox", bbox_vals)
  }
  run_cmd(cfg$python, fc_args, "forecast")
  must_exist("docs/data/forecast_chunks/index.geojson")
}

cat("\nPipeline complete.\n")
cat(sprintf("- train input: %s\n", train_input))
cat("- metrics: analysis_output/model_metrics.json\n")
cat("- map data: docs/data/risk_chunks/index.geojson, docs/data/forecast_chunks/index.geojson\n")
