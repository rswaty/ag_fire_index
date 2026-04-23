#!/usr/bin/env python3
"""
Train and evaluate a baseline logistic model from model_table.csv.

Inputs:
- analysis_output/model_table.csv

Outputs:
- analysis_output/model_metrics.json
- analysis_output/model_coefficients.csv
- analysis_output/model_scored_test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline logistic model.")
    parser.add_argument(
        "--input-csv",
        default="analysis_output/model_table.csv",
        help="Input model table CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Output directory for model artifacts.",
    )
    parser.add_argument(
        "--test-start-year",
        type=int,
        default=2021,
        help="Holdout test split starts from this year (inclusive). Used when --test-split year.",
    )
    parser.add_argument(
        "--test-split",
        choices=["year", "spatial"],
        default="year",
        help="year: time holdout. spatial: longitude holdout (generalizes across geography).",
    )
    parser.add_argument(
        "--spatial-test-min-lon",
        type=float,
        default=-96.0,
        help="If test-split=spatial: test rows have longitude >= this value.",
    )
    parser.add_argument(
        "--norm-bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=(-97.60, 41.80, -94.90, 43.85),
        help="BBox for normalizing lat/lon features (match ingest region).",
    )
    parser.add_argument("--epochs", type=int, default=1800, help="Gradient steps.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--l2", type=float, default=0.0005, help="L2 regularization.")
    return parser.parse_args()


def to_float(v: str | None, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def row_to_features(row: dict, norm_bbox: tuple[float, float, float, float]) -> list[float]:
    """
    Features usable at inference time for a gridded risk map (no label/source leakage).
    """
    min_lon, min_lat, max_lon, max_lat = norm_bbox
    lat = to_float(row.get("latitude"), (min_lat + max_lat) / 2)
    lon = to_float(row.get("longitude"), (min_lon + max_lon) / 2)
    lat_n = (lat - min_lat) / max(max_lat - min_lat, 1e-6)
    lon_n = (lon - min_lon) / max(max_lon - min_lon, 1e-6)

    doy = to_float(row.get("day_of_year"), 180.0)
    angle = 2.0 * math.pi * doy / 365.25
    sin_doy = math.sin(angle)
    cos_doy = math.cos(angle)

    temp_stress = to_float(row.get("temp_stress"), 0.0)
    humidity_stress = to_float(row.get("humidity_stress"), 0.0)
    wind_stress = to_float(row.get("wind_stress"), 0.0)
    rain_deficit = to_float(row.get("rain_deficit"), 0.0)
    has_weather = 1.0 if (row.get("temp_stress") not in (None, "", "null")) else 0.0

    return [
        1.0,
        lat_n,
        lon_n,
        to_float(row.get("season_shoulder"), 0.0),
        to_float(row.get("is_spring_or_fall"), 0.0),
        sin_doy,
        cos_doy,
        has_weather,
        temp_stress,
        humidity_stress,
        wind_stress,
        rain_deficit,
    ]


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic(X: np.ndarray, y: np.ndarray, epochs: int, lr: float, l2: float) -> np.ndarray:
    w = np.zeros(X.shape[1], dtype=float)
    n = float(X.shape[0])
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = (X.T @ (p - y)) / n
        reg = l2 * w
        reg[0] = 0.0  # don't regularize intercept
        w -= lr * (grad + reg)
    return w


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    pos = int(np.sum(y_sorted == 1))
    neg = int(len(y_sorted) - pos)
    if pos == 0 or neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_sorted) + 1)
    rank_sum = float(np.sum(ranks[y_sorted == 1]))
    return (rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg)


def logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_hat = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_hat == 1)))
    fp = int(np.sum((y_true == 0) & (y_hat == 1)))
    tn = int(np.sum((y_true == 0) & (y_hat == 0)))
    fn = int(np.sum((y_true == 1) & (y_hat == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_bbox = tuple(args.norm_bbox)
    rows = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = int(to_float(row.get("label"), 0.0))
            year = int(to_float(row.get("year"), 0.0))
            lon = to_float(row.get("longitude"), 0.0)
            x = row_to_features(row, norm_bbox)
            rows.append((row, x, y, year, lon))

    if not rows:
        raise ValueError("No rows found in model table.")

    if args.test_split == "year":
        train = [r for r in rows if r[3] < args.test_start_year]
        test = [r for r in rows if r[3] >= args.test_start_year]
        split_note = f"year>={args.test_start_year} is test"
    else:
        thr = args.spatial_test_min_lon
        train = [r for r in rows if r[4] < thr]
        test = [r for r in rows if r[4] >= thr]
        split_note = f"lon>={thr} is test (spatial holdout)"

    if not train or not test:
        raise ValueError(
            f"Train/test split is empty ({split_note}). Adjust split parameters."
        )

    X_train = np.array([r[1] for r in train], dtype=float)
    y_train = np.array([r[2] for r in train], dtype=float)
    X_test = np.array([r[1] for r in test], dtype=float)
    y_test = np.array([r[2] for r in test], dtype=float)

    w = train_logistic(X_train, y_train, args.epochs, args.learning_rate, args.l2)
    p_train = sigmoid(X_train @ w)
    p_test = sigmoid(X_test @ w)

    feature_names = [
        "intercept",
        "lat_norm",
        "lon_norm",
        "season_shoulder",
        "is_spring_or_fall",
        "sin_doy",
        "cos_doy",
        "has_weather",
        "temp_stress",
        "humidity_stress",
        "wind_stress",
        "rain_deficit",
    ]

    metrics = {
        "input_csv": str(input_csv),
        "test_split": args.test_split,
        "test_split_description": split_note,
        "test_start_year": args.test_start_year,
        "spatial_test_min_lon": args.spatial_test_min_lon
        if args.test_split == "spatial"
        else None,
        "norm_bbox": list(norm_bbox),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": round(float(np.mean(y_train)), 4),
        "test_positive_rate": round(float(np.mean(y_test)), 4),
        "train_auc": round(auc_roc(y_train, p_train), 4),
        "test_auc": round(auc_roc(y_test, p_test), 4),
        "train_logloss": round(logloss(y_train, p_train), 4),
        "test_logloss": round(logloss(y_test, p_test), 4),
        "test_classification_at_0_5": classification_metrics(y_test, p_test, 0.5),
        "notes": "Logistic model: calendar + location only (no source/FIRMS metadata leakage).",
    }

    metrics_path = output_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    coef_path = output_dir / "model_coefficients.csv"
    with coef_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "coefficient", "odds_multiplier"])
        writer.writeheader()
        for name, coef in zip(feature_names, w):
            writer.writerow(
                {
                    "feature": name,
                    "coefficient": round(float(coef), 6),
                    "odds_multiplier": round(float(math.exp(coef)), 6),
                }
            )

    scored_test_path = output_dir / "model_scored_test.csv"
    with scored_test_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(test[0][0].keys()) + ["pred_prob"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (raw_row, _, _, _, _), prob in zip(test, p_test):
            row = dict(raw_row)
            row["pred_prob"] = round(float(prob), 6)
            writer.writerow(row)

    print(f"Wrote {metrics_path}")
    print(f"Wrote {coef_path}")
    print(f"Wrote {scored_test_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
