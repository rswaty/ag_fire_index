#!/usr/bin/env python3
"""
Historical fire feature analysis and baseline risk model calibration.

This script builds an event-vs-control dataset from historical fire records,
fetches weather features, optionally joins NDVI history, and reports model skill.

Input fire CSV columns:
- fire_id (optional)
- date (YYYY-MM-DD)
- latitude
- longitude

Optional NDVI CSV columns:
- latitude
- longitude
- date (YYYY-MM-DD)
- ndvi
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


@dataclass
class Event:
    fire_id: str
    dt: date
    latitude: float
    longitude: float
    label: int


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze historical fires for model calibration.")
    parser.add_argument("--fires-csv", required=True, help="Path to historical fires CSV.")
    parser.add_argument("--output-dir", default="analysis_output", help="Directory for outputs.")
    parser.add_argument("--controls-per-fire", type=int, default=3, help="Control samples per fire.")
    parser.add_argument(
        "--control-radius-km",
        type=float,
        default=12.0,
        help="Radius for random control points around each fire.",
    )
    parser.add_argument(
        "--ndvi-csv",
        default="",
        help="Optional NDVI history CSV (lat,lon,date,ndvi) for fuel anomaly features.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def read_fire_events(path: Path) -> list[Event]:
    events: list[Event] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            dt = datetime.strptime(row["date"], "%Y-%m-%d").date()
            events.append(
                Event(
                    fire_id=row.get("fire_id") or f"fire_{i + 1}",
                    dt=dt,
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    label=1,
                )
            )
    if not events:
        raise ValueError("No fire rows found in input CSV.")
    return events


def jitter_point_km(lat: float, lon: float, radius_km: float) -> tuple[float, float]:
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(0.1, radius_km)
    dlat = (dist * math.sin(angle)) / 111.32
    dlon = (dist * math.cos(angle)) / (111.32 * max(0.2, math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def fetch_json_with_retry(url: str, retries: int = 4) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "burnable-breadbasket/analysis"})
    last_err = None
    for n in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep((2**n) + random.uniform(0, 0.5))
    raise RuntimeError(f"Failed API request after retries: {last_err}")


def weather_features_for_event(lat: float, lon: float, target: date, cache: dict) -> dict:
    key = (round(lat, 4), round(lon, 4), target.isoformat())
    if key in cache:
        return cache[key]

    start = (target - timedelta(days=14)).isoformat()
    end = target.isoformat()
    query = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "start_date": start,
        "end_date": end,
        "daily": ",".join(
            [
                "temperature_2m_max",
                "precipitation_sum",
                "relative_humidity_2m_min",
                "wind_speed_10m_max",
            ]
        ),
        "timezone": "America/Chicago",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(query)}"
    payload = fetch_json_with_retry(url)
    daily = payload["daily"]
    times = daily["time"]
    idx = times.index(target.isoformat())

    temp = float(daily["temperature_2m_max"][idx])
    rh_min = float(daily["relative_humidity_2m_min"][idx])
    wind = float(daily["wind_speed_10m_max"][idx])
    precip_14 = sum(float(x) for x in daily["precipitation_sum"])

    feat = {
        "temp_c_max": temp,
        "rh_min_pct": rh_min,
        "wind_kph_max": wind,
        "precip_14d_mm": precip_14,
        "temp_stress": clamp((temp - 10.0) / 22.0, 0.0, 1.0),
        "humidity_stress": clamp((55.0 - rh_min) / 40.0, 0.0, 1.0),
        "wind_stress": clamp((wind - 12.0) / 35.0, 0.0, 1.0),
        "rain_deficit": clamp((30.0 - precip_14) / 30.0, 0.0, 1.0),
    }
    cache[key] = feat
    return feat


def season_features(dt: date) -> dict:
    doy = dt.timetuple().tm_yday
    # shoulder-season priors for ag burning windows
    spring_score = math.exp(-((doy - 100) / 35) ** 2)
    fall_score = math.exp(-((doy - 290) / 35) ** 2)
    return {
        "day_of_year": doy,
        "season_shoulder": max(spring_score, fall_score),
        "is_spring_or_fall": 1 if (70 <= doy <= 160 or 250 <= doy <= 330) else 0,
    }


def read_optional_ndvi(path: str) -> list[dict]:
    if not path:
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "lat": float(row["latitude"]),
                    "lon": float(row["longitude"]),
                    "date": datetime.strptime(row["date"], "%Y-%m-%d").date(),
                    "ndvi": float(row["ndvi"]),
                }
            )
    return rows


def nearest_ndvi(ndvi_rows: list[dict], lat: float, lon: float, dt: date) -> tuple[float | None, float | None]:
    if not ndvi_rows:
        return None, None

    best = None
    best_dist = float("inf")
    for row in ndvi_rows:
        days = abs((row["date"] - dt).days)
        if days > 21:
            continue
        d = math.hypot((row["lat"] - lat), (row["lon"] - lon)) + 0.01 * days
        if d < best_dist:
            best = row
            best_dist = d
    if best is None:
        return None, None

    doy_vals = [r["ndvi"] for r in ndvi_rows if abs((r["date"].timetuple().tm_yday - dt.timetuple().tm_yday)) <= 7]
    seasonal_med = statistics.median(doy_vals) if doy_vals else best["ndvi"]
    return best["ndvi"], best["ndvi"] - seasonal_med


def baseline_score(row: dict) -> float:
    fuel_component = 0.0
    if row["ndvi"] is not None:
        fuel_component = clamp((0.55 - row["ndvi"]) / 0.45, 0.0, 1.0)
    if row["ndvi_anomaly"] is not None:
        fuel_component = clamp(fuel_component + clamp(-row["ndvi_anomaly"], 0.0, 0.4), 0.0, 1.0)

    return (
        0.30 * row["temp_stress"]
        + 0.22 * row["humidity_stress"]
        + 0.23 * row["wind_stress"]
        + 0.15 * row["rain_deficit"]
        + 0.10 * row["season_shoulder"]
        + 0.10 * fuel_component
    )


def auc_roc(rows: list[dict]) -> float:
    scored = sorted(((r["score"], r["label"]) for r in rows), key=lambda x: x[0])
    pos = sum(1 for _, y in scored if y == 1)
    neg = len(scored) - pos
    if pos == 0 or neg == 0:
        return 0.5

    rank_sum = 0.0
    for i, (_, y) in enumerate(scored, start=1):
        if y == 1:
            rank_sum += i
    return (rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    fire_events = read_fire_events(Path(args.fires_csv))
    ndvi_rows = read_optional_ndvi(args.ndvi_csv)
    weather_cache = {}

    rows: list[dict] = []
    for event in fire_events:
        sample_events = [event]
        for i in range(args.controls_per_fire):
            lat, lon = jitter_point_km(event.latitude, event.longitude, args.control_radius_km)
            sample_events.append(
                Event(
                    fire_id=f"{event.fire_id}_ctrl_{i + 1}",
                    dt=event.dt,
                    latitude=lat,
                    longitude=lon,
                    label=0,
                )
            )

        for sample in sample_events:
            w = weather_features_for_event(sample.latitude, sample.longitude, sample.dt, weather_cache)
            s = season_features(sample.dt)
            ndvi, ndvi_anom = nearest_ndvi(ndvi_rows, sample.latitude, sample.longitude, sample.dt)
            row = {
                "sample_id": sample.fire_id,
                "label": sample.label,
                "date": sample.dt.isoformat(),
                "latitude": round(sample.latitude, 6),
                "longitude": round(sample.longitude, 6),
                **w,
                **s,
                "ndvi": None if ndvi is None else round(ndvi, 4),
                "ndvi_anomaly": None if ndvi_anom is None else round(ndvi_anom, 4),
            }
            row["score"] = round(baseline_score(row), 5)
            rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = output_dir / "historical_fire_features.csv"
    write_csv(dataset_csv, rows)

    fire_scores = [r["score"] for r in rows if r["label"] == 1]
    ctrl_scores = [r["score"] for r in rows if r["label"] == 0]
    summary = {
        "rows": len(rows),
        "fires": sum(1 for r in rows if r["label"] == 1),
        "controls": sum(1 for r in rows if r["label"] == 0),
        "auc_roc": round(auc_roc(rows), 4),
        "mean_score_fire": round(statistics.mean(fire_scores), 4) if fire_scores else None,
        "mean_score_control": round(statistics.mean(ctrl_scores), 4) if ctrl_scores else None,
        "uses_ndvi": bool(args.ndvi_csv),
        "notes": "AUC from baseline weighted risk score. Replace with ML model after validating features.",
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {dataset_csv}")
    print(f"Wrote {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
