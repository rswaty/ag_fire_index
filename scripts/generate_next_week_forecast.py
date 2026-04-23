#!/usr/bin/env python3
"""
Generate a next-week wildfire risk forecast layer (real forecast weather only).

Output:
- docs/data/forecast_chunks/index.geojson
- docs/data/forecast_chunks/chunks/*.geojson

Default forecast window:
- 2026-04-22 .. 2026-04-29
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path


OUT_INDEX_PATH = Path("docs/data/forecast_chunks/index.geojson")
OUT_CHUNK_DIR = Path("docs/data/forecast_chunks/chunks")

MIN_LON, MAX_LON = -97.60, -94.90
MIN_LAT, MAX_LAT = 41.80, 43.85
CELL_SIZE_KM = 1.0
DEFAULT_WEATHER_SAMPLE_KM = 6.0
CHUNK_SIZE_DEG = 0.20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate weekly forecast risk grid.")
    p.add_argument("--start-date", default="2026-04-22", help="Forecast start date (YYYY-MM-DD).")
    p.add_argument("--end-date", default="2026-04-29", help="Forecast end date (YYYY-MM-DD).")
    p.add_argument(
        "--weather-sample-km",
        type=float,
        default=DEFAULT_WEATHER_SAMPLE_KM,
        help="Forecast weather sample spacing in km (smaller = smoother, slower).",
    )
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT),
        help="Bounding box for grid generation (WGS84).",
    )
    return p.parse_args()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def quantile_sorted(sorted_vals: list[float], q: float) -> float:
    """q in [0, 1]; sorted_vals must be sorted ascending."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo_i = int(math.floor(pos))
    hi_i = int(math.ceil(pos))
    if lo_i == hi_i:
        return sorted_vals[lo_i]
    w = pos - lo_i
    return sorted_vals[lo_i] * (1.0 - w) + sorted_vals[hi_i] * w


def percentile_rank(sorted_vals: list[float], value: float) -> float:
    if not sorted_vals:
        return 0.5
    if len(sorted_vals) == 1:
        return 0.5
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo / (len(sorted_vals) - 1)


def km_to_lat_deg(km: float) -> float:
    return km / 111.32


def km_to_lon_deg(km: float, lat_deg: float) -> float:
    return km / (111.32 * max(0.2, math.cos(math.radians(lat_deg))))


def fetch_json_with_retry(url: str, retries: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "burnable-breadbasket/forecast"})
    last_err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(min(10.0, (2**i) + 0.4))
    raise RuntimeError(f"Forecast request failed after retries: {last_err}")


def open_meteo_forecast(lat: float, lon: float, start_dt: str, end_dt: str, cache: dict) -> dict:
    key = (round(lat, 4), round(lon, 4), start_dt, end_dt)
    if key in cache:
        return cache[key]
    query = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "start_date": start_dt,
        "end_date": end_dt,
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
    url = f"https://api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(query)}"
    payload = fetch_json_with_retry(url)
    cache[key] = payload
    return payload


def chunk_key(lat: float, lon: float) -> tuple[int, int]:
    y = math.floor((lat - MIN_LAT) / CHUNK_SIZE_DEG)
    x = math.floor((lon - MIN_LON) / CHUNK_SIZE_DEG)
    return y, x


def chunk_bounds(y: int, x: int) -> tuple[float, float, float, float]:
    min_lat = MIN_LAT + y * CHUNK_SIZE_DEG
    min_lon = MIN_LON + x * CHUNK_SIZE_DEG
    max_lat = min(min_lat + CHUNK_SIZE_DEG, MAX_LAT)
    max_lon = min(min_lon + CHUNK_SIZE_DEG, MAX_LON)
    return min_lon, min_lat, max_lon, max_lat


def cell_polygon(lon0: float, lon1: float, lat0: float, lat1: float) -> list:
    return [[
        [lon0, lat0],
        [lon1, lat0],
        [lon1, lat1],
        [lon0, lat1],
        [lon0, lat0],
    ]]


def risk_from_weather(temp_c: float, rh_min: float, wind_kph: float, precip_7d: float) -> float:
    temp_stress = clamp((temp_c - 10.0) / 22.0, 0.0, 1.0)
    humidity_stress = clamp((55.0 - rh_min) / 40.0, 0.0, 1.0)
    wind_stress = clamp((wind_kph - 12.0) / 35.0, 0.0, 1.0)
    rain_deficit = clamp((20.0 - precip_7d) / 20.0, 0.0, 1.0)
    return 100.0 * (0.32 * temp_stress + 0.23 * humidity_stress + 0.25 * wind_stress + 0.20 * rain_deficit)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def bilinear(a00: float, a10: float, a01: float, a11: float, tx: float, ty: float) -> float:
    return lerp(lerp(a00, a10, tx), lerp(a01, a11, tx), ty)


def main() -> None:
    global MIN_LON, MIN_LAT, MAX_LON, MAX_LAT
    args = parse_args()
    MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = tuple(args.bbox)
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end-date must be >= start-date")

    ref_lat = (MIN_LAT + MAX_LAT) / 2.0
    dy = km_to_lat_deg(CELL_SIZE_KM)
    dx = km_to_lon_deg(CELL_SIZE_KM, ref_lat)
    weather_sample_km = float(args.weather_sample_km)
    wy = km_to_lat_deg(weather_sample_km)
    wx = km_to_lon_deg(weather_sample_km, ref_lat)

    nx = math.ceil((MAX_LON - MIN_LON) / dx)
    ny = math.ceil((MAX_LAT - MIN_LAT) / dy)
    nsx = math.ceil((MAX_LON - MIN_LON) / wx) + 1
    nsy = math.ceil((MAX_LAT - MIN_LAT) / wy) + 1

    weather_cache = {}
    # Fetch sample lattice forecasts once.
    sample_daily = {}
    for sy in range(nsy):
        for sx in range(nsx):
            lon = min(MIN_LON + sx * wx, MAX_LON)
            lat = min(MIN_LAT + sy * wy, MAX_LAT)
            payload = open_meteo_forecast(lat, lon, args.start_date, args.end_date, weather_cache)
            sample_daily[(sy, sx)] = payload["daily"]

    chunks: dict[str, list] = {}
    all_weekly = []
    for iy in range(ny):
        for ix in range(nx):
            lon0 = MIN_LON + ix * dx
            lon1 = min(lon0 + dx, MAX_LON)
            lat0 = MIN_LAT + iy * dy
            lat1 = min(lat0 + dy, MAX_LAT)
            c_lon = (lon0 + lon1) / 2
            c_lat = (lat0 + lat1) / 2

            fx = (c_lon - MIN_LON) / wx
            fy = (c_lat - MIN_LAT) / wy
            sx0 = int(max(0, min(nsx - 1, math.floor(fx))))
            sy0 = int(max(0, min(nsy - 1, math.floor(fy))))
            sx1 = int(max(0, min(nsx - 1, sx0 + 1)))
            sy1 = int(max(0, min(nsy - 1, sy0 + 1)))
            tx = 0.0 if sx1 == sx0 else (fx - sx0)
            ty = 0.0 if sy1 == sy0 else (fy - sy0)

            d00 = sample_daily[(sy0, sx0)]
            d10 = sample_daily[(sy0, sx1)]
            d01 = sample_daily[(sy1, sx0)]
            d11 = sample_daily[(sy1, sx1)]
            n_days = len(d00["temperature_2m_max"])
            temps: list[float] = []
            rhs: list[float] = []
            winds: list[float] = []
            precs: list[float] = []
            for di in range(n_days):
                t = bilinear(
                    float(d00["temperature_2m_max"][di]),
                    float(d10["temperature_2m_max"][di]),
                    float(d01["temperature_2m_max"][di]),
                    float(d11["temperature_2m_max"][di]),
                    tx,
                    ty,
                )
                rh = bilinear(
                    float(d00["relative_humidity_2m_min"][di]),
                    float(d10["relative_humidity_2m_min"][di]),
                    float(d01["relative_humidity_2m_min"][di]),
                    float(d11["relative_humidity_2m_min"][di]),
                    tx,
                    ty,
                )
                w = bilinear(
                    float(d00["wind_speed_10m_max"][di]),
                    float(d10["wind_speed_10m_max"][di]),
                    float(d01["wind_speed_10m_max"][di]),
                    float(d11["wind_speed_10m_max"][di]),
                    tx,
                    ty,
                )
                pr = bilinear(
                    float(d00["precipitation_sum"][di]),
                    float(d10["precipitation_sum"][di]),
                    float(d01["precipitation_sum"][di]),
                    float(d11["precipitation_sum"][di]),
                    tx,
                    ty,
                )
                temps.append(t)
                rhs.append(rh)
                winds.append(w)
                precs.append(pr)

            weekly_precip = float(sum(precs))
            daily_scores = [risk_from_weather(t, rh, w, weekly_precip) for t, rh, w in zip(temps, rhs, winds)]
            weekly_max = int(round(clamp(max(daily_scores), 0.0, 100.0)))
            weekly_mean = float(sum(daily_scores) / len(daily_scores))
            all_weekly.append(weekly_mean)

            cy, cx = chunk_key(c_lat, c_lon)
            key = f"{cy}_{cx}"
            chunks.setdefault(key, []).append(
                {
                    "type": "Feature",
                    "properties": {
                        "cell_id": f"FC-{iy + 1:03d}-{ix + 1:03d}",
                        "forecast_start": args.start_date,
                        "forecast_end": args.end_date,
                        "risk_weekly_max": weekly_max,
                        "risk_weekly_mean": round(weekly_mean, 3),
                        "temp_max_mean_c": round(sum(temps) / len(temps), 3),
                        "rh_min_mean_pct": round(sum(rhs) / len(rhs), 3),
                        "wind_max_mean_kph": round(sum(winds) / len(winds), 3),
                        "precip_sum_mm": round(weekly_precip, 3),
                        "weather_source": "Open-Meteo forecast API (bilinear interpolated)",
                        "interpolation": "bilinear",
                        "generated_on": date.today().isoformat(),
                    },
                    "geometry": {"type": "Polygon", "coordinates": cell_polygon(lon0, lon1, lat0, lat1)},
                }
            )

    # Stretch weekly mean for map contrast without rank remapping (rank remap
    # creates artificial diagonal banding on smooth fields).
    sorted_means = sorted(all_weekly)
    p_lo = quantile_sorted(sorted_means, 0.05)
    p_hi = quantile_sorted(sorted_means, 0.95)
    span = max(p_hi - p_lo, 1e-6)

    for feats in chunks.values():
        for f in feats:
            m = float(f["properties"]["risk_weekly_mean"])
            r = percentile_rank(sorted_means, m)
            t = clamp((m - p_lo) / span, 0.0, 1.0)
            f["properties"]["risk_display"] = int(round(clamp(15 + 80 * t, 0, 100)))
            f["properties"]["risk_percentile"] = round(r, 3)
            f["properties"]["risk_display_scaling"] = "quantile_5_95_linear"

    OUT_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for key, feats in chunks.items():
        (OUT_CHUNK_DIR / f"{key}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "name": f"forecast_chunk_{key}", "features": feats}, separators=(",", ":")),
            encoding="utf-8",
        )

    idx_features = []
    for key, feats in chunks.items():
        cy, cx = [int(x) for x in key.split("_")]
        min_lon, min_lat, max_lon, max_lat = chunk_bounds(cy, cx)
        idx_features.append(
            {
                "type": "Feature",
                "properties": {
                    "chunk_id": key,
                    "path": f"./data/forecast_chunks/chunks/{key}.geojson",
                    "feature_count": len(feats),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]],
                },
            }
        )

    forecast_start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    # MODIS Terra 8-day NDVI in GIBS is requested by period start date; use a
    # composite immediately before the forecast week for vegetation context.
    ndvi_gibs_time = (forecast_start - timedelta(days=8)).isoformat()

    OUT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDEX_PATH.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": "forecast_chunk_index",
                "features": idx_features,
                "meta": {
                    "forecast_start": args.start_date,
                    "forecast_end": args.end_date,
                    "ndvi_gibs_time": ndvi_gibs_time,
                    "cell_size_km": CELL_SIZE_KM,
                    "weather_sample_km": weather_sample_km,
                    "interpolation": "bilinear",
                    "synthetic_data": False,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_INDEX_PATH} and {len(chunks)} forecast chunk files")


if __name__ == "__main__":
    main()
