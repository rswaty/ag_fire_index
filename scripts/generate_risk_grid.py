#!/usr/bin/env python3
"""
Generate a chunked NW Iowa + SE South Dakota wildfire risk grid for 2026-03-28.

This script is real-data-first and defaults to strict mode:
- Pull real historical weather from Open-Meteo Archive API.
- Fail if real weather cannot be fetched.
- Optional flag `--allow-fallback` enables synthetic fallback for offline testing.
"""

from __future__ import annotations

import json
import math
import random
import time
import argparse
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path


OUT_INDEX_PATH = Path("docs/data/risk_chunks/index.geojson")
OUT_CHUNK_DIR = Path("docs/data/risk_chunks/chunks")

# Approximate NW Iowa + SE South Dakota extent for this prototype.
# Less south, more north, and extends west into SD.
MIN_LON, MAX_LON = -97.60, -94.90
MIN_LAT, MAX_LAT = 41.80, 43.85

# Approximate target cell size in kilometers.
CELL_SIZE_KM = 1.0

# Weather sample spacing in kilometers for API fetches.
# 1 km cell values are bilinearly interpolated from this lattice.
WEATHER_SAMPLE_KM = 8.0
CHUNK_SIZE_DEG = 0.20

TARGET_DATE = "2026-03-28"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate chunked historical risk grid.")
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT),
        help="Bounding box for grid generation (WGS84).",
    )
    p.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow synthetic weather fallback when real fetch fails.",
    )
    return p.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def cell_polygon(lon0: float, lon1: float, lat0: float, lat1: float) -> list:
    return [[
        [lon0, lat0],
        [lon1, lat0],
        [lon1, lat1],
        [lon0, lat1],
        [lon0, lat0],
    ]]


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "woodbury-risk-prototype/1.0"})
    last_error = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as err:
            last_error = err
            sleep_s = min(10.0, (2**attempt) + random.uniform(0.0, 0.6))
            time.sleep(sleep_s)
    raise urllib.error.URLError(f"Request failed after retries: {last_error}")


def _open_meteo_cell(lat: float, lon: float) -> dict:
    # Pull day-level weather around the target date to estimate fuel dryness risk.
    query = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "start_date": "2026-03-14",
        "end_date": TARGET_DATE,
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
    url = f"https://archive-api.open-meteo.com/v1/archive?{urllib.parse.urlencode(query)}"
    payload = _fetch_json(url)
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    if TARGET_DATE not in times:
        raise ValueError(f"Target date {TARGET_DATE} not in API response")

    idx = times.index(TARGET_DATE)
    temp_c = float(daily["temperature_2m_max"][idx])
    rh_min = float(daily["relative_humidity_2m_min"][idx])
    wind_max = float(daily["wind_speed_10m_max"][idx])
    precip_series = [float(x) for x in daily["precipitation_sum"]]
    precip_14d = float(sum(precip_series))

    # Normalize proxies to 0..1 using pragmatic thresholds.
    temp_stress = clamp((temp_c - 10.0) / 22.0, 0.0, 1.0)
    humidity_stress = clamp((55.0 - rh_min) / 40.0, 0.0, 1.0)
    wind_stress = clamp((wind_max - 12.0) / 35.0, 0.0, 1.0)
    rain_deficit = clamp((30.0 - precip_14d) / 30.0, 0.0, 1.0)

    risk = int(round(100.0 * (0.32 * temp_stress + 0.23 * humidity_stress + 0.25 * wind_stress + 0.20 * rain_deficit)))
    risk = int(clamp(risk, 0, 100))

    return {
        "risk_index": risk,
        "temp_c_max": round(temp_c, 2),
        "rh_min_pct": round(rh_min, 2),
        "wind_max_kph": round(wind_max, 2),
        "precip_14d_mm": round(precip_14d, 2),
        "temp_stress": round(temp_stress, 3),
        "humidity_stress": round(humidity_stress, 3),
        "wind_stress": round(wind_stress, 3),
        "rain_deficit": round(rain_deficit, 3),
        "data_mode": "open_meteo_archive",
    }


def _fallback_cell(lat: float, lon: float) -> dict:
    # Offline fallback so the app remains usable if network calls fail.
    north = (lat - MIN_LAT) / (MAX_LAT - MIN_LAT)
    east = (lon - MIN_LON) / (MAX_LON - MIN_LON)
    temp_stress = clamp(0.45 + 0.35 * (1.0 - north), 0.0, 1.0)
    humidity_stress = clamp(0.35 + 0.30 * east, 0.0, 1.0)
    wind_stress = clamp(0.30 + 0.30 * (1.0 - north), 0.0, 1.0)
    rain_deficit = clamp(0.40 + 0.25 * east, 0.0, 1.0)
    risk = int(round(100.0 * (0.32 * temp_stress + 0.23 * humidity_stress + 0.25 * wind_stress + 0.20 * rain_deficit)))
    risk = int(clamp(risk, 0, 100))
    return {
        "risk_index": risk,
        "temp_c_max": None,
        "rh_min_pct": None,
        "wind_max_kph": None,
        "precip_14d_mm": None,
        "temp_stress": round(temp_stress, 3),
        "humidity_stress": round(humidity_stress, 3),
        "wind_stress": round(wind_stress, 3),
        "rain_deficit": round(rain_deficit, 3),
        "data_mode": "offline_fallback",
    }


def _km_to_lat_deg(km: float) -> float:
    return km / 111.32


def _km_to_lon_deg(km: float, ref_lat_deg: float) -> float:
    return km / (111.32 * math.cos(math.radians(ref_lat_deg)))


def _round_cell(value: float, step: float) -> float:
    return round(value / step) * step


def _weather_at_point(lat: float, lon: float, cache: dict, allow_fallback: bool) -> dict:
    key = (round(lat, 4), round(lon, 4))
    if key in cache:
        return cache[key]
    try:
        obs = _open_meteo_cell(lat, lon)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as err:
        if not allow_fallback:
            raise RuntimeError(
                f"Real weather fetch failed at lat={lat:.4f}, lon={lon:.4f}: {err}"
            ) from err
        obs = _fallback_cell(lat, lon)
    cache[key] = obs
    return obs


def _obs_to_stress(obs: dict) -> dict:
    return {
        "temp_stress": float(obs["temp_stress"]),
        "humidity_stress": float(obs["humidity_stress"]),
        "wind_stress": float(obs["wind_stress"]),
        "rain_deficit": float(obs["rain_deficit"]),
    }


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _bilinear(a00: float, a10: float, a01: float, a11: float, tx: float, ty: float) -> float:
    return _lerp(_lerp(a00, a10, tx), _lerp(a01, a11, tx), ty)


def _interpolate_stress(samples: dict, sx0: int, sx1: int, sy0: int, sy1: int, tx: float, ty: float) -> dict:
    p00 = samples[(sy0, sx0)]
    p10 = samples[(sy0, sx1)]
    p01 = samples[(sy1, sx0)]
    p11 = samples[(sy1, sx1)]
    out = {}
    for k in ("temp_stress", "humidity_stress", "wind_stress", "rain_deficit"):
        out[k] = _bilinear(p00[k], p10[k], p01[k], p11[k], tx, ty)
    return out


def _percentile_rank(sorted_values: list[float], value: float) -> float:
    if not sorted_values:
        return 0.5
    lo = 0
    hi = len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    idx = lo
    if len(sorted_values) == 1:
        return 0.5
    return idx / (len(sorted_values) - 1)


def _quantile_sorted(sorted_values: list[float], q: float) -> float:
    """q in [0, 1]; sorted_values must be sorted ascending."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo_i = int(math.floor(pos))
    hi_i = int(math.ceil(pos))
    if lo_i == hi_i:
        return sorted_values[lo_i]
    w = pos - lo_i
    return sorted_values[lo_i] * (1.0 - w) + sorted_values[hi_i] * w


def _chunk_key(lat: float, lon: float) -> tuple[int, int]:
    y = math.floor((lat - MIN_LAT) / CHUNK_SIZE_DEG)
    x = math.floor((lon - MIN_LON) / CHUNK_SIZE_DEG)
    return y, x


def _chunk_bounds(y: int, x: int) -> tuple[float, float, float, float]:
    min_lat = MIN_LAT + y * CHUNK_SIZE_DEG
    min_lon = MIN_LON + x * CHUNK_SIZE_DEG
    max_lat = min(min_lat + CHUNK_SIZE_DEG, MAX_LAT)
    max_lon = min(min_lon + CHUNK_SIZE_DEG, MAX_LON)
    return min_lon, min_lat, max_lon, max_lat


def build_chunked_features(allow_fallback: bool) -> tuple[dict, dict]:
    chunks = {}
    pending_cells = []
    ref_lat = (MIN_LAT + MAX_LAT) / 2.0
    dy = _km_to_lat_deg(CELL_SIZE_KM)
    dx = _km_to_lon_deg(CELL_SIZE_KM, ref_lat)
    wy = _km_to_lat_deg(WEATHER_SAMPLE_KM)
    wx = _km_to_lon_deg(WEATHER_SAMPLE_KM, ref_lat)

    nx = math.ceil((MAX_LON - MIN_LON) / dx)
    ny = math.ceil((MAX_LAT - MIN_LAT) / dy)
    weather_cache = {}

    # Build weather sample lattice for interpolation.
    nsx = math.ceil((MAX_LON - MIN_LON) / wx) + 1
    nsy = math.ceil((MAX_LAT - MIN_LAT) / wy) + 1
    sample_stress = {}
    for sy in range(nsy):
        for sx in range(nsx):
            sample_lon = min(MIN_LON + sx * wx, MAX_LON)
            sample_lat = min(MIN_LAT + sy * wy, MAX_LAT)
            obs = _weather_at_point(sample_lat, sample_lon, weather_cache, allow_fallback)
            sample_stress[(sy, sx)] = _obs_to_stress(obs)
    data_mode = "open_meteo_archive" if not allow_fallback else "mixed_or_fallback"

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
            stress = _interpolate_stress(sample_stress, sx0, sx1, sy0, sy1, tx, ty)
            raw_risk = 100.0 * (
                0.32 * stress["temp_stress"]
                + 0.23 * stress["humidity_stress"]
                + 0.25 * stress["wind_stress"]
                + 0.20 * stress["rain_deficit"]
            )

            cy, cx = _chunk_key(c_lat, c_lon)
            key = f"{cy}_{cx}"
            pending_cells.append(
                (
                    key,
                    raw_risk,
                    {
                        "cell_id": f"RGN-{iy + 1:03d}-{ix + 1:03d}",
                        "week_ending": TARGET_DATE,
                        "temp_c_max": None,
                        "rh_min_pct": None,
                        "wind_max_kph": None,
                        "precip_14d_mm": None,
                        "temp_stress": round(stress["temp_stress"], 3),
                        "humidity_stress": round(stress["humidity_stress"], 3),
                        "wind_stress": round(stress["wind_stress"], 3),
                        "rain_deficit": round(stress["rain_deficit"], 3),
                        "data_mode": data_mode,
                        "weather_sampling_km": WEATHER_SAMPLE_KM,
                        "cell_resolution_km": CELL_SIZE_KM,
                        "interpolation": "bilinear",
                        "generated_on": str(date.today()),
                    },
                    {"type": "Polygon", "coordinates": cell_polygon(lon0, lon1, lat0, lat1)},
                )
            )

    sorted_raw = sorted(raw for _, raw, _, _ in pending_cells)
    p_lo = _quantile_sorted(sorted_raw, 0.05)
    p_hi = _quantile_sorted(sorted_raw, 0.95)
    span = max(p_hi - p_lo, 1e-6)
    for key, raw_risk, props, geom in pending_cells:
        rank = _percentile_rank(sorted_raw, raw_risk)
        # Linear quantile stretch preserves geographic structure. (Pure rank remap
        # turns any smooth gradient into parallel "diagonal stripe" color bands.)
        t = clamp((raw_risk - p_lo) / span, 0.0, 1.0)
        stretched = int(round(clamp(15.0 + 80.0 * t, 0.0, 100.0)))
        props["risk_index_raw"] = int(round(clamp(raw_risk, 0.0, 100.0)))
        props["risk_index"] = stretched
        props["risk_percentile"] = round(rank, 3)
        props["risk_scaling"] = "regional_quantile_5_95_linear"

        if key not in chunks:
            chunks[key] = []
        chunks[key].append({"type": "Feature", "properties": props, "geometry": geom})

    chunk_index_features = []
    for key, features in chunks.items():
        cy, cx = [int(part) for part in key.split("_")]
        min_lon, min_lat, max_lon, max_lat = _chunk_bounds(cy, cx)
        chunk_index_features.append(
            {
                "type": "Feature",
                "properties": {
                    "chunk_id": key,
                    "path": f"./data/risk_chunks/chunks/{key}.geojson",
                    "feature_count": len(features),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]],
                },
            }
        )

    index_data = {
        "type": "FeatureCollection",
        "name": "risk_chunk_index",
        "features": chunk_index_features,
        "meta": {
            "target_date": TARGET_DATE,
            "cell_size_km": CELL_SIZE_KM,
            "weather_sample_km": WEATHER_SAMPLE_KM,
            "chunk_size_deg": CHUNK_SIZE_DEG,
            "allow_fallback": allow_fallback,
            "interpolation": "bilinear_to_1km_cells",
            "display_scaling": "quantile_5_95_linear_not_rank",
        },
    }
    return chunks, index_data


def main() -> None:
    global MIN_LON, MIN_LAT, MAX_LON, MAX_LAT
    args = parse_args()
    MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = tuple(args.bbox)
    allow_fallback = bool(args.allow_fallback)
    chunks, index_data = build_chunked_features(allow_fallback)
    OUT_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for key, features in chunks.items():
        chunk_data = {"type": "FeatureCollection", "name": f"risk_chunk_{key}", "features": features}
        chunk_path = OUT_CHUNK_DIR / f"{key}.geojson"
        chunk_path.write_text(json.dumps(chunk_data, separators=(",", ":")), encoding="utf-8")

    OUT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDEX_PATH.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    mode = "fallback-allowed" if allow_fallback else "strict-real-data"
    print(f"Wrote {OUT_INDEX_PATH} and {len(chunks)} chunk files ({mode})")


if __name__ == "__main__":
    main()
