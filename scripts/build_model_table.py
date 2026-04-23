#!/usr/bin/env python3
"""
Build a cleaned model table from historical fire points.

Inputs:
- docs/data/historical_fires.geojson (real sourced records)

Outputs:
- analysis_output/model_table.csv
- analysis_output/model_table_meta.json

This preprocessing step:
- Filters to the target season window and year range
- Deduplicates FIRMS detections into spatiotemporal event clusters
- Keeps source provenance and quality-weight fields
- Generates matched control points for supervised training
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class EventRow:
    event_id: str
    source: str
    source_class: str
    dt: datetime
    lat: float
    lon: float
    year: int
    day_of_year: int
    is_spring_or_fall: int
    season_shoulder: float
    acres: float | None
    frp: float | None
    confidence: float | None
    source_weight: float
    firms_cluster_size: int
    temp_c_max: float | None = None
    rh_min_pct: float | None = None
    wind_kph_max: float | None = None
    precip_14d_mm: float | None = None
    temp_stress: float | None = None
    humidity_stress: float | None = None
    wind_stress: float | None = None
    rain_deficit: float | None = None
    label: int
    control_strategy: str = ""


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model table from historical fire points.")
    parser.add_argument(
        "--input-geojson",
        default="docs/data/historical_fires.geojson",
        help="Path to ingested historical fires GeoJSON.",
    )
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory.")
    parser.add_argument("--start-year", type=int, default=2004, help="Start year filter.")
    parser.add_argument("--end-year", type=int, default=2023, help="End year filter.")
    parser.add_argument(
        "--season",
        choices=["all", "spring_fall"],
        default="spring_fall",
        help="Season filter for training table.",
    )
    parser.add_argument(
        "--firms-max-confidence",
        type=float,
        default=999.0,
        help="Optional upper confidence bound to allow FIRMS confidence as numeric (kept broad by default).",
    )
    parser.add_argument(
        "--firms-dedupe-km",
        type=float,
        default=2.0,
        help="Spatial threshold (km) for FIRMS event deduping.",
    )
    parser.add_argument(
        "--firms-dedupe-days",
        type=int,
        default=1,
        help="Temporal threshold (days) for FIRMS event deduping.",
    )
    parser.add_argument(
        "--controls-per-positive",
        type=int,
        default=2,
        help="Number of matched controls to generate per positive event.",
    )
    parser.add_argument(
        "--control-radius-km",
        type=float,
        default=12.0,
        help="(Legacy) Only used if --control-mode=jitter.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["hard_spatial", "jitter"],
        default="hard_spatial",
        help="hard_spatial: uniform random in bbox, min distance from any positive. "
        "jitter: legacy local jitter (easy negatives, not recommended).",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=(-97.60, 41.80, -94.90, 43.85),
        help="Study bbox for hard negatives (WGS84), same region as fire ingest.",
    )
    parser.add_argument(
        "--hard-negative-min-km",
        type=float,
        default=18.0,
        help="Controls must be at least this far (km) from every positive location.",
    )
    parser.add_argument(
        "--hard-negative-max-tries",
        type=int,
        default=400,
        help="Max random draws per control before relaxing distance slightly.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--join-weather",
        action="store_true",
        help="Join real historical weather features via Open-Meteo archive API.",
    )
    parser.add_argument(
        "--weather-sample-km",
        type=float,
        default=8.0,
        help="Snap points to this grid spacing (km) before weather fetch for caching.",
    )
    return parser.parse_args()


def source_class(source: str) -> str:
    s = (source or "").lower()
    if "firms" in s:
        return "firms"
    if "nifc" in s or "historic wildfires" in s:
        return "nifc"
    return "other"


def source_weight(source_cls: str) -> float:
    if source_cls == "nifc":
        return 1.0
    if source_cls == "firms":
        return 0.7
    return 0.5


def parse_date(props: dict) -> datetime | None:
    raw = props.get("date")
    if isinstance(raw, str) and len(raw) >= 10:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d")
        except ValueError:
            return None
    yr = props.get("year")
    if yr is None:
        return None
    try:
        # Year-only sources use mid-season proxy date for features.
        return datetime(int(yr), 7, 1)
    except ValueError:
        return None


def shoulder_score(doy: int) -> float:
    spring = math.exp(-((doy - 100) / 35.0) ** 2)
    fall = math.exp(-((doy - 290) / 35.0) ** 2)
    return max(spring, fall)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def jitter_point(lat: float, lon: float, radius_km: float) -> tuple[float, float]:
    ang = random.uniform(0.0, 2 * math.pi)
    dist = random.uniform(0.2, radius_km)
    dlat = (dist * math.sin(ang)) / 111.32
    dlon = (dist * math.cos(ang)) / (111.32 * max(0.2, math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def km_to_lat_deg(km: float) -> float:
    return km / 111.32


def km_to_lon_deg(km: float, ref_lat: float) -> float:
    return km / (111.32 * max(0.2, math.cos(math.radians(ref_lat))))


def fetch_json_with_retry(url: str, retries: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "burnable-breadbasket/model-table"})
    last_err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(min(12.0, (2**i) + 0.4))
    raise RuntimeError(f"Weather request failed after retries: {last_err}")


def weather_for_point_day(lat: float, lon: float, dt: datetime, cache: dict) -> dict:
    key = (round(lat, 4), round(lon, 4), dt.date().isoformat())
    if key in cache:
        return cache[key]
    start = dt.date().replace(day=dt.day).isoformat()
    # Pull 14-day context ending at event date
    start_dt = (dt.date().fromordinal(dt.date().toordinal() - 13)).isoformat()
    end_dt = dt.date().isoformat()
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
    url = f"https://archive-api.open-meteo.com/v1/archive?{urllib.parse.urlencode(query)}"
    payload = fetch_json_with_retry(url)
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    target = end_dt
    if target not in times:
        raise RuntimeError(f"Weather date missing in response: {target}")
    idx = times.index(target)
    temp = float(daily["temperature_2m_max"][idx])
    rh = float(daily["relative_humidity_2m_min"][idx])
    wind = float(daily["wind_speed_10m_max"][idx])
    precip_14 = float(sum(float(x) for x in daily["precipitation_sum"]))
    out = {
        "temp_c_max": temp,
        "rh_min_pct": rh,
        "wind_kph_max": wind,
        "precip_14d_mm": precip_14,
        "temp_stress": clamp((temp - 10.0) / 22.0, 0.0, 1.0),
        "humidity_stress": clamp((55.0 - rh) / 40.0, 0.0, 1.0),
        "wind_stress": clamp((wind - 12.0) / 35.0, 0.0, 1.0),
        "rain_deficit": clamp((30.0 - precip_14) / 30.0, 0.0, 1.0),
    }
    cache[key] = out
    return out


def join_weather(rows: list[EventRow], sample_km: float) -> tuple[int, int]:
    if not rows:
        return 0, 0
    ref_lat = sum(r.lat for r in rows) / len(rows)
    dlat = km_to_lat_deg(sample_km)
    dlon = km_to_lon_deg(sample_km, ref_lat)
    cache = {}
    attached = 0
    for r in rows:
        slat = round(r.lat / dlat) * dlat
        slon = round(r.lon / dlon) * dlon
        w = weather_for_point_day(slat, slon, r.dt, cache)
        r.temp_c_max = w["temp_c_max"]
        r.rh_min_pct = w["rh_min_pct"]
        r.wind_kph_max = w["wind_kph_max"]
        r.precip_14d_mm = w["precip_14d_mm"]
        r.temp_stress = w["temp_stress"]
        r.humidity_stress = w["humidity_stress"]
        r.wind_stress = w["wind_stress"]
        r.rain_deficit = w["rain_deficit"]
        attached += 1
    return attached, len(cache)


def read_events(path: Path, start_year: int, end_year: int, season: str, firms_conf_max: float) -> list[EventRow]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(
            f"Input file is empty: {path}. Re-run fire ingestion before building model table."
        )
    data = json.loads(text)
    out: list[EventRow] = []
    for i, feat in enumerate(data.get("features", []), start=1):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) != 2:
            continue
        lon = to_float(coords[0])
        lat = to_float(coords[1])
        if lon is None or lat is None:
            continue

        dt = parse_date(props)
        if dt is None:
            continue
        yr = dt.year
        if yr < start_year or yr > end_year:
            continue

        doy = dt.timetuple().tm_yday
        is_shoulder = 1 if (70 <= doy <= 160 or 250 <= doy <= 330) else 0
        if season == "spring_fall" and not is_shoulder:
            continue

        src = props.get("source") or "unknown"
        src_cls = source_class(src)
        conf = to_float(props.get("confidence"))
        if src_cls == "firms" and conf is not None and conf > firms_conf_max:
            continue

        out.append(
            EventRow(
                event_id=str(props.get("event_id") or f"evt_{i}"),
                source=src,
                source_class=src_cls,
                dt=dt,
                lat=lat,
                lon=lon,
                year=yr,
                day_of_year=doy,
                is_spring_or_fall=is_shoulder,
                season_shoulder=round(shoulder_score(doy), 6),
                acres=to_float(props.get("acres")),
                frp=to_float(props.get("frp")),
                confidence=conf,
                source_weight=source_weight(src_cls),
                firms_cluster_size=1,
                label=1,
                control_strategy="",
            )
        )
    return out


def dedupe_firms(events: list[EventRow], dist_km: float, days: int) -> list[EventRow]:
    firms = [e for e in events if e.source_class == "firms"]
    non_firms = [e for e in events if e.source_class != "firms"]
    firms.sort(key=lambda e: e.dt)

    used = [False] * len(firms)
    clusters: list[EventRow] = []
    for i, e in enumerate(firms):
        if used[i]:
            continue
        idxs = [i]
        used[i] = True
        for j in range(i + 1, len(firms)):
            if used[j]:
                continue
            o = firms[j]
            if abs((o.dt - e.dt).days) > days:
                if (o.dt - e.dt).days > days:
                    break
                continue
            if haversine_km(e.lat, e.lon, o.lat, o.lon) <= dist_km:
                idxs.append(j)
                used[j] = True

        cluster_rows = [firms[k] for k in idxs]
        lat = sum(r.lat for r in cluster_rows) / len(cluster_rows)
        lon = sum(r.lon for r in cluster_rows) / len(cluster_rows)
        frp_vals = [r.frp for r in cluster_rows if r.frp is not None]
        conf_vals = [r.confidence for r in cluster_rows if r.confidence is not None]
        c = cluster_rows[0]
        clusters.append(
            EventRow(
                event_id=f"{c.event_id}_cluster",
                source=c.source,
                source_class=c.source_class,
                dt=c.dt,
                lat=lat,
                lon=lon,
                year=c.year,
                day_of_year=c.day_of_year,
                is_spring_or_fall=c.is_spring_or_fall,
                season_shoulder=c.season_shoulder,
                acres=None,
                frp=max(frp_vals) if frp_vals else None,
                confidence=max(conf_vals) if conf_vals else None,
                source_weight=c.source_weight,
                firms_cluster_size=len(cluster_rows),
                label=1,
                control_strategy="",
            )
        )
    return non_firms + clusters


def min_dist_km_to_any(lat: float, lon: float, coords: list[tuple[float, float]]) -> float:
    return min(haversine_km(lat, lon, a, b) for a, b in coords)


def sample_hard_spatial_point(
    bbox: tuple[float, float, float, float],
    pos_coords: list[tuple[float, float]],
    min_km: float,
    max_tries: int,
) -> tuple[float, float] | None:
    min_lon, min_lat, max_lon, max_lat = bbox
    threshold = min_km
    for relax in range(8):
        t = threshold - relax * 2.5
        if t < 3.0:
            t = 3.0
        for _ in range(max(max_tries, 2500)):
            lat = random.uniform(min_lat, max_lat)
            lon = random.uniform(min_lon, max_lon)
            if min_dist_km_to_any(lat, lon, pos_coords) >= t:
                return lat, lon
    return None


def grid_sparse_cell_centers(
    bbox: tuple[float, float, float, float],
    positives: list[EventRow],
    divisions: int = 32,
) -> list[tuple[float, float]]:
    """
    Cell centers in grid cells with zero recorded positives (coarse hard negatives).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    w = max_lon - min_lon
    h = max_lat - min_lat
    if w <= 0 or h <= 0:
        return []

    occupied: set[tuple[int, int]] = set()
    for e in positives:
        i = int((e.lon - min_lon) / w * divisions)
        j = int((e.lat - min_lat) / h * divisions)
        i = max(0, min(divisions - 1, i))
        j = max(0, min(divisions - 1, j))
        occupied.add((i, j))

    centers: list[tuple[float, float]] = []
    for i in range(divisions):
        for j in range(divisions):
            if (i, j) in occupied:
                continue
            lon = min_lon + (i + 0.5) / divisions * w
            lat = min_lat + (j + 0.5) / divisions * h
            centers.append((lat, lon))
    return centers


def sample_hard_spatial_point_robust(
    bbox: tuple[float, float, float, float],
    positives: list[EventRow],
    min_km: float,
    max_tries: int,
) -> tuple[float, float]:
    pos_coords = [(e.lat, e.lon) for e in positives]
    pt = sample_hard_spatial_point(bbox, pos_coords, min_km, max_tries)
    if pt is not None:
        return pt

    sparse = grid_sparse_cell_centers(bbox, positives)
    if sparse:
        lat, lon = random.choice(sparse)
        half_cell_lat = (bbox[3] - bbox[1]) / 32.0 / 2.0
        half_cell_lon = (bbox[2] - bbox[0]) / 32.0 / 2.0
        lat += random.uniform(-half_cell_lat, half_cell_lat)
        lon += random.uniform(-half_cell_lon, half_cell_lon)
        lat = clamp(lat, bbox[1], bbox[3])
        lon = clamp(lon, bbox[0], bbox[2])
        return lat, lon

    for _ in range(5000):
        lat = random.uniform(bbox[1], bbox[3])
        lon = random.uniform(bbox[0], bbox[2])
        if min_dist_km_to_any(lat, lon, pos_coords) >= 3.0:
            return lat, lon
    raise RuntimeError("Could not sample any hard negative; check bbox and positive density.")


def generate_controls_jitter(positives: list[EventRow], per_pos: int, radius_km: float) -> list[EventRow]:
    controls: list[EventRow] = []
    for p in positives:
        for i in range(per_pos):
            lat, lon = jitter_point(p.lat, p.lon, radius_km)
            controls.append(
                EventRow(
                    event_id=f"{p.event_id}_ctrl_{i + 1}",
                    source="synthetic_control_for_training",
                    source_class="control",
                    dt=p.dt,
                    lat=lat,
                    lon=lon,
                    year=p.year,
                    day_of_year=p.day_of_year,
                    is_spring_or_fall=p.is_spring_or_fall,
                    season_shoulder=p.season_shoulder,
                    acres=None,
                    frp=None,
                    confidence=None,
                    source_weight=1.0,
                    firms_cluster_size=0,
                    label=0,
                    control_strategy="jitter_local",
                )
            )
    return controls


def generate_controls_hard_spatial(
    positives: list[EventRow],
    per_pos: int,
    bbox: tuple[float, float, float, float],
    min_km: float,
    max_tries: int,
) -> list[EventRow]:
    by_year_season: dict[tuple[int, int], list[EventRow]] = {}
    for e in positives:
        key = (e.year, e.is_spring_or_fall)
        by_year_season.setdefault(key, []).append(e)

    controls: list[EventRow] = []
    for p in positives:
        for i in range(per_pos):
            lat, lon = sample_hard_spatial_point_robust(bbox, positives, min_km, max_tries)
            pool = by_year_season.get((p.year, p.is_spring_or_fall)) or positives
            tmpl = random.choice(pool)
            controls.append(
                EventRow(
                    event_id=f"{p.event_id}_hardctrl_{i + 1}",
                    source="synthetic_control_for_training",
                    source_class="control",
                    dt=tmpl.dt,
                    lat=lat,
                    lon=lon,
                    year=tmpl.year,
                    day_of_year=tmpl.day_of_year,
                    is_spring_or_fall=tmpl.is_spring_or_fall,
                    season_shoulder=tmpl.season_shoulder,
                    acres=None,
                    frp=None,
                    confidence=None,
                    source_weight=1.0,
                    firms_cluster_size=0,
                    label=0,
                    control_strategy="hard_spatial_bbox_min_dist",
                )
            )
    return controls


def write_csv(path: Path, rows: list[EventRow]) -> None:
    fields = [
        "event_id",
        "label",
        "source",
        "source_class",
        "source_weight",
        "date",
        "year",
        "day_of_year",
        "is_spring_or_fall",
        "season_shoulder",
        "latitude",
        "longitude",
        "acres",
        "frp",
        "confidence",
        "firms_cluster_size",
        "temp_c_max",
        "rh_min_pct",
        "wind_kph_max",
        "precip_14d_mm",
        "temp_stress",
        "humidity_stress",
        "wind_stress",
        "rain_deficit",
        "control_strategy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "event_id": r.event_id,
                    "label": r.label,
                    "source": r.source,
                    "source_class": r.source_class,
                    "source_weight": round(r.source_weight, 4),
                    "date": r.dt.date().isoformat(),
                    "year": r.year,
                    "day_of_year": r.day_of_year,
                    "is_spring_or_fall": r.is_spring_or_fall,
                    "season_shoulder": round(r.season_shoulder, 6),
                    "latitude": round(r.lat, 6),
                    "longitude": round(r.lon, 6),
                    "acres": None if r.acres is None else round(r.acres, 3),
                    "frp": None if r.frp is None else round(r.frp, 3),
                    "confidence": None if r.confidence is None else round(r.confidence, 3),
                    "firms_cluster_size": r.firms_cluster_size,
                    "temp_c_max": None if r.temp_c_max is None else round(r.temp_c_max, 3),
                    "rh_min_pct": None if r.rh_min_pct is None else round(r.rh_min_pct, 3),
                    "wind_kph_max": None if r.wind_kph_max is None else round(r.wind_kph_max, 3),
                    "precip_14d_mm": None if r.precip_14d_mm is None else round(r.precip_14d_mm, 3),
                    "temp_stress": None if r.temp_stress is None else round(r.temp_stress, 5),
                    "humidity_stress": None if r.humidity_stress is None else round(r.humidity_stress, 5),
                    "wind_stress": None if r.wind_stress is None else round(r.wind_stress, 5),
                    "rain_deficit": None if r.rain_deficit is None else round(r.rain_deficit, 5),
                    "control_strategy": r.control_strategy or "",
                }
            )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    in_path = Path(args.input_geojson)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = read_events(in_path, args.start_year, args.end_year, args.season, args.firms_max_confidence)
    before = len(events)
    events = dedupe_firms(events, args.firms_dedupe_km, args.firms_dedupe_days)
    after = len(events)

    positives = events
    bbox = tuple(args.bbox)
    if args.control_mode == "hard_spatial":
        controls = generate_controls_hard_spatial(
            positives,
            args.controls_per_positive,
            bbox,
            args.hard_negative_min_km,
            args.hard_negative_max_tries,
        )
    else:
        controls = generate_controls_jitter(
            positives, args.controls_per_positive, args.control_radius_km
        )
    rows = positives + controls
    weather_attached = 0
    weather_unique_fetches = 0
    if args.join_weather:
        weather_attached, weather_unique_fetches = join_weather(rows, args.weather_sample_km)

    csv_path = out_dir / "model_table.csv"
    meta_path = out_dir / "model_table_meta.json"
    write_csv(csv_path, rows)

    counts = {}
    for r in positives:
        counts[r.source_class] = counts.get(r.source_class, 0) + 1

    meta = {
        "input_geojson": str(in_path),
        "output_csv": str(csv_path),
        "year_range": [args.start_year, args.end_year],
        "season_filter": args.season,
        "positives_before_firms_dedupe": before,
        "positives_after_firms_dedupe": after,
        "controls_per_positive": args.controls_per_positive,
        "total_rows": len(rows),
        "positive_rows": len(positives),
        "control_rows": len(controls),
        "positive_source_breakdown": counts,
        "synthetic_incidents_included": False,
        "controls_are_synthetic_training_negatives": True,
        "control_mode": args.control_mode,
        "bbox_wgs84": list(bbox),
        "hard_negative_min_km": args.hard_negative_min_km
        if args.control_mode == "hard_spatial"
        else None,
        "join_weather": args.join_weather,
        "weather_sample_km": args.weather_sample_km if args.join_weather else None,
        "weather_rows_attached": weather_attached,
        "weather_unique_fetches": weather_unique_fetches,
        "weather_source": "Open-Meteo archive API" if args.join_weather else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
