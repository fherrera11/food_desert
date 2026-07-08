import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "input_locations.csv"
OUTPUT_JSON = "restaurant_data.json"

# If X/Y are longitude/latitude in WGS84, keep this True.
# If X/Y are projected coordinates, such as Web Mercator or State Plane, set this False.
USE_LOCATION_BIAS = True

# Search radius around each X/Y coordinate, in meters.
LOCATION_BIAS_RADIUS_METERS = 500.0

# Keep this None for broader matching.
# If all rows are definitely restaurants, you can set:
# INCLUDED_TYPE = "restaurant"
INCLUDED_TYPE = None

# Pause between requests. Helps avoid hammering the API.
REQUEST_DELAY_SECONDS = 0.10

# Set to an integer for testing, for example 10. Set to None for full file.
MAX_ROWS = None

TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# Photo settings.
# The Places Text Search response returns photo metadata. If FETCH_PHOTO_URI is True,
# this script also calls Place Photos (New) to get a short-lived image URL.
FETCH_PHOTO_URI = True
PHOTO_MAX_WIDTH_PX = 640
PHOTO_MAX_HEIGHT_PX = 400

FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.priceRange",
    "places.priceLevel",
    "places.googleMapsUri",
    # Returns photo metadata only. The actual image is retrieved separately
    # through Place Photos (New) using the returned photos[].name value.
    "places.photos",
])


# ============================================================
# SETUP
# ============================================================

load_dotenv()
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_MAPS_API_KEY in .env file.")


# ============================================================
# HELPERS
# ============================================================

def normalize_header(value: str) -> str:
    """
    Normalizes headers so these can all be matched:
    USER_current_rating
    USER_current rating
    user current rating
    """
    return (
        value.strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )


def make_column_lookup(fieldnames: list[str]) -> dict[str, str]:
    """Returns a normalized-header -> original-header lookup."""
    return {normalize_header(name): name for name in fieldnames}


def get_value(row: dict[str, str], col_lookup: dict[str, str], *possible_names: str) -> str:
    """Case/underscore/space-insensitive column getter."""
    for name in possible_names:
        original = col_lookup.get(normalize_header(name))
        if original is not None:
            return (row.get(original) or "").strip()
    return ""


def parse_float(value: Any) -> float | None:
    if value is None:
        return None

    cleaned = str(value).strip().replace(",", "")

    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def is_probably_lon_lat(x: float | None, y: float | None) -> bool:
    """Basic sanity check for longitude/latitude."""
    if x is None or y is None:
        return False

    return -180 <= x <= 180 and -90 <= y <= 90


def money_to_float(money: dict[str, Any] | None) -> float | None:
    """
    Converts Google's Money object to a float.

    Example:
    {"currencyCode": "USD", "units": "10"} -> 10.0
    {"currencyCode": "USD", "units": "10", "nanos": 500000000} -> 10.5
    """
    if not money:
        return None

    units = float(money.get("units", 0) or 0)
    nanos = float(money.get("nanos", 0) or 0) / 1_000_000_000

    return units + nanos


def format_price_range(place: dict[str, Any]) -> tuple[str, str, str, str]:
    """
    Returns:
    price_label, price_bucket, start_price, end_price

    Prefer Google's numeric priceRange.
    Fallback to priceLevel.
    """
    price_range = place.get("priceRange") or {}

    start_price = money_to_float(price_range.get("startPrice"))
    end_price = money_to_float(price_range.get("endPrice"))

    if start_price is not None and end_price is not None:
        label = f"${start_price:.0f}-${end_price:.0f}"
        return label, price_bucket_from_range(start_price, end_price), f"{start_price:.0f}", f"{end_price:.0f}"

    if start_price is not None:
        label = f"${start_price:.0f}+"
        return label, price_bucket_from_range(start_price, None), f"{start_price:.0f}", ""

    price_level = place.get("priceLevel", "")

    price_level_labels = {
        "PRICE_LEVEL_FREE": "Free",
        "PRICE_LEVEL_INEXPENSIVE": "$",
        "PRICE_LEVEL_MODERATE": "$$",
        "PRICE_LEVEL_EXPENSIVE": "$$$",
        "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
    }

    fallback_label = price_level_labels.get(price_level, "Price unavailable")
    return fallback_label, fallback_label, "", ""


def price_bucket_from_range(start: float, end: float | None) -> str:
    """
    Converts a numeric Google priceRange into broad user-facing buckets.
    Keeps the bucket intentionally coarse for map display/filtering.
    """
    if end is None:
        if start >= 30:
            return "$30+"
        if start >= 20:
            return "$20-$30+"
        if start >= 10:
            return "$10-$20+"
        return "$0-$10+"

    if end <= 10:
        return "$0-$10"
    if start < 20 and end <= 20:
        return "$10-$20" if start >= 10 else "$0-$20"
    if start < 30 and end <= 30:
        return "$20-$30" if start >= 20 else "$10-$30"
    return "$30+"


def quality_label_from_score(score: str) -> str:
    """
    Source inspection quality score mapping:
    0 = Unsatisfactory
    1 = Satisfactory
    2 = Good
    """
    cleaned = str(score).strip()

    if cleaned == "0":
        return "Unsatisfactory"
    if cleaned == "1":
        return "Satisfactory"
    if cleaned == "2":
        return "Good"

    return ""


def build_search_query(
    facility: str,
    match_addr: str,
    postal: str,
    city: str = "",
    region: str = "",
) -> str:
    """Builds a strong Google search query from your existing geocoder fields."""
    parts = [facility, match_addr, city, region, postal]
    return " ".join(part for part in parts if part).strip()


def haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Distance between two WGS84 coordinate pairs in meters."""
    radius_m = 6_371_000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius_m * c


def google_text_search(
    query: str,
    x_lon: float | None,
    y_lat: float | None,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    body: dict[str, Any] = {
        "textQuery": query,
        "maxResultCount": 1,
        "languageCode": "en",
        "regionCode": "US",
    }

    if INCLUDED_TYPE:
        body["includedType"] = INCLUDED_TYPE

    if USE_LOCATION_BIAS and is_probably_lon_lat(x_lon, y_lat):
        body["locationBias"] = {
            "circle": {
                "center": {
                    "latitude": y_lat,
                    "longitude": x_lon,
                },
                "radius": LOCATION_BIAS_RADIUS_METERS,
            }
        }

    response = requests.post(
        TEXT_SEARCH_URL,
        headers=headers,
        json=body,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()
    places = data.get("places", [])

    if not places:
        return {}

    return places[0]


def primary_photo_fields(place: dict[str, Any]) -> dict[str, Any]:
    """
    Extracts the first Google Places photo metadata record.

    Important: photos[].name is the reusable photo resource name returned by
    Places. The photoUri returned by Place Photos is short-lived, so keep the
    photo name and author attributions in the JSON for future refreshes.
    """
    photos = place.get("photos") or []
    if not photos:
        return {
            "google_photo_name": "",
            "google_photo_width_px": None,
            "google_photo_height_px": None,
            "google_photo_author_attributions": [],
        }

    first = photos[0] or {}
    return {
        "google_photo_name": first.get("name", ""),
        "google_photo_width_px": first.get("widthPx"),
        "google_photo_height_px": first.get("heightPx"),
        "google_photo_author_attributions": first.get("authorAttributions") or [],
    }


def google_photo_media(photo_name: str) -> dict[str, Any]:
    """
    Calls Place Photos (New) and returns a short-lived renderable photo URI.
    """
    if not photo_name:
        return {"google_photo_media_name": "", "google_photo_uri": ""}

    url = f"https://places.googleapis.com/v1/{photo_name}/media"
    params = {
        "key": API_KEY,
        "maxWidthPx": PHOTO_MAX_WIDTH_PX,
        "maxHeightPx": PHOTO_MAX_HEIGHT_PX,
        "skipHttpRedirect": "true",
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()
    return {
        "google_photo_media_name": data.get("name", ""),
        "google_photo_uri": data.get("photoUri", ""),
    }


def selected_source_fields(row: dict[str, str], col_lookup: dict[str, str]) -> dict[str, Any]:
    """
    Pulls only the source columns you said you care about.
    Handles exact names and common spelling/header variants.
    """
    match_addr = get_value(row, col_lookup, "Match_addr", "match_addr")
    postal = get_value(row, col_lookup, "Postal", "postal")
    x = get_value(row, col_lookup, "X", "x")
    y = get_value(row, col_lookup, "Y", "y")
    facility = get_value(row, col_lookup, "USER_facility", "facility")
    current_rating = get_value(
        row,
        col_lookup,
        "USER_current_rating",
        "USER_current rating",
        "current_rating",
    )
    last_inspection_date = get_value(
        row,
        col_lookup,
        "USER_last_inspection_date",
        "USER_last inspection date",
        "last_inspection_date",
    )
    quality_score = get_value(
        row,
        col_lookup,
        "quality_score",
        "quality score",
    )

    return {
        "Match_addr": match_addr,
        "Postal": postal,
        "X": x,
        "Y": y,
        "USER_facility": facility,
        "USER_current_rating": current_rating,
        "USER_last_inspection_date": last_inspection_date,
        "quality_score": quality_score,
        "quality_label": quality_label_from_score(quality_score),
    }


def build_feature(
    source: dict[str, Any],
    google_fields: dict[str, Any],
    source_row_number: int,
) -> dict[str, Any]:
    x_lon = parse_float(source.get("X"))
    y_lat = parse_float(source.get("Y"))

    geometry = None
    if is_probably_lon_lat(x_lon, y_lat):
        geometry = {
            "longitude": x_lon,
            "latitude": y_lat,
            "spatialReference": {"wkid": 4326},
        }

    attributes = {
        "source_row_number": source_row_number,
        **source,
        **google_fields,
    }

    return {
        "type": "Feature",
        "geometry": geometry,
        "attributes": attributes,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    features: list[dict[str, Any]] = []
    source_row_count = 0

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)

        if not reader.fieldnames:
            raise RuntimeError("Input CSV has no headers.")

        col_lookup = make_column_lookup(reader.fieldnames)

        for index, row in enumerate(reader, start=1):
            if MAX_ROWS is not None and index > MAX_ROWS:
                break

            source_row_count += 1
            source = selected_source_fields(row, col_lookup)

            match_addr = source["Match_addr"]
            postal = source["Postal"]
            x_raw = source["X"]
            y_raw = source["Y"]
            facility = source["USER_facility"]

            city = get_value(row, col_lookup, "City", "USER_city", "city")
            region = get_value(row, col_lookup, "RegionAbbr", "Region", "USER_state", "state")

            x_lon = parse_float(x_raw)
            y_lat = parse_float(y_raw)

            search_query = build_search_query(
                facility=facility,
                match_addr=match_addr,
                postal=postal,
                city=city,
                region=region,
            )

            print(f"[{index}] Searching: {search_query}")

            google_fields: dict[str, Any] = {
                "google_status": "",
                "google_place_id": "",
                "google_name": "",
                "google_address": "",
                "google_lat": None,
                "google_lng": None,
                "google_distance_meters": None,
                "google_rating": None,
                "google_review_count": None,
                "google_price_label": "",
                "google_price_bucket": "",
                "google_price_level": "",
                "google_price_range_start": None,
                "google_price_range_end": None,
                "google_maps_uri": "",
                "google_photo_name": "",
                "google_photo_width_px": None,
                "google_photo_height_px": None,
                "google_photo_author_attributions": [],
                "google_photo_media_name": "",
                "google_photo_uri": "",
                "google_photo_error": "",
                "google_search_query": search_query,
                "google_error": "",
            }

            if not search_query:
                google_fields["google_status"] = "SKIPPED"
                google_fields["google_error"] = "No facility/address data available for search."
                features.append(build_feature(source, google_fields, index))
                continue

            try:
                place = google_text_search(search_query, x_lon, y_lat)

                if not place:
                    google_fields["google_status"] = "NOT_FOUND"
                    google_fields["google_error"] = "No Google Places match returned."
                    features.append(build_feature(source, google_fields, index))
                    time.sleep(REQUEST_DELAY_SECONDS)
                    continue

                price_label, price_bucket, price_start, price_end = format_price_range(place)
                location = place.get("location") or {}
                photo_fields = primary_photo_fields(place)

                if FETCH_PHOTO_URI and photo_fields["google_photo_name"]:
                    try:
                        photo_fields.update(
                            google_photo_media(photo_fields["google_photo_name"])
                        )
                    except Exception as photo_exc:
                        photo_fields["google_photo_media_name"] = ""
                        photo_fields["google_photo_uri"] = ""
                        photo_fields["google_photo_error"] = str(photo_exc)

                google_lat = parse_float(location.get("latitude"))
                google_lng = parse_float(location.get("longitude"))

                distance_meters = None
                if (
                    is_probably_lon_lat(x_lon, y_lat)
                    and is_probably_lon_lat(google_lng, google_lat)
                ):
                    distance_meters = round(haversine_meters(x_lon, y_lat, google_lng, google_lat), 1)

                google_fields.update({
                    "google_status": "OK",
                    "google_place_id": place.get("id", ""),
                    "google_name": (place.get("displayName") or {}).get("text", ""),
                    "google_address": place.get("formattedAddress", ""),
                    "google_lat": google_lat,
                    "google_lng": google_lng,
                    "google_distance_meters": distance_meters,
                    "google_rating": place.get("rating"),
                    "google_review_count": place.get("userRatingCount"),
                    "google_price_label": price_label,
                    "google_price_bucket": price_bucket,
                    "google_price_level": place.get("priceLevel", ""),
                    "google_price_range_start": parse_float(price_start),
                    "google_price_range_end": parse_float(price_end),
                    "google_maps_uri": place.get("googleMapsUri", ""),
                    **photo_fields,
                    "google_error": "",
                })

            except requests.HTTPError as exc:
                response_text = ""
                status_code = ""

                if exc.response is not None:
                    status_code = str(exc.response.status_code)
                    response_text = exc.response.text

                google_fields["google_status"] = "ERROR"
                google_fields["google_error"] = f"HTTP {status_code}: {response_text}"

            except Exception as exc:
                google_fields["google_status"] = "ERROR"
                google_fields["google_error"] = str(exc)

            features.append(build_feature(source, google_fields, index))
            time.sleep(REQUEST_DELAY_SECONDS)

    mapped_count = sum(1 for feature in features if feature.get("geometry"))
    google_ok_count = sum(1 for feature in features if feature["attributes"].get("google_status") == "OK")

    output_data = {
        "type": "RestaurantInspectionGoogleJoin",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": INPUT_CSV,
        "output_file": OUTPUT_JSON,
        "metadata": {
            "source_row_count": source_row_count,
            "feature_count": len(features),
            "mapped_feature_count": mapped_count,
            "google_ok_count": google_ok_count,
            "coordinate_assumption": "X=longitude, Y=latitude, WGS84/EPSG:4326",
            "quality_score_mapping": {
                "0": "Unsatisfactory",
                "1": "Satisfactory",
                "2": "Good",
            },
            "google_fields_requested": FIELD_MASK.split(","),
        },
        "features": features,
    }

    # ensure_ascii=False preserves accented characters and other UTF-8 text naturally.
    # JSON escaping still protects quotes, backslashes, and control characters correctly.
    with open(OUTPUT_JSON, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=2)

    print()
    print(f"Done. Wrote {len(features)} features to: {OUTPUT_JSON}")
    print(f"Mapped features with valid X/Y: {mapped_count}")
    print(f"Google OK matches: {google_ok_count}")


if __name__ == "__main__":
    main()
