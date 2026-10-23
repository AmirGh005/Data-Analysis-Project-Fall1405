"""
Real Data Fetcher and Image Downloader for Track C (iNaturalist Open Data - Fungi Subset).
Connects directly to the official iNaturalist v1 API to:
  1. Resolve canonical iNaturalist taxon_ids for all 40 target fungal species.
  2. Fetch real research-grade observation and photo records across North America and Europe.
  3. Download the actual mushroom photographs to disk (`data/raw/images/`) using a thread pool
     and verify image integrity using Pillow.

Note: Synthetic generation is strictly isolated behind `--mode dev_synthetic_only` and is NEVER
triggered in default/real/auto runs.
"""

import os
import sys
import time
import yaml
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_real_inat_taxon_ids(
    species_list: List[Dict[str, Any]], session: requests.Session
) -> List[Dict[str, Any]]:
    """
    Queries `https://api.inaturalist.org/v1/taxa` to retrieve the official iNaturalist taxon_id
    for each of the 40 target fungal species.
    """
    logger.info("Resolving official iNaturalist taxon_id for all 40 target species...")
    taxa_url = "https://api.inaturalist.org/v1/taxa"
    resolved = []

    for idx, sp in enumerate(tqdm(species_list, desc="Resolving Taxa"), start=1):
        sci_name = sp["scientific_name"]
        real_taxon_id = None
        for attempt in range(3):
            try:
                resp = session.get(
                    taxa_url,
                    params={"q": sci_name, "rank": "species", "is_active": "true", "per_page": 5},
                    timeout=12,
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    for r in results:
                        if r.get("name", "").lower() == sci_name.lower():
                            real_taxon_id = int(r["id"])
                            break
                    if real_taxon_id is None and results:
                        real_taxon_id = int(results[0]["id"])
                    break
                time.sleep(1.0)
            except Exception as exc:
                logger.warning(f"Attempt {attempt+1} failed for taxon {sci_name}: {exc}")
                time.sleep(1.5)

        if real_taxon_id is None:
            raise RuntimeError(
                f"Could not resolve real iNaturalist taxon_id for '{sci_name}'. "
                f"Please check your internet connection."
            )

        sp_copy = dict(sp)
        sp_copy["species_idx"] = idx
        sp_copy["taxon_id"] = real_taxon_id
        sp_copy["genus"] = sci_name.split()[0]
        sp_copy["is_toxic_or_inedible"] = int(sp["risk_level"] > 0)
        sp_copy["is_deadly"] = int(sp["risk_level"] == 3)
        resolved.append(sp_copy)
        time.sleep(0.25)

    return resolved


def build_species_metadata_table(config: Dict[str, Any], session: requests.Session) -> pd.DataFrame:
    """
    Builds and saves `data/species_toxicity_metadata.csv` with official iNaturalist taxon_ids.
    If the file already exists with valid real iNaturalist taxon_ids, reuses the resolved IDs
    to save API calls.
    """
    out_path = config["data_paths"]["species_metadata_csv"]
    species_list = config["target_species"]

    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path)
        # Check if existing table already has real iNaturalist IDs (not synthetic 50000+idx*137)
        synthetic_ids = {50000 + i * 137 for i in range(1, len(species_list) + 1)}
        if not set(df_existing["taxon_id"].tolist()).issubset(synthetic_ids) and len(df_existing) == len(species_list):
            logger.info(f"Loaded existing verified iNaturalist species table from {out_path}")
            return df_existing

    resolved_records = resolve_real_inat_taxon_ids(species_list, session)
    df_species = pd.DataFrame(resolved_records)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_species.to_csv(out_path, index=False)
    logger.info(f"Saved official iNaturalist species metadata ({len(df_species)} species) to {out_path}")
    return df_species


def convert_photo_url_size(url: str, target_size: str = "small") -> str:
    """Converts iNaturalist square thumbnail URL to 'small' (240px) or 'medium' (500px) URL."""
    if not url:
        return ""
    for sz in ["square", "thumb", "small", "medium", "large", "original"]:
        if f"/{sz}." in url:
            return url.replace(f"/{sz}.", f"/{target_size}.")
    return url


def fetch_real_inat_observations(
    df_species: pd.DataFrame, config: Dict[str, Any], session: requests.Session
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches real observations and associated photo metadata from `https://api.inaturalist.org/v1/observations`
    across North America and Europe for all 40 species.
    """
    base_url = "https://api.inaturalist.org/v1/observations"
    subset_cfg = config["subset_selection"]
    regions = subset_cfg["regions"]
    per_page = int(subset_cfg.get("per_page", 100))
    img_variant = subset_cfg.get("image_size_variant", "small")

    obs_records = []
    photo_records = []

    logger.info("Fetching real observations from iNaturalist v1 API across North America & Europe...")
    for _, sp_row in tqdm(df_species.iterrows(), total=len(df_species), desc="Fetching Species Observations"):
        taxon_id = int(sp_row["taxon_id"])
        sci_name = sp_row["scientific_name"]

        for reg_name, bbox in regions.items():
            max_pages = (
                int(subset_cfg.get("max_pages_per_species_na", 1))
                if reg_name == "north_america"
                else int(subset_cfg.get("max_pages_per_species_eu", 1))
            )

            for page_num in range(1, max_pages + 1):
                params = {
                    "taxon_id": taxon_id,
                    "quality_grade": subset_cfg["quality_grade"],
                    "photos": "true",
                    "d1": subset_cfg["date_start"],
                    "d2": subset_cfg["date_end"],
                    "nelat": bbox["max_lat"],
                    "nelng": bbox["max_lon"],
                    "swlat": bbox["min_lat"],
                    "swlng": bbox["min_lon"],
                    "per_page": per_page,
                    "page": page_num,
                    "order_by": "id",
                    "order": "desc",
                }

                results = []
                for attempt in range(4):
                    try:
                        resp = session.get(base_url, params=params, timeout=15)
                        if resp.status_code == 200:
                            results = resp.json().get("results", [])
                            break
                        elif resp.status_code == 429:
                            logger.warning("iNaturalist API rate limit reached (HTTP 429). Sleeping 5s...")
                            time.sleep(5.0)
                        else:
                            time.sleep(1.5)
                    except Exception as exc:
                        logger.warning(f"Network retry {attempt+1} for {sci_name} ({reg_name}): {exc}")
                        time.sleep(2.0)

                if not results:
                    break

                for item in results:
                    obs_id_num = item.get("id")
                    obs_uuid = str(item.get("uuid") or obs_id_num)
                    coords = item.get("geojson", {}).get("coordinates", [None, None]) if item.get("geojson") else [None, None]
                    lon, lat = coords[0], coords[1]
                    user_info = item.get("user") or {}
                    observer_id = user_info.get("id")
                    observer_login = user_info.get("login") or f"user_{observer_id}"
                    photos = item.get("photos") or []

                    if not photos or observer_id is None:
                        continue

                    primary_photo = photos[0]
                    primary_url = convert_photo_url_size(primary_photo.get("url", ""), target_size=img_variant)

                    obs_records.append(
                        {
                            "observation_id": obs_id_num,
                            "observation_uuid": obs_uuid,
                            "inaturalist_url": f"https://www.inaturalist.org/observations/{obs_id_num}",
                            "taxon_id": taxon_id,
                            "scientific_name": sci_name,
                            "observer_id": int(observer_id),
                            "observer_handle": str(observer_login),
                            "observed_on": item.get("observed_on"),
                            "time_observed_at": item.get("time_observed_at"),
                            "latitude": float(lat) if lat is not None else np.nan,
                            "longitude": float(lon) if lon is not None else np.nan,
                            "positional_accuracy": (
                                float(item["positional_accuracy"])
                                if item.get("positional_accuracy") is not None
                                else np.nan
                            ),
                            "quality_grade": item.get("quality_grade", "research"),
                            "region": reg_name,
                            "num_photos": len(photos),
                            "captive_cultivated": int(bool(item.get("captive", False))),
                            "primary_image_url": primary_url,
                            "local_image_path": os.path.join(
                                config["data_paths"]["raw_images_dir"], f"{obs_uuid}.jpg"
                            ).replace("\\", "/"),
                            "data_source": "iNaturalist_v1_API_Real",
                        }
                    )

                    for p_idx, p in enumerate(photos):
                        p_id = p.get("id")
                        dims = p.get("original_dimensions") or {}
                        p_url = convert_photo_url_size(p.get("url", ""), target_size=img_variant)
                        lic = p.get("license_code") or "All-Rights-Reserved"
                        photo_records.append(
                            {
                                "photo_id": int(p_id) if p_id is not None else 0,
                                "observation_uuid": obs_uuid,
                                "taxon_id": taxon_id,
                                "observer_id": int(observer_id),
                                "photo_position": p_idx + 1,
                                "image_url": p_url,
                                "width": int(dims["width"]) if dims.get("width") else np.nan,
                                "height": int(dims["height"]) if dims.get("height") else np.nan,
                                "license_code": str(lic).upper(),
                            }
                        )

                time.sleep(0.45)

    df_obs = pd.DataFrame(obs_records)
    df_photos = pd.DataFrame(photo_records)
    logger.info(f"Retrieved {len(df_obs):,} real observations and {len(df_photos):,} photo records from iNaturalist.")
    return df_obs, df_photos


def _download_single_image(task: Tuple[str, str, str]) -> Tuple[str, bool, int, int]:
    """Worker function to download and verify a single real observation image."""
    obs_uuid, url, local_path = task
    if os.path.exists(local_path) and os.path.getsize(local_path) > 512:
        try:
            with Image.open(local_path) as im:
                w, h = im.size
                return obs_uuid, True, w, h
        except Exception:
            pass

    if not url:
        return obs_uuid, False, 0, 0

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 512:
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                with Image.open(local_path) as im:
                    im_rgb = im.convert("RGB")
                    w, h = im_rgb.size
                return obs_uuid, True, w, h
            time.sleep(0.8)
        except Exception:
            time.sleep(1.0)

    return obs_uuid, False, 0, 0


def download_and_verify_real_images(
    df_obs: pd.DataFrame, df_photos: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads the primary photograph for every real observation into `data/raw/images/{observation_uuid}.jpg`
    using a thread pool, and records the verified image dimensions.
    """
    img_dir = config["data_paths"]["raw_images_dir"]
    os.makedirs(img_dir, exist_ok=True)
    max_workers = int(config["subset_selection"].get("max_download_workers", 12))

    unique_obs = df_obs.drop_duplicates(subset=["observation_uuid"])
    tasks = [
        (row["observation_uuid"], row["primary_image_url"], row["local_image_path"])
        for _, row in unique_obs.iterrows()
    ]

    logger.info(f"Downloading & verifying {len(tasks):,} real mushroom images to {img_dir} (workers={max_workers})...")
    verified_status = {}
    verified_dims = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_single_image, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading Images"):
            obs_uuid, ok, w, h = fut.result()
            verified_status[obs_uuid] = int(ok)
            verified_dims[obs_uuid] = (w, h)

    df_obs = df_obs.copy()
    df_obs["image_downloaded_ok"] = df_obs["observation_uuid"].map(verified_status).fillna(0).astype(int)

    # Backfill missing original_dimensions in df_photos from verified downloaded image dimensions
    def resolve_w(row: pd.Series) -> float:
        if pd.notna(row["width"]) and row["width"] > 0:
            return float(row["width"])
        return float(verified_dims.get(row["observation_uuid"], (0, 0))[0])

    def resolve_h(row: pd.Series) -> float:
        if pd.notna(row["height"]) and row["height"] > 0:
            return float(row["height"])
        return float(verified_dims.get(row["observation_uuid"], (0, 0))[1])

    df_photos = df_photos.copy()
    df_photos["width"] = df_photos.apply(resolve_w, axis=1)
    df_photos["height"] = df_photos.apply(resolve_h, axis=1)

    success_count = int(df_obs["image_downloaded_ok"].sum())
    logger.info(
        f"Image download complete: {success_count:,} / {len(df_obs):,} observations have verified local images."
    )
    return df_obs, df_photos


def main():
    parser = argparse.ArgumentParser(description="Fetch real iNaturalist Fungi subset and images for Phase 1.")
    parser.add_argument("--config", type=str, default="configs/phase1_config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["real", "auto"],
        default="real",
        help="Execution mode. Both 'real' and 'auto' fetch real data from iNaturalist v1 API.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-fetching metadata from iNaturalist API even if raw CSV already exists.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    obs_path = config["data_paths"]["raw_metadata_csv"]
    photos_path = config["data_paths"]["raw_photos_csv"]

    session = requests.Session()
    session.headers.update({"User-Agent": "SharifDataAnalysisProject/1.0 (Academic Research)"})

    df_species = build_species_metadata_table(config, session)

    # Check if real raw data is already cached on disk
    use_cached = False
    if os.path.exists(obs_path) and os.path.exists(photos_path) and not args.force_refresh:
        df_obs_existing = pd.read_csv(obs_path)
        if (
            "data_source" in df_obs_existing.columns
            and (df_obs_existing["data_source"] == "iNaturalist_v1_API_Real").all()
            and len(df_obs_existing) >= 1000
        ):
            logger.info(f"Found existing real iNaturalist metadata at {obs_path} ({len(df_obs_existing):,} rows).")
            df_obs = df_obs_existing
            df_photos = pd.read_csv(photos_path)
            use_cached = True

    if not use_cached:
        df_obs, df_photos = fetch_real_inat_observations(df_species, config, session)

    # Download and verify actual image files
    df_obs, df_photos = download_and_verify_real_images(df_obs, df_photos, config)

    os.makedirs(os.path.dirname(obs_path), exist_ok=True)
    df_obs.to_csv(obs_path, index=False)
    df_photos.to_csv(photos_path, index=False)

    logger.info(f"Saved REAL iNaturalist observations table: {obs_path} ({len(df_obs):,} rows)")
    logger.info(f"Saved REAL iNaturalist photos table:       {photos_path} ({len(df_photos):,} rows)")


if __name__ == "__main__":
    main()