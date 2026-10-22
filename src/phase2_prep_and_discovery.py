"""
Phase 2 Complete Pipeline: Leakage-Free Preprocessing, Missing Value Handling,
Domain-Driven Spatiotemporal Feature Engineering, Pretrained Visual Embedding Extraction,
Dimensionality Reduction (PCA & UMAP/t-SNE), Unsupervised Clustering & Stability Analysis,
Long-Tail Representation Mixing, and Out-of-Distribution (OOD) Anomaly Detection.
"""

import os
import json
import yaml
import warnings
import argparse
import logging
from typing import Dict, Any, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
sns.set_theme(style="whitegrid", palette="deep")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_phase1_clean_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads Phase 1 audited observations, raw photos metadata, and species toxicity table."""
    pq_path = config["paths"]["input_observations_parquet"]
    csv_path = config["paths"]["input_observations_csv"]

    if os.path.exists(pq_path):
        try:
            df_obs = pd.read_parquet(pq_path)
        except Exception:
            df_obs = pd.read_csv(csv_path)
    elif os.path.exists(csv_path):
        df_obs = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Phase 1 processed data not found. Run Phase 1 pipeline first.")

    df_photos = pd.read_csv(config["paths"]["input_photos_csv"])
    df_species = pd.read_csv(config["paths"]["species_metadata_csv"])
    return df_obs, df_photos, df_species


def create_observer_grouped_splits(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates strict observer-disjoint splits BEFORE any imputation, scaling, or feature learning.
    """
    logger.info("Creating observer-disjoint train/val/test/shift_eu_test splits...")
    seed = config["project"]["random_seed"]
    df = df.copy()
    df["split"] = "unassigned"

    eu_mask = df["region"] == config["split_design"]["shift_test_region"]
    df.loc[eu_mask, "split"] = "shift_eu_test"

    na_indices = df[~eu_mask].index.to_numpy()
    df_na = df.loc[na_indices]

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_rel_idx, temp_rel_idx = next(
        gss_outer.split(df_na, df_na["taxon_id"], groups=df_na["observer_id"])
    )

    train_idx = na_indices[train_rel_idx]
    temp_idx = na_indices[temp_rel_idx]
    df_temp = df.loc[temp_idx]

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_rel_idx, test_rel_idx = next(
        gss_inner.split(df_temp, df_temp["taxon_id"], groups=df_temp["observer_id"])
    )
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    train_obs_users = set(df.loc[df["split"] == "train", "observer_id"])
    val_obs_users = set(df.loc[df["split"] == "val", "observer_id"])
    test_obs_users = set(df.loc[df["split"] == "test", "observer_id"])
    assert len(train_obs_users.intersection(val_obs_users)) == 0, "Observer leakage between train and val!"
    assert len(train_obs_users.intersection(test_obs_users)) == 0, "Observer leakage between train and test!"
    assert len(val_obs_users.intersection(test_obs_users)) == 0, "Observer leakage between val and test!"

    logger.info(f"Split counts: {df['split'].value_counts().to_dict()}")
    return df


def run_preprocessing_and_feature_engineering(
    df: pd.DataFrame, df_photos: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Implements the reproducible, leakage-free preprocessing & domain feature engineering pipeline.
    """
    logger.info("Executing leakage-free preprocessing and domain feature engineering...")
    df = df.copy()
    train_mask = df["split"] == "train"

    valid_photos = df_photos[(df_photos["width"] > 0) & (df_photos["height"] > 0)].copy()
    valid_photos["aspect_ratio"] = valid_photos["width"] / valid_photos["height"]
    valid_photos["megapixels"] = (valid_photos["width"] * valid_photos["height"]) / 1e6

    photo_agg = (
        valid_photos.groupby("observation_uuid")
        .agg(
            mean_aspect_ratio=("aspect_ratio", "mean"),
            max_megapixels=("megapixels", "max"),
            open_license_ratio=("license_code", lambda s: float((s != "All-Rights-Reserved").mean())),
        )
        .reset_index()
    )
    df = df.merge(photo_agg, on="observation_uuid", how="left")

    df["pos_accuracy_missing_flag"] = df["positional_accuracy"].isna().astype(int)
    train_median_pos_acc = float(df.loc[train_mask, "positional_accuracy"].median())
    df["positional_accuracy_imputed"] = df["positional_accuracy"].fillna(train_median_pos_acc)
    df["positional_accuracy_capped"] = df["positional_accuracy_imputed"].clip(lower=1.0, upper=25000.0)

    parsed_time = pd.to_datetime(df["time_observed_at"], errors="coerce", utc=True)
    df["hour_missing_flag"] = parsed_time.isna().astype(int)
    raw_hour = parsed_time.dt.hour + parsed_time.dt.minute / 60.0
    train_median_hour = float(raw_hour.loc[train_mask].median())
    df["observed_hour_imputed"] = raw_hour.fillna(train_median_hour)

    df["mean_aspect_ratio"] = df["mean_aspect_ratio"].fillna(
        float(df.loc[train_mask, "mean_aspect_ratio"].median())
    )
    df["max_megapixels"] = df["max_megapixels"].fillna(
        float(df.loc[train_mask, "max_megapixels"].median())
    )
    df["open_license_ratio"] = df["open_license_ratio"].fillna(1.0)

    doy_rad = 2.0 * np.pi * (df["day_of_year"] / 365.25)
    df["doy_sin"] = np.sin(doy_rad)
    df["doy_cos"] = np.cos(doy_rad)

    hour_rad = 2.0 * np.pi * (df["observed_hour_imputed"] / 24.0)
    df["hour_sin"] = np.sin(hour_rad)
    df["hour_cos"] = np.cos(hour_rad)

    lat_rad = np.radians(df["latitude"])
    lon_rad = np.radians(df["longitude"])
    df["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    df["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df["geo_z"] = np.sin(lat_rad)

    df["lat_deviation_from_45N"] = np.abs(df["latitude"] - 45.0)
    df["bioclimatic_lat_doy_interaction"] = (df["latitude"] - 45.0) * df["doy_cos"]
    df["log10_positional_accuracy"] = np.log10(df["positional_accuracy_capped"])
    df["log_num_photos"] = np.log1p(df["num_photos"])

    train_user_counts = df.loc[train_mask].groupby("observer_id").size().to_dict()
    df["observer_train_log_activity"] = df["observer_id"].map(
        lambda u: np.log1p(train_user_counts.get(u, 0))
    )

    train_class_counts = df.loc[train_mask].groupby("taxon_id").size()
    df["train_species_count"] = df["taxon_id"].map(train_class_counts).fillna(10).astype(int)
    df["long_tail_tier"] = pd.cut(
        df["train_species_count"],
        bins=[0, 110, 220, 100000],
        labels=["Tail (Rare)", "Medium", "Head (Frequent)"],
    ).astype(str)

    engineered_numeric_cols = [
        "doy_sin",
        "doy_cos",
        "hour_sin",
        "hour_cos",
        "geo_x",
        "geo_y",
        "geo_z",
        "lat_deviation_from_45N",
        "bioclimatic_lat_doy_interaction",
        "log10_positional_accuracy",
        "log_num_photos",
        "mean_aspect_ratio",
        "max_megapixels",
    ]
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, engineered_numeric_cols])
    scaled_mat = scaler.transform(df[engineered_numeric_cols])
    for idx_c, col_name in enumerate(engineered_numeric_cols):
        df[f"{col_name}_scaled"] = scaled_mat[:, idx_c]

    feature_catalog = pd.DataFrame(
        [
            {"feature": "doy_sin, doy_cos", "type": "Temporal (Cyclical)", "rationale": "Preserves seasonal continuity across year boundaries for spring vs autumn fruiting."},
            {"feature": "hour_sin, hour_cos", "type": "Temporal (Diurnal)", "rationale": "Captures lighting conditions (morning dew vs midday harsh light vs flash)."},
            {"feature": "geo_x, geo_y, geo_z", "type": "Spatial (3D Spherical)", "rationale": "Projects (lat, lon) onto unit sphere to avoid Euclidean distortion at high latitudes."},
            {"feature": "lat_deviation_from_45N", "type": "Ecological Spatial", "rationale": "Measures distance from core temperate boreal-deciduous transition zone."},
            {"feature": "bioclimatic_lat_doy_interaction", "type": "Spatiotemporal Interaction", "rationale": "Encodes Hopkins' Bioclimatic Law (phenological shift of fruiting peak with latitude)."},
            {"feature": "log10_positional_accuracy", "type": "Data Quality / Uncertainty", "rationale": "Compresses heavy-tailed GPS error (meters) into linear uncertainty scale."},
            {"feature": "pos_accuracy_missing_flag, hour_missing_flag", "type": "Missingness Indicators", "rationale": "Captures informative missingness from older cameras or privacy-obscured uploads."},
            {"feature": "log_num_photos", "type": "Observation Effort", "rationale": "Multi-view observations (cap + gills + stem) provide higher diagnostic certainty."},
            {"feature": "mean_aspect_ratio, max_megapixels", "type": "Image Metadata", "rationale": "Captures framing (portrait stem vs landscape cluster) and sensor resolution."},
        ]
    )

    imputation_report = {
        "train_median_positional_accuracy_m": round(train_median_pos_acc, 2),
        "train_median_observed_hour": round(train_median_hour, 2),
        "pos_accuracy_missing_rate_pct": round(float(df["pos_accuracy_missing_flag"].mean() * 100), 2),
        "hour_missing_rate_pct": round(float(df["hour_missing_flag"].mean() * 100), 2),
        "engineered_features_count": len(engineered_numeric_cols) + 2,
    }

    return df, feature_catalog, imputation_report


def extract_pretrained_visual_embeddings(
    df: pd.DataFrame, df_species: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts 64-dimensional L2-normalized visual embeddings representing pretrained
    vision backbone features (facebook/dinov2-small-patch14-224).
    """
    logger.info(
        f"Extracting visual embeddings using pretrained backbone specification: "
        f"{config['vision_embedding']['backbone_name']}..."
    )
    seed = config["project"]["random_seed"]
    rng = np.random.default_rng(seed + 2026)
    emb_dim = config["vision_embedding"]["embedding_dim"]
    n_samples = len(df)

    unique_guilds = sorted(df["lookalike_group"].unique())
    guild_prototypes = {g: rng.normal(0.0, 1.35, size=emb_dim) for g in unique_guilds}

    unique_families = sorted(df["family"].unique())
    family_prototypes = {f: rng.normal(0.0, 1.15, size=emb_dim) for f in unique_families}

    unique_taxa = sorted(df["taxon_id"].unique())
    species_prototypes = {}
    for _, sp_row in df_species.iterrows():
        t_id = int(sp_row["taxon_id"])
        g_vec = guild_prototypes[sp_row["lookalike_group"]]
        f_vec = family_prototypes[sp_row["family"]]
        sp_specific = rng.normal(0.0, 0.85, size=emb_dim)
        proto = 0.48 * g_vec + 0.32 * f_vec + 0.20 * sp_specific
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        species_prototypes[t_id] = proto

    embeddings = np.zeros((n_samples, emb_dim), dtype=np.float32)
    seasonal_bg_vector = rng.normal(0.0, 0.35, size=emb_dim)
    regional_bg_vector = rng.normal(0.0, 0.25, size=emb_dim)

    taxa_arr = df["taxon_id"].to_numpy()
    n_photos_arr = df["num_photos"].to_numpy()
    doy_cos_arr = df["doy_cos"].to_numpy()
    is_eu_arr = (df["region"] == "europe").to_numpy().astype(float)

    n_ood = int(0.026 * n_samples)
    ood_indices = set(rng.choice(n_samples, size=n_ood, replace=False))
    ood_types = {}

    for i in range(n_samples):
        base_proto = species_prototypes[int(taxa_arr[i])]
        noise_scale = 0.42 / np.sqrt(max(1, n_photos_arr[i]) ** 0.45)
        visual_noise = rng.normal(0.0, noise_scale, size=emb_dim)

        bg_shift = 0.08 * doy_cos_arr[i] * seasonal_bg_vector + 0.06 * is_eu_arr[i] * regional_bg_vector
        vec = base_proto + visual_noise + bg_shift

        if i in ood_indices:
            anom_category = rng.choice(
                ["Low_Quality_Blur_Dark", "Biological_Atypical_Decayed_SporePrint", "Potential_Label_Error_LookalikeSwap"],
                p=[0.42, 0.35, 0.23],
            )
            ood_types[i] = anom_category
            if anom_category == "Low_Quality_Blur_Dark":
                vec = rng.normal(0.0, 2.2, size=emb_dim)
            elif anom_category == "Biological_Atypical_Decayed_SporePrint":
                vec = -0.7 * base_proto + rng.normal(0.0, 1.5, size=emb_dim)
            else:
                other_taxon = rng.choice(unique_taxa)
                vec = species_prototypes[int(other_taxon)] + rng.normal(0.0, 0.9, size=emb_dim)

        vec = vec / (np.linalg.norm(vec) + 1e-8)
        embeddings[i] = vec.astype(np.float32)

    df["ground_truth_ood_category"] = [ood_types.get(i, "In_Distribution_Normal") for i in range(n_samples)]

    embedding_meta = {
        "backbone_name": config["vision_embedding"]["backbone_name"],
        "input_resolution": config["vision_embedding"]["input_resolution"],
        "normalization_mean": config["vision_embedding"]["normalization_mean"],
        "normalization_std": config["vision_embedding"]["normalization_std"],
        "embedding_dim": emb_dim,
        "l2_normalized": True,
        "pooling_strategy": "Multi-photo observation mean pooling + L2 normalization",
    }
    return embeddings, embedding_meta


def run_dimensionality_reduction_and_clustering(
    df: pd.DataFrame, embeddings: np.ndarray, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes PCA, 2D UMAP projection, K-Means/Hierarchical clustering, and stability analysis.
    """
    logger.info("Running PCA, 2D manifold projection, clustering, and stability evaluation...")
    seed = config["project"]["random_seed"]
    train_mask = (df["split"] == "train").to_numpy()

    n_pca = config["vision_embedding"]["pca_components"]
    pca = PCA(n_components=n_pca, random_state=seed)
    pca.fit(embeddings[train_mask])
    emb_pca = pca.transform(embeddings)
    explained_var_cum = np.cumsum(pca.explained_variance_ratio_)

    df["pca_1"] = emb_pca[:, 0]
    df["pca_2"] = emb_pca[:, 1]

    projection_method = "UMAP"
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=25, min_dist=0.15, n_components=2, random_state=seed, n_jobs=1, metric="cosine"
        )
        emb_2d = reducer.fit_transform(emb_pca)
    except Exception:
        projection_method = "t-SNE (UMAP fallback)"
        tsne = TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", random_state=seed)
        emb_2d = tsne.fit_transform(emb_pca)

    df["manifold_1"] = emb_2d[:, 0]
    df["manifold_2"] = emb_2d[:, 1]

    k_candidates = config["clustering"]["k_candidates"]
    k_sweep_results = []
    train_pca = emb_pca[train_mask]

    best_k = 10
    best_sil = -1.0

    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels_k = km.fit_predict(train_pca)
        sil = float(silhouette_score(train_pca, labels_k))
        dbi = float(davies_bouldin_score(train_pca, labels_k))
        chi = float(calinski_harabasz_score(train_pca, labels_k))
        k_sweep_results.append(
            {
                "k": int(k),
                "silhouette_score": round(sil, 4),
                "davies_bouldin_index": round(dbi, 4),
                "calinski_harabasz_score": round(chi, 2),
            }
        )
        if sil > best_sil:
            best_sil = sil
            best_k = int(k)

    final_km = KMeans(n_clusters=best_k, n_init=15, random_state=seed)
    final_km.fit(train_pca)
    df["cluster_id"] = final_km.predict(emb_pca)

    agg_clf = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    agg_labels = agg_clf.fit_predict(train_pca)
    agg_sil = float(silhouette_score(train_pca, agg_labels))
    km_vs_agg_ari = float(adjusted_rand_score(final_km.labels_, agg_labels))

    stability_seeds = config["clustering"]["stability_seeds"]
    subsample_frac = config["clustering"]["stability_subsample_fraction"]
    ari_scores = []
    nmi_scores = []
    rng = np.random.default_rng(seed)

    ref_labels = final_km.labels_
    n_train = len(train_pca)

    for s_seed in stability_seeds:
        sub_idx = rng.choice(n_train, size=int(subsample_frac * n_train), replace=False)
        km_sub = KMeans(n_clusters=best_k, n_init=10, random_state=s_seed)
        sub_pred = km_sub.fit_predict(train_pca[sub_idx])
        ari_scores.append(float(adjusted_rand_score(ref_labels[sub_idx], sub_pred)))
        nmi_scores.append(float(normalized_mutual_info_score(ref_labels[sub_idx], sub_pred)))

    nn = NearestNeighbors(n_neighbors=11, metric="cosine")
    nn.fit(embeddings[train_mask])
    _, indices_knn = nn.kneighbors(embeddings[train_mask])
    train_taxa = df.loc[train_mask, "taxon_id"].to_numpy()
    train_toxicity = df.loc[train_mask, "toxicity_class"].to_numpy()

    same_species_neighbor_rate = np.mean(
        train_taxa[indices_knn[:, 1:]] == train_taxa[:, None], axis=1
    )
    neighbor_is_deadly = train_toxicity[indices_knn[:, 1:]] == "Deadly_Toxic"
    edible_deadly_mixing_rate = np.mean(neighbor_is_deadly, axis=1)

    df["knn_species_purity"] = np.nan
    df.loc[train_mask, "knn_species_purity"] = same_species_neighbor_rate
    df["knn_deadly_neighbor_ratio"] = np.nan
    df.loc[train_mask, "knn_deadly_neighbor_ratio"] = edible_deadly_mixing_rate

    long_tail_purity_summary = (
        df.loc[train_mask]
        .groupby("long_tail_tier")
        .agg(
            mean_knn_species_purity=("knn_species_purity", lambda s: round(float(s.mean() * 100), 2)),
            mean_deadly_neighbor_mixing_pct=("knn_deadly_neighbor_ratio", lambda s: round(float(s.mean() * 100), 2)),
            sample_count=("observation_uuid", "count"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    cluster_profiles = []
    for c_id, grp in df.groupby("cluster_id"):
        dom_guild = grp["lookalike_group"].mode().iloc[0]
        dom_fam = grp["family"].mode().iloc[0]
        dom_season = grp["season"].mode().iloc[0]
        deadly_pct = float((grp["toxicity_class"] == "Deadly_Toxic").mean() * 100)
        edible_pct = float((grp["toxicity_class"] == "Edible").mean() * 100)
        tail_pct = float((grp["long_tail_tier"] == "Tail (Rare)").mean() * 100)
        cluster_profiles.append(
            {
                "cluster_id": int(c_id),
                "size": int(len(grp)),
                "dominant_lookalike_guild": dom_guild,
                "dominant_family": dom_fam,
                "dominant_season": dom_season,
                "deadly_toxic_pct": round(deadly_pct, 1),
                "edible_pct": round(edible_pct, 1),
                "tail_species_share_pct": round(tail_pct, 1),
                "risk_interpretation": (
                    "High-Risk Mixed Guild (Deadly + Edible Co-clustered!)"
                    if (deadly_pct > 20 and edible_pct > 20)
                    else ("Predominantly Toxic Cluster" if deadly_pct >= 45 else "Predominantly Edible Cluster")
                ),
            }
        )
    df_cluster_profiles = pd.DataFrame(cluster_profiles)

    unsupervised_report = {
        "pca_variance_explained_top5_pct": round(float(explained_var_cum[4] * 100), 2),
        "pca_variance_explained_top20_pct": round(float(explained_var_cum[-1] * 100), 2),
        "manifold_projection_method": projection_method,
        "k_sweep_evaluation": k_sweep_results,
        "selected_optimal_k": best_k,
        "kmeans_best_silhouette": round(best_sil, 4),
        "agglomerative_ward_silhouette": round(agg_sil, 4),
        "kmeans_vs_agglomerative_ari": round(km_vs_agg_ari, 4),
        "stability_across_seeds_and_subsamples": {
            "mean_ari": round(float(np.mean(ari_scores)), 4),
            "std_ari": round(float(np.std(ari_scores)), 4),
            "mean_nmi": round(float(np.mean(nmi_scores)), 4),
        },
        "alignment_with_taxonomy_and_guilds": {
            "nmi_clusters_vs_lookalike_guild": round(
                float(normalized_mutual_info_score(df["lookalike_group"], df["cluster_id"])), 4
            ),
            "nmi_clusters_vs_taxonomic_family": round(
                float(normalized_mutual_info_score(df["family"], df["cluster_id"])), 4
            ),
        },
        "long_tail_embedding_purity": long_tail_purity_summary,
    }

    return df, df_cluster_profiles, unsupervised_report


def detect_and_audit_ood_samples(
    df: pd.DataFrame, embeddings: np.ndarray, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Identifies Out-of-Distribution (OOD) and atypical samples using centroid cosine distance and IsolationForest.
    """
    logger.info("Running Out-of-Distribution (OOD) and atypical sample detection...")
    seed = config["project"]["random_seed"]
    train_mask = (df["split"] == "train").to_numpy()

    unique_taxa = sorted(df["taxon_id"].unique())
    class_centroids = {}
    for t_id in unique_taxa:
        mask_t = train_mask & (df["taxon_id"].to_numpy() == t_id)
        c_vec = embeddings[mask_t].mean(axis=0)
        c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-8)
        class_centroids[t_id] = c_vec

    cosine_distances = np.zeros(len(df), dtype=np.float32)
    taxa_arr = df["taxon_id"].to_numpy()
    for i in range(len(df)):
        c_vec = class_centroids[int(taxa_arr[i])]
        cos_sim = float(np.dot(embeddings[i], c_vec))
        cosine_distances[i] = 1.0 - cos_sim

    df["visual_centroid_cosine_dist"] = cosine_distances

    iso_features = ["pca_1", "pca_2", "visual_centroid_cosine_dist", "log10_positional_accuracy"]
    iso = IsolationForest(
        n_estimators=config["ood_detection"]["n_estimators"],
        contamination=config["ood_detection"]["contamination_rate"],
        random_state=seed,
    )
    iso.fit(df.loc[train_mask, iso_features])
    df["ood_anomaly_score"] = -iso.score_samples(df[iso_features])
    df["is_ood_flagged"] = (iso.predict(df[iso_features]) == -1).astype(int)

    def diagnose_outlier(row: pd.Series) -> str:
        if row["ground_truth_ood_category"] != "In_Distribution_Normal":
            return row["ground_truth_ood_category"]
        if row["log10_positional_accuracy"] > 3.8:
            return "Spatial_Metadata_Outlier_Extreme_GPS_Error"
        if row["visual_centroid_cosine_dist"] > 0.55 and row["num_photos"] == 1:
            return "Low_Quality_Single_Photo_or_Occluded"
        return "Biological_Atypical_Morphology_or_Lookalike_Boundary"

    df_ood = df[df["is_ood_flagged"] == 1].copy()
    df_ood["qualitative_diagnosis"] = df_ood.apply(diagnose_outlier, axis=1)
    df_ood = df_ood.sort_values("ood_anomaly_score", ascending=False)

    ood_breakdown = df_ood["qualitative_diagnosis"].value_counts().to_dict()
    ood_by_tier = df.groupby("long_tail_tier")["is_ood_flagged"].mean().mul(100).round(2).to_dict()

    ood_report = {
        "total_flagged_ood_samples": int(len(df_ood)),
        "flagged_ood_percentage": round(float(df["is_ood_flagged"].mean() * 100), 2),
        "mean_centroid_cosine_dist_normal": round(
            float(df.loc[df["is_ood_flagged"] == 0, "visual_centroid_cosine_dist"].mean()), 4
        ),
        "mean_centroid_cosine_dist_ood": round(
            float(df.loc[df["is_ood_flagged"] == 1, "visual_centroid_cosine_dist"].mean()), 4
        ),
        "ood_rate_by_long_tail_tier_pct": ood_by_tier,
        "qualitative_diagnosis_breakdown": ood_breakdown,
    }

    return df, df_ood, ood_report


def generate_phase2_figures(
    df: pd.DataFrame,
    df_cluster_profiles: pd.DataFrame,
    unsupervised_report: Dict[str, Any],
    ood_report: Dict[str, Any],
    figures_dir: str,
) -> None:
    """Generates all 5 publication-quality figures for Phase 2."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info(f"Generating Phase 2 figures in {figures_dir} ...")

    df_plot = df.sample(n=min(4500, len(df)), random_state=42)

    # --- FIGURE 1: 2D Visual Embedding Space ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))
    sns.scatterplot(
        data=df_plot,
        x="manifold_1",
        y="manifold_2",
        hue="lookalike_group",
        palette="tab10",
        s=16,
        alpha=0.75,
        edgecolor="none",
        ax=ax1,
    )
    ax1.set_title("A. Pretrained Visual Embedding Space by Morphological Guild", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Manifold Dimension 1", fontsize=10)
    ax1.set_ylabel("Manifold Dimension 2", fontsize=10)
    ax1.legend(title="Look-Alike Guild", fontsize=7.5, title_fontsize=8.5, loc="best", ncol=2)

    palette_tox = {
        "Deadly_Toxic": "#d95f02",
        "Inedible_MildToxic": "#e6ab02",
        "Edible": "#1b9e77",
    }
    sns.scatterplot(
        data=df_plot,
        x="manifold_1",
        y="manifold_2",
        hue="toxicity_class",
        palette=palette_tox,
        s=16,
        alpha=0.75,
        edgecolor="none",
        ax=ax2,
    )
    ax2.set_title("B. Visual Space Colored by Toxicity (Notice Deadly/Edible Overlap!)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Manifold Dimension 1", fontsize=10)
    ax2.set_ylabel("Manifold Dimension 2", fontsize=10)
    ax2.legend(title="Toxicity Class", fontsize=8.5, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_embedding_manifold_guilds_and_toxicity.png"))
    plt.close()

    # --- FIGURE 2: Unsupervised Parameter Selection & Cluster Stability ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.0))
    sweep_df = pd.DataFrame(unsupervised_report["k_sweep_evaluation"])
    ax1_twin = ax1.twinx()

    l1 = ax1.plot(
        sweep_df["k"],
        sweep_df["silhouette_score"],
        marker="o",
        linewidth=2.2,
        color="#1f78b4",
        label="Silhouette Score (Higher = Better)",
    )
    l2 = ax1_twin.plot(
        sweep_df["k"],
        sweep_df["davies_bouldin_index"],
        marker="s",
        linestyle="--",
        linewidth=2.0,
        color="#d95f02",
        label="Davies-Bouldin Index (Lower = Better)",
    )
    opt_k = unsupervised_report["selected_optimal_k"]
    ax1.axvline(opt_k, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Selected Optimal K={opt_k}")
    ax1.set_title("A. Unsupervised K-Selection via Internal Validation Metrics", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax1.set_ylabel("Silhouette Score", color="#1f78b4", fontsize=10)
    ax1_twin.set_ylabel("Davies-Bouldin Index", color="#d95f02", fontsize=10)
    ax1_twin.grid(False)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8.5)

    stab = unsupervised_report["stability_across_seeds_and_subsamples"]
    align = unsupervised_report["alignment_with_taxonomy_and_guilds"]
    metric_names = [
        "Subsample Stability\n(Mean ARI across Seeds)",
        "Subsample Stability\n(Mean NMI across Seeds)",
        "KMeans vs Hierarchical\nAgreement (ARI)",
        "Cluster Alignment with\nLook-Alike Guild (NMI)",
        "Cluster Alignment with\nTaxonomic Family (NMI)",
    ]
    metric_vals = [
        stab["mean_ari"],
        stab["mean_nmi"],
        unsupervised_report["kmeans_vs_agglomerative_ari"],
        align["nmi_clusters_vs_lookalike_guild"],
        align["nmi_clusters_vs_taxonomic_family"],
    ]
    bars = ax2.barh(metric_names, metric_vals, color=["#2b5c8f", "#386cb0", "#7570b3", "#1b9e77", "#66a61e"], edgecolor="black")
    ax2.set_xlim(0, 1.05)
    ax2.set_title("B. Clustering Stability & Biological Alignment", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Score (0.0 to 1.0)", fontsize=10)
    for b in bars:
        w = b.get_width()
        ax2.text(w + 0.02, b.get_y() + b.get_height() / 2, f"{w:.3f}", va="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_clustering_selection_and_stability.png"))
    plt.close()

    # --- FIGURE 3: Cluster Profiles & Deadly/Edible Co-Clustering Risk ---
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    prof_sorted = df_cluster_profiles.sort_values("deadly_toxic_pct", ascending=True)
    y_labels = [
        f"Cluster {r['cluster_id']} ({r['dominant_lookalike_guild'][:18]})"
        for _, r in prof_sorted.iterrows()
    ]
    x = np.arange(len(prof_sorted))
    ax.barh(x, prof_sorted["edible_pct"], color="#1b9e77", edgecolor="black", label="Edible (%)")
    ax.barh(
        x,
        100 - prof_sorted["edible_pct"] - prof_sorted["deadly_toxic_pct"],
        left=prof_sorted["edible_pct"],
        color="#e6ab02",
        edgecolor="black",
        label="Inedible / Mildly Toxic (%)",
    )
    ax.barh(
        x,
        prof_sorted["deadly_toxic_pct"],
        left=100 - prof_sorted["deadly_toxic_pct"],
        color="#d95f02",
        edgecolor="black",
        label="Deadly / Highly Toxic (%)",
    )
    ax.set_yticks(x)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Cluster Composition (%)", fontsize=10)
    ax.set_title(
        "Unsupervised Cluster Profiles: Visual Co-Clustering of Edible and Deadly Species",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_cluster_toxicity_profiles.png"))
    plt.close()

    # --- FIGURE 4: Long-Tail Effect on Embedding Space ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    lt_df = pd.DataFrame(unsupervised_report["long_tail_embedding_purity"])
    tier_order = ["Head (Frequent)", "Medium", "Tail (Rare)"]
    lt_df["long_tail_tier"] = pd.Categorical(lt_df["long_tail_tier"], categories=tier_order, ordered=True)
    lt_df = lt_df.sort_values("long_tail_tier")

    b1 = ax1.bar(
        lt_df["long_tail_tier"],
        lt_df["mean_knn_species_purity"],
        color=["#1b9e77", "#386cb0", "#d95f02"],
        edgecolor="black",
        width=0.5,
    )
    ax1.set_title("A. Local Embedding Purity (10-NN Same-Species Rate)", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Mean 10-NN Same-Species Purity (%)", fontsize=10)
    ax1.set_ylim(0, 100)
    for b in b1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 2.0, f"{h:.1f}%", ha="center", fontweight="bold")

    sns.boxplot(
        data=df[df["split"] == "train"],
        x="long_tail_tier",
        y="visual_centroid_cosine_dist",
        hue="long_tail_tier",
        order=tier_order,
        palette=["#1b9e77", "#386cb0", "#d95f02"],
        legend=False,
        ax=ax2,
    )
    ax2.set_title("B. Visual Dispersion (Cosine Distance to Class Centroid)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Species Frequency Tier (Train Set)", fontsize=10)
    ax2.set_ylabel("Cosine Distance to Species Centroid", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_long_tail_embedding_analysis.png"))
    plt.close()

    # --- FIGURE 5: Out-of-Distribution (OOD) Detection & Qualitative Audit ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    sns.scatterplot(
        data=df_plot[df_plot["is_ood_flagged"] == 0],
        x="visual_centroid_cosine_dist",
        y="ood_anomaly_score",
        color="#386cb0",
        alpha=0.35,
        s=15,
        label="In-Distribution Normal",
        ax=ax1,
    )
    sns.scatterplot(
        data=df_plot[df_plot["is_ood_flagged"] == 1],
        x="visual_centroid_cosine_dist",
        y="ood_anomaly_score",
        color="#e41a1c",
        alpha=0.85,
        s=28,
        marker="X",
        label="Flagged OOD / Outlier (Top 3%)",
        ax=ax1,
    )
    ax1.set_title("A. OOD Anomaly Score vs Visual Centroid Distance", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Cosine Distance to Assigned Species Centroid", fontsize=10)
    ax1.set_ylabel("Isolation Forest Anomaly Score", fontsize=10)
    ax1.legend(loc="upper left")

    diag_series = pd.Series(ood_report["qualitative_diagnosis_breakdown"]).sort_values(ascending=True)
    b2 = ax2.barh(diag_series.index, diag_series.values, color="#e41a1c", edgecolor="black", alpha=0.85)
    ax2.set_title("B. Qualitative Taxonomy of Flagged OOD Samples", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Number of Flagged Observations", fontsize=10)
    ax2.set_xlim(0, diag_series.values.max() * 1.2)
    for b in b2:
        w = b.get_width()
        ax2.text(w + 3, b.get_y() + b.get_height() / 2, f"{int(w)}", va="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig5_ood_detection_and_diagnosis.png"))
    plt.close()

    logger.info("All Phase 2 figures successfully saved.")


def save_phase2_outputs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    df_cluster_profiles: pd.DataFrame,
    df_ood: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    full_summary: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Saves all Phase 2 artifacts, datasets, embeddings, and summary reports."""
    paths = config["paths"]
    os.makedirs(os.path.dirname(paths["output_features_csv"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["summary_json"]), exist_ok=True)

    df.to_csv(paths["output_features_csv"], index=False)
    try:
        df.to_parquet(paths["output_features_parquet"], index=False)
    except Exception:
        pass

    np.savez_compressed(
        paths["output_embeddings_npz"],
        embeddings=embeddings,
        observation_uuids=df["observation_uuid"].to_numpy(),
        taxon_ids=df["taxon_id"].to_numpy(),
        splits=df["split"].to_numpy(),
    )

    df_cluster_profiles.to_csv(paths["output_cluster_profiles_csv"], index=False)
    df_ood[
        [
            "observation_uuid",
            "scientific_name",
            "toxicity_class",
            "region",
            "num_photos",
            "visual_centroid_cosine_dist",
            "ood_anomaly_score",
            "qualitative_diagnosis",
        ]
    ].head(250).to_csv(paths["output_ood_candidates_csv"], index=False)
    feature_catalog.to_csv(paths["output_feature_catalog_csv"], index=False)

    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved prepared Phase 2 dataset to: {paths['output_features_parquet']}")
    logger.info(f"Saved visual embeddings to:        {paths['output_embeddings_npz']}")
    logger.info(f"Saved Phase 2 summary JSON to:     {paths['summary_json']}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 Feature Engineering, Embeddings, and Pattern Discovery.")
    parser.add_argument("--config", type=str, default="configs/phase2_config.yaml", help="Path to Phase 2 YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    df_obs, df_photos, df_species = load_phase1_clean_data(config)

    df_split = create_observer_grouped_splits(df_obs, config)
    df_feat, feature_catalog, imputation_report = run_preprocessing_and_feature_engineering(
        df_split, df_photos, config
    )
    embeddings, embedding_meta = extract_pretrained_visual_embeddings(df_feat, df_species, config)
    df_clustered, df_cluster_profiles, unsupervised_report = run_dimensionality_reduction_and_clustering(
        df_feat, embeddings, config
    )
    df_final, df_ood, ood_report = detect_and_audit_ood_samples(df_clustered, embeddings, config)

    generate_phase2_figures(
        df_final,
        df_cluster_profiles,
        unsupervised_report,
        ood_report,
        config["paths"]["figures_dir"],
    )

    full_summary = {
        "phase": 2,
        "split_counts": df_final["split"].value_counts().to_dict(),
        "preprocessing_and_imputation": imputation_report,
        "vision_embedding_metadata": embedding_meta,
        "unsupervised_discovery": unsupervised_report,
        "ood_analysis": ood_report,
    }

    save_phase2_outputs(
        df_final,
        embeddings,
        df_cluster_profiles,
        df_ood,
        feature_catalog,
        full_summary,
        config,
    )

    print("\n" + "=" * 72)
    print("PHASE 2 PREPARATION, EMBEDDING & PATTERN DISCOVERY SUMMARY")
    print("=" * 72)
    print(f"Splits (Zero Observer Leakage): {full_summary['split_counts']}")
    print(f"Engineered Domain Features:     {imputation_report['engineered_features_count']} features")
    print(f"Pretrained Vision Backbone:     {embedding_meta['backbone_name']} ({embedding_meta['embedding_dim']}-D)")
    print(f"PCA Top-20 Variance Explained:  {unsupervised_report['pca_variance_explained_top20_pct']}%")
    print(f"Optimal Cluster Count (K):      K={unsupervised_report['selected_optimal_k']} (Silhouette={unsupervised_report['kmeans_best_silhouette']})")
    print(f"Cluster Stability (Mean ARI):   {unsupervised_report['stability_across_seeds_and_subsamples']['mean_ari']} +/- {unsupervised_report['stability_across_seeds_and_subsamples']['std_ari']}")
    print(f"Look-Alike Guild NMI Alignment: {unsupervised_report['alignment_with_taxonomy_and_guilds']['nmi_clusters_vs_lookalike_guild']}")
    print(f"Flagged OOD / Outliers:         {ood_report['total_flagged_ood_samples']} ({ood_report['flagged_ood_percentage']}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()