"""
Phase 1 Complete Pipeline: Data Structure Audit, Data Quality Audit,
Univariate & Multivariate EDA, Long-Tail Analysis, Spatiotemporal Shortcut Audit,
Data Leakage Verification, and Initial Baselines for Track C (iNaturalist Fungi).
"""

import os
import json
import yaml
import argparse
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set publication-quality matplotlib defaults
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
sns.set_theme(style="whitegrid", palette="deep")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_gini_coefficient(counts: np.ndarray) -> float:
    """Computes Gini coefficient of class frequency distribution to quantify long-tail imbalance."""
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    cum_c = np.cumsum(sorted_c, dtype=float)
    return float((n + 1 - 2 * np.sum(cum_c) / cum_c[-1]) / n)


def save_processed_dataframe(df: pd.DataFrame, out_parquet_path: str) -> str:
    """
    Saves processed DataFrame to Parquet if pyarrow/fastparquet is installed,
    and always saves a synchronized CSV copy so the pipeline never fails due to missing C++ engines.
    """
    os.makedirs(os.path.dirname(out_parquet_path), exist_ok=True)
    csv_fallback_path = out_parquet_path.replace(".parquet", ".csv")
    df.to_csv(csv_fallback_path, index=False)

    try:
        df.to_parquet(out_parquet_path, index=False)
        logger.info(f"Saved audited observations to Parquet ({out_parquet_path}) and CSV ({csv_fallback_path})")
        return out_parquet_path
    except ImportError:
        logger.warning(
            f"Optional library 'pyarrow' not found in current Python environment. "
            f"Saved clean dataset as standard CSV instead: {csv_fallback_path}"
        )
        return csv_fallback_path


def audit_structure_and_quality(
    df_obs: pd.DataFrame, df_photos: pd.DataFrame, df_species: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs rigorous structure and quality audit required by Phase 1.
    """
    logger.info("Running Phase 1 Data Structure and Quality Audit...")
    raw_obs_count = len(df_obs)
    raw_photo_count = len(df_photos)

    # 1. Duplicate check
    exact_duplicates = int(df_obs.duplicated(subset=["observation_uuid"]).sum())
    df_dedup = df_obs.drop_duplicates(subset=["observation_uuid"], keep="first").copy()

    # 2. Merge with species toxicity metadata
    df_merged = df_dedup.merge(
        df_species[
            [
                "taxon_id",
                "common_name",
                "genus",
                "family",
                "toxicity_class",
                "is_toxic_or_inedible",
                "is_deadly",
                "risk_level",
                "lookalike_group",
            ]
        ],
        on="taxon_id",
        how="left",
    )

    # 3. Missing values quantification
    missing_summary = {
        col: {
            "count": int(df_merged[col].isna().sum()),
            "percentage": round(float(df_merged[col].isna().mean() * 100), 2),
        }
        for col in df_merged.columns
    }

    # 4. Identify impossible / anomalous records
    impossible_gps_mask = (
        df_merged["latitude"].isna()
        | df_merged["longitude"].isna()
        | (df_merged["latitude"].abs() > 90.0)
        | (df_merged["longitude"].abs() > 180.0)
        | ((df_merged["latitude"] == 0.0) & (df_merged["longitude"] == 0.0))
    )
    extreme_pos_acc_mask = df_merged["positional_accuracy"] > 10000.0  # > 10 km uncertainty
    captive_mask = df_merged["captive_cultivated"] == 1
    corrupt_photos_count = int(((df_photos["width"] <= 0) | (df_photos["height"] <= 0)).sum())

    quality_issues = {
        "raw_observations_total": raw_obs_count,
        "raw_photos_total": raw_photo_count,
        "exact_duplicate_observations": exact_duplicates,
        "missing_or_impossible_gps_count": int(impossible_gps_mask.sum()),
        "extreme_positional_accuracy_gt_10km_count": int(extreme_pos_acc_mask.sum()),
        "captive_or_cultivated_count": int(captive_mask.sum()),
        "corrupt_photo_dimensions_count": corrupt_photos_count,
        "missing_values_by_column": missing_summary,
    }

    # 5. Apply Phase 1 transparent filtering rules (remove impossible GPS & captive specimens)
    df_clean = df_merged[~impossible_gps_mask & ~captive_mask].copy()
    df_clean["is_high_gps_uncertainty"] = extreme_pos_acc_mask.loc[df_clean.index].astype(int)

    # Parse dates and temporal variables
    df_clean["observed_on_dt"] = pd.to_datetime(df_clean["observed_on"], errors="coerce")
    df_clean["year"] = df_clean["observed_on_dt"].dt.year
    df_clean["month"] = df_clean["observed_on_dt"].dt.month
    df_clean["day_of_year"] = df_clean["observed_on_dt"].dt.dayofyear
    df_clean["season"] = df_clean["month"].map(
        {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn",
        }
    )

    # 6. Structure hierarchy summary (Track C specific requirement)
    photos_per_obs = df_clean["num_photos"]
    obs_per_observer = df_clean.groupby("observer_id")["observation_uuid"].count()

    structure_summary = {
        "clean_observations_count": int(len(df_clean)),
        "unique_taxa_count": int(df_clean["taxon_id"].nunique()),
        "unique_families_count": int(df_clean["family"].nunique()),
        "unique_observers_count": int(df_clean["observer_id"].nunique()),
        "valid_photos_count": int(raw_photo_count - corrupt_photos_count),
        "photos_per_observation": {
            "mean": round(float(photos_per_obs.mean()), 2),
            "median": float(photos_per_obs.median()),
            "max": int(photos_per_obs.max()),
            "pct_multi_photo_observations": round(float((photos_per_obs > 1).mean() * 100), 2),
        },
        "observations_per_observer": {
            "mean": round(float(obs_per_observer.mean()), 2),
            "median": float(obs_per_observer.median()),
            "max": int(obs_per_observer.max()),
            "top_10_pct_observers_share_of_data": round(
                float(obs_per_observer.nlargest(int(0.1 * len(obs_per_observer))).sum() / len(df_clean) * 100), 2
            ),
        },
        "regional_split": df_clean["region"].value_counts().to_dict(),
        "toxicity_class_distribution": df_clean["toxicity_class"].value_counts().to_dict(),
    }

    return df_clean, {"quality_audit": quality_issues, "structure_audit": structure_summary}


def analyze_long_tail_distribution(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyzes class imbalance and defines Head (>480), Medium (220-480), and Tail (<220) tiers.
    """
    class_counts = (
        df_clean.groupby(["taxon_id", "scientific_name", "common_name", "toxicity_class", "family"])
        .size()
        .reset_index(name="obs_count")
        .sort_values("obs_count", ascending=False)
        .reset_index(drop=True)
    )
    class_counts["rank"] = np.arange(1, len(class_counts) + 1)
    class_counts["frequency_tier"] = pd.cut(
        class_counts["obs_count"],
        bins=[0, 220, 480, 100000],
        labels=["Tail (<220)", "Medium (220-480)", "Head (>480)"],
    )

    counts_arr = class_counts["obs_count"].values
    imbalance_ratio = float(counts_arr.max() / counts_arr.min())
    gini = compute_gini_coefficient(counts_arr)

    tier_summary = (
        class_counts.groupby("frequency_tier", observed=False)
        .agg(
            num_species=("taxon_id", "count"),
            total_observations=("obs_count", "sum"),
            mean_observations=("obs_count", "mean"),
        )
        .reset_index()
    )

    long_tail_stats = {
        "max_class_count": int(counts_arr.max()),
        "min_class_count": int(counts_arr.min()),
        "imbalance_ratio_max_to_min": round(imbalance_ratio, 2),
        "gini_coefficient": round(gini, 4),
        "top_5_species_share_pct": round(float(counts_arr[:5].sum() / counts_arr.sum() * 100), 2),
        "bottom_10_species_share_pct": round(float(counts_arr[-10:].sum() / counts_arr.sum() * 100), 2),
        "tiers": tier_summary.to_dict(orient="records"),
    }

    return class_counts, long_tail_stats


def audit_leakage_and_baselines(
    df_clean: pd.DataFrame, df_photos: pd.DataFrame, seed: int = 42
) -> Dict[str, Any]:
    """
    Quantifies the exact risk of Data Leakage across three split strategies and evaluates Phase 1 Baselines.
    """
    logger.info("Auditing Data Leakage and evaluating Phase 1 Initial Baselines...")

    valid_photos = df_photos[df_photos["observation_uuid"].isin(df_clean["observation_uuid"])].copy()
    p_train, p_test = train_test_split(valid_photos, test_size=0.2, random_state=seed)
    shared_obs_in_photo_split = set(p_train["observation_uuid"]).intersection(set(p_test["observation_uuid"]))
    pct_test_photos_leaked_obs = float(
        p_test["observation_uuid"].isin(shared_obs_in_photo_split).mean() * 100
    )

    o_train, o_test = train_test_split(
        df_clean, test_size=0.2, stratify=df_clean["taxon_id"], random_state=seed
    )
    shared_observers = set(o_train["observer_id"]).intersection(set(o_test["observer_id"]))
    pct_test_obs_leaked_observer = float(o_test["observer_id"].isin(shared_observers).mean() * 100)

    df_na = df_clean[df_clean["region"] == "north_america"].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(df_na, df_na["taxon_id"], groups=df_na["observer_id"]))
    na_train = df_na.iloc[train_idx]
    na_val = df_na.iloc[val_idx]

    assert len(set(na_train["observation_uuid"]).intersection(set(na_val["observation_uuid"]))) == 0
    assert len(set(na_train["observer_id"]).intersection(set(na_val["observer_id"]))) == 0

    feature_cols = ["latitude", "longitude", "month", "day_of_year"]
    X_train, y_train = na_train[feature_cols], na_train["taxon_id"]
    X_val, y_val = na_val[feature_cols], na_val["taxon_id"]

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    y_pred_dummy = dummy_clf.predict(X_val)

    shortcut_clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=25, random_state=seed)
    shortcut_clf.fit(X_train[["latitude", "month", "day_of_year"]], y_train)
    y_pred_shortcut = shortcut_clf.predict(X_val[["latitude", "month", "day_of_year"]])

    df_eu = df_clean[df_clean["region"] == "europe"].copy()
    shortcut_full = DecisionTreeClassifier(max_depth=6, min_samples_leaf=25, random_state=seed)
    shortcut_full.fit(X_train[feature_cols], y_train)
    y_pred_val_full_meta = shortcut_full.predict(X_val[feature_cols])
    y_pred_eu_full_meta = shortcut_full.predict(df_eu[feature_cols])

    leakage_and_baseline_report = {
        "leakage_audit": {
            "naive_photo_split_test_leakage_pct": round(pct_test_photos_leaked_obs, 2),
            "naive_observation_split_observer_overlap_pct": round(pct_test_obs_leaked_observer, 2),
            "observer_grouped_split_observation_overlap_pct": 0.0,
            "observer_grouped_split_observer_overlap_pct": 0.0,
        },
        "phase1_baselines_40_classes": {
            "random_guess_theoretical_accuracy_pct": 2.50,
            "majority_class_baseline": {
                "val_accuracy_pct": round(float(accuracy_score(y_val, y_pred_dummy) * 100), 2),
                "val_macro_f1_pct": round(float(f1_score(y_val, y_pred_dummy, average="macro") * 100), 2),
            },
            "season_plus_latitude_baseline": {
                "val_accuracy_pct": round(float(accuracy_score(y_val, y_pred_shortcut) * 100), 2),
                "val_macro_f1_pct": round(float(f1_score(y_val, y_pred_shortcut, average="macro") * 100), 2),
                "val_balanced_acc_pct": round(float(balanced_accuracy_score(y_val, y_pred_shortcut) * 100), 2),
            },
            "full_lat_lon_season_shortcut_model": {
                "in_domain_na_val_accuracy_pct": round(float(accuracy_score(y_val, y_pred_val_full_meta) * 100), 2),
                "in_domain_na_val_macro_f1_pct": round(float(f1_score(y_val, y_pred_val_full_meta, average="macro") * 100), 2),
                "geographic_shift_eu_accuracy_pct": round(
                    float(accuracy_score(df_eu["taxon_id"], y_pred_eu_full_meta) * 100), 2
                ),
                "geographic_shift_eu_macro_f1_pct": round(
                    float(f1_score(df_eu["taxon_id"], y_pred_eu_full_meta, average="macro") * 100), 2
                ),
            },
        },
    }
    return leakage_and_baseline_report


def generate_phase1_figures(
    df_clean: pd.DataFrame,
    class_counts: pd.DataFrame,
    audit_results: Dict[str, Any],
    leakage_results: Dict[str, Any],
    figures_dir: str,
) -> None:
    """
    Generates 5 multi-panel publication-quality figures for Phase 1.
    """
    os.makedirs(figures_dir, exist_ok=True)
    logger.info(f"Generating Phase 1 figures in {figures_dir} ...")

    # --- FIGURE 1: Long-Tail Distribution & Toxicity Class Breakdown ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2.2, 1]})
    palette_map = {
        "Deadly_Toxic": "#d95f02",
        "Inedible_MildToxic": "#e6ab02",
        "Edible": "#1b9e77",
    }
    colors = [palette_map[c] for c in class_counts["toxicity_class"]]
    ax1.bar(
        class_counts["rank"],
        class_counts["obs_count"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
    )
    ax1.axhline(480, color="#444444", linestyle="--", linewidth=1.0, label="Head / Medium Threshold (480)")
    ax1.axhline(220, color="#888888", linestyle=":", linewidth=1.0, label="Medium / Tail Threshold (220)")
    ax1.set_title("A. Long-Tail Distribution Across 40 Target Fungi Species", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Species Rank (Ordered by Observation Frequency)", fontsize=10)
    ax1.set_ylabel("Number of Research-Grade Observations", fontsize=10)

    legend_elements = [
        Patch(facecolor="#d95f02", edgecolor="black", label="Deadly / Highly Toxic (12 spp.)"),
        Patch(facecolor="#e6ab02", edgecolor="black", label="Inedible / Mildly Toxic (8 spp.)"),
        Patch(facecolor="#1b9e77", edgecolor="black", label="Choice Edible (20 spp.)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9)

    sorted_counts = np.sort(class_counts["obs_count"].values)[::-1]
    cum_pct = np.cumsum(sorted_counts) / np.sum(sorted_counts) * 100
    x_pct = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
    ax2.plot(x_pct, cum_pct, marker="o", markersize=3.5, color="#2b5c8f", linewidth=2, label="Observed Lorenz Curve")
    ax2.plot([0, 100], [0, 100], linestyle="--", color="gray", label="Uniform Balance Line")
    ax2.fill_between(x_pct, x_pct, cum_pct, color="#2b5c8f", alpha=0.15)
    ax2.set_title("B. Cumulative Data Share (Imbalance)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Percentage of Species (%)", fontsize=10)
    ax2.set_ylabel("Cumulative Share of Observations (%)", fontsize=10)
    ax2.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_long_tail_distribution.png"))
    plt.close()

    # --- FIGURE 2: Observation Hierarchy & Observer Power-Law (Univariate EDA) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    photo_counts = df_clean["num_photos"].value_counts().sort_index()
    x_pos = np.arange(len(photo_counts))
    axes[0].bar(x_pos, photo_counts.values, color="#386cb0", edgecolor="black", alpha=0.85)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"{k} photo(s)" for k in photo_counts.index])
    axes[0].set_title("A. Photos per Observation (Multi-View Rate)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Photos Attached to Observation", fontsize=10)
    axes[0].set_ylabel("Number of Observations", fontsize=10)
    axes[0].set_ylim(0, photo_counts.values.max() * 1.15)
    for idx_b, val in enumerate(photo_counts.values):
        axes[0].text(idx_b, val + (photo_counts.values.max() * 0.02), f"{val/len(df_clean)*100:.1f}%", ha="center", fontsize=8.5)

    obs_per_user = df_clean.groupby("observer_id").size()
    axes[1].hist(obs_per_user, bins=35, color="#7570b3", edgecolor="black", log=True, alpha=0.85)
    axes[1].set_title("B. Observer Contribution Distribution (Log Y)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Observations Contributed per Observer", fontsize=10)
    axes[1].set_ylabel("Number of Observers (Log Scale)", fontsize=10)

    valid_acc = df_clean["positional_accuracy"].dropna()
    axes[2].hist(np.log10(valid_acc.clip(lower=1.0)), bins=35, color="#e7298a", edgecolor="black", alpha=0.85)
    axes[2].axvline(np.log10(10000), color="red", linestyle="--", linewidth=1.2, label="10 km Outlier Cutoff")
    axes[2].set_title("C. GPS Positional Accuracy (log10 Meters)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("log10(Positional Accuracy in Meters)", fontsize=10)
    axes[2].set_ylabel("Frequency", fontsize=10)
    axes[2].legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_hierarchy_and_univariate_audit.png"))
    plt.close()

    # --- FIGURE 3: Spatiotemporal Distribution & Shortcut Risk (Multivariate EDA) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.5))
    top_families = ["Morchellaceae", "Discinaceae", "Amanitaceae", "Boletaceae", "Cantharellaceae", "Physalacriaceae"]
    df_fam_month = (
        df_clean[df_clean["family"].isin(top_families)]
        .groupby(["family", "month"])
        .size()
        .unstack(fill_value=0)
    )
    df_fam_month_pct = df_fam_month.div(df_fam_month.sum(axis=1), axis=0) * 100
    sns.heatmap(
        df_fam_month_pct,
        annot=True,
        fmt=".1f",
        cmap="YlOrBr",
        cbar_kws={"label": "Row Percentage (%)"},
        ax=ax1,
    )
    ax1.set_title("A. Seasonal Phenology by Fungal Family (% per Month)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Month of Year (1=Jan, 12=Dec)", fontsize=10)
    ax1.set_ylabel("Fungal Taxonomic Family", fontsize=10)

    sns.violinplot(
        data=df_clean,
        x="region",
        y="latitude",
        hue="toxicity_class",
        palette=palette_map,
        inner="quartile",
        ax=ax2,
    )
    ax2.set_title("B. Latitudinal Distribution Across Regions & Toxicity Classes", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Geographic Domain (Train/Val vs Shift Test Domain)", fontsize=10)
    ax2.set_ylabel("Latitude (Degrees North)", fontsize=10)
    ax2.legend(title="Toxicity Class", loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_spatiotemporal_multivariate_patterns.png"))
    plt.close()

    # --- FIGURE 4: Look-Alike Groups (Deadly/Toxic vs Edible Co-Occurrence) ---
    fig, ax = plt.subplots(figsize=(13, 5.5))
    lookalike_ct = (
        df_clean.groupby(["lookalike_group", "toxicity_class"])
        .size()
        .unstack(fill_value=0)[["Edible", "Inedible_MildToxic", "Deadly_Toxic"]]
    )
    lookalike_ct.plot(
        kind="barh",
        stacked=True,
        color=["#1b9e77", "#e6ab02", "#d95f02"],
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title(
        "Morphological Look-Alike Groups: Co-occurrence of Choice Edible and Deadly/Toxic Species",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Observations", fontsize=10)
    ax.set_ylabel("Morphological Look-Alike Guild", fontsize=10)
    ax.legend(["Edible", "Inedible / Mildly Toxic", "Deadly / Highly Toxic"], loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_lookalike_risk_guilds.png"))
    plt.close()

    # --- FIGURE 5: Data Leakage Audit & Baseline Shortcut Comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    leak_data = leakage_results["leakage_audit"]
    split_names = [
        "Naive Photo Split\n(Shared Observation)",
        "Observation Split\n(Shared Observer)",
        "Observer-Grouped Split\n(Track C Standard)",
    ]
    leak_vals = [
        leak_data["naive_photo_split_test_leakage_pct"],
        leak_data["naive_observation_split_observer_overlap_pct"],
        leak_data["observer_grouped_split_observer_overlap_pct"],
    ]
    bars1 = ax1.bar(split_names, leak_vals, color=["#d95f02", "#e6ab02", "#1b9e77"], edgecolor="black", width=0.55)
    ax1.set_title("A. Information Leakage Rate Across Split Designs", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Test Samples Sharing Group with Train (%)", fontsize=10)
    ax1.set_ylim(0, 100)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 2.0, f"{h:.1f}%", ha="center", fontweight="bold", fontsize=9.5)

    base_data = leakage_results["phase1_baselines_40_classes"]
    model_labels = [
        "Majority Class\nBaseline",
        "Season + Lat\nShortcut (NA Val)",
        "Full GPS + Season\nShortcut (NA Val)",
        "Full GPS + Season\nTested on Europe (Shift!)",
    ]
    acc_vals = [
        base_data["majority_class_baseline"]["val_accuracy_pct"],
        base_data["season_plus_latitude_baseline"]["val_accuracy_pct"],
        base_data["full_lat_lon_season_shortcut_model"]["in_domain_na_val_accuracy_pct"],
        base_data["full_lat_lon_season_shortcut_model"]["geographic_shift_eu_accuracy_pct"],
    ]
    bars2 = ax2.bar(
        model_labels,
        acc_vals,
        color=["#888888", "#386cb0", "#1f78b4", "#e41a1c"],
        edgecolor="black",
        width=0.55,
    )
    ax2.set_title("B. Metadata Shortcut Power & Collapse Under Geographic Shift", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("40-Class Top-1 Accuracy (%)", fontsize=10)
    ax2.set_ylim(0, max(acc_vals) * 1.25)
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.2f}%", ha="center", fontweight="bold", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig5_leakage_and_shortcut_baselines.png"))
    plt.close()

    logger.info("All Phase 1 figures successfully saved.")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Data Quality Audit, EDA, and Baselines.")
    parser.add_argument("--config", type=str, default="configs/phase1_config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["random_seed"]

    obs_path = config["data_paths"]["raw_metadata_csv"]
    photos_path = config["data_paths"]["raw_photos_csv"]
    species_path = config["data_paths"]["species_metadata_csv"]

    if not (os.path.exists(obs_path) and os.path.exists(photos_path) and os.path.exists(species_path)):
        raise FileNotFoundError(
            "Raw data files not found. Please run `python src/data_fetcher.py` first."
        )

    df_obs = pd.read_csv(obs_path)
    df_photos = pd.read_csv(photos_path)
    df_species = pd.read_csv(species_path)

    # 1. Audit structure and quality
    df_clean, audit_results = audit_structure_and_quality(df_obs, df_photos, df_species)

    # 2. Analyze long-tail distribution
    class_counts, long_tail_stats = analyze_long_tail_distribution(df_clean)

    # 3. Audit leakage & evaluate Phase 1 baselines
    leakage_and_baselines = audit_leakage_and_baselines(df_clean, df_photos, seed=seed)

    # 4. Generate publication-ready figures
    figures_dir = config["data_paths"]["figures_dir"]
    generate_phase1_figures(df_clean, class_counts, audit_results, leakage_and_baselines, figures_dir)

    # 5. Save processed audited dataset (with resilient Parquet + CSV support)
    out_parquet = config["data_paths"]["audited_observations_parquet"]
    saved_data_path = save_processed_dataframe(df_clean, out_parquet)

    full_summary = {
        "phase": 1,
        "dataset_track": config["project"]["track"],
        "saved_processed_data_path": saved_data_path,
        "structure_and_quality": audit_results,
        "long_tail_analysis": long_tail_stats,
        "leakage_and_baselines": leakage_and_baselines,
    }

    summary_json_path = config["data_paths"]["summary_stats_json"]
    os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved complete Phase 1 audit summary JSON to: {summary_json_path}")
    print("\n" + "=" * 70)
    print("PHASE 1 AUDIT & BASELINE SUMMARY")
    print("=" * 70)
    print(f"Raw Observations:           {audit_results['quality_audit']['raw_observations_total']:,}")
    print(f"Clean Observations:         {audit_results['structure_audit']['clean_observations_count']:,}")
    print(f"Valid Photos:               {audit_results['structure_audit']['valid_photos_count']:,}")
    print(f"Unique Observers:           {audit_results['structure_audit']['unique_observers_count']:,}")
    print(f"Long-Tail Imbalance Ratio:  {long_tail_stats['imbalance_ratio_max_to_min']}x (Gini={long_tail_stats['gini_coefficient']})")
    print(f"Naive Photo-Split Leakage:  {leakage_and_baselines['leakage_audit']['naive_photo_split_test_leakage_pct']}% of test photos leaked!")
    print(f"Majority Baseline Acc:      {leakage_and_baselines['phase1_baselines_40_classes']['majority_class_baseline']['val_accuracy_pct']}%")
    print(f"Metadata Shortcut (NA Val): {leakage_and_baselines['phase1_baselines_40_classes']['full_lat_lon_season_shortcut_model']['in_domain_na_val_accuracy_pct']}%")
    print(f"Metadata Shortcut (EU Test):{leakage_and_baselines['phase1_baselines_40_classes']['full_lat_lon_season_shortcut_model']['geographic_shift_eu_accuracy_pct']}% (Collapse under Geographic Shift!)")
    print("=" * 70)


if __name__ == "__main__":
    main()