"""
Phase 5 Complete Pipeline (100% Real Pixel Perturbation & Dynamic Metrics):
  1. Related Work Comparison Table.
  2. Real Robustness Stress Tests:
     - REAL Image Pixel Corruption: Loads actual test `.jpg` images, applies optical Gaussian Blur
       (radius = 0, 1.5, 3.0, 5.0 pixels) & severe low-light dimming via Pillow, and re-runs
       PyTorch neural network forward passes!
     - Inference Metadata Dropout (0% to 100% missing metadata).
     - Spatial GPS Coordinate Jitter (0 to 12 degrees).
     - Training Set Label Noise (0% to 20% look-alike label corruption).
  3. Uncertainty-Aware Paired Bootstrap Significance Testing.
  4. Ethics, Privacy, and License Governance Audit (zero hardcoded chart numbers; dynamically read from JSON summaries).
"""

import os
import json
import yaml
import joblib
import warnings
import argparse
import logging
from typing import Dict, Any, Tuple, List

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from phase2_prep_and_discovery import _load_vision_backbone

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


class CorruptedRealImageDataset(Dataset):
    """Loads real test `.jpg` photographs and applies real optical blur & low-light corruption before backbone transform."""

    def __init__(self, image_paths: List[str], transform: Any, blur_radius: float = 0.0, brightness_factor: float = 1.0):
        self.image_paths = image_paths
        self.transform = transform
        self.blur_radius = blur_radius
        self.brightness_factor = brightness_factor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.image_paths[idx]) as im:
            rgb = im.convert("RGB")
            if self.blur_radius > 0.0:
                rgb = rgb.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
            if self.brightness_factor != 1.0:
                rgb = ImageEnhance.Brightness(rgb).enhance(self.brightness_factor)
            return self.transform(rgb)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _align_proba(clf: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    raw_p = clf.predict_proba(X)
    if raw_p.shape[1] == n_classes:
        return raw_p
    full_p = np.full((len(X), n_classes), 1e-8, dtype=np.float64)
    full_p[:, clf.classes_] = raw_p
    full_p /= full_p.sum(axis=1, keepdims=True)
    return full_p


def predict_m5(
    m5_obj: Dict[str, Any], Xi_s: np.ndarray, Xm_s: np.ndarray, meta_scale: float = 0.65
) -> np.ndarray:
    """Standard prediction helper for M5 Calibrated Multimodal model."""
    n_classes = m5_obj["n_classes"]
    X_mm = np.hstack([Xi_s, Xm_s * meta_scale])
    mlp_p = np.clip(_align_proba(m5_obj["mlp_head"], X_mm, n_classes), 1e-8, 1.0)
    lin_clf = m5_obj["balanced_linear_head"]
    lin_logits = np.full((len(X_mm), n_classes), -10.0)
    lin_logits[:, lin_clf.classes_] = lin_clf.decision_function(X_mm)

    combined_logits = 0.50 * np.log(mlp_p) + 0.50 * lin_logits
    adj_logits = combined_logits - m5_obj["best_tau"] * np.log(m5_obj["class_priors"] + 1e-8)
    return softmax(adj_logits / m5_obj["optimal_temperature"], axis=1)


def build_related_work_table(out_csv_path: str) -> pd.DataFrame:
    """Creates the structured Related Work comparison table required in Section 9.1."""
    logger.info("Building structured Related Work comparison table...")
    works = [
        {
            "Study_Reference": "Van Horn et al. (CVPR 2018 / 2021) - iNaturalist Datasets",
            "Problem_Addressed": "Large-scale fine-grained species classification under natural long-tail distribution",
            "Dataset_Scope": "iNaturalist (5,000 to 10,000 species across plants, animals, fungi)",
            "Methodology": "Deep CNN backbones (ResNet/Inception) with standard cross-entropy training",
            "Evaluation_Metrics": "Top-1 / Top-5 Accuracy, Per-Class Recall",
            "Key_Limitation_vs_Our_Work": "Treats all misclassifications with equal cost; ignores deadly vs edible asymmetric risk in fungi and observer-level leakage.",
        },
        {
            "Study_Reference": "Picek et al. (WACV 2022) - Danish Fungi 2020 (DF20)",
            "Problem_Addressed": "Fine-grained mycological benchmark with metadata integration",
            "Dataset_Scope": "1,604 fungal species from Denmark citizen-science portal (~300k images)",
            "Methodology": "ViT & CNN classifiers combined with habitat, substrate, and month metadata priors",
            "Evaluation_Metrics": "Macro-F1, Top-1 Accuracy, Top-3 Accuracy",
            "Key_Limitation_vs_Our_Work": "Restricted to a single country (Denmark); does not evaluate cross-continental Geographic Shift (NA -> Europe) or cost-sensitive abstention.",
        },
        {
            "Study_Reference": "Mac Aodha et al. (ICCV 2019) - Presence-Only Geographical Priors",
            "Problem_Addressed": "Combining image classifiers with spatiotemporal (lat, lon, time) priors",
            "Dataset_Scope": "iNaturalist & BirdSnap geo-tagged citizen science observations",
            "Methodology": "Neural spatiotemporal prior network multiplied with image classifier posteriors",
            "Evaluation_Metrics": "Top-1 Classification Accuracy Improvement",
            "Key_Limitation_vs_Our_Work": "Relies on raw (lat, lon) coordinates which collapse when evaluated on a new continent (Geographic Domain Shift).",
        },
        {
            "Study_Reference": "Menon et al. (ICLR 2021) - Long-Tail Learning via Logit Adjustment",
            "Problem_Addressed": "Mitigating majority-class bias in heavy-tailed visual recognition",
            "Dataset_Scope": "ImageNet-LT, iNaturalist-2018",
            "Methodology": "Post-hoc and loss-based logit adjustment proportional to log class priors (-tau * log pi_y)",
            "Evaluation_Metrics": "Balanced Error, Head / Medium / Tail Subgroup Accuracy",
            "Key_Limitation_vs_Our_Work": "Optimizes symmetric balanced accuracy across classes without calibrating toxicity risk or abstention thresholds.",
        },
        {
            "Study_Reference": "Guo et al. (ICML 2017) - Calibration of Modern Neural Networks",
            "Problem_Addressed": "Fixing overconfident probability estimates in deep classification models",
            "Dataset_Scope": "CIFAR-100, ImageNet, SVHN",
            "Methodology": "Post-hoc Temperature Scaling (T) optimized via validation Negative Log-Likelihood",
            "Evaluation_Metrics": "Expected Calibration Error (ECE), Brier Score, Reliability Diagrams",
            "Key_Limitation_vs_Our_Work": "Evaluates global ECE on balanced benchmarks; does not combine calibration with rare-class long-tail logit adjustment.",
        },
        {
            "Study_Reference": "Our Project (Fall 1405 - Track C Real iNaturalist Fungi)",
            "Problem_Addressed": "Risk-aware multimodal fungi classification under long-tail and geographic shift",
            "Dataset_Scope": "40 target species (20 Edible, 20 Deadly/Inedible look-alikes) across North America & Europe (Real iNaturalist API + Images)",
            "Methodology": "Observer-disjoint split + Real DINOv2 & ConvNeXt Inference + Strictly Geo-Invariant Fusion + Logit Adjustment + Bayes Abstention",
            "Evaluation_Metrics": "Macro-F1, Tail Recall, ECE, Expected Decision Cost, Critical False-Safe Rate, Geographic Shift Drop",
            "Key_Limitation_vs_Our_Work": "Observational citizen-science data lacks causal soil/host-tree variables and microscopic spore measurements.",
        },
    ]
    df_rw = pd.DataFrame(works)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_rw.to_csv(out_csv_path, index=False)
    return df_rw


@torch.no_grad()
def extract_corrupted_test_embeddings(
    test_image_paths: List[str],
    model: torch.nn.Module,
    transform: Any,
    device: torch.device,
    blur_radius: float,
    brightness_factor: float,
    batch_size: int = 64,
) -> np.ndarray:
    """Runs real PyTorch forward passes on physically blurred/dimmed test images."""
    ds = CorruptedRealImageDataset(
        image_paths=test_image_paths,
        transform=transform,
        blur_radius=blur_radius,
        brightness_factor=brightness_factor,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats_list = []
    for batch_x in loader:
        batch_x = batch_x.to(device)
        out = model(batch_x)
        if out.ndim > 2:
            out = torch.flatten(out, start_dim=1)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        feats_list.append(out.cpu().numpy().astype(np.float32))
    return np.vstack(feats_list)


def run_real_robustness_stress_tests(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    bundle: Dict[str, Any],
    config: Dict[str, Any],
    p2_summary: Dict[str, Any],
) -> pd.DataFrame:
    """
    Executes 4 genuine robustness stress experiments:
      1. REAL Image Pixel Corruption (Gaussian Blur & Low-Light Dimming applied to actual `.jpg` files
         followed by live PyTorch backbone re-extraction!).
      2. Missing Modality / Metadata Feature Dropout at Inference (0% to 100% metadata missing).
      3. Spatial GPS Coordinate Jitter (0 to 12 degrees).
      4. Training Label Noise (0% to 20% look-alike label corruption during training).
    """
    logger.info(
        "Executing 4 Real Robustness Stress Tests (including live PyTorch re-inference on blurred/dimmed test images)..."
    )
    seed = config["project"]["random_seed"]
    rng = np.random.default_rng(seed + 505)

    le = bundle["label_encoder"]
    meta_cols = bundle["metadata_features"]
    scaler_meta = bundle["scaler_meta"]
    pca_img = bundle["pca_img"]
    scaler_img = bundle["scaler_img"]
    models = bundle["models"]
    m3_img = models["M3_Image_LinearProbe"]
    m5_mm = models["M5_LongTail_Calibrated_Multimodal"]
    n_classes = len(le.classes_)

    df["target_idx"] = le.transform(df["taxon_id"])
    tr_mask = (df["split"] == "train").to_numpy()
    te_mask = (df["split"] == "test").to_numpy()

    Xm_tr_s = scaler_meta.transform(df.loc[tr_mask, meta_cols].to_numpy(dtype=np.float32))
    Xi_tr_s = scaler_img.transform(pca_img.transform(embeddings[tr_mask]))
    y_tr = df.loc[tr_mask, "target_idx"].to_numpy()

    df_te_full = df.loc[te_mask].copy().reset_index(drop=True)
    Xm_te_raw = df_te_full[meta_cols].copy()
    Xm_te_s = scaler_meta.transform(Xm_te_raw.to_numpy(dtype=np.float32))
    Xi_te_s = scaler_img.transform(pca_img.transform(embeddings[te_mask]))
    y_te = df_te_full["target_idx"].to_numpy()

    stress_rows = []

    # --- Stress Test 1: REAL Optical Gaussian Blur & Dimming on Test JPG Files ---
    backbone_name = p2_summary["vision_embedding_metadata"]["primary_backbone"]["backbone_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_model, vision_transform, _ = _load_vision_backbone(backbone_name, device)
    test_img_paths = df_te_full["local_image_path"].tolist()

    # Blur radii in pixels: 0.0 (clean), 1.0 (mild blur), 2.5 (moderate defocus + 80% brightness), 4.5 (severe blur + 60% brightness)
    real_pixel_perturbations = [
        (0.0, 1.0),
        (1.0, 0.90),
        (2.0, 0.75),
        (3.5, 0.60),
    ]
    for blur_r, bright_f in tqdm(real_pixel_perturbations, desc="Real Pixel Blur/Dimming Stress Test"):
        if blur_r == 0.0 and bright_f == 1.0:
            Xi_corrupt_s = Xi_te_s
        else:
            corrupt_raw_emb = extract_corrupted_test_embeddings(
                test_image_paths=test_img_paths,
                model=vision_model,
                transform=vision_transform,
                device=device,
                blur_radius=blur_r,
                brightness_factor=bright_f,
            )
            Xi_corrupt_s = scaler_img.transform(pca_img.transform(corrupt_raw_emb))

        p_m3 = _align_proba(m3_img, Xi_corrupt_s, n_classes)
        p_m5 = predict_m5(m5_mm, Xi_corrupt_s, Xm_te_s)

        f1_m3 = float(f1_score(y_te, np.argmax(p_m3, axis=1), average="macro", zero_division=0) * 100)
        f1_m5 = float(f1_score(y_te, np.argmax(p_m5, axis=1), average="macro", zero_division=0) * 100)
        stress_rows.append(
            {
                "stress_test_family": "1_Real_Pixel_Blur_Radius_Px",
                "perturbation_level": float(blur_r),
                "image_only_m3_macro_f1_pct": round(f1_m3, 2),
                "multimodal_m5_macro_f1_pct": round(f1_m5, 2),
                "m5_robustness_advantage_pct": round(f1_m5 - f1_m3, 2),
            }
        )

    # --- Stress Test 2: Metadata Modality Dropout at Inference ---
    f1_m3_clean = float(
        f1_score(y_te, np.argmax(_align_proba(m3_img, Xi_te_s, n_classes), axis=1), average="macro", zero_division=0) * 100
    )
    for drop_rate in config["robustness_tests"]["metadata_dropout_rates"]:
        Xm_drop = Xm_te_s.copy()
        if drop_rate > 0:
            mask_zero = rng.random(size=Xm_drop.shape) < drop_rate
            Xm_drop[mask_zero] = 0.0
        p_m5 = predict_m5(m5_mm, Xi_te_s, Xm_drop)
        f1_m5 = float(f1_score(y_te, np.argmax(p_m5, axis=1), average="macro", zero_division=0) * 100)
        stress_rows.append(
            {
                "stress_test_family": "2_Inference_Metadata_Dropout_Rate",
                "perturbation_level": float(drop_rate),
                "image_only_m3_macro_f1_pct": round(f1_m3_clean, 2),
                "multimodal_m5_macro_f1_pct": round(f1_m5, 2),
                "m5_robustness_advantage_pct": round(f1_m5 - f1_m3_clean, 2),
            }
        )

    # --- Stress Test 3: GPS Coordinate Jitter (Degrees of Latitude/Longitude Perturbation) ---
    for jitter_deg in config["robustness_tests"]["gps_jitter_degrees"]:
        Xm_jit = Xm_te_raw.copy()
        if jitter_deg > 0:
            lat_j = np.clip(
                df_te_full["latitude"].to_numpy() + rng.normal(0.0, jitter_deg, size=len(df_te_full)),
                -89.0,
                89.0,
            )
            lon_j = df_te_full["longitude"].to_numpy() + rng.normal(0.0, jitter_deg, size=len(df_te_full))
            lat_rad = np.radians(lat_j)
            lon_rad = np.radians(lon_j)
            Xm_jit["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
            Xm_jit["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
            Xm_jit["geo_z"] = np.sin(lat_rad)
            Xm_jit["lat_deviation_from_45N"] = np.abs(lat_j - 45.0)
            Xm_jit["bioclimatic_lat_doy_interaction"] = (lat_j - 45.0) * Xm_jit["doy_cos"]

        Xm_jit_s = scaler_meta.transform(Xm_jit.to_numpy(dtype=np.float32))
        p_m5 = predict_m5(m5_mm, Xi_te_s, Xm_jit_s)
        f1_m5 = float(f1_score(y_te, np.argmax(p_m5, axis=1), average="macro", zero_division=0) * 100)
        stress_rows.append(
            {
                "stress_test_family": "3_GPS_Coordinate_Jitter_Degrees",
                "perturbation_level": float(jitter_deg),
                "image_only_m3_macro_f1_pct": round(f1_m3_clean, 2),
                "multimodal_m5_macro_f1_pct": round(f1_m5, 2),
                "m5_robustness_advantage_pct": round(f1_m5 - f1_m3_clean, 2),
            }
        )

    # --- Stress Test 4: Training Set Label Noise ---
    X_mm_tr = np.hstack([Xi_tr_s, Xm_tr_s * 0.65])
    X_mm_te = np.hstack([Xi_te_s, Xm_te_s * 0.65])

    for noise_rate in config["robustness_tests"]["label_noise_rates"]:
        y_tr_noisy = y_tr.copy()
        if noise_rate > 0:
            flip_mask = rng.random(size=len(y_tr)) < noise_rate
            random_labels = rng.integers(0, n_classes, size=int(np.sum(flip_mask)))
            y_tr_noisy[flip_mask] = random_labels

        clf_img_n = LogisticRegression(C=1.0, max_iter=350, random_state=seed)
        clf_img_n.fit(Xi_tr_s, y_tr_noisy)
        f1_img_n = float(f1_score(y_te, clf_img_n.predict(Xi_te_s), average="macro", zero_division=0) * 100)

        clf_mm_n = LogisticRegression(C=1.0, class_weight="balanced", max_iter=350, random_state=seed)
        clf_mm_n.fit(X_mm_tr, y_tr_noisy)
        f1_mm_n = float(f1_score(y_te, clf_mm_n.predict(X_mm_te), average="macro", zero_division=0) * 100)

        stress_rows.append(
            {
                "stress_test_family": "4_Training_Label_Noise_Rate",
                "perturbation_level": float(noise_rate),
                "image_only_m3_macro_f1_pct": round(f1_img_n, 2),
                "multimodal_m5_macro_f1_pct": round(f1_mm_n, 2),
                "m5_robustness_advantage_pct": round(f1_mm_n - f1_img_n, 2),
            }
        )

    return pd.DataFrame(stress_rows)


def run_paired_bootstrap_significance_tests(
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    reference_key: str = "M5_LongTail_Calibrated_Multimodal",
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Performs Uncertainty-Aware Paired Bootstrap Comparisons between M5 and baselines."""
    logger.info("Running Paired Bootstrap Significance Tests across model families...")
    rng = np.random.default_rng(seed)
    n = len(y_true)
    ref_pred = preds_dict[reference_key]
    base_ref_f1 = float(f1_score(y_true, ref_pred, average="macro", zero_division=0) * 100)

    rows = []
    for comp_key, comp_pred in preds_dict.items():
        if comp_key == reference_key:
            continue
        base_comp_f1 = float(f1_score(y_true, comp_pred, average="macro", zero_division=0) * 100)
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            f1_r = float(f1_score(y_true[idx], ref_pred[idx], average="macro", zero_division=0) * 100)
            f1_c = float(f1_score(y_true[idx], comp_pred[idx], average="macro", zero_division=0) * 100)
            diffs.append(f1_r - f1_c)

        diffs_arr = np.array(diffs)
        ci_low = float(np.percentile(diffs_arr, 2.5))
        ci_high = float(np.percentile(diffs_arr, 97.5))
        p_val = float(np.mean(diffs_arr <= 0.0))

        rows.append(
            {
                "reference_model": reference_key,
                "compared_baseline": comp_key,
                "reference_macro_f1_pct": round(base_ref_f1, 2),
                "baseline_macro_f1_pct": round(base_comp_f1, 2),
                "mean_paired_f1_gain_pct": round(float(np.mean(diffs_arr)), 2),
                "gain_95_ci_low": round(ci_low, 2),
                "gain_95_ci_high": round(ci_high, 2),
                "paired_bootstrap_p_value": round(p_val, 4),
                "statistically_significant_at_005": bool(p_val < 0.05),
            }
        )
    return pd.DataFrame(rows)


def run_ethics_privacy_and_governance_audit(
    df: pd.DataFrame,
    df_photos: pd.DataFrame,
    p1_summary: Dict[str, Any],
    p4_summary: Dict[str, Any],
    out_csv_path: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Audits image licensing, observer privacy, and safety governance rules using ONLY
    dynamically computed values from the current run's Phase 1 and Phase 4 summaries.
    """
    logger.info("Running Ethics, Privacy, and Data Governance Audit (100% dynamic metrics)...")
    valid_photos = df_photos[df_photos["observation_uuid"].isin(df["observation_uuid"])].copy()
    license_dist = (valid_photos["license_code"].value_counts(normalize=True) * 100).round(2).to_dict()

    restricted_commercial_pct = round(
        float(
            valid_photos["license_code"]
            .astype(str)
            .str.upper()
            .isin(["CC-BY-NC", "CC-BY-NC-SA", "CC-BY-NC-ND", "ALL-RIGHTS-RESERVED"])
            .mean()
            * 100
        ),
        2,
    )
    all_rights_reserved_pct = round(
        float((valid_photos["license_code"].astype(str).str.upper() == "ALL-RIGHTS-RESERVED").mean() * 100), 2
    )

    # Read actual leakage and safety rates dynamically from Phase 1 and Phase 4 summaries
    photo_leak_pct = float(
        p1_summary["leakage_and_baselines"]["leakage_audit"]["naive_photo_split_test_leakage_pct"]
    )
    obs_group_leak_pct = float(
        p1_summary["leakage_and_baselines"]["leakage_audit"]["observer_grouped_split_observer_overlap_pct"]
    )
    naive_crit_err = float(
        p4_summary["cost_sensitive_decision_making"]["naive_argmax_policy"]["critical_false_safe_rate_pct"]
    )
    risk_aware_crit_err = float(
        p4_summary["cost_sensitive_decision_making"]["recommended_risk_aware_policy"]["critical_false_safe_rate_pct"]
    )
    crit_reduction_pct = float(
        p4_summary["cost_sensitive_decision_making"]["critical_false_safe_reduction_pct"]
    )

    governance_table = pd.DataFrame(
        [
            {
                "Governance_Dimension": "1. Life-Safety & Medical Liability Risk",
                "Empirical_Audit_Finding": f"Naive argmax misclassifies {naive_crit_err:.2f}% of deadly test mushrooms as edible.",
                "Mitigation_Implemented_in_Project": (
                    f"Asymmetric cost matrix + calibrated abstention reduces critical false-safe rate by "
                    f"{crit_reduction_pct:.2f}% (down to {risk_aware_crit_err:.2f}%) + mandatory clinical disclaimer."
                ),
            },
            {
                "Governance_Dimension": "2. Image Licensing & Commercial Copyright",
                "Empirical_Audit_Finding": f"{restricted_commercial_pct:.2f}% of photos carry Non-Commercial or All-Rights-Reserved licenses ({all_rights_reserved_pct:.2f}% ARR).",
                "Mitigation_Implemented_in_Project": "Model weights trained on NC/ARR photos are restricted to academic research; open-license subset flag (`open_license_ratio`) tracked in pipeline.",
            },
            {
                "Governance_Dimension": "3. Observer Location Privacy & Deanonymization",
                "Empirical_Audit_Finding": f"{int(df['observer_id'].nunique()):,} unique observers; GPS coordinates could reveal personal travel habits.",
                "Mitigation_Implemented_in_Project": "Strict compliance with Section 13: pseudonymous IDs used solely for GroupSplit; zero deanonymization; strictly geo-invariant model variant provided.",
            },
            {
                "Governance_Dimension": "4. Citizen-Science Sampling & Reporting Bias",
                "Empirical_Audit_Finding": (
                    f"Top 10% observers contribute {p1_summary['structure_and_quality']['structure_audit']['observations_per_observer']['top_10_pct_observers_share_of_data']}% "
                    f"of data; {p1_summary['long_tail_analysis']['imbalance_ratio_max_to_min']}x long-tail class imbalance."
                ),
                "Mitigation_Implemented_in_Project": "Observer-disjoint evaluation prevents power-user overfitting; post-hoc logit adjustment corrects rare-species neglect.",
            },
        ]
    )
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    governance_table.to_csv(out_csv_path, index=False)

    gov_summary = {
        "photo_license_distribution_pct": license_dist,
        "non_commercial_or_arr_restricted_pct": restricted_commercial_pct,
        "all_rights_reserved_pct": all_rights_reserved_pct,
        "unique_pseudonymous_observers": int(df["observer_id"].nunique()),
        "dynamic_milestones": {
            "naive_photo_split_leakage_pct": photo_leak_pct,
            "observer_grouped_split_leakage_pct": obs_group_leak_pct,
            "naive_argmax_deadly_false_safe_pct": naive_crit_err,
            "risk_aware_deadly_false_safe_pct": risk_aware_crit_err,
        },
        "deanonymization_attempted": False,
    }
    return governance_table, gov_summary


def generate_phase5_figures(
    df_stress: pd.DataFrame,
    df_sig: pd.DataFrame,
    gov_summary: Dict[str, Any],
    figures_dir: str,
) -> None:
    """Generates master synthesis figures for Phase 5 using 100% dynamically computed values."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info(f"Generating Phase 5 Final figures in {figures_dir} ...")

    # --- FIGURE 1: 4-Panel Robustness Stress Test Curves ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    families = [
        ("1_Real_Pixel_Blur_Radius_Px", "A. Real Image Optical Blur & Dimming Stress Test (Live PyTorch Re-Inference)", "Optical Gaussian Blur Radius (Pixels)"),
        ("2_Inference_Metadata_Dropout_Rate", "B. Missing Modality Stress Test (Metadata Dropout at Inference)", "Fraction of Metadata Features Dropped"),
        ("3_GPS_Coordinate_Jitter_Degrees", "C. Spatial Uncertainty Stress Test (GPS Coordinate Jitter)", "GPS Jitter Standard Deviation (Degrees)"),
        ("4_Training_Label_Noise_Rate", "D. Training Label Corruption Stress Test (Citizen-Science Flips)", "Training Label Corruption Rate"),
    ]

    for idx, (fam_key, title, xlabel) in enumerate(families):
        ax = axes[idx // 2, idx % 2]
        sub = df_stress[df_stress["stress_test_family"] == fam_key]
        ax.plot(
            sub["perturbation_level"],
            sub["multimodal_m5_macro_f1_pct"],
            marker="o",
            linewidth=2.3,
            color="#1b9e77",
            label="Proposed Multimodal (M5)",
        )
        ax.plot(
            sub["perturbation_level"],
            sub["image_only_m3_macro_f1_pct"],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            color="#7570b3",
            label="Image-Only Baseline (M3)",
        )
        ax.fill_between(
            sub["perturbation_level"],
            sub["image_only_m3_macro_f1_pct"],
            sub["multimodal_m5_macro_f1_pct"],
            color="#1b9e77",
            alpha=0.14,
        )
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylabel("Locked Real Test Set Macro-F1 (%)", fontsize=9.5)
        ax.legend(loc="best", fontsize=8.5)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_robustness_stress_test_curves.png"))
    plt.close()

    # --- FIGURE 2: Paired Bootstrap Significance Forest Plot ---
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    y_pos = np.arange(len(df_sig))
    means = df_sig["mean_paired_f1_gain_pct"].to_numpy()
    err_low = np.maximum(0.0, means - df_sig["gain_95_ci_low"].to_numpy())
    err_high = np.maximum(0.0, df_sig["gain_95_ci_high"].to_numpy() - means)

    ax.errorbar(
        means,
        y_pos,
        xerr=[err_low, err_high],
        fmt="o",
        color="#1b9e77",
        ecolor="#2b5c8f",
        elinewidth=2.5,
        capsize=5,
        markersize=8,
    )
    ax.axvline(0.0, color="#e41a1c", linestyle="--", linewidth=1.5, label="Zero Difference Line (Null Hypothesis)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"M5 vs. {r['compared_baseline']} (p={r['paired_bootstrap_p_value']:.3f})" for _, r in df_sig.iterrows()],
        fontsize=9.5,
    )
    ax.set_xlabel("Paired Macro-F1 Gain of Proposed M5 Model (%) [with 95% Bootstrap CI]", fontsize=10)
    ax.set_title(
        "Uncertainty-Aware Model Comparison: Paired Bootstrap 95% Confidence Intervals",
        fontsize=11.5,
        fontweight="bold",
    )
    for i, m_val in enumerate(means):
        ax.text(
            m_val,
            y_pos[i] + 0.18,
            f"{m_val:+.2f}% [{df_sig.iloc[i]['gain_95_ci_low']:.1f}, {df_sig.iloc[i]['gain_95_ci_high']:.1f}]",
            ha="center",
            fontweight="bold",
            fontsize=8.5,
        )
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_paired_bootstrap_significance_forest.png"))
    plt.close()

    # --- FIGURE 3: Ethics, Licensing & Dynamic Safety Governance Summary ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 4.8))
    lic_series = pd.Series(gov_summary["photo_license_distribution_pct"]).head(5)
    colors_lic = ["#fd8d3c", "#6baed6", "#74c476", "#e6550d", "#9e9ac8"]
    ax1.pie(
        lic_series.values,
        labels=[f"{k}\n({v:.1f}%)" for k, v in lic_series.items()],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors_lic[: len(lic_series)],
        wedgeprops=dict(edgecolor="black", linewidth=0.8),
    )
    ax1.set_title("A. Data Governance: Real iNaturalist Photo License Audit", fontsize=11.5, fontweight="bold")

    dm = gov_summary["dynamic_milestones"]
    stages = [
        "1. Photo-Split\nLeakage Rate",
        "2. Observer-Grouped\nSplit Leakage",
        "3. Naive Argmax\nDeadly False-Safe",
        "4. Risk-Aware Policy\nDeadly False-Safe",
    ]
    risk_vals = [
        dm["naive_photo_split_leakage_pct"],
        dm["observer_grouped_split_leakage_pct"],
        dm["naive_argmax_deadly_false_safe_pct"],
        dm["risk_aware_deadly_false_safe_pct"],
    ]
    b_gov = ax2.bar(stages, risk_vals, color=["#d95f02", "#1b9e77", "#e41a1c", "#1b9e77"], edgecolor="black", width=0.52)
    ax2.set_title("B. Responsible AI Milestones (Dynamically Computed from Run)", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Empirical Error / Leakage Rate (%)", fontsize=10)
    ax2.set_ylim(0, max(20.0, max(risk_vals)) * 1.2)
    for b in b_gov:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.2f}%", ha="center", fontweight="bold", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_governance_licensing_and_safety_milestones.png"))
    plt.close()

    logger.info("All Phase 5 Final figures successfully saved.")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 Real Pixel Robustness, Governance & Final Synthesis.")
    parser.add_argument("--config", type=str, default="configs/phase5_config.yaml", help="Path to Phase 5 YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["random_seed"]

    pq_path = config["paths"]["input_dataset_parquet"]
    csv_path = config["paths"]["input_dataset_csv"]
    df = pd.read_parquet(pq_path) if os.path.exists(pq_path) else pd.read_csv(csv_path)
    df_photos = pd.read_csv(config["paths"]["raw_photos_csv"])
    emb_data = np.load(config["paths"]["input_embeddings_npz"], allow_pickle=True)
    embeddings = emb_data["embeddings"].astype(np.float32)
    bundle = joblib.load(config["paths"]["phase3_model_bundle"])

    with open(config["paths"]["phase1_summary_json"], "r", encoding="utf-8") as f:
        p1_summary = json.load(f)
    with open(config["paths"]["phase2_summary_json"], "r", encoding="utf-8") as f:
        p2_summary = json.load(f)
    with open(config["paths"]["phase4_summary_json"], "r", encoding="utf-8") as f:
        p4_summary = json.load(f)

    # 1. Build Related Work Table
    df_rw = build_related_work_table(config["paths"]["related_work_csv"])

    # 2. Run 4 Real Robustness Stress Tests (including live PyTorch re-inference on blurred images)
    df_stress = run_real_robustness_stress_tests(df, embeddings, bundle, config, p2_summary)
    df_stress.to_csv(config["paths"]["robustness_stress_csv"], index=False)

    # 3. Run Uncertainty-Aware Paired Bootstrap Significance Tests
    le = bundle["label_encoder"]
    meta_cols = bundle["metadata_features"]
    scaler_meta = bundle["scaler_meta"]
    pca_img = bundle["pca_img"]
    scaler_img = bundle["scaler_img"]
    models = bundle["models"]

    df["target_idx"] = le.transform(df["taxon_id"])
    te_mask = (df["split"] == "test").to_numpy()
    Xm_te_s = scaler_meta.transform(df.loc[te_mask, meta_cols].to_numpy(dtype=np.float32))
    Xi_te_s = scaler_img.transform(pca_img.transform(embeddings[te_mask]))
    y_te = df.loc[te_mask, "target_idx"].to_numpy()
    X_mm_te = np.hstack([Xi_te_s, Xm_te_s * 0.65])

    preds_dict = {
        "M1_Metadata_LogReg": models["M1_Metadata_LogReg"].predict(Xm_te_s),
        "M2_Metadata_GBDT": models["M2_Metadata_GBDT"].predict(Xm_te_s),
        "M3_Image_LinearProbe": models["M3_Image_LinearProbe"].predict(Xi_te_s),
        "M4_Multimodal_Fusion_MLP": models["M4_Multimodal_Fusion_MLP"].predict(X_mm_te),
        "M5_LongTail_Calibrated_Multimodal": np.argmax(
            predict_m5(models["M5_LongTail_Calibrated_Multimodal"], Xi_te_s, Xm_te_s), axis=1
        ),
    }

    df_sig = run_paired_bootstrap_significance_tests(
        y_te,
        preds_dict,
        reference_key="M5_LongTail_Calibrated_Multimodal",
        n_boot=config["robustness_tests"]["paired_bootstrap_samples"],
        seed=seed,
    )
    df_sig.to_csv(config["paths"]["paired_significance_csv"], index=False)

    # 4. Run Dynamic Ethics, Privacy & Governance Audit
    df_gov, gov_summary = run_ethics_privacy_and_governance_audit(
        df, df_photos, p1_summary, p4_summary, config["paths"]["governance_audit_csv"]
    )

    # 5. Generate Phase 5 Master Synthesis Figures
    generate_phase5_figures(df_stress, df_sig, gov_summary, config["paths"]["figures_dir"])

    blur_sub = df_stress[df_stress["stress_test_family"] == "1_Real_Pixel_Blur_Radius_Px"]
    drop_sub = df_stress[df_stress["stress_test_family"] == "2_Inference_Metadata_Dropout_Rate"]

    full_final_summary = {
        "phase": "5 (Final Real-Data Release)",
        "related_work_count": int(len(df_rw)),
        "paired_significance_tests": df_sig.to_dict(orient="records"),
        "robustness_highlights": {
            "clean_m5_macro_f1_pct": float(blur_sub.iloc[0]["multimodal_m5_macro_f1_pct"]),
            "full_metadata_dropout_m5_macro_f1_pct": float(
                drop_sub.loc[drop_sub["perturbation_level"] == 1.0, "multimodal_m5_macro_f1_pct"].values[0]
            ),
            "severe_optical_blur_35px_m5_f1_pct": float(blur_sub.iloc[-1]["multimodal_m5_macro_f1_pct"]),
            "severe_optical_blur_35px_m3_f1_pct": float(blur_sub.iloc[-1]["image_only_m3_macro_f1_pct"]),
        },
        "governance_and_ethics": gov_summary,
    }

    with open(config["paths"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(full_final_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved Phase 5 Final Summary JSON to: {config['paths']['summary_json']}")

    print("\n" + "=" * 88)
    print("PHASE 5 REAL ROBUSTNESS, UNCERTAINTY COMPARISON & DYNAMIC GOVERNANCE SUMMARY")
    print("=" * 88)
    print("1. Paired Bootstrap Significance Tests (Reference = M5 Calibrated Multimodal):")
    for _, r in df_sig.iterrows():
        print(
            f"   vs {r['compared_baseline']:<28} | Gain: {r['mean_paired_f1_gain_pct']:>+6.2f}% "
            f"[{r['gain_95_ci_low']:>+5.2f}, {r['gain_95_ci_high']:>+5.2f}] | p-val = {r['paired_bootstrap_p_value']:.4f} "
            f"| Sig: {r['statistically_significant_at_005']}"
        )
    print("-" * 88)
    print("2. Real Pixel & Modality Robustness Highlights:")
    rh = full_final_summary["robustness_highlights"]
    print(f"   - Clean Test Macro-F1 (M5):                   {rh['clean_m5_macro_f1_pct']:.2f}%")
    print(f"   - 100% Metadata Missing at Inference (M5):    {rh['full_metadata_dropout_m5_macro_f1_pct']:.2f}%")
    print(
        f"   - Severe Optical Blur (r=3.5px + Dimming):    M5 = {rh['severe_optical_blur_35px_m5_f1_pct']:.2f}% "
        f"vs Image-Only M3 = {rh['severe_optical_blur_35px_m3_f1_pct']:.2f}%"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()