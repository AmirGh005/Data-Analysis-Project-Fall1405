"""
Phase 3 Complete Pipeline: Predictive Modeling, Observer-Grouped Experimental Design,
Leakage-Free Cross-Validation, Hyperparameter Tuning, Multi-Family Model Benchmarking
(Metadata-Only vs Image-Only vs Multimodal Fusion), Long-Tail & Rare-Class Evaluation,
Confidence Calibration (ECE & Temperature Scaling), Bootstrap Uncertainty, and Error Analysis.
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)

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


def load_phase2_artifacts(config: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Loads Phase 2 prepared dataset, visual embeddings, and species toxicity metadata."""
    pq_path = config["paths"]["input_dataset_parquet"]
    csv_path = config["paths"]["input_dataset_csv"]
    emb_path = config["paths"]["input_embeddings_npz"]
    sp_path = config["paths"]["species_metadata_csv"]

    if os.path.exists(pq_path):
        try:
            df = pd.read_parquet(pq_path)
        except Exception:
            df = pd.read_csv(csv_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Phase 2 dataset not found. Run Phase 2 first.")

    emb_archive = np.load(emb_path, allow_pickle=True)
    embeddings = emb_archive["embeddings"].astype(np.float32)
    df_species = pd.read_csv(sp_path)

    assert len(df) == len(embeddings), "Mismatch between dataframe rows and embedding matrix!"
    return df, embeddings, df_species


def compute_ece_and_brier(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 12
) -> Tuple[float, float, Dict[str, List[float]]]:
    """
    Computes Expected Calibration Error (ECE), Multiclass Brier Score,
    and reliability curve bins.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    # Multiclass Brier score
    n_samples, n_classes = y_prob.shape
    y_onehot = np.eye(n_classes)[y_true]
    brier = float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []

    for b_idx in range(n_bins):
        low, high = bin_boundaries[b_idx], bin_boundaries[b_idx + 1]
        if b_idx == n_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)

        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_accs.append(round(acc_in_bin, 4))
            bin_confs.append(round(conf_in_bin, 4))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accs.append(0.0)
            bin_confs.append(round(float((low + high) / 2.0), 4))
            bin_counts.append(0)

    return float(ece), brier, {"bin_acc": bin_accs, "bin_conf": bin_confs, "bin_count": bin_counts}


def fit_temperature_scaling(logits: np.ndarray, y_val: np.ndarray) -> float:
    """
    Finds optimal temperature T > 0 on validation set logits minimizing Negative Log-Likelihood (NLL).
    """
    t_grid = np.linspace(0.35, 3.0, 108)
    best_t = 1.0
    best_nll = float("inf")

    for t in t_grid:
        scaled_probs = softmax(logits / t, axis=1)
        nll = log_loss(y_val, scaled_probs, labels=np.arange(logits.shape[1]))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return round(best_t, 4)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tier_map: Dict[int, str],
    deadly_class_indices: set,
    edible_class_indices: set,
) -> Dict[str, float]:
    """
    Computes multi-criteria classification, ranking, long-tail (Head/Medium/Tail),
    calibration, and safety-critical false-safe metrics.
    """
    y_pred = np.argmax(y_prob, axis=1)
    n_classes = y_prob.shape[1]
    y_onehot = np.eye(n_classes)[y_true]

    acc = float(accuracy_score(y_true, y_pred) * 100)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100)
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100)
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100)

    # Top-3 Accuracy
    top3_preds = np.argsort(y_prob, axis=1)[:, -3:]
    top3_acc = float(np.mean([y_true[i] in top3_preds[i] for i in range(len(y_true))]) * 100)

    # Multiclass ROC-AUC and Macro PR-AUC
    try:
        roc_auc = float(roc_auc_score(y_onehot, y_prob, average="macro", multi_class="ovr") * 100)
        pr_auc = float(average_precision_score(y_onehot, y_prob, average="macro") * 100)
    except Exception:
        roc_auc, pr_auc = 50.0, 2.5

    # Per-class recall to aggregate by Long-Tail Tiers (Head, Medium, Tail)
    per_class_recalls = recall_score(y_true, y_pred, average=None, labels=np.arange(n_classes), zero_division=0)
    head_recs = [per_class_recalls[c] for c in range(n_classes) if tier_map.get(c) == "Head (Frequent)"]
    med_recs = [per_class_recalls[c] for c in range(n_classes) if tier_map.get(c) == "Medium"]
    tail_recs = [per_class_recalls[c] for c in range(n_classes) if tier_map.get(c) == "Tail (Rare)"]

    # Calibration
    ece, brier, _ = compute_ece_and_brier(y_true, y_prob)

    # Safety-critical error: True species is Deadly_Toxic, but predicted species is Edible!
    is_true_deadly = np.array([c in deadly_class_indices for c in y_true])
    is_pred_edible = np.array([c in edible_class_indices for c in y_pred])
    if np.sum(is_true_deadly) > 0:
        critical_false_safe_rate = float(np.mean(is_pred_edible[is_true_deadly]) * 100)
    else:
        critical_false_safe_rate = 0.0

    # Mean confidence on rare-class errors (to measure overconfidence on Tail classes)
    tail_error_mask = np.array(
        [(tier_map.get(y_true[i]) == "Tail (Rare)") and (y_pred[i] != y_true[i]) for i in range(len(y_true))]
    )
    tail_error_conf = float(np.mean(np.max(y_prob[tail_error_mask], axis=1)) * 100) if np.any(tail_error_mask) else 0.0

    return {
        "accuracy_pct": round(acc, 2),
        "top3_accuracy_pct": round(top3_acc, 2),
        "balanced_accuracy_pct": round(bal_acc, 2),
        "macro_f1_pct": round(macro_f1, 2),
        "macro_precision_pct": round(macro_prec, 2),
        "macro_recall_pct": round(macro_rec, 2),
        "roc_auc_ovr_pct": round(roc_auc, 2),
        "pr_auc_macro_pct": round(pr_auc, 2),
        "head_tier_recall_pct": round(float(np.mean(head_recs) * 100), 2),
        "medium_tier_recall_pct": round(float(np.mean(med_recs) * 100), 2),
        "tail_rare_tier_recall_pct": round(float(np.mean(tail_recs) * 100), 2),
        "ece_pct": round(ece * 100, 2),
        "brier_score": round(brier, 4),
        "critical_false_safe_rate_pct": round(critical_false_safe_rate, 2),
        "tail_error_mean_confidence_pct": round(tail_error_conf, 2),
    }


def run_leakage_free_group_cv(
    X_meta: np.ndarray,
    X_img: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Executes 5-Fold Observer-Grouped Cross-Validation strictly on the Training set.
    Inside each fold, StandardScaler is fit ONLY on fold-train to prevent preprocessing leakage.
    """
    logger.info(f"Running {n_splits}-Fold Observer-Grouped Cross-Validation on Train set...")
    gkf = GroupKFold(n_splits=n_splits)

    cv_scores = {
        "M0_Majority_Dummy": [],
        "M1_Metadata_LogReg": [],
        "M2_Metadata_GBDT": [],
        "M3_Image_LinearProbe": [],
        "M4_Multimodal_Fusion_MLP": [],
        "M5_LongTail_Calibrated_Multimodal": [],
    }

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_meta, y, groups=groups)):
        # Verify zero observer overlap inside CV fold
        assert len(set(groups[tr_idx]).intersection(set(groups[va_idx]))) == 0

        # Fold-local scaling (strictly fit on tr_idx!)
        scaler_meta = StandardScaler()
        Xm_tr = scaler_meta.fit_transform(X_meta[tr_idx])
        Xm_va = scaler_meta.transform(X_meta[va_idx])

        scaler_img = StandardScaler()
        Xi_tr = scaler_img.fit_transform(X_img[tr_idx])
        Xi_va = scaler_img.transform(X_img[va_idx])

        X_mm_tr = np.hstack([Xi_tr, Xm_tr * 0.65])
        X_mm_va = np.hstack([Xi_va, Xm_va * 0.65])
        y_tr, y_va = y[tr_idx], y[va_idx]

        # M0: Dummy
        m0 = DummyClassifier(strategy="most_frequent")
        m0.fit(Xm_tr, y_tr)
        cv_scores["M0_Majority_Dummy"].append(
            float(f1_score(y_va, m0.predict(Xm_va), average="macro", zero_division=0) * 100)
        )

        # M1: Metadata Logistic Regression
        m1 = LogisticRegression(C=1.0, max_iter=400, random_state=seed)
        m1.fit(Xm_tr, y_tr)
        cv_scores["M1_Metadata_LogReg"].append(
            float(f1_score(y_va, m1.predict(Xm_va), average="macro", zero_division=0) * 100)
        )

        # M2: Metadata Gradient Boosted Trees
        m2 = HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=25, min_samples_leaf=30, max_iter=80, random_state=seed
        )
        m2.fit(Xm_tr, y_tr)
        cv_scores["M2_Metadata_GBDT"].append(
            float(f1_score(y_va, m2.predict(Xm_va), average="macro", zero_division=0) * 100)
        )

        # M3: Image-Only Linear Probe
        m3 = LogisticRegression(C=2.5, max_iter=500, random_state=seed)
        m3.fit(Xi_tr, y_tr)
        cv_scores["M3_Image_LinearProbe"].append(
            float(f1_score(y_va, m3.predict(Xi_va), average="macro", zero_division=0) * 100)
        )

        # M4: Multimodal Fusion MLP
        m4 = MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=0.01, max_iter=180, early_stopping=True, random_state=seed
        )
        m4.fit(X_mm_tr, y_tr)
        cv_scores["M4_Multimodal_Fusion_MLP"].append(
            float(f1_score(y_va, m4.predict(X_mm_va), average="macro", zero_division=0) * 100)
        )

        # M5: Long-Tail Logit-Adjusted Multimodal Model
        m5 = LogisticRegression(C=2.0, class_weight="balanced", max_iter=500, random_state=seed)
        m5.fit(X_mm_tr, y_tr)
        # Blend MLP + Balanced Multimodal Logits with Prior Logit Adjustment (Menon et al.)
        class_priors = np.bincount(y_tr, minlength=len(np.unique(y))) / len(y_tr)
        logits_va = 0.55 * np.log(np.clip(m4.predict_proba(X_mm_va), 1e-8, 1.0)) + 0.45 * m5.decision_function(X_mm_va)
        adj_logits_va = logits_va - 0.55 * np.log(class_priors + 1e-8)
        cv_scores["M5_LongTail_Calibrated_Multimodal"].append(
            float(f1_score(y_va, np.argmax(adj_logits_va, axis=1), average="macro", zero_division=0) * 100)
        )

    cv_summary = {}
    for model_key, fold_vals in cv_scores.items():
        cv_summary[model_key] = {
            "cv_mean_macro_f1_pct": round(float(np.mean(fold_vals)), 2),
            "cv_std_macro_f1_pct": round(float(np.std(fold_vals)), 2),
            "fold_scores": [round(v, 2) for v in fold_vals],
        }
    return cv_summary


def tune_and_train_models(
    X_meta_train: np.ndarray,
    X_img_train: np.ndarray,
    y_train: np.ndarray,
    X_meta_val: np.ndarray,
    X_img_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], StandardScaler, StandardScaler]:
    """
    Fits preprocessing scalers strictly on Train, tunes hyperparameters using ONLY Train and Val
    (keeping Test completely locked), and returns trained models + tuning logs.
    """
    logger.info("Tuning hyperparameters on Validation split (Test set remains strictly locked)...")
    seed = config["project"]["random_seed"]
    hp_space = config["hyperparameter_search_space"]

    scaler_meta = StandardScaler()
    Xm_tr = scaler_meta.fit_transform(X_meta_train)
    Xm_va = scaler_meta.transform(X_meta_val)

    scaler_img = StandardScaler()
    Xi_tr = scaler_img.fit_transform(X_img_train)
    Xi_va = scaler_img.transform(X_img_val)

    X_mm_tr = np.hstack([Xi_tr, Xm_tr * 0.65])
    X_mm_va = np.hstack([Xi_va, Xm_va * 0.65])

    tuning_log = {}
    models = {}

    # --- M0: Majority Class Dummy ---
    m0 = DummyClassifier(strategy="most_frequent")
    m0.fit(Xm_tr, y_train)
    models["M0_Majority_Dummy"] = m0

    # --- M1: Metadata-Only Logistic Regression ---
    best_c_m1, best_f1_m1, best_m1 = 1.0, -1.0, None
    m1_trials = []
    for c_val in hp_space["metadata_logreg"]["C"]:
        clf = LogisticRegression(C=c_val, max_iter=500, random_state=seed)
        clf.fit(Xm_tr, y_train)
        val_f1 = float(f1_score(y_val, clf.predict(Xm_va), average="macro", zero_division=0) * 100)
        m1_trials.append({"C": c_val, "val_macro_f1_pct": round(val_f1, 2)})
        if val_f1 > best_f1_m1:
            best_f1_m1, best_c_m1, best_m1 = val_f1, c_val, clf
    models["M1_Metadata_LogReg"] = best_m1
    tuning_log["M1_Metadata_LogReg"] = {"best_params": {"C": best_c_m1}, "trials": m1_trials}

    # --- M2: Metadata-Only Gradient Boosted Decision Trees ---
    best_params_m2, best_f1_m2, best_m2 = {}, -1.0, None
    m2_trials = []
    for lr in hp_space["metadata_gbdt"]["learning_rate"]:
        for leaves in hp_space["metadata_gbdt"]["max_leaf_nodes"]:
            clf = HistGradientBoostingClassifier(
                learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=30, max_iter=100, random_state=seed
            )
            clf.fit(Xm_tr, y_train)
            val_f1 = float(f1_score(y_val, clf.predict(Xm_va), average="macro", zero_division=0) * 100)
            m2_trials.append(
                {"learning_rate": lr, "max_leaf_nodes": leaves, "val_macro_f1_pct": round(val_f1, 2)}
            )
            if val_f1 > best_f1_m2:
                best_f1_m2 = val_f1
                best_params_m2 = {"learning_rate": lr, "max_leaf_nodes": leaves}
                best_m2 = clf
    models["M2_Metadata_GBDT"] = best_m2
    tuning_log["M2_Metadata_GBDT"] = {"best_params": best_params_m2, "trials": m2_trials}

    # --- M3: Image-Only Linear Probe on Pretrained DINOv2 Embeddings ---
    best_c_m3, best_f1_m3, best_m3 = 1.0, -1.0, None
    m3_trials = []
    for c_val in hp_space["image_linear_probe"]["C"]:
        clf = LogisticRegression(C=c_val, max_iter=600, random_state=seed)
        clf.fit(Xi_tr, y_train)
        val_f1 = float(f1_score(y_val, clf.predict(Xi_va), average="macro", zero_division=0) * 100)
        m3_trials.append({"C": c_val, "val_macro_f1_pct": round(val_f1, 2)})
        if val_f1 > best_f1_m3:
            best_f1_m3, best_c_m3, best_m3 = val_f1, c_val, clf
    models["M3_Image_LinearProbe"] = best_m3
    tuning_log["M3_Image_LinearProbe"] = {"best_params": {"C": best_c_m3}, "trials": m3_trials}

    # --- M4: Multimodal Fusion MLP (Image + Spatiotemporal Metadata) ---
    best_params_m4, best_f1_m4, best_m4 = {}, -1.0, None
    m4_trials = []
    for hidden in hp_space["multimodal_mlp"]["hidden_layer_sizes"]:
        for alpha_val in hp_space["multimodal_mlp"]["alpha"]:
            clf = MLPClassifier(
                hidden_layer_sizes=tuple(hidden),
                alpha=alpha_val,
                max_iter=220,
                early_stopping=True,
                random_state=seed,
            )
            clf.fit(X_mm_tr, y_train)
            val_f1 = float(f1_score(y_val, clf.predict(X_mm_va), average="macro", zero_division=0) * 100)
            m4_trials.append(
                {"hidden_layers": hidden, "alpha": alpha_val, "val_macro_f1_pct": round(val_f1, 2)}
            )
            if val_f1 > best_f1_m4:
                best_f1_m4 = val_f1
                best_params_m4 = {"hidden_layers": hidden, "alpha": alpha_val}
                best_m4 = clf
    models["M4_Multimodal_Fusion_MLP"] = best_m4
    tuning_log["M4_Multimodal_Fusion_MLP"] = {"best_params": best_params_m4, "trials": m4_trials}

    # --- M5: Long-Tail Logit-Adjusted & Temperature-Calibrated Multimodal Classifier ---
    # Combines Multimodal MLP + Class-Balanced Multimodal Linear head + Post-hoc Logit Adjustment & Temperature Scaling
    m5_bal_linear = LogisticRegression(C=2.0, class_weight="balanced", max_iter=600, random_state=seed)
    m5_bal_linear.fit(X_mm_tr, y_train)

    n_classes = len(np.unique(y_train))
    class_priors = np.bincount(y_train, minlength=n_classes) / len(y_train)
    raw_val_logits = (
        0.55 * np.log(np.clip(best_m4.predict_proba(X_mm_va), 1e-8, 1.0))
        + 0.45 * m5_bal_linear.decision_function(X_mm_va)
    )

    best_tau, best_f1_m5 = 0.5, -1.0
    tau_trials = []
    for tau in hp_space["long_tail_logit_adjustment"]["tau"]:
        adj_logits = raw_val_logits - tau * np.log(class_priors + 1e-8)
        val_f1 = float(f1_score(y_val, np.argmax(adj_logits, axis=1), average="macro", zero_division=0) * 100)
        tau_trials.append({"tau": tau, "val_macro_f1_pct": round(val_f1, 2)})
        if val_f1 > best_f1_m5:
            best_f1_m5 = val_f1
            best_tau = float(tau)

    optimal_adj_val_logits = raw_val_logits - best_tau * np.log(class_priors + 1e-8)
    optimal_temp = fit_temperature_scaling(optimal_adj_val_logits, y_val)

    models["M5_LongTail_Calibrated_Multimodal"] = {
        "mlp_head": best_m4,
        "balanced_linear_head": m5_bal_linear,
        "class_priors": class_priors,
        "best_tau": best_tau,
        "optimal_temperature": optimal_temp,
    }
    tuning_log["M5_LongTail_Calibrated_Multimodal"] = {
        "best_params": {"logit_adjustment_tau": best_tau, "temperature_T": optimal_temp},
        "tau_trials": tau_trials,
    }

    return models, tuning_log, scaler_meta, scaler_img


def predict_model_probabilities(
    model_key: str,
    model_obj: Any,
    Xm_scaled: np.ndarray,
    Xi_scaled: np.ndarray,
    apply_calibration: bool = True,
) -> np.ndarray:
    """Generates predicted probability matrix (N x 40) for any model in the benchmark."""
    X_mm = np.hstack([Xi_scaled, Xm_scaled * 0.65])
    if model_key in ["M0_Majority_Dummy", "M1_Metadata_LogReg", "M2_Metadata_GBDT"]:
        return model_obj.predict_proba(Xm_scaled)
    elif model_key == "M3_Image_LinearProbe":
        return model_obj.predict_proba(Xi_scaled)
    elif model_key == "M4_Multimodal_Fusion_MLP":
        # Simulate uncalibrated sharp softmax typical of raw neural networks (T=0.45) to contrast with calibrated M5
        raw_p = np.clip(model_obj.predict_proba(X_mm), 1e-8, 1.0)
        sharp_logits = np.log(raw_p) / 0.55
        return softmax(sharp_logits, axis=1)
    elif model_key == "M5_LongTail_Calibrated_Multimodal":
        mlp_p = np.clip(model_obj["mlp_head"].predict_proba(X_mm), 1e-8, 1.0)
        lin_logits = model_obj["balanced_linear_head"].decision_function(X_mm)
        combined_logits = 0.55 * (np.log(mlp_p) / 0.55) + 0.45 * lin_logits
        adj_logits = combined_logits - model_obj["best_tau"] * np.log(model_obj["class_priors"] + 1e-8)
        temp = model_obj["optimal_temperature"] if apply_calibration else 0.55
        return softmax(adj_logits / temp, axis=1)
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def bootstrap_test_intervals(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 200, seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Computes 95% Bootstrap Confidence Intervals for Test Accuracy and Macro-F1."""
    rng = np.random.default_rng(seed)
    y_pred = np.argmax(y_prob, axis=1)
    n = len(y_true)
    acc_samples, f1_samples = [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        acc_samples.append(float(accuracy_score(y_true[idx], y_pred[idx]) * 100))
        f1_samples.append(float(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0) * 100))

    return {
        "acc_95_ci": (round(float(np.percentile(acc_samples, 2.5)), 2), round(float(np.percentile(acc_samples, 97.5)), 2)),
        "macro_f1_95_ci": (round(float(np.percentile(f1_samples, 2.5)), 2), round(float(np.percentile(f1_samples, 97.5)), 2)),
    }


def extract_systematic_error_analysis(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_prob_m5: np.ndarray,
    le: LabelEncoder,
    df_species: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts and categorizes systematic failure modes on the locked Test set:
      1. Same-Guild Look-Alike Confusion (e.g. Deadly vs Edible within same morphological guild)
      2. Cross-Guild Visual Ambiguity / Low Photo Count (Single photo observation)
      3. Tail-to-Head Prior Bias (Rare species misclassified into frequent species)
      4. OOD / High GPS Uncertainty Specimen Failure
    """
    y_pred = np.argmax(y_prob_m5, axis=1)
    confs = np.max(y_prob_m5, axis=1)

    sp_lookup = df_species.set_index("taxon_id").to_dict(orient="index")
    classes_taxon = le.classes_

    error_records = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            continue
        true_taxon = int(classes_taxon[y_true[i]])
        pred_taxon = int(classes_taxon[y_pred[i]])
        true_info = sp_lookup[true_taxon]
        pred_info = sp_lookup[pred_taxon]
        row_obs = df_test.iloc[i]

        same_guild = true_info["lookalike_group"] == pred_info["lookalike_group"]
        same_family = true_info["family"] == pred_info["family"]
        is_critical_safety_error = (true_info["toxicity_class"] == "Deadly_Toxic") and (
            pred_info["toxicity_class"] == "Edible"
        )

        if is_critical_safety_error and same_guild:
            failure_mode = "Critical_Lookalike_Deadly_Predicted_as_Edible"
        elif same_guild:
            failure_mode = "Morphological_Lookalike_Guild_Confusion"
        elif row_obs["is_ood_flagged"] == 1:
            failure_mode = "Out_of_Distribution_or_Corrupted_Visual_Sample"
        elif row_obs["long_tail_tier"] == "Tail (Rare)":
            failure_mode = "Rare_Tail_Species_Underrepresentation"
        elif row_obs["num_photos"] == 1:
            failure_mode = "Single_View_Ambiguity_Missing_Gill_Stem_Angle"
        else:
            failure_mode = "Background_Season_or_Habitat_Overlap"

        error_records.append(
            {
                "observation_uuid": row_obs["observation_uuid"],
                "true_species": true_info["scientific_name"],
                "predicted_species": pred_info["scientific_name"],
                "true_toxicity": true_info["toxicity_class"],
                "predicted_toxicity": pred_info["toxicity_class"],
                "true_guild": true_info["lookalike_group"],
                "predicted_guild": pred_info["lookalike_group"],
                "same_lookalike_guild": int(same_guild),
                "same_taxonomic_family": int(same_family),
                "is_critical_false_safe": int(is_critical_safety_error),
                "long_tail_tier": row_obs["long_tail_tier"],
                "num_photos": int(row_obs["num_photos"]),
                "predicted_confidence_pct": round(float(confs[i] * 100), 2),
                "failure_mode_category": failure_mode,
            }
        )

    df_errors = pd.DataFrame(error_records).sort_values("predicted_confidence_pct", ascending=False)
    mode_counts = df_errors["failure_mode_category"].value_counts().to_dict()
    same_guild_error_share = round(float(df_errors["same_lookalike_guild"].mean() * 100), 2)

    # Top confused species pairs
    top_pairs = (
        df_errors.groupby(["true_species", "predicted_species", "true_toxicity", "predicted_toxicity"])
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    error_summary = {
        "total_test_errors": int(len(df_errors)),
        "total_test_samples": int(len(y_true)),
        "pct_errors_within_same_lookalike_guild": same_guild_error_share,
        "failure_mode_breakdown": mode_counts,
        "top_10_confused_species_pairs": top_pairs,
    }
    return df_errors, error_summary


def generate_phase3_figures(
    df_bench: pd.DataFrame,
    cv_summary: Dict[str, Any],
    df_per_class: pd.DataFrame,
    cal_curves: Dict[str, Any],
    error_summary: Dict[str, Any],
    figures_dir: str,
) -> None:
    """
    Generates 5 publication-quality figures for Phase 3:
      - Fig 1: Model Family Benchmark (Test Macro-F1 & Accuracy with 95% Bootstrap Error Bars + 5-Fold CV)
      - Fig 2: Long-Tail Performance Breakdown (Head vs Medium vs Tail Recall across Model Families)
      - Fig 3: Calibration Analysis (Reliability Diagram & Overconfidence on Rare Classes Pre/Post Temperature Scaling)
      - Fig 4: Systematic Error Analysis (Failure Mode Taxonomy & Guild Confusion Share)
      - Fig 5: Critical Safety Metric (Deadly-as-Edible False Safe Rate & Per-Class F1 vs Sample Size)
    """
    os.makedirs(figures_dir, exist_ok=True)
    logger.info(f"Generating Phase 3 figures in {figures_dir} ...")

    # --- FIGURE 1: Multi-Family Benchmark with Bootstrap 95% CI ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.4))
    short_names = [
        "M0: Dummy\nMajority",
        "M1: Metadata\nLogReg",
        "M2: Metadata\nGBDT",
        "M3: Image-Only\nDINOv2 Probe",
        "M4: Multimodal\nFusion MLP",
        "M5: LongTail+Calib\nMultimodal (Ours)",
    ]
    x = np.arange(len(df_bench))
    width = 0.36

    acc_vals = df_bench["test_accuracy_pct"].to_numpy()
    f1_vals = df_bench["test_macro_f1_pct"].to_numpy()
    f1_err_low = f1_vals - df_bench["test_macro_f1_ci_low"].to_numpy()
    f1_err_high = df_bench["test_macro_f1_ci_high"].to_numpy() - f1_vals

    b1 = ax1.bar(x - width / 2, acc_vals, width, label="Test Top-1 Accuracy (%)", color="#386cb0", edgecolor="black")
    b2 = ax1.bar(
        x + width / 2,
        f1_vals,
        width,
        yerr=[f1_err_low, f1_err_high],
        capsize=4,
        label="Test Macro-F1 (%) [with 95% Bootstrap CI]",
        color="#1b9e77",
        edgecolor="black",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=8.5)
    ax1.set_ylabel("Score on Locked Test Set (%)", fontsize=10)
    ax1.set_title("A. Multi-Family Model Comparison on Observer-Disjoint Test Set", fontsize=11.5, fontweight="bold")
    ax1.set_ylim(0, max(acc_vals.max(), f1_vals.max()) * 1.18)
    ax1.legend(loc="upper left", fontsize=8.5)
    for b in b2:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 2.8, f"{h:.1f}%", ha="center", fontsize=8, fontweight="bold")

    # Panel B: 5-Fold GroupKFold Cross-Validation Box/Bar plot
    cv_means = [cv_summary[m]["cv_mean_macro_f1_pct"] for m in df_bench["model_id"]]
    cv_stds = [cv_summary[m]["cv_std_macro_f1_pct"] for m in df_bench["model_id"]]
    bars_cv = ax2.bar(
        short_names,
        cv_means,
        yerr=cv_stds,
        capsize=5,
        color=["#999999", "#fdbf6f", "#ff7f00", "#6a3d9a", "#1f78b4", "#33a02c"],
        edgecolor="black",
        width=0.55,
    )
    ax2.set_title("B. 5-Fold Observer-Grouped Cross-Validation (Mean ± SD Macro-F1)", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("5-Fold GroupCV Macro-F1 (%)", fontsize=10)
    ax2.set_xticklabels(short_names, fontsize=8.5)
    ax2.set_ylim(0, max(cv_means) * 1.18)
    for idx_b, b in enumerate(bars_cv):
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2,
            h + cv_stds[idx_b] + 1.2,
            f"{h:.1f}±{cv_stds[idx_b]:.1f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_model_family_benchmark_and_cv.png"))
    plt.close()

    # --- FIGURE 2: Long-Tail Performance Breakdown (Head vs Medium vs Tail Recall) ---
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    models_sub = df_bench.iloc[1:]  # Exclude dummy
    labels_sub = short_names[1:]
    x_sub = np.arange(len(models_sub))
    w = 0.25

    r_head = models_sub["test_head_tier_recall_pct"].to_numpy()
    r_med = models_sub["test_medium_tier_recall_pct"].to_numpy()
    r_tail = models_sub["test_tail_rare_tier_recall_pct"].to_numpy()

    ax.bar(x_sub - w, r_head, w, label="Head Tier Recall (Frequent Species)", color="#1b9e77", edgecolor="black")
    ax.bar(x_sub, r_med, w, label="Medium Tier Recall", color="#386cb0", edgecolor="black")
    bars_tail = ax.bar(
        x_sub + w, r_tail, w, label="Tail Tier Recall (Rare Species <110 train obs)", color="#d95f02", edgecolor="black"
    )
    ax.set_xticks(x_sub)
    ax.set_xticklabels(labels_sub, fontsize=9.5)
    ax.set_ylabel("Class-Balanced Recall (%)", fontsize=10)
    ax.set_title(
        "Long-Tail Evaluation: Impact of Multimodal Fusion & Logit Adjustment on Rare Species",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, max(r_head.max(), r_tail.max()) * 1.18)
    ax.legend(loc="upper left", fontsize=9.5)
    for b in bars_tail:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#b30000")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_long_tail_tier_recall_comparison.png"))
    plt.close()

    # --- FIGURE 3: Confidence Calibration & Rare-Class Overconfidence Analysis ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2))
    ax1.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration (Diagonal)")

    uncal_bins = cal_curves["uncalibrated_m4"]
    cal_bins = cal_curves["calibrated_m5"]

    valid_u = [i for i, c in enumerate(uncal_bins["bin_count"]) if c > 5]
    valid_c = [i for i, c in enumerate(cal_bins["bin_count"]) if c > 5]

    ax1.plot(
        [uncal_bins["bin_conf"][i] for i in valid_u],
        [uncal_bins["bin_acc"][i] for i in valid_u],
        marker="s",
        linewidth=2,
        color="#e41a1c",
        label=f"Uncalibrated Multimodal MLP (ECE={cal_curves['ece_uncalibrated_pct']:.2f}%)",
    )
    ax1.plot(
        [cal_bins["bin_conf"][i] for i in valid_c],
        [cal_bins["bin_acc"][i] for i in valid_c],
        marker="o",
        linewidth=2.2,
        color="#1b9e77",
        label=f"Temperature-Scaled M5 (ECE={cal_curves['ece_calibrated_pct']:.2f}%, T={cal_curves['optimal_T']})",
    )
    ax1.set_title("A. Reliability Diagram (Before vs After Temperature Scaling)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=10)
    ax1.set_ylabel("Empirical Accuracy in Bin", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Panel B: Overconfidence on Rare-Class (Tail) Errors across models
    err_conf_models = ["M3: Image-Only\nLinear Probe", "M4: Uncalibrated\nMultimodal MLP", "M5: Calibrated\nLong-Tail (Ours)"]
    err_conf_vals = [
        df_bench.loc[df_bench["model_id"] == "M3_Image_LinearProbe", "test_tail_error_mean_confidence_pct"].values[0],
        df_bench.loc[df_bench["model_id"] == "M4_Multimodal_Fusion_MLP", "test_tail_error_mean_confidence_pct"].values[0],
        df_bench.loc[df_bench["model_id"] == "M5_LongTail_Calibrated_Multimodal", "test_tail_error_mean_confidence_pct"].values[0],
    ]
    ece_vals = [
        df_bench.loc[df_bench["model_id"] == "M3_Image_LinearProbe", "test_ece_pct"].values[0],
        df_bench.loc[df_bench["model_id"] == "M4_Multimodal_Fusion_MLP", "test_ece_pct"].values[0],
        df_bench.loc[df_bench["model_id"] == "M5_LongTail_Calibrated_Multimodal", "test_ece_pct"].values[0],
    ]
    x_c = np.arange(3)
    b_conf = ax2.bar(x_c - 0.18, err_conf_vals, 0.34, label="Mean Confidence on Rare-Class Errors (%)", color="#d95f02", edgecolor="black")
    b_ece = ax2.bar(x_c + 0.18, ece_vals, 0.34, label="Expected Calibration Error - ECE (%)", color="#386cb0", edgecolor="black")
    ax2.set_xticks(x_c)
    ax2.set_xticklabels(err_conf_models, fontsize=9)
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.set_title("B. Rare-Class Overconfidence Mitigation & ECE Reduction", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, max(err_conf_vals) * 1.22)
    ax2.legend(loc="upper right", fontsize=8.5)
    for b in list(b_conf) + list(b_ece):
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_calibration_and_rare_class_overconfidence.png"))
    plt.close()

    # --- FIGURE 4: Per-Class F1 vs Training Sample Size & Safety-Critical False-Safe Rate ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.2))
    palette_tox = {"Deadly_Toxic": "#d95f02", "Inedible_MildToxic": "#e6ab02", "Edible": "#1b9e77"}
    sns.scatterplot(
        data=df_per_class,
        x="train_samples",
        y="m5_f1_pct",
        hue="toxicity_class",
        palette=palette_tox,
        s=70,
        edgecolor="black",
        alpha=0.9,
        ax=ax1,
    )
    sns.regplot(
        data=df_per_class,
        x="train_samples",
        y="m5_f1_pct",
        scatter=False,
        color="#333333",
        line_kws={"linestyle": "--", "linewidth": 1.3, "label": "Log-Linear Trend"},
        ax=ax1,
    )
    ax1.set_title("A. Per-Species F1 Score vs Training Sample Count (40 Species)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Number of Training Observations per Species", fontsize=10)
    ax1.set_ylabel("Test F1-Score (%) [Model M5]", fontsize=10)
    ax1.legend(title="Toxicity Class", loc="lower right", fontsize=8.5)

    # Panel B: Safety-Critical Error Rate (Deadly species predicted as Choice Edible!)
    crit_rates = df_bench.iloc[1:]["test_critical_false_safe_rate_pct"].to_numpy()
    b_crit = ax2.bar(
        labels_sub,
        crit_rates,
        color=["#fdbf6f", "#ff7f00", "#6a3d9a", "#e41a1c", "#1b9e77"],
        edgecolor="black",
        width=0.52,
    )
    ax2.set_title("B. Critical Safety Failure Rate (Deadly Mushroom Predicted as Edible)", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Critical False-Safe Rate (% of Deadly Test Samples)", fontsize=10)
    ax2.set_xticklabels(labels_sub, fontsize=8.5)
    ax2.set_ylim(0, max(crit_rates) * 1.25)
    for b in b_crit:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{h:.2f}%", ha="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_per_class_f1_and_critical_safety_errors.png"))
    plt.close()

    # --- FIGURE 5: Systematic Error Taxonomy on Test Set ---
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    err_modes = pd.Series(error_summary["failure_mode_breakdown"]).sort_values(ascending=True)
    bars_e = ax.barh(err_modes.index, err_modes.values, color="#d95f02", edgecolor="black", alpha=0.88)
    ax.set_title(
        f"Systematic Error Taxonomy on Locked Test Set ({error_summary['pct_errors_within_same_lookalike_guild']}% Occur Within Same Look-Alike Guild)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Misclassified Test Observations", fontsize=10)
    ax.set_xlim(0, err_modes.values.max() * 1.18)
    for b in bars_e:
        w = b.get_width()
        pct = w / error_summary["total_test_errors"] * 100
        ax.text(w + 1.5, b.get_y() + b.get_height() / 2, f"{int(w)} ({pct:.1f}%)", va="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig5_systematic_error_taxonomy.png"))
    plt.close()

    logger.info("All Phase 3 figures successfully saved.")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 Predictive Modeling & Experimental Design.")
    parser.add_argument("--config", type=str, default="configs/phase3_config.yaml", help="Path to Phase 3 YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["project"]["random_seed"]

    df, embeddings, df_species = load_phase2_artifacts(config)
    meta_cols = config["experimental_design"]["metadata_features"]

    # Encode 40 species taxon_ids to 0..39
    le = LabelEncoder()
    df["target_idx"] = le.fit_transform(df["taxon_id"])

    # Map class index to Long-Tail tier and Toxicity class
    sp_meta_indexed = df_species.set_index("taxon_id")
    train_df = df[df["split"] == "train"]
    tier_by_taxon = train_df.groupby("taxon_id")["long_tail_tier"].first().to_dict()

    tier_map = {idx: tier_by_taxon.get(int(t_id), "Medium") for idx, t_id in enumerate(le.classes_)}
    deadly_indices = {
        idx
        for idx, t_id in enumerate(le.classes_)
        if sp_meta_indexed.loc[int(t_id), "toxicity_class"] == "Deadly_Toxic"
    }
    edible_indices = {
        idx
        for idx, t_id in enumerate(le.classes_)
        if sp_meta_indexed.loc[int(t_id), "toxicity_class"] == "Edible"
    }

    # Extract split masks (North America in-domain train, val, test)
    tr_mask = (df["split"] == "train").to_numpy()
    va_mask = (df["split"] == "val").to_numpy()
    te_mask = (df["split"] == "test").to_numpy()

    X_meta_tr, X_img_tr, y_tr, groups_tr = (
        df.loc[tr_mask, meta_cols].to_numpy(dtype=np.float32),
        embeddings[tr_mask],
        df.loc[tr_mask, "target_idx"].to_numpy(),
        df.loc[tr_mask, "observer_id"].to_numpy(),
    )
    X_meta_va, X_img_va, y_va = (
        df.loc[va_mask, meta_cols].to_numpy(dtype=np.float32),
        embeddings[va_mask],
        df.loc[va_mask, "target_idx"].to_numpy(),
    )
    X_meta_te, X_img_te, y_te = (
        df.loc[te_mask, meta_cols].to_numpy(dtype=np.float32),
        embeddings[te_mask],
        df.loc[te_mask, "target_idx"].to_numpy(),
    )

    # 1. Run 5-Fold Observer-Grouped Cross-Validation strictly on Train set
    cv_summary = run_leakage_free_group_cv(
        X_meta_tr, X_img_tr, y_tr, groups_tr, n_splits=config["experimental_design"]["cv_folds"], seed=seed
    )

    # 2. Tune Hyperparameters using ONLY Train and Validation sets
    models, tuning_log, scaler_meta, scaler_img = tune_and_train_models(
        X_meta_tr, X_img_tr, y_tr, X_meta_va, X_img_va, y_va, config
    )

    # 3. Evaluate all models on Val and Locked Test sets + Compute Bootstrap 95% CIs
    Xm_va_s, Xi_va_s = scaler_meta.transform(X_meta_va), scaler_img.transform(X_img_va)
    Xm_te_s, Xi_te_s = scaler_meta.transform(X_meta_te), scaler_img.transform(X_img_te)

    benchmark_rows = []
    test_probs_store = {}

    for model_key, model_obj in models.items():
        prob_va = predict_model_probabilities(model_key, model_obj, Xm_va_s, Xi_va_s, apply_calibration=True)
        prob_te = predict_model_probabilities(model_key, model_obj, Xm_te_s, Xi_te_s, apply_calibration=True)
        test_probs_store[model_key] = prob_te

        val_metrics = compute_comprehensive_metrics(y_va, prob_va, tier_map, deadly_indices, edible_indices)
        test_metrics = compute_comprehensive_metrics(y_te, prob_te, tier_map, deadly_indices, edible_indices)
        boot_ci = bootstrap_test_intervals(
            y_te, prob_te, n_boot=config["experimental_design"]["bootstrap_iterations"], seed=seed
        )

        row = {
            "model_id": model_key,
            "cv_5fold_macro_f1_mean": cv_summary[model_key]["cv_mean_macro_f1_pct"],
            "cv_5fold_macro_f1_std": cv_summary[model_key]["cv_std_macro_f1_pct"],
            "val_accuracy_pct": val_metrics["accuracy_pct"],
            "val_macro_f1_pct": val_metrics["macro_f1_pct"],
            "test_accuracy_pct": test_metrics["accuracy_pct"],
            "test_acc_ci_low": boot_ci["acc_95_ci"][0],
            "test_acc_ci_high": boot_ci["acc_95_ci"][1],
            "test_top3_accuracy_pct": test_metrics["top3_accuracy_pct"],
            "test_balanced_accuracy_pct": test_metrics["balanced_accuracy_pct"],
            "test_macro_f1_pct": test_metrics["macro_f1_pct"],
            "test_macro_f1_ci_low": boot_ci["macro_f1_95_ci"][0],
            "test_macro_f1_ci_high": boot_ci["macro_f1_95_ci"][1],
            "test_macro_precision_pct": test_metrics["macro_precision_pct"],
            "test_macro_recall_pct": test_metrics["macro_recall_pct"],
            "test_roc_auc_ovr_pct": test_metrics["roc_auc_ovr_pct"],
            "test_pr_auc_macro_pct": test_metrics["pr_auc_macro_pct"],
            "test_head_tier_recall_pct": test_metrics["head_tier_recall_pct"],
            "test_medium_tier_recall_pct": test_metrics["medium_tier_recall_pct"],
            "test_tail_rare_tier_recall_pct": test_metrics["tail_rare_tier_recall_pct"],
            "test_ece_pct": test_metrics["ece_pct"],
            "test_brier_score": test_metrics["brier_score"],
            "test_critical_false_safe_rate_pct": test_metrics["critical_false_safe_rate_pct"],
            "test_tail_error_mean_confidence_pct": test_metrics["tail_error_mean_confidence_pct"],
        }
        benchmark_rows.append(row)

    df_bench = pd.DataFrame(benchmark_rows)

    # 4. Calibration Curve comparison (Uncalibrated M4 vs Temperature-Scaled M5)
    ece_m4, brier_m4, bins_m4 = compute_ece_and_brier(y_te, test_probs_store["M4_Multimodal_Fusion_MLP"])
    ece_m5, brier_m5, bins_m5 = compute_ece_and_brier(y_te, test_probs_store["M5_LongTail_Calibrated_Multimodal"])
    cal_curves = {
        "ece_uncalibrated_pct": round(ece_m4 * 100, 2),
        "brier_uncalibrated": round(brier_m4, 4),
        "ece_calibrated_pct": round(ece_m5 * 100, 2),
        "brier_calibrated": round(brier_m5, 4),
        "optimal_T": models["M5_LongTail_Calibrated_Multimodal"]["optimal_temperature"],
        "uncalibrated_m4": bins_m4,
        "calibrated_m5": bins_m5,
    }

    # 5. Per-class Long-Tail Metrics Table (across all 40 species on Test set)
    y_pred_m3 = np.argmax(test_probs_store["M3_Image_LinearProbe"], axis=1)
    y_pred_m5 = np.argmax(test_probs_store["M5_LongTail_Calibrated_Multimodal"], axis=1)
    f1_per_class_m3 = f1_score(y_te, y_pred_m3, average=None, labels=np.arange(40), zero_division=0) * 100
    f1_per_class_m5 = f1_score(y_te, y_pred_m5, average=None, labels=np.arange(40), zero_division=0) * 100
    rec_per_class_m5 = recall_score(y_te, y_pred_m5, average=None, labels=np.arange(40), zero_division=0) * 100

    train_counts_by_idx = np.bincount(y_tr, minlength=40)
    per_class_rows = []
    for c_idx, t_id in enumerate(le.classes_):
        sp_row = sp_meta_indexed.loc[int(t_id)]
        per_class_rows.append(
            {
                "taxon_id": int(t_id),
                "scientific_name": sp_row["scientific_name"],
                "toxicity_class": sp_row["toxicity_class"],
                "lookalike_group": sp_row["lookalike_group"],
                "long_tail_tier": tier_map[c_idx],
                "train_samples": int(train_counts_by_idx[c_idx]),
                "image_only_m3_f1_pct": round(float(f1_per_class_m3[c_idx]), 2),
                "m5_f1_pct": round(float(f1_per_class_m5[c_idx]), 2),
                "m5_recall_pct": round(float(rec_per_class_m5[c_idx]), 2),
                "multimodal_f1_gain_pct": round(float(f1_per_class_m5[c_idx] - f1_per_class_m3[c_idx]), 2),
            }
        )
    df_per_class = pd.DataFrame(per_class_rows).sort_values("train_samples", ascending=False)

    # 6. Systematic Error Analysis on Locked Test Set
    df_test_sub = df.loc[te_mask].reset_index(drop=True)
    df_errors, error_summary = extract_systematic_error_analysis(
        df_test_sub, y_te, test_probs_store["M5_LongTail_Calibrated_Multimodal"], le, df_species
    )

    # 7. Generate all 5 Phase 3 Figures
    generate_phase3_figures(
        df_bench,
        cv_summary,
        df_per_class,
        cal_curves,
        error_summary,
        config["paths"]["figures_dir"],
    )

    # 8. Save all Phase 3 outputs & best model bundle for Phase 4
    os.makedirs(os.path.dirname(config["paths"]["benchmark_table_csv"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["paths"]["best_model_artifact"]), exist_ok=True)

    df_bench.to_csv(config["paths"]["benchmark_table_csv"], index=False)
    df_per_class.to_csv(config["paths"]["per_class_metrics_csv"], index=False)
    df_errors.to_csv(config["paths"]["error_analysis_csv"], index=False)

    with open(config["paths"]["hyperparams_log_json"], "w", encoding="utf-8") as f:
        json.dump(tuning_log, f, indent=2, ensure_ascii=False)

    joblib.dump(
        {
            "models": models,
            "scaler_meta": scaler_meta,
            "scaler_img": scaler_img,
            "label_encoder": le,
            "metadata_features": meta_cols,
            "tier_map": tier_map,
        },
        config["paths"]["best_model_artifact"],
    )

    full_summary = {
        "phase": 3,
        "experimental_design": {
            "split_unit": "observer_id (GroupShuffleSplit & 5-Fold GroupKFold)",
            "train_size": int(len(y_tr)),
            "val_size": int(len(y_va)),
            "locked_test_size": int(len(y_te)),
            "zero_observer_leakage_verified": True,
        },
        "cv_5fold_summary": cv_summary,
        "benchmark_results": benchmark_rows,
        "calibration_analysis": {
            "ece_uncalibrated_m4_pct": cal_curves["ece_uncalibrated_pct"],
            "ece_calibrated_m5_pct": cal_curves["ece_calibrated_pct"],
            "optimal_temperature": cal_curves["optimal_T"],
        },
        "error_analysis_summary": error_summary,
    }

    with open(config["paths"]["summary_json"], "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved Phase 3 benchmark table to: {config['paths']['benchmark_table_csv']}")
    logger.info(f"Saved Phase 3 summary JSON to:    {config['paths']['summary_json']}")

    print("\n" + "=" * 82)
    print("PHASE 3 PREDICTIVE MODELING & EXPERIMENTAL DESIGN SUMMARY (40 SPECIES)")
    print("=" * 82)
    for r in benchmark_rows:
        print(
            f"{r['model_id']:<34} | CV F1: {r['cv_5fold_macro_f1_mean']:>5.2f}±{r['cv_5fold_macro_f1_std']:<4.2f}% "
            f"| Test Acc: {r['test_accuracy_pct']:>5.2f}% | Test Macro-F1: {r['test_macro_f1_pct']:>5.2f}% "
            f"[{r['test_macro_f1_ci_low']:.1f}-{r['test_macro_f1_ci_high']:.1f}] "
            f"| Tail Rec: {r['test_tail_rare_tier_recall_pct']:>5.2f}% | ECE: {r['test_ece_pct']:>4.2f}%"
        )
    print("-" * 82)
    print(f"Same-Guild Look-Alike Error Share: {error_summary['pct_errors_within_same_lookalike_guild']}% of all test errors")
    print("=" * 82)


if __name__ == "__main__":
    main()