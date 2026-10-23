"""
Master End-to-End Reproducibility Runner (Phase 1 through Phase 5 - 100% REAL iNaturalist Pipeline).
Automatically detects and uses the project's `venv` Python interpreter if available.
"""

import os
import subprocess
import sys
import time


def get_python_executable() -> str:
    """Prefers the local virtualenv Python interpreter on Windows/Linux/macOS."""
    win_venv = os.path.join("venv", "Scripts", "python.exe")
    unix_venv = os.path.join("venv", "bin", "python")
    if os.path.exists(win_venv):
        return os.path.abspath(win_venv)
    if os.path.exists(unix_venv):
        return os.path.abspath(unix_venv)
    return sys.executable


PY_EXEC = get_python_executable()

STEPS = [
    ("Phase 1: Real iNaturalist Data & Image Verification", [PY_EXEC, "src/data_fetcher.py", "--config", "configs/phase1_config.yaml", "--mode", "real"]),
    ("Phase 1: Data Quality Audit, EDA & Baselines", [PY_EXEC, "src/phase1_audit_and_eda.py", "--config", "configs/phase1_config.yaml"]),
    ("Phase 1: Generate Walkthrough Notebook", [PY_EXEC, "src/build_phase1_notebook.py"]),
    ("Phase 2: Real PyTorch Vision Inference (DINO & ConvNeXt) & Pattern Discovery", [PY_EXEC, "src/phase2_prep_and_discovery.py", "--config", "configs/phase2_config.yaml"]),
    ("Phase 2: Generate Pattern Discovery Notebook", [PY_EXEC, "src/build_phase2_notebook.py"]),
    ("Phase 3: Predictive Modeling, GroupCV, Long-Tail & Calibration", [PY_EXEC, "src/phase3_modeling_and_eval.py", "--config", "configs/phase3_config.yaml"]),
    ("Phase 3: Generate Modeling Notebook", [PY_EXEC, "src/build_phase3_notebook.py"]),
    ("Phase 4: Interpretability, True Geo-Invariance, Real ConvNeXt Ablation & Decision Policy", [PY_EXEC, "src/phase4_interpretability_and_decision.py", "--config", "configs/phase4_config.yaml"]),
    ("Phase 4: Generate Interpretability & Decision Notebook", [PY_EXEC, "src/build_phase4_notebook.py"]),
    ("Phase 5: Related Work, Real Pixel Blur Stress Tests & Dynamic Governance Audit", [PY_EXEC, "src/phase5_robustness_and_final.py", "--config", "configs/phase5_config.yaml"]),
    ("Phase 5: Generate Final Synthesis Notebook", [PY_EXEC, "src/build_phase5_notebook.py"]),
    ("Final Step: Synchronize README & Reports with Exact Real Empirical Metrics", [PY_EXEC, "src/sync_reports_with_real_results.py"]),
]


def main() -> None:
    start_all = time.time()
    print("=" * 84)
    print(f"STARTING 100% REAL iNATURALIST END-TO-END PIPELINE")
    print(f"Using Python Interpreter: {PY_EXEC}")
    print("=" * 84)

    for step_name, cmd in STEPS:
        print(f"\n>>> Running: {step_name} ...")
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            print(f"[ERROR] Step failed: {step_name}")
            sys.exit(res.returncode)

    elapsed = time.time() - start_all
    print("\n" + "=" * 84)
    print(f"ALL 5 REAL-DATA PHASES COMPLETED & SYNCHRONIZED IN {elapsed:.1f} SECONDS!")
    print("=" * 84)


if __name__ == "__main__":
    main()