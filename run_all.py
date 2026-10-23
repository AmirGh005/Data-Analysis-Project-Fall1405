"""
Master End-to-End Reproducibility Runner (Phase 1 through Phase 5).
Executes the entire data product pipeline from data acquisition and auditing
to final robustness stress testing and notebook generation.
"""

import subprocess
import sys
import time


STEPS = [
    ("Phase 1: Data Acquisition & Subset Builder", [sys.executable, "src/data_fetcher.py", "--config", "configs/phase1_config.yaml", "--mode", "auto"]),
    ("Phase 1: Data Quality Audit, EDA & Baselines", [sys.executable, "src/phase1_audit_and_eda.py", "--config", "configs/phase1_config.yaml"]),
    ("Phase 1: Generate Walkthrough Notebook", [sys.executable, "src/build_phase1_notebook.py"]),
    ("Phase 2: Preprocessing, Feature Engineering, Embeddings & Discovery", [sys.executable, "src/phase2_prep_and_discovery.py", "--config", "configs/phase2_config.yaml"]),
    ("Phase 2: Generate Pattern Discovery Notebook", [sys.executable, "src/build_phase2_notebook.py"]),
    ("Phase 3: Predictive Modeling, GroupCV, Long-Tail & Calibration", [sys.executable, "src/phase3_modeling_and_eval.py", "--config", "configs/phase3_config.yaml"]),
    ("Phase 3: Generate Modeling Notebook", [sys.executable, "src/build_phase3_notebook.py"]),
    ("Phase 4: Interpretability, Geographic Shift & Cost-Sensitive Decision", [sys.executable, "src/phase4_interpretability_and_decision.py", "--config", "configs/phase4_config.yaml"]),
    ("Phase 4: Generate Interpretability & Decision Notebook", [sys.executable, "src/build_phase4_notebook.py"]),
    ("Phase 5: Related Work, Robustness Stress Tests & Governance Audit", [sys.executable, "src/phase5_robustness_and_final.py", "--config", "configs/phase5_config.yaml"]),
    ("Phase 5: Generate Final Synthesis Notebook", [sys.executable, "src/build_phase5_notebook.py"]),
]


def main() -> None:
    start_all = time.time()
    print("=" * 80)
    print("STARTING END-TO-END REPRODUCIBLE PIPELINE (PHASES 1 TO 5)")
    print("=" * 80)

    for step_name, cmd in STEPS:
        print(f"\n>>> Running: {step_name} ...")
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            print(f"[ERROR] Step failed: {step_name}")
            sys.exit(res.returncode)

    elapsed = time.time() - start_all
    print("\n" + "=" * 80)
    print(f"ALL 5 PHASES COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS!")
    print("=" * 80)


if __name__ == "__main__":
    main()