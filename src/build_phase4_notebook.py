"""
Generates the reproducible Jupyter Notebook for Phase 4 Interpretability, Shift & Decision Analysis
(notebooks/04_phase4_interpretability_and_decision.ipynb).
"""

import os
import json

NOTEBOOK_PATH = "notebooks/04_phase4_interpretability_and_decision.ipynb"


def create_phase4_notebook() -> None:
    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# فاز چهارم: تفسیرپذیری، تحلیل تغییر دامنه (Domain Shift)، کالبدشکافی و تصمیم‌گیری آگاه از ریسک\n",
                "**پروژه درس تحلیل داده — پاییز ۱۴۰۵ (مسیر ج: iNaturalist Fungi)**\n",
                "\n",
                "این نوت‌بوک نتایج تفسیرپذیری کلی و موردی، کالبدشکافی وجوه داده و انکودرهای تصویری، ارزیابی تغییر دامنه جغرافیایی روی قاره اروپا، تحلیل زیرگروه‌ها و سیاست تصمیم‌گیری حساس به هزینه (Cost-Sensitive Decision Making) را نمایش می‌دهد."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import pandas as pd\n",
                "from IPython.display import Image, display\n",
                "\n",
                "os.chdir('..') if os.path.basename(os.getcwd()) == 'notebooks' else None\n",
                "with open('reports/phase4/phase4_summary.json', 'r', encoding='utf-8') as f:\n",
                "    p4_summary = json.load(f)\n",
                "print('Modality Group Importance Drops (% Macro-F1):')\n",
                "display(pd.Series(p4_summary['global_interpretability']['modality_group_macro_f1_drops_pct']))"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ۱. کالبدشکافی وجوه داده، تغییر دامنه جغرافیایی (آمریکای شمالی به اروپا) و مقایسه انکودرها"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_ablation = pd.read_csv('reports/phase4/phase4_modality_and_ shift_ablation.csv')\n",
                "display(df_ablation)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ۲. تفسیرپذیری موردی (Case-Level Interpretability: Rescue, Failure, and Boundary Cases)"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_cases = pd.read_csv('reports/phase4/phase4_case_level_interpretability.csv')\n",
                "display(df_cases)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ۳. عملکرد زیرگروه‌ها (Subgroup Audit) و سیاست تصمیم‌گیری آگاه از ریسک"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_sub = pd.read_csv('reports/phase4/phase4_subgroup_performance.csv')\n",
                "print('=== Subgroup Performance ===')\n",
                "display(df_sub)\n",
                "\n",
                "df_dec = pd.read_csv('reports/phase4/phase4_cost_sensitive_decision_curve.csv')\n",
                "print('\\n=== Cost-Sensitive Decision Policy Curve ===')\n",
                "display(df_dec)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ۴. نمایش نمودارهای فاز چهارم"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "figs = [\n",
                "    'figures/phase4/fig1_global_interpretability_and_feature_importance.png',\n",
                "    'figures/phase4/fig2_geographic_shift_and_representation_ablation.png',\n",
                "    'figures/phase4/fig3_subgroup_audit_and_sensitivity_heatmap.png',\n",
                "    'figures/phase4/fig4_cost_sensitive_decision_and_risk_coverage.png',\n",
                "    'figures/phase4/fig5_causal_dag_vs_predictive_shortcuts.png',\n",
                "]\n",
                "for fp in figs:\n",
                "    if os.path.exists(fp):\n",
                "        print(f'\\n--- {os.path.basename(fp)} ---')\n",
                "        display(Image(filename=fp))"
            ],
        },
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated Phase 4 Notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    create_phase4_notebook()