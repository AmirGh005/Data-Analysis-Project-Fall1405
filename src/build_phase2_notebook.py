"""
Generates the reproducible Jupyter Notebook for Phase 2 Data Preparation & Pattern Discovery
(notebooks/02_phase2_pattern_discovery.ipynb).
"""

import os
import json

NOTEBOOK_PATH = "notebooks/02_phase2_pattern_discovery.ipynb"


def create_phase2_notebook() -> None:
    os.makedirs(os.path.dirname(NOTEBOOK_PATH), exist_ok=True)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# فاز دوم: آماده‌سازی داده، مهندسی ویژگی، استخراج Embedding و کشف الگو\n",
                "**پروژه درس تحلیل داده — پاییز ۱۴۰۵ (مسیر ج: iNaturalist Fungi)**\n",
                "\n",
                "در این نوت‌بوک، خروجی‌های زنجیره پیش‌پردازش بدون نشت، جدول ویژگی‌های مهندسی‌شده، تحلیل فضای بازنمایی تصویری (PCA / UMAP)، انتخاب بهینه خوشه‌ها، پایداری خوشه‌بندی و نمونه‌های غیرمعمول (OOD) بررسی می‌شوند."
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
                "with open('reports/phase2/phase2_summary.json', 'r', encoding='utf-8') as f:\n",
                "    p2_summary = json.load(f)\n",
                "print('Split Distribution:', p2_summary['split_counts'])"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ۱. کاتالوگ ویژگی‌های مهندسی‌شده (Domain-Driven Engineered Features)"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_features_cat = pd.read_csv('reports/phase2/phase2_engineered_features.csv')\n",
                "display(df_features_cat)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ۲. پروفایل خوشه‌های کشف‌شده و هم‌خوشه‌شدن گونه‌های سمی و خوراکی"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_clusters = pd.read_csv('reports/phase2/phase2_cluster_profiles.csv')\n",
                "display(df_clusters)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ۳. بررسی نمونه‌های غیرمعمول و خارج از توزیع (Flagged OOD / Outliers)"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_ood = pd.read_csv('reports/phase2/phase2_ood_outliers.csv')\n",
                "print('Qualitative OOD Diagnosis Breakdown:')\n",
                "display(pd.Series(p2_summary['ood_analysis']['qualitative_diagnosis_breakdown']))\n",
                "display(df_ood.head(10))"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ۴. نمایش نمودارهای کاهش بعد، خوشه‌بندی، دنباله بلند و OOD"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "figs = [\n",
                "    'figures/phase2/fig1_embedding_manifold_guilds_and_toxicity.png',\n",
                "    'figures/phase2/fig2_clustering_selection_and_stability.png',\n",
                "    'figures/phase2/fig3_cluster_toxicity_profiles.png',\n",
                "    'figures/phase2/fig4_long_tail_embedding_analysis.png',\n",
                "    'figures/phase2/fig5_ood_detection_and_diagnosis.png',\n",
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
    print(f"Successfully generated Phase 2 Notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    create_phase2_notebook()