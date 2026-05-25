# OSS-Supply-Demand-Dynamics: Ghost Maintenance Modeling

**Authors:** Emin Açıkgöz & Bilge Oğuz

## Overview
This repository contains a complete Python-based data mining pipeline designed to analyze the supply and demand dynamics of Open Source Software (OSS). The primary goal is to quantify and predict **"Ghost Maintenance"**—a phenomenon where repositories appear superficially active but have functionally ceased active stewardship, posing a silent supply chain risk to enterprise systems. 

While the project features a robust, end-to-end data engineering architecture, **it is primarily a proof-of-concept**. Our analysis reveals that standard, aggregated GitHub event counts (constrained by free-tier API limits) are insufficient for accurate predictive modeling of maintenance cessation without a high false-positive rate.

---

## Dataset
* **Source:** Google BigQuery (`githubarchive.month.*` tables)
* **Scope:** 586 repositories filtered from an initial 1,728 candidates
* **Languages:** Java, Go, Rust, Python, JavaScript, TypeScript
* **Observation Window:** March 2024 to May 2026 (26 months)
* **Event Types Analyzed:** Issues, Pull Requests, Push, Create, Watch

---

## Technical Pipeline

### 1. Data Processing & Bot Filtering
The pipeline implements strict data cleaning to isolate authentic human stewardship:
* **Format Conversion:** Initial raw data is processed from `.parquet` to `.csv` formats using `convert_parquet_to_csv.py`.
* **Three-Layer Bot Filtering:** Removes 65% of event noise using suffix matching (`[bot]`), a legacy 55-account blocklist, and offline statistical heuristics (e.g., flagging accounts with ≥30 PRs and ≤2 users in a single month).
* **Temporal Leakage Prevention:** The 26-month dataset is strictly split into a 13-month Feature Window (March 2024 – March 2025) and a 13-month Target Window (April 2025 – May 2026) to prevent the models from accessing future data.

### 2. Economic Feature Engineering
We engineered five core features derived from economic theory to evaluate project health:
* **M-Ratio (Maintenance Ratio):** Pull Requests (Supply) ÷ Issues (Demand).
* **Gini Index:** Measures labor concentration and temporal inequality of activity.
* **Velocity:** Total operational heartbeat (PRs + Issues) normalized by active months.
* **Activity Volatility:** Coefficient of variation for monthly event counts.
* **Maintenance Burden:** Measures how heavily demand outstrips supply capacity.

### 3. Machine Learning Models
* **Unsupervised Clustering (K-Means & t-SNE):** Segments repositories into behavioral typologies based on language and ecosystem (e.g., JavaScript Frontend vs. Java Infrastructure).
* **Supervised Classification:** Predicts binary stagnation using Logistic Regression, XGBoost, and Random Forest models.

---

## Key Findings & Descriptive Analytics
The most valuable outputs of this pipeline are the descriptive, ecosystem-level insights:
* **Language Burnout:** JavaScript repositories stagnate at roughly twice the rate of Java repositories (50% vs. 23%).
* **The "Zero-Issue" Blindspot:** 23.4% of major repositories log zero Issues on GitHub, indicating heavy reliance on external trackers like Jira.
* **Persistent Stress:** The vast majority of OSS projects operate under persistent stewardship stress, clustering in an M-Ratio range of 0.2 to 0.8.

---

## Known Limitations & Data Realities
Please read before attempting to deploy this as an enterprise alerting tool:
1. **The Sparsity Tautology:** The predictive model achieves a high AUC-ROC (0.870) due to a mathematical tautology. Repositories with extremely low baseline activity in Year 1 naturally show low activity in Year 2. The primary driver is a Gini Index of 0.0 (which occurs by default when a repo only has 1 or 2 events). 
2. **High False-Positive Rate:** Ground-truth API validation showed an 80% false-positive rate. The model flags projects that lack sufficient GitHub data, rather than strictly identifying abandoned projects.
3. **API Constraints:** Due to BigQuery's 1 TB/month free-tier limits, we could not query raw payload data (e.g., commit SHAs, comment text, issue resolution speed). Accurate prediction requires deep-payload scanning, which incurs cloud computing costs.

---

## Project Structure & Execution

* `Data_Mining_Project_Script.py` - The main execution script containing the extraction, feature engineering, and modeling logic.
* `setup_and_run.py` - Orchestration script to configure the environment and run the pipeline.
* `convert_parquet_to_csv.py` - Utility script for formatting raw GitHub Archive data.
* `data/` - Directory for storing raw and processed datasets.
* `output/` - Directory for generated figures, t-SNE plots, and CSV reports.
* `requirements.txt` - Python package dependencies.

### Installation & Usage
1. Clone this repository:
   ```bash
   git clone [https://github.com/BilgeOguz/OSS-Supply-Demand-Dynamics.git](https://github.com/BilgeOguz/OSS-Supply-Demand-Dynamics.git)
   cd OSS-Supply-Demand-Dynamics
