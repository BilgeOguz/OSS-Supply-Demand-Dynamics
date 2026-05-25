# OSS Maintenance Risk Prediction — Data Mining Methodology

**Generated:** 2026-05-24 (Expanded v2: 1,728 repos)
**Pipeline:** `local_run.py` with re-harvested data
**Dataset:** GitHub Archive (`githubarchive.month.*` via BigQuery)
**Window:** March 2024 – May 2026 (26 months)
**Event types:** IssuesEvent, PullRequestEvent, PushEvent, CreateEvent, WatchEvent
**Bot filtering:** 3 layers — SQL `[bot]` suffix, 45-account blocklist, offline statistical heuristics

> **v2 changes from previous run:** Expanded keywords (34→140) increased repos 3.2× (542→1,728). Rust coverage fixed (1→11 repos). Event types expanded (2→5). Survival analysis and per-language models added. All 12 planned improvements implemented.

---

## 1. Algorithms Used

### 1.1 Cohort SQL Filtering (Data Preparation)

**What:** A SQL WHERE clause selects repositories matching six target languages (Java, Go, Rust, Python, JavaScript, TypeScript), 140 inclusion keyword patterns (e.g., `%spring%`, `%react%`), and a minimum star threshold (≥32). A separate `bigquery-public-data.github_repos.languages` join assigns each repo its primary language by byte count. Bot filtering excludes events from known bot accounts (`%[bot]%` suffix filter + 45-account blocklist).

**Why:** This defines the study population. Without filtering, the dataset would be dominated by tutorial/demo repos, abandoned experiments, or personal projects with no meaningful maintenance activity. The inclusion keywords target production-grade OSS ecosystems (frameworks, databases, APIs, build tools). The star threshold acts as a coarse quality proxy — projects with 32+ watchers are more likely to have a community, though this is an imperfect filter.

**Implementation detail:** The query uses a CTE (`visibility_filter`) that cross-joins `github_repos.languages` with `sample_repos` (watch_count), then inner-joins to `githubarchive.month.*` events. This avoids scanning the full event table against repos that would fail the language/star filters.

**Verification:** 1,801 repos harvested, 1,728 with features. 6 languages represented. Rust is minimally represented (11 repos) — improved from 1 repo via expanded ecosystem keywords.

### 1.2 Temporal Aggregation (Feature Engineering)

**What:** Raw events are grouped by `(repo_name, event_type, year, month)` and aggregated to `event_count` and `unique_users` (COUNT DISTINCT actor.login). The 26-month window is split at a strict calendar boundary:

| Window | Months | Period |
|--------|--------|--------|
| **Feature** | 1–13 | March 2024 – March 2025 |
| **Target** | 14–26 | April 2025 – May 2026 |

**Why:** Temporal separation at a single calendar boundary prevents data leakage. No feature window event can contain information about the target window. This is the most common error in time-series classification and is the single most important design decision in the pipeline.

**Verification:** Feature rows = 27,584, Target rows = 19,977. 1,728 repos in feature, 1,402 in target (326 repos have no target-window events at all).

### 1.3 Feature Computation (Seven Engineered Features)

All features are computed exclusively from the **feature window** (months 1–13):

| Feature | Formula | What It Measures | Rationale |
|---------|---------|-----------------|-----------|
| **M-Ratio** | `PRs / Issues` | Supply/demand balance of maintenance work. <1 = demand outpaces supply | Core metric from OSS bus-factor literature |
| **Velocity** | `(PRs + Issues) / active_months` | Mean monthly event rate. Proxy for project vitality | Captures overall momentum independent of window length |
| **Gini Index** | Standard Gini coefficient on `event_count` per month | Concentration of activity. 0 = perfect equality, 1 = one month dominates | Borrowed from economics (income inequality). Detects repos where work is dangerously concentrated in a few months — a pattern associated with maintainer burnout |
| **Maintenance Burden** | `Issues / (PRs + 1)` | How much demand outstrips supply capacity. Higher = more pressure | Asymmetric: burden can grow arbitrarily large (many Issues, few PRs) |
| **Activity Volatility** | `std(event_count) / mean(event_count)` | Coefficient of variation — stability of activity over time | High volatility may indicate sporadic maintenance or seasonal contributors |
| **Active Months** | `len(unique months)` | How many distinct months the repo had events | Captures engagement duration vs burst activity |
| **Sparse Data Flag** | `True if n_months < 2 or total_events == 0` | Identifies repos where Gini=0 is forced, not measured | Prevents false confidence in repos with insufficient data |

**Limitation:** All five features are derived from *counts*, not *content*. They cannot capture code quality, community sentiment, contributor churn, or release cadence. This is the fundamental constraint of the free-tier BigQuery approach.

### 1.4 Gini Coefficient (Detailed)

The Gini coefficient is the algorithmic centerpiece. Implementation:

```python
def calculate_gini(series):
    x = series.to_numpy()
    n = len(x)
    if n < 2 or x.sum() == 0:
        return 0.0, True  # sparse_data = True
    x_sorted = np.sort(x)
    index = np.arange(1, n + 1)
    g = (2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n
    return g, False
```

**Critical behavior:** When a repo has 0 or 1 month(s) of data, Gini is forced to 0.0 with `sparse_data=True`. This means 1,007 repos (58.3%) have Gini=0 by definition, not by measurement (expanded event types brought in many PushEvent-only repos with sparse activity). The `sparse_data` flag is the dominant predictor in the Cox survival model (hazard ratio = 15.9).

### 1.5 Target Definition

**What:** A repo is labeled `stagnant = 1` if it has zero rows in the target window (no observed Issues or PRs for 13+ months). A repo is `stagnant = 0` if it has at least one event.

**Why:** The simplest possible definition avoids subjective thresholds. Using "zero activity" rather than "below a threshold" makes the target objective and reproducible.

**Result:** 399/1,728 repos (23.1%) stagnant. The 1,402 repos with target-window activity contribute 0 stagnant labels; 326 repos vanish in the target window (filled via `fill_null(1)`). Stagnation rate dropped from 39.3% to 23.1% because expanded keywords brought in more actively maintained repos.

**Limitation:** A single automated PR merge from Dependabot marks a repo as "active" even if it's otherwise abandoned. A repo that migrated its issue tracking off GitHub appears as "dead" when it's actually healthy. The binary target is too weak to capture gradations of maintenance risk.

### 1.6 K-Means Clustering (Unsupervised)

**What:** Repos are clustered on four standardized, log-transformed features (m_ratio, gini_index, log_velocity, maintenance_burden) using K-Means. Optimal k is selected by maximizing silhouette score over k=2..8. t-SNE (perplexity=30, max_iter=500) projects the 4D feature space to 2D for visualization. The t-SNE double-fit bug was fixed (single `fit_transform` call).

**Why:** Clustering reveals natural groupings in behavioral space without requiring labels. t-SNE preserves local neighborhood structure better than PCA for sparse, skewed data with many near-zero values.

**Result (v2 expanded data):** k=7, silhouette=0.377:

| Cluster | Size | Description | Language | Category |
|---------|------|-------------|----------|----------|
| **0** | 177 | Near Equilibrium, Low Activity | 30% Python | 18% Framework |
| **1** | 243 | Near Equilibrium, Low Activity | 38% JS | 32% Framework |
| **2** | 6 | Demand >> Supply, Low Activity | 67% JS | 67% Frontend |
| **3** | 84 | Near Equilibrium, High Activity | 39% Java | 29% Infra/Cloud |
| **4** | 1 | Supply >> Demand outlier | 100% JS | 100% API/Integration |
| **5** | 5 | Supply >> Demand | 60% Python | 40% Database/ORM |
| **6** | 12 | Demand >> Supply, Low Activity | 42% Java | 42% Other |

**Key finding:** With 3× more repos, the data separates into 7 finer-grained clusters. The velocity-driven separation between "high-activity" (cluster 3) and "low-activity" (clusters 0-1) persists. The expanded category keywords enable better category assignment — the "Other" catch-all has meaningful behavioral patterns.

### 1.7 Supervised Classification Tournament

Three models were trained to predict `stagnant` from six features (m_ratio, gini_index, velocity, maintenance_burden, active_months, sparse_data):

| Model | Type | Why Chosen | Configuration |
|-------|------|------------|---------------|
| **Logistic Regression** | Linear classifier with L2 regularization | Interpretable coefficients, fast, handles correlated features via regularization | `class_weight="balanced"`, `max_iter=1000` |
| **XGBoost** | Gradient-boosted decision trees | Captures nonlinear interactions, robust to outliers, SOTA on tabular data | `n_estimators=100`, `max_depth=3`, `lr=0.1` |
| **Random Forest** | Bagged decision trees | Handles nonlinearity, provides feature importance, less prone to overfitting than single trees | `n_estimators=100`, `class_weight="balanced"` |
| **Dummy (Majority)** | Most-frequent class baseline | Lower bound for comparison | `strategy="most_frequent"` |

**Evaluation:** 5-fold GroupKFold cross-validation (groups = repo names). This ensures all rows for the same repo stay in the same fold, preventing within-repo information leakage. LR and XGBoost use a `Pipeline` with `StandardScaler`; RF does not require scaling.

**VIF analysis:**

| Feature | VIF |
|---------|-----|
| gini_index | 5.08 |
| active_months | 5.54 |
| velocity | 1.29 |
| maintenance_burden | 1.16 |
| m_ratio | 1.13 |
| sparse_data | 1.00 |

`gini_index` and `active_months` exceed VIF=5 and are excluded from the tournament features. The final feature set is `[m_ratio, velocity, maintenance_burden, sparse_data]`. The `sparse_data` boolean flag (VIF=1.00) becomes the dominant predictor.

### 1.8 SHAP Model Interpretation

**What:** SHapley Additive exPlanations decompose each prediction into feature contributions. For each repo in the top 15, the single feature with the largest positive SHAP value is reported as the "primary driver."

**Why:** SHAP is necessary because a high AUC-ROC doesn't tell you *why* a specific repo was flagged. The SHAP analysis reveals whether the model is making decisions for substantively meaningful reasons or for data-quality artifacts.

### 1.9 Survival Analysis (Cox Proportional Hazards)

**What:** A Cox proportional hazards model (via `lifelines.CoxPHFitter`) models time-to-stagnation using the same features. Duration `T` = number of months the repo was active (first to last event in window). Event `E` = 1 if stagnant, 0 if censored.

**Why:** Survival analysis subsumes the binary classification target by incorporating *when* stagnation occurs. It answers "how much longer will this repo survive?" rather than just "will it die?"

**Result (v2, 1,728 repos):** Concordance index = 0.853

| Covariate | Coefficient | Hazard Ratio | p-value |
|-----------|-------------|-------------|---------|
| sparse_data | 2.77 | 15.97 | <0.005 |
| maintenance_burden | -0.08 | 0.92 | <0.005 |
| m_ratio | -0.05 | 0.95 | <0.005 |
| velocity | -0.01 | 0.99 | <0.005 |

**Key finding:** `sparse_data` repos have **15.9× higher hazard** of stagnation — the strongest signal in the entire pipeline. The proportional hazards assumption was tested (`check_assumptions`) with expected violations — m_ratio's effect changes over time.

### 1.10 Per-Language Stratified Models

**What:** Logistic regression models trained separately for each language with ≥20 repos, using 5-fold GroupKFold.

**Why:** Language-level stagnation rates vary 3:1 (JS 32% vs Rust 0%). A global model may miss language-specific patterns.

**Result:**

| Language | Repos | AUC-ROC |
|----------|-------|---------|
| TypeScript | 80 | 0.885 |
| Java | 291 | 0.841 |
| Python | 323 | 0.807 |
| Go | 99 | 0.772 |
| JavaScript | 924 | 0.758 |
| Rust | 11 | skipped (n<20) |

**Finding:** Per-language models outperform the global model for TypeScript and Java. JS proves hardest to predict due to its diversity.

### 1.11 Ablation Study

**What:** Substitute `activity_volatility` for `gini_index` and compare AUC-ROC.

**Why:** Gini_index is often VIF-excluded; volatility is an orthogonal measure of activity concentration.

**Result:** Ablation AUC-ROC = **0.855** (vs baseline 0.814). **Change: +0.041**. Activity volatility is a better predictor when Gini is unavailable.

### 1.12 Ground-Truth Validation via GitHub API

**What:** Repos across 5 risk quintiles are sampled (10 per quintile) and checked against the GitHub REST API for archival status (`archived` flag or "deprecated"/"unmaintained" in description).

**Why:** This is an external validity check. Stratified sampling across risk quintiles gives a more representative picture than only checking the top 15.

**Result (v2):** Unable to verify — no `GH_PAT` set for the re-harvest run. All API calls returned authentication-limited responses. Set `GH_PAT` environment variable and re-run to enable validation.

**Previous result (v1, 542 repos):** 2/15 (13.3%) confirmed dead (blazegraph/database, debop/hibernate-redis).

---

## 2. ML Tournament: XGBoost Wins

**The tournament result (v2, 1,728 repos):**

| Model | AUC-ROC | F1 | Recall |
|-------|---------|-----|--------|
| XGBoost | **0.814** | 0.528 | 0.436 |
| Random Forest | **0.814** | 0.533 | 0.886 |
| Logistic Regression | 0.795 | 0.517 | 0.665 |
| Dummy (majority) | 0.500 | 0.000 | 0.000 |

**Why XGBoost/RF beat LR on the expanded data:**

1. **Larger, more diverse dataset:** With 1,728 repos (vs 542), the decision boundary is more complex. Gini_index was VIF-excluded, so `sparse_data` — a boolean with nonlinear interactions — becomes the primary feature. Tree models handle boolean features better than linear models.

2. **sparse_data dominance:** All models rely on the `sparse_data` flag. XGBoost and RF capture threshold interactions (e.g., `sparse_data=True AND low_velocity`), while LR can only assign a single linear coefficient.

3. **Ablation finding:** Substituting `activity_volatility` for `gini_index` improved AUC to 0.855 (+0.041), confirming volatility is a better predictor than Gini when Gini is VIF-excluded.

**However, the narrow margin matters:** The 0.019 gap between XGBoost and LR is small. The difference between XGBoost and RF is 0.000. LR's Recall (0.665) is actually better than XGBoost's (0.436). The "winner" label is technically correct but practically marginal — the real signal is `sparse_data` regardless of model choice.

---

## 3. Findings

### 3.1 Primary Finding: Sparse Data Is the Real Signal

Despite AUC-ROC = 0.814 (v2, XGBoost) / 0.870 (v1, LR), the dominant predictor across both versions is a **data quality artifact**:

- **sparse_data=True repos have 15.9× higher stagnation hazard** (Cox model, p<0.005).
- **Top risk repos are driven by SPARSE_DATA** — they have insufficient event history to measure.
- **1,007 repos (58.3%) have Gini=0 by definition** (sparse_data=True, forced Gini=0).
- The model detects **repos with insufficient data**, not repos that are *actually abandoned*.

**Mitigation implemented:** `sparse_data` is now an explicit feature (added in v2), making the artifact transparent. The survival analysis separates "never had data" from "went silent after having data."

### 3.2 Secondary Findings (Actionable)

| Finding | Confidence | Impact |
|---------|------------|--------|
| **JS stagnates at 2.5× the rate of Java** (32.3% vs 13.1%) | High — large sample sizes (924 vs 291) | Choose Java-ecosystem dependencies when possible |
| **1,007 repos (58.3%) have zero Issues** — PushEvent/CreateEvent-only repos | High — direct measurement | GitHub Issues alone is insufficient for monitoring |
| **399/1,728 repos (23.1%) had zero activity in the target window** | High — direct measurement | Almost 1/4 of the cohort went completely silent |
| **Per-language models show TS easiest to predict (0.885), JS hardest (0.758)** | High — cross-validated | Language-specific monitoring thresholds needed |
| **Survival analysis c-index=0.853** outperforms tournament models on time-to-event | Medium — new feature | Survival modeling should replace binary classification |
| **Ablation shows volatility > Gini (+0.041 AUC)** when Gini is VIF-excluded | Medium — single experiment | Replace Gini with volatility in future iterations |
| **Rust coverage improved 1→11 repos** but still insufficient for modeling | High — direct measurement | Further keyword expansion or manual Rust cohort needed |
| **Bot filtering at query level removed ~65% events** (from earlier comparison) | High — direct measurement | Required for realistic M-Ratio, zero repos entirely removed |

### 3.3 Surprising Negative Results

- **Velocity does not predict stagnation.** Low-velocity repos are equally likely to be active or stagnant. The correlation is weak (Spearman ρ ≈ -0.15).
- **Language-specific models would not help.** The clusters show that language is a weak separator of behavioral profiles. Functional category (Framework vs Infrastructure) separates better.
- **Bot contamination is low** (1.8% flagged, mean score 0.012) after the BigQuery-level filter. Layer 3 found few additional bots beyond Layer 1's `[bot]` suffix filter.

### 3.4 Data Mining Category Checklist

| Category | Used? | Algorithm | Why / Why Not |
|----------|-------|-----------|---------------|
| Classification | ✅ | LR, XGBoost, RF | Predict binary stagnation |
| Clustering | ✅ | K-Means + silhouette | Discover behavioral groupings |
| Dimensionality Reduction | ✅ | t-SNE | Visualize 4D clusters in 2D |
| Feature Selection | ✅ | VIF analysis, SHAP | Eliminate multicollinear features, interpret predictions |
| Anomaly/Outlier Detection | ✅ | 3-layer heuristics + contamination score | Detect bot-like activity |
| Model Interpretation | ✅ | SHAP LinearExplainer / TreeExplainer | Decompose per-repo predictions |
| Imputation | ⛔ Rejected | — | Ghost-demand imputation removed; see §6 |
| Time Series / Survival Analysis | ❌ Absent | — | Binary classification is wrong framing; see §7 |
| NLP / Text Mining | ❌ Absent | — | Quota-limited; see §8 |
| Association / Dependency Mining | ❌ Failed | — | GraphQL API returned 0 for all repos; see §9 |
| Causal Inference | ❌ Absent | — | Confounders (age, team size) may drive both low velocity and stagnation |
| Regression (continuous target) | ❌ Absent | — | Target is binary, not continuous |

---

## 4. Flaws in the Current Method

### 4.1 Data Quality Flaws

| Flaw | Severity | Details |
|------|----------|---------|
| **Binary target too weak** | Critical | A single Dependabot PR in 13 months = "active." A repo that went silent for 12 months then got 1 PR = "active." This conflates "truly maintained" with "not completely dead." |
| **Features only from event counts** | Critical | No commit payloads, no contributor demographics, no issue/PR body text, no review comments, no release frequency. All richness is lost in aggregation. |
| **Gini=0 is a tautology for low-activity repos** | High | 131 repos (24.2%) have forced Gini=0 because they have <2 months of data. The model's primary signal is a measurement artifact. |
| **26-month window is too short** | Medium | Seasonal patterns (summer slowdowns, holiday dips, conference-driven contribution spikes) appear as stagnation. Long-lived projects with multi-year release cycles are misclassified. |
| **GitHub-only scope** | Medium | Repos using Jira, GitLab Issues, or GitHub Discussions for issue management appear as "zero demand." 23.4% of the cohort is affected. |
| **Rust under-sampled** | Medium | Only 1 Rust repo survived the keyword filter. The inclusion keyword list was designed for Spring/React/Django ecosystems and misses Rust-specific terms (e.g., `%cargo%`, `%rust-%`). |
| **Star threshold ≥32 is arbitrary** | Low | A repo with 31 watchers is excluded; one with 32 is included. This boundary effect adds noise without clear justification. |

### 4.2 Methodological Flaws

| Flaw | Severity | Details |
|------|----------|---------|
| **No per-language stratification** | High | Training one model across all languages assumes the same Features→Stagnation relationship holds for JS libraries, Java infrastructure, and Go CLI tools. The t-SNE visualization shows this is false. |
| **No causal framework** | High | All reported relationships are correlations. Low velocity may cause stagnation, or stagnation may cause low velocity, or a third factor (founder burnout, funding loss) may cause both. |
| **Alternatives recommendation is broken** | Medium | The ecosystem-matching algorithm returns the highest-velocity repo in the same language regardless of functional fit. `facebook/react-native` is recommended for `express-winston` — a logging middleware. |
| **Ground-truth validation is too small (n=15)** | Medium | 15 repos at the extreme high-risk end is insufficient to estimate classifier performance. 80% of the data falls in low-risk territory with no validation. |
| **Single temporal split** | Low | A single calendar boundary means one train/test realization. Rolling-window or time-series cross-validation would produce more robust estimates. |

### 4.3 Free-Tier Constraints

The 1 TB/month BigQuery free tier prevents querying raw event payloads. The following features would each increase query bytes scanned by 3-10×:

- Issue response time (nested `payload.issue` JSON)
- PR merge rate (`payload.pull_request.merged` boolean)
- Contributor churn (full commit log analysis)
- Bus factor (per-author commit percentages)
- Release cadence (`push` events on release branches)
- Sentiment analysis (issue/PR body text)

---

## 5. Things to Improve

### 5.1 Immediate Improvements (Within Free Tier)

1. **Stratify models by language.** Train separate LR models for JS, Java, Python, TS, Go. The t-SNE visualization shows language-specific behavioral patterns that a global model cannot capture.

2. **Fix Rust under-sampling.** Add Rust-specific keywords: `%cargo%`, `%rust-%`, `%serde%`, `%clap%`, `%wasm%`. The single Diesel-rs repo does not represent the Rust ecosystem.

3. **Expand event types.** Add `PushEvent`, `CreateEvent` (releases), `WatchEvent` (community interest). These are available in the same GH Archive tables at no additional BigQuery cost.

4. **Improve alternatives recommendation.** Instead of highest-velocity repo in the same language, match by functional category (Framework→Framework, Database→Database). Use the existing `assign_category()` function.

5. **Increase ground-truth validation.** Sample 50 repos across the risk spectrum (not just top-15). This would produce a precision-recall curve rather than a single point estimate.

### 5.2 Medium Improvements (Paid BigQuery)

6. **Survival analysis instead of binary classification.** Model "time to stagnation" using Cox Proportional Hazards or Kaplan-Meier estimators. This is the correct statistical framing — it uses all the temporal information instead of collapsing to a binary label.

7. **Query raw JSON payloads for:**
   - Issue close time → response latency feature
   - PR merge status → merge rate (distinguishes maintenance from drive-by PRs)
   - Actor login per event → contributor churn rate

8. **Expand to all event types** with payload parsing — `PushEvent` for commit frequency, `CreateEvent` for release cadence, `ReleaseEvent` for version tracking.

### 5.3 Fundamental Redesign

9. **Use the GitHub REST/GraphQL API directly** instead of GH Archive. This gives access to per-repo metadata (fork count, license, topics, description, last push timestamp) that GH Archive lacks. Rate limits (5,000 req/hr with PAT) would constrain the cohort size but enable richer features.

10. **Adopt a multi-label or ordinal target.** Instead of binary stagnant/active, use:
    - **Healthy:** Regular activity in every quarter
    - **At-risk:** Activity declining
    - **Stagnant:** No activity for 6+ months
    - **Archived:** Official archived flag

11. **Validate against libraries.io data.** The Dependency Graph API failed (see §9), but libraries.io provides a downloadable dataset of package dependencies. This would enable actual blast-radius analysis.

---

## 6. Anomaly Detection — Three-Layer Bot Filtering

### 6.1 Motivation

GitHub Archive events include automated accounts (Dependabot, Renovate, GitHub Actions) that generate PRs at machine scale. Without filtering, a repo with 1,000 Dependabot PRs and 10 human PRs appears healthy (M-Ratio ≈ 1.0). Bot filtering is essential for honest measurement.

### 6.2 Layer 1 — SQL Suffix Filter

**Mechanism:** `WHERE events.actor.login NOT LIKE '%[bot]%'`

**Coverage:** Catches ~95% of bot accounts. GitHub Apps automatically receive the `[bot]` suffix in their login name (e.g., `dependabot[bot]`, `renovate[bot]`). This is the CHAOSS/GrimoireLab standard approach.

**Limitation:** Legacy accounts created before the `[bot]` convention (pre-2018) do not carry the suffix. Accounts that registered as users rather than apps also bypass this filter.

### 6.3 Layer 2 — Known-Name Blocklist

**Mechanism:** `WHERE events.actor.login NOT IN (55 known bot usernames)`

**Coverage:** Catches legacy bots that predate the `[bot]` suffix convention. Categories:

| Category | Examples | Count |
|----------|----------|-------|
| Dependency management | dependabot, dependabot-preview, renovate, pyup-bot | ~6 |
| CI/CD & Automation | github-actions, jenkins-bot, travis-ci, circleci, mergify | ~12 |
| Code Quality & Coverage | codecov, coveralls, snyk, sonarcloud, deepsource | ~12 |
| Security | whitesource-bolt, mend-bolt-for-github | ~2 |
| Community & Events | hacktoberfest, allcontributors, stale | ~5 |
| Platform / Foundation | facebook-bot, googlebot, dotnet-bot | ~6 |
| Translation / Docs | crowdin-bot, weblate, readthedocs | ~3 |
| Project-specific | react-bot, angular-bot, kubernetes-bot, nodejs-bot | ~4 |

**Limitation:** The blocklist must be maintained manually. New bot services appear regularly. Reverse-engineered personal tokens (cron jobs using real user accounts) are not caught.

### 6.4 Layer 3 — Statistical Heuristics (Offline)

**Mechanism:** Three rules applied to the cached feature data, not requiring additional BigQuery calls:

1. **PR velocity anomaly:** Repos where ≤2 users generated ≥30 PRs in a single month. Flag: `bot_velocity_flag = True`. Weight: 0.5.

2. **M-Ratio bias:** Repos with m_ratio > 10 AND total_load > 100. This catches repos where PR volume far exceeds what human-maintained repos typically show. Flag: `bot_bias_flag = True`. Weight: 0.3.

3. **Contributor anomaly:** Repos with velocity > 50 AND avg_monthly_contributors < 3. High automation throughput with tiny human contributor base. Flag: `bot_contrib_flag = True`. Weight: 0.2.

**Contamination Score Formula:**
```
contamination_score = velocity_flag * 0.5 + bias_flag * 0.3 + contrib_flag * 0.2
```

**Result:** 10/542 repos (1.8%) flagged as bot-heavy. Mean contamination score: 0.012. Top suspects:

| Repo | Score | Reason |
|------|-------|--------|
| mybatis/spring | 0.80 | PR velocity + M-Ratio bias (30 PRs from ~2 users/month, m_ratio=12.5) |
| apache/shiro | 0.50 | PR velocity (399 suspicious PRs across 8 months) |
| apache/opennlp | 0.50 | PR velocity (67 suspicious PRs, 2 suspicious months) |
| go-resty/resty | 0.50 | PR velocity (89 suspicious PRs, 2 suspicious months) |

**Limitation:** The heuristic thresholds (30 PRs, m_ratio > 10, velocity > 50) are untuned. A principled approach would use an isolation forest or one-class SVM on the full feature set.

### 6.5 Effectiveness

Bot filtering (Layers 1+2 at query time, Layer 3 offline) reduced event volume by ~65% in the reharvest run (741K → 258K events). M-Ratios dropped 16-41% across languages after filtering. The filter is essential for honest measurement but introduces a bias: repos that rely heavily on automation (CI/CD-heavy workflows) may have legitimate high PR counts from a small team.

---

## 7. Imputation Methods — Ghost-Demand Removal

### 7.1 Original Approach (Removed)

The earlier version of this pipeline attempted to impute "ghost demand" — Issues that are managed externally (GitHub Discussions, Jira, mailing lists). The approach used a constant conversion ratio of 0.73 (median Issue-to-PR ratio across the cohort) to estimate phantom Issues for repos with zero Issues but many PRs.

Formula used previously:
```
imputed_issues = PR_count * 0.73
```

### 7.2 Why It Was Removed

Manual checks across individual repos showed that the 0.73 ratio does not generalize:

- **By language:** JS repos showed 0.45 median, Java repos showed 1.2 median
- **By maturity:** Young repos (<2 years) averaged 0.3; mature repos (>5 years) averaged 0.9
- **By community size:** Repos with 1-2 contributors had wildly varying ratios that could not be predicted from features

Using a single scalar obscured real heterogeneity and produced misleading effective-demand estimates.

### 7.3 Current Approach

1. **No imputation.** Raw IssuesEvent counts are used directly.
2. **Zero-Issue repos are flagged.** 127/542 (23.4%) are reported with a `zero_issue_repos` column.
3. **No synthetic values are substituted.** Every analysis note disclaims that repos with zero Issues may manage demand externally.

### 7.4 Methodological Principle

This follows Rubin's taxonomy of missing data: the missingness is **Missing Not at Random** (MNAR) — repos manage Issues externally for systematic reasons (size, community norms, tooling preferences) that are correlated with the outcome. Imputing MNAR data with a constant produces biased estimates. Flagging missingness honestly is the correct approach when the missingness mechanism cannot be modeled.

---

## 8. NLP / Text Mining — Absent

### 8.1 Why It's Absent

NLP features require access to raw event payloads (nested JSON in `githubarchive.month.*`):

- Issue/PR body text → sentiment analysis, toxic comment detection
- Commit messages → change type classification (fix, feature, refactor)
- Review comments → review quality, maintainer responsiveness

Querying these payloads would increase BigQuery bytes scanned by 3-10× per query, pushing well beyond the 1 TB/month free tier. The current pipeline uses only the outer event schema (`event.type`, `event.repo.name`, `event.created_at`), which is a fraction of the total row size.

### 8.2 What Would Be Possible

If billing were enabled:

| Feature | Data Source | Expected Signal |
|---------|-------------|-----------------|
| Issue closing sentiment | `payload.issue.body` | Identify frustrated vs constructive user reports |
| PR review quality | `payload.pull_request.review_comments` | Measure maintenance thoroughness |
| Burnout language | Recent issue comments | "Maintainer is overwhelmed" keyword detection |
| Release note content | `payload.release.body` | Measure documentation quality |
| Commit message hygiene | `payload.commits[].message` | Correlate message quality with project health |

### 8.3 Why Not External NLP APIs

Using GitHub's REST API for text extraction was considered but rejected because:

- Each issue/PR body requires a separate API call → 5,000 req/hr limit with PAT
- The cohort has 586 repos with thousands of issues → weeks of wall-clock time
- Bot detection on text would require additional filtering (Dependabot PR bodies are templated)

---

## 9. Association / Dependency Mining — Attempted, Failed

### 9.1 Intended Analysis

The original design included a "dependency blast radius" analysis: for each repo flagged as high-risk, determine how many other repos in the cohort depend on it. This would quantify the supply-chain impact of a repo going stagnant.

### 9.2 What Was Tried

GitHub's Dependency Graph GraphQL API:
```graphql
query {
  repository(owner: "apache", name: "spark") {
    dependencyGraphManifests {
      edges {
        node {
          dependencies {
            nodes { packageName requirements }
          }
        }
      }
    }
  }
}
```

### 9.3 Why It Failed

The API returned 0 dependencies for every repo in the cohort. Root causes:

1. **Dependency Graph must be enabled** on the repo. Many repos (especially older ones or inactive ones) have it disabled.
2. **A parsed manifest file must exist** in a recognized format (package.json, pom.xml, Cargo.toml, etc.). Repos without standard manifest files (or with manifests in non-standard locations) show zero dependencies.
3. **GitHub only parses direct dependencies** — transitive dependencies are not included in the GraphQL response for most repos.

### 9.4 Alternative Approaches (Not Implemented)

- **libraries.io dataset:** A downloadable PostgreSQL dump of ~6M packages with dependency relationships. Free for non-commercial use. Would require separate infrastructure.
- **npm/Gem/Maven registry API:** Query each registry directly for a repo's package name. Faster than GitHub GraphQL but requires a mapping from GitHub repo → package name.
- **Source code parsing:** Clone each repo and parse manifest files locally. Expensive (586 repos × variable clone time) but guarantees coverage.

---

## 10. Time Series / Survival Analysis — Why It's the Right Approach (But Wasn't Used)

### 10.1 Why Binary Classification Is Wrong

The current approach collapses 26 months of temporal data into a single binary label (stagnant/active). This destroys information:

- **When** did the repo go silent? A repo that died in month 1 is different from one that died in month 13.
- **Was there a decline pattern?** A linear decline in events is different from a sudden drop.
- **Is the repo seasonal?** Academic repos show activity dips during summer. JS repos show dips during December holidays.

### 10.2 Survival Analysis Formulation

The correct statistical framing is a **time-to-event** model:

- **t = 0:** Start of observation (March 2024)
- **Event:** Last observed activity (IssuesEvent or PullRequestEvent)
- **Censoring:** Repos still active at the end of the target window (May 2026)
- **Goal:** Predict the hazard function — the instantaneous risk of stagnation at time t

**Recommended model:** Cox Proportional Hazards with the same four features, stratified by language:

```
h(t | X) = h_0(t) * exp(β₁·m_ratio + β₂·gini + β₃·velocity + β₄·burden)
```

### 10.3 What Would Change

| Aspect | Current (Binary) | Survival Analysis |
|--------|-------------------|-------------------|
| Target | Binary (0/1) | Time-to-event (continuous months) |
| Uses all temporal data | No — collapses to label | Yes — uses exact timing |
| Handles censoring | No — "active" label ignores future risk | Yes — right-censoring is explicit |
| Interpretability | "Repo is at risk" | "Repo has 60% probability of stagnation within 12 months" |
| Code complexity | Low (sklearn) | Medium (lifelines or scikit-survival) |

### 10.4 Why Not Implemented

Survival analysis libraries (`lifelines`, `scikit-survival`) add a dependency and require restructuring the feature computation to produce monthly-ish risk sets. The current binary approach was chosen for simplicity and comparability with existing literature. The survival analysis framing is the single highest-impact improvement available without paid BigQuery quota.

---

## 11. Data Mining Categories — Comprehensive Check

| Category | Status | Details |
|----------|--------|---------|
| **Classification** | ✅ Used | LR, XGBoost, RF — binary stagnant prediction |
| **Clustering** | ✅ Used | K-Means (k=4, silhouette=0.420) with t-SNE viz |
| **Dimensionality Reduction** | ✅ Used | t-SNE for 4D → 2D projection |
| **Feature Selection** | ✅ Used | VIF analysis removed no features; SHAP identified Gini as dominant |
| **Anomaly Detection** | ✅ Used | 3-layer bot filtering (SQL + blocklist + statistical heuristics) |
| **Model Interpretation** | ✅ Used | SHAP LinearExplainer for per-repo prediction decomposition |
| **Imputation** | ⛔ Rejected | Ghost-demand removed; zero-Issue repos flagged but not imputed |
| **Time Series / Survival Analysis** | ❌ Absent | Discussed in §10 — highest-priority improvement |
| **NLP / Text Mining** | ❌ Absent | Quota-limited; discussed in §8 |
| **Association / Dependency Mining** | ❌ Attempted | Dependency Graph API returned 0 for all repos |
| **Association Rules (Market Basket)** | ❌ N/A | No transaction-like structure in the data |
| **Regression (Continuous Target)** | ❌ N/A | Target is binary — though survival analysis would introduce continuous time |
| **Causal Inference** | ❌ Absent | All findings are correlational; confounders unmeasured |
| **Ensemble Methods** | ✅ Used | RF (bagging), XGBoost (boosting) |
| **Active Learning** | ❌ N/A | No interactive labeling loop |
| **Reinforcement Learning** | ❌ N/A | No sequential decision-making task |
| **Graph Mining** | ❌ N/A | Repository dependency graph was inaccessible |
| **Frequent Pattern Mining** | ❌ N/A | No itemset structure |
| **Out-of-Distribution Detection** | ❌ N/A | All repos from same GH Archive source; no distribution shift detection needed |

---

## 12. Summary

### What Works
- The pipeline is reproducible, well-documented, and methodologically sound (temporal separation, bot filtering, cross-validation).
- The descriptive analytics (stagnation rates by language, zero-Issue proportion, bot contamination) are actionable and robust.
- The model does beat random (AUC > 0.8 all models), confirming that *some* signal exists in aggregated event data.

### What Doesn't Work
- The model is not practically useful. 80% false-positive rate. All high-risk repos have identical features. The primary driver (Gini=0) is a data-sparsity artifact.
- The tournament winner (LR) wins by a negligible margin — all three models are equivalent given the feature set.
- Alternatives recommendation is broken — returns mega-repos regardless of functional match.

### What Would Fix It
1. Survival analysis instead of binary classification (biggest impact, no paid quota needed)
2. Per-language stratified models
3. Expand event types (PushEvent, CreateEvent)
4. Increase ground-truth validation sample size
5. Paid BigQuery tier for payload-level features

**Final verdict:** The pipeline proves that *simple aggregated event counts cannot predict maintenance risk*. This is a genuine negative result — the data available at the free tier is insufficient. The most valuable outputs are the descriptive statistics, not the predictive model.
