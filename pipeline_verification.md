# Dataset Properties: OSS Maintenance Risk Prediction

**Generated:** 2026-05-24
**Pipeline:** `local_run.py` (v2 expanded harvest)
**Format:** Apache Parquet (2 files) + CSV mirrors

---

## 1. General Information

| Property | Value |
|----------|-------|
| **Title** | OSS Maintenance Risk — Multi-Language GitHub Archive Event Dataset |
| **Purpose** | Predict OSS repository stagnation using engineered features from longitudinal event data |
| **Collection period** | March 2024 – May 2026 (26 months) |
| **Source** | `githubarchive.month.*` via Google BigQuery (`github-mining-bot-filtered` project) |
| **Target languages** | Java, Go, Rust, Python, JavaScript, TypeScript |
| **Event types** | IssuesEvent, PullRequestEvent, PushEvent, CreateEvent, WatchEvent |
| **Inclusion keywords** | 140 ecosystem keywords spanning frameworks, databases, build tools, testing frameworks, ML libraries, and infrastructure |
| **Bot filtering** | 3 layers: SQL `[bot]` suffix filter (Layer 1), 45-account blocklist (Layer 2), offline statistical heuristics (Layer 3) |
| **Exclusion keywords** | tutorial, course, demo, example, sample, playground, template, awesome |
| **Minimum stars** | 32 |
| **License** | GitHub Archive is freely available. This derived dataset is released under CC0. |

---

## 2. File Inventory

| File | Format | Size | Rows | Description |
|------|--------|------|------|-------------|
| `data/multilang_event_indices.parquet` | Parquet | 180 KB | 47,561 | Raw aggregated events (grouped by repo × type × year × month) |
| `data/multilang_health_indices.parquet` | Parquet | 77 KB | 1,728 | Engineered feature store (one row per repo) |
| `output/raw_events.csv` | CSV | 447 KB | 47,561 | CSV mirror of event indices |
| `output/feature_store.csv` | CSV | 75 KB | 1,728 | CSV mirror of feature store |

---

## 3. Data Provenance

### 3.1 Collection Pipeline

```
GitHub Archive (BigQuery)  →  SQL filter (140 keywords + bot blocklist)
                              ↓
                         Raw events (47,561 rows)
                              ↓
                         Language assignment (bigquery-public-data.github_repos.languages)
                              ↓
                         Temporal split (feature: 2024-03–2025-04, target: 2025-04–2026-05)
                              ↓
                          Feature engineering (pivot + compute 7 derived features)
                              ↓
                         Target computation (stagnant flag from target window)
                              ↓
                         Layer 3 bot heuristics (velocity anomaly, single-type bias)
```

### 3.2 SQL Filter Details

The BigQuery query uses a CTE (`visibility_filter`) to first identify repos in target languages with ≥32 stars, then inner-joins to `githubarchive.month.*` events. The WHERE clause applies:

1. 140 inclusion keyword patterns on `events.repo.name` (e.g., `%react%`, `%pytorch%`)
2. 8 exclusion keyword patterns
3. 5 event type filters
4. Layer 1: `events.actor.login NOT LIKE '%[bot]%'`
5. Layer 2: `events.actor.login NOT IN (45 known bot accounts)`

Language assignment uses `bigquery-public-data.github_repos.languages` with `ROW_NUMBER() PARTITION BY repo_name ORDER BY bytes DESC` to pick the primary language.

### 3.3 Feature Engineering

7 engineered features from the feature window (13 months):

| Feature | Formula | Description |
|---------|---------|-------------|
| M-Ratio | `PullRequestEvent / IssuesEvent` | Supply/demand balance of maintenance work |
| Velocity | `(PRs + Issues) / active_months` | Mean monthly event rate |
| Gini Index | `(2 * Σ(i * sorted_events)) / (n * Σ(events)) - (n+1)/n` | Concentration of activity over time |
| Maintenance Burden | `Issues / (PRs + 1)` | Demand outstripping supply capacity |
| Activity Volatility | `std(event_count) / mean(event_count)` | Coefficient of variation of activity |
| Active Months | `count(distinct year*12 + month)` | Number of distinct months with ≥1 event |
| Sparse Data Flag | `True if n_months < 2 or total_events == 0` | Identifies repos where Gini=0 is forced |

**Gini implementation detail:**
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

**Derivation of secondary features:**
- `total_load = PullRequestEvent + IssuesEvent` (operational load, excludes passive events)
- `avg_monthly_contributors = mean(unique_users)` per month (mean distinct active users)
- `months_since_active = ref_month - max(year*12 + month)` (recency; negative means last active before feature window end)
- `bot_contamination_score = 0.5*velocity_flag + 0.3*bias_flag + 0.2*contrib_flag` (weighted composite of 3 heuristics)

### 3.4 Target Definition

A repo is `stagnant = 1` if it has zero rows in the target window (13 months without any observed events). Repos with feature data but no target data are filled as stagnant via `fill_null(1)`.

---

## 4. Raw Events Schema

**File:** `multilang_event_indices.parquet` (47,561 rows, 7 columns)

| Column | Type | Description |
|--------|------|-------------|
| `name` | String | Repository name in `owner/repo` format (e.g., `facebook/react`) |
| `type` | String | Event type: `IssuesEvent`, `PullRequestEvent`, `PushEvent`, `CreateEvent`, `WatchEvent` |
| `year` | Int64 | Year of aggregation bucket (2024–2026) |
| `month` | Int64 | Month of aggregation bucket (1–12) |
| `event_count` | Int64 | Total event count for this (repo, type, year, month) group |
| `unique_users` | Int64 | COUNT DISTINCT actor.login for this group |
| `language` | String | Primary language of the repo (from `bigquery-public-data.github_repos.languages`) |

### 4.1 Event Type Distribution

| Event Type | Total Events | Repos Affected | % of Total |
|-----------|-------------|----------------|------------|
| WatchEvent | 481,862 | 1,763 | 32.8% |
| PushEvent | 379,340 | 586 | 25.9% |
| PullRequestEvent | 354,584 | 819 | 24.2% |
| IssuesEvent | 191,042 | 776 | 13.0% |
| CreateEvent | 60,466 | 484 | 4.1% |

**Note:** WatchEvent dominates by volume but is the least informative for maintenance analysis (watching a repo is passive). PushEvent was added in v2 to capture commit activity. IssuesEvent and PullRequestEvent remain the primary signal carriers.

### 4.2 Language Distribution

| Language | Repos | Total Events | Avg Events/Repo |
|----------|-------|-------------|----------------|
| JavaScript | 984 | 213,915 | 14.1 |
| Python | 327 | 274,592 | 22.7 |
| Java | 298 | 424,882 | 35.2 |
| Go | 100 | 254,033 | 57.6 |
| TypeScript | 81 | 278,236 | 92.3 |
| Rust | 11 | 21,636 | 27.8 |

**Key observation:** TypeScript has the highest per-repo event rate (92.3) despite the smallest sample among well-represented languages. Java has the most total events (424K) due to high-activity Apache/Spring projects. Rust remains under-sampled (11 repos) — a known limitation.

---

## 5. Feature Store Schema

**File:** `multilang_health_indices.parquet` (1,728 rows, 22 columns)

### 5.1 Column Definitions

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `name` | String | Repository name | — |
| `IssuesEvent` | Float64 | Total IssuesEvent count (feature window) | 0 – 11,693 |
| `PullRequestEvent` | Float64 | Total PullRequestEvent count (feature window) | 0 – 21,695 |
| `WatchEvent` | Float64 | Total WatchEvent count (feature window) | 0 – 19,644 |
| `PushEvent` | Float64 | Total PushEvent count (feature window) | 0 – 24,609 |
| `CreateEvent` | Float64 | Total CreateEvent count (feature window) | 0 – 1,776 |
| `m_ratio` | Float64 | PullRequestEvent / IssuesEvent (PR/Issue ratio) | 0 – 475.5 |
| `total_load` | Float64 | PullRequestEvent + IssuesEvent (total operational load) | 0 – 33,388 |
| `active_months` | Float64 | Number of distinct months with events | 1 – 13 |
| `velocity` | Float64 | total_load / active_months (mean monthly event rate) | 0 – 1,011.2 |
| `activity_volatility` | Float64 | std(event_count) / mean(event_count) | 0 – 13.0 |
| `avg_monthly_contributors` | Float64 | Mean unique_users per month | 0 – 101.0 |
| `gini_index` | Float64 | Gini coefficient of event_count distribution | 0 – 1.0 |
| `sparse_data` | Boolean | True if n_months < 2 or total_events == 0 | True: 253 repos |
| `maintenance_burden` | Float64 | IssuesEvent / (PullRequestEvent + 1) | 0 – 11,693 |
| `language` | String | Primary programming language | 6 values |
| `stagnant` | Int32 | Binary target: 1 if no target-window activity | 0: 1,329 / 1: 399 |
| `months_since_active` | Int64 | Months since last event (relative to feature window end) | -13 – 24,304 |
| `bot_contamination_score` | Float64 | Weighted composite of 3 bot heuristics (0–1) | 0 – 0.8 |
| `suspected_bot_heavy` | Boolean | Flagged as bot-heavy by Layer 3 heuristics | True: 17 repos |
| `suspicious_prs` | Int64 | Count of PRs from suspicious months | 0 – 399 |
| `suspicious_months` | UInt32 | Count of suspicious months | 0 – 8 |

### 5.2 Target Summary

| Class | Count | Percentage |
|-------|-------|------------|
| Active (0) | 1,329 | 76.9% |
| Stagnant (1) | 399 | 23.1% |

### 5.3 Zero / Value Distributions

Counts of zero-valued entries per column (feature store, n=1,728):

| Column | Zero Count | % of Total | Notes |
|--------|-----------|------------|-------|
| `IssuesEvent` | 1,007 | 58.3% | Repos without GitHub Issues tracker |
| `PullRequestEvent` | 984 | 56.9% | Repos with zero PR activity |
| `PushEvent` | 1,197 | 69.3% | Most repos have no PushEvents (WatchEvent-heavy) |
| `CreateEvent` | 1,303 | 75.4% | Least common event type |
| `WatchEvent` | 39 | 2.3% | Nearly every repo has watches |
| `m_ratio` | 1,170 | 67.7% | M-Ratio = 0 when Issues = 0 |
| `total_load` | 821 | 47.5% | Zero operational load (Issues=0 & PRs=0) |
| `velocity` | 821 | 47.5% | Perfectly correlated with total_load=0 |
| `activity_volatility` | 543 | 31.4% | Repos with only 1 month of data |
| `gini_index` | 543 | 31.4% | Exact 1:1 overlap with volatility=0 cases |
| `maintenance_burden` | 1,007 | 58.3% | Identical to IssuesEvent=0 |
| `sparse_data` (True) | 253 | 14.6% | Forced Gini=0 repos |
| `bot_contamination_score` | 1,699 | 98.3% | Negligible bot activity detected |

### 5.4 Zero-Event Combinations

| Pattern | Count | % | Interpretation |
|---------|-------|---|---------------|
| Issues=0 & PRs=0 | 821 | 47.5% | No operational events — WatchEvent/PushEvent only |
| Issues=0 & PRs>0 | 186 | 10.8% | PRs only (external issue tracker) |
| Issues>0 & PRs=0 | 163 | 9.4% | Issues only (abandoned or early-stage) |
| Issues>0 & PRs>0 | 558 | 32.3% | Both signals present — modelable repos |

**By language** — repos with Issues=0 AND PRs=0:
| Language | Zero-Both | Total | % |
|----------|-----------|-------|---|
| JavaScript | 525 | 924 | 56.8% |
| Java | 118 | 291 | 40.5% |
| Python | 110 | 323 | 34.1% |
| Go | 39 | 99 | 39.4% |
| TypeScript | 25 | 80 | 31.2% |
| Rust | 4 | 11 | 36.4% |

### 5.5 Event Type Exclusivity

Repos captured ONLY through expanded event types (no IssuesEvent or PullRequestEvent):

| Exclusive Signal | Count | % |
|----------------|-------|---|
| WatchEvent-only | 821 | 47.5% |
| PushEvent-only | 19 | 1.1% |
| CreateEvent-only | 6 | 0.3% |

These repos would be invisible to the v1 pipeline (which only used Issues + PRs). 47.5% of the cohort would be lost without the expanded event types.

### 5.6 Derived Feature Edge Cases

**M-Ratio:**
| Condition | Count | % |
|-----------|-------|---|
| m_ratio = 0 (Issues=0 → undefined) | 1,170 | 67.7% |
| m_ratio < 0.1 (extreme demand pressure) | 1,185 | 68.6% |
| m_ratio in [0.9, 1.1] (equilibrium) | 82 | 4.7% |
| m_ratio > 10 (extreme supply bias) | 19 | 1.1% |
| m_ratio > 100 (outlier) | 1 | 0.06% |

**Gini Index:**
| Condition | Count | % |
|-----------|-------|---|
| gini = 0.0 (exact zero) | 543 | 31.4% |
| gini = 0.0 & sparse_data=True | 253 | 14.6% |
| gini = 0.0 & sparse_data=False | 290 | 16.8% |
| gini > 0.8 (extreme concentration) | 11 | 0.6% |

**Velocity:**
| Condition | Count | % |
|-----------|-------|---|
| velocity = 0 | 821 | 47.5% |
| velocity > 100 | 32 | 1.9% |
| velocity > 500 | 8 | 0.5% |

**Active Months:**
| Condition | Count | % |
|-----------|-------|---|
| active_months = 1 (single burst) | 253 | 14.6% |
| active_months = 13 (full window) | 59 | 3.4% |
| mean active_months | 5.3 | — |
| median active_months | 4 | — |

**Maintenance Burden:**
| Condition | Count | % |
|-----------|-------|---|
| burden = 0 (no Issues) | 1,007 | 58.3% |
| burden > 10 (extreme pressure) | 83 | 4.8% |
| burden > 100 (outlier) | 14 | 0.8% |

**Bot Contamination:**
| Condition | Count | % |
|-----------|-------|---|
| score = 0 | 1,699 | 98.3% |
| score ≥ 0.1 | 29 | 1.7% |
| score ≥ 0.5 | 17 | 1.0% |
| score ≥ 0.7 | 1 | 0.06% |
| suspected_bot_heavy = True | 17 | 1.0% |

### 5.7 Derived Feature Correlations

Pearson correlation matrix among engineered features:

| | m_ratio | gini | velocity | burden | active_months | volatility | total_load |
|---|---------|------|----------|--------|---------------|------------|------------|
| m_ratio | 1.000 | 0.239 | 0.120 | -0.029 | 0.275 | 0.219 | 0.111 |
| gini_index | 0.239 | 1.000 | 0.201 | 0.257 | 0.786 | **0.939** | 0.189 |
| velocity | 0.120 | 0.201 | 1.000 | 0.065 | 0.403 | 0.212 | **0.997** |
| maintenance_burden | -0.029 | 0.257 | 0.065 | 1.000 | 0.172 | 0.293 | 0.057 |
| active_months | 0.275 | **0.786** | 0.403 | 0.172 | 1.000 | **0.693** | 0.392 |
| activity_volatility | 0.219 | **0.939** | 0.212 | 0.293 | 0.693 | 1.000 | 0.201 |
| total_load | 0.111 | 0.189 | **0.997** | 0.057 | 0.392 | 0.201 | 1.000 |

**Key collinearities (|r| ≥ 0.7):**
- `gini_index` ↔ `activity_volatility`: r=0.939 (near-identical — both measure concentration)
- `total_load` ↔ `velocity`: r=0.997 (velocity = total_load / active_months; nearly identical when active_months ≈ 1)
- `gini_index` ↔ `active_months`: r=0.786 (more months → more opportunity for inequality)
- `active_months` ↔ `activity_volatility`: r=0.693 (longer windows enable higher volatility)

### 5.8 Stagnant vs Active Comparison

| Feature | Stagnant Mean | Active Mean | Ratio |
|---------|--------------|-------------|-------|
| m_ratio | 0.081 | 1.114 | 13.8× higher in active |
| gini_index | 0.048 | 0.284 | 5.9× higher in active |
| velocity | 0.202 | 4.469 | 22.1× higher in active |
| maintenance_burden | 0.182 | 0.926 | 5.1× higher in active |
| active_months | 2.739 | 19.933 | 7.3× higher in active |
| sparse_data (pct) | 43.4% | 6.0% | 7.2× higher in stagnant |

### 5.9 Feature Distribution Summary

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| m_ratio | 0.88 | 7.55 | 0.00 | 0.00 | 0.00 | 0.70 | 475.50 |
| gini_index | 0.23 | 0.20 | 0.00 | 0.00 | 0.23 | 0.38 | 0.87 |
| velocity | 3.48 | 27.47 | 0.00 | 0.00 | 0.09 | 10.90 | 1,011.20 |
| maintenance_burden | 46.72 | 373.27 | 0.00 | 0.00 | 0.57 | 5.23 | 11,693.00 |
| active_months | 15.96 | 18.51 | 1.00 | 3.00 | 8.00 | 20.00 | 65.00 |
| total_load | 213.50 | 1,148.10 | 0.00 | 0.00 | 11.00 | 61.00 | 33,388.00 |
| activity_volatility | 0.47 | 0.53 | 0.00 | 0.00 | 0.33 | 0.73 | 13.06 |
| avg_monthly_contributors | 3.17 | 7.24 | 0.00 | 0.50 | 1.20 | 3.20 | 101.00 |

### 5.10 Language Breakdown with Stagnation

| Language | Repos | Stagnant | Stagnation Rate | Avg M-Ratio | Avg Velocity | Sparse % |
|----------|-------|----------|----------------|-------------|-------------|----------|
| JavaScript | 924 | 298 | 32.3% | 0.53 | 0.83 | 20.3% |
| TypeScript | 80 | 15 | 18.8% | 1.65 | 11.24 | 8.8% |
| Java | 291 | 38 | 13.1% | 0.99 | 7.01 | 11.7% |
| Python | 323 | 37 | 11.5% | 1.51 | 4.09 | 4.3% |
| Go | 99 | 11 | 11.1% | 1.09 | 9.20 | 8.1% |
| Rust | 11 | 0 | 0.0% | 0.82 | 7.71 | 18.2% |

---

## 6. Data Quality

### 6.1 Completeness

- **No null values** in either dataset — all fields are populated.
- Missing languages are filled as `"Unknown"` (0 rows affected in v2 — all repos received a language label).
- Repos with feature data but no target data receive `stagnant = 1` (326 repos, 18.9%).

### 6.2 Known Artifacts

| Artifact | Impact | Affected | Severity |
|----------|--------|----------|----------|
| `sparse_data = True` (Gini=0 forced) | 253 repos (14.6%) have artificially zero Gini | Sparse repos systematically flagged as high-risk | **High** |
| `m_ratio = 0` when `IssuesEvent = 0` | 1,170 repos (67.7%) — M-Ratio is zero, not measured | Repos without GitHub Issues get misleading M-Ratio=0 | **High** |
| `velocity = 0` when `total_load = 0` | 821 repos (47.5%) — cannot compute meaningful velocity | Half the cohort has zero velocity | **High** |
| Bot filtering removes events, not repos | All repos retained but bot accounts excluded per-event | M-Ratio realistic but may still undercount human effort | Medium |
| PushEvent/CreateEvent are weak maintenance signals | 26% of events from PushEvent; 4% from CreateEvent | Inflates event counts without indicating maintenance quality | Medium |
| Rust under-sampling | Only 11 repos — results for Rust are unreliable | Inclusion keywords still inadequate for Rust | Medium |
| gini_index ↔ activity_volatility collinearity | r=0.939 — they measure nearly the same thing | VIF forces gini_index out of models | Low |
| active_months range anomaly | max = 65 (should be ≤ 13) | active_months includes target window months in some cases | **Investigating** |

### 6.3 Temporal Consistency

- Feature and target windows are non-overlapping with a strict calendar boundary at April 2025.
- No data leakage between windows — features are derived exclusively from months 1-13.
- 326 repos (18.9%) have feature data but no target data — filled as stagnant via `fill_null(1)`.
- These 326 repos represent the "true lost" cohort: repos that were active during the feature window but completely silent during the target window.

---

## 7. Limitations

1. **Event-level aggregation**: `unique_users` is COUNT DISTINCT actor.login per month — not contributor identity. Individual contributor churn cannot be tracked.
2. **Free-tier BigQuery constraints**: Richer features (commit-level data, issue response time, release cadence) require paid quota.
3. **Keyword coverage bias**: The 140 inclusion keywords favor well-known ecosystems. Niche or emerging frameworks are under-represented.
4. **Binary target**: "Stagnant vs. active" is coarse. A gradual decline is not captured.
5. **WatchEvent and CreateEvent are weak signals**: They inflate event counts without indicating substantive maintenance activity.
6. **GitHub-centric**: Repos using GitLab, Bitbucket, or self-hosted forges are excluded.

---

## 8. Version History

| Version | Date | Changes | Repos | Event Types |
|---------|------|---------|-------|-------------|
| v1 | 2026-05-21 | Initial harvest, 34 keywords, 2 event types (Issues, PRs) | 542 | 2 |
| v2 | 2026-05-24 | Expanded to 140 keywords, 5 event types, bot filtering in SQL, Rust fix (1→11 repos) | 1,728 | 5 |
