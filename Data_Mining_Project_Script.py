WARNING: GH_PAT not set. Unauthenticated rate limits (60 req/hr).
Feature window (features only):  202403 to 202504
Target window  (target only):    202504 to 202605
Storage dir:  /home/mint/Documents/GitHub/Data Mining Notebook/data
Output dir:   /home/mint/Documents/GitHub/Data Mining Notebook/output
Log file:     /home/mint/Documents/GitHub/Data Mining Notebook/output/pipeline.log


╔══════════════════════════════════════════════════════════════════════╗
║  DISCLAIMER: Ghost-demand imputation REMOVED                        ║
║  The previous 0.73 median Issue-to-PR conversion rate was found     ║
║  not to generalize. Manual checks across individual repos showed    ║
║  consistently lower ratios that varied widely by language and       ║
║  project maturity. Using a single scalar obscures real heterogeneity║
║  and produces misleading effective-demand estimates. All analyses   ║
║  now use raw IssuesEvent counts. Zero-Issue repos are reported      ║
║  separately with a flag; no synthetic values are substituted.       ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║  DISCLAIMER: M-Ratio includes bot activity                          ║
║  PullRequestEvent counts include automated PRs from Dependabot,     ║
║  Renovate, and similar bots. This inflates the supply side of the   ║
║  ratio, making repos appear healthier than they are. A repo with    ║
║  1000 bot PRs and 10 human PRs gets M ≈ 1.0 (looks balanced) but   ║
║  is actually fragile — all substantive work falls on a tiny core.   ║
║  Results should be interpreted with this bias in mind.              ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║  DISCLAIMER: Dependency blast radius data unavailable               ║
║  GitHub's Dependency Graph API (GraphQL dependencyGraphManifests)   ║
║  returned 0 dependencies for every repo in this cohort. The feature ║
║  requires repos to have the Dependency Graph enabled and a manifest ║
║  file parsed by GitHub. We lack reliable API access to this signal. ║
║  The blast-radius / mitigation-matrix section has been removed.     ║
╚══════════════════════════════════════════════════════════════════════╝

Loaded cached feature store from /home/mint/Documents/GitHub/Data Mining Notebook/data/multilang_health_indices.parquet (1728 repos)
Skipping BigQuery, feature engineering, and bot heuristics — using cached data.

--- Per-language stagnation rates ---
shape: (6, 6)
┌────────────┬───────┬─────────────────┬─────────────┬──────────────┬──────────────────┐
│ language   ┆ repos ┆ stagnation_rate ┆ avg_m_ratio ┆ avg_velocity ┆ zero_issue_repos │
│ ---        ┆ ---   ┆ ---             ┆ ---         ┆ ---          ┆ ---              │
│ str        ┆ u32   ┆ f64             ┆ f64         ┆ f64          ┆ u32              │
╞════════════╪═══════╪═════════════════╪═════════════╪══════════════╪══════════════════╡
│ JavaScript ┆ 924   ┆ 0.322511        ┆ 0.525848    ┆ 0.825682     ┆ 626              │
│ TypeScript ┆ 80    ┆ 0.1875          ┆ 1.645982    ┆ 11.235183    ┆ 31               │
│ Java       ┆ 291   ┆ 0.130584        ┆ 0.994206    ┆ 7.012764     ┆ 159              │
│ Python     ┆ 323   ┆ 0.114551        ┆ 1.513286    ┆ 4.092822     ┆ 139              │
│ Go         ┆ 99    ┆ 0.111111        ┆ 1.0912      ┆ 9.196057     ┆ 48               │
│ Rust       ┆ 11    ┆ 0.0             ┆ 0.82257     ┆ 7.714877     ┆ 4                │
└────────────┴───────┴─────────────────┴─────────────┴──────────────┴──────────────────┘

============================================================
CLUSTERING ANALYSIS
============================================================

  Optimal k=3 (silhouette=0.404)

── Cluster size & central tendency ──
         m_ratio  gini_index  velocity  maintenance_burden  total_load  count
cluster                                                                      
0           1.00        0.43      1.18                0.78        31.0    389
1           0.02        0.46      2.89               33.50        82.5      8
2           1.46        0.49     16.66                0.45       986.0    131

── Language makeup per cluster (proportion) ──
language    Go  Java  JavaScript  Python  Rust  TypeScript
cluster                                                   
0          5.4  19.5        35.5    31.9   0.8         6.9
1         12.5  12.5        50.0    12.5   0.0        12.5
2         15.3  37.4        13.0    23.7   3.1         7.6

── Category makeup per cluster (proportion) ──
category  API / Integration  Build / Tooling  CLI / Terminal  Data Science / ML  Database / ORM  Framework  Frontend  Infrastructure / Cloud  Mobile  Other  Security
cluster                                                                                                                                                              
0                       4.9              4.9             0.3                2.8             7.5       26.5      15.9                    11.3     1.0   24.7       0.3
1                      12.5              0.0             0.0               12.5             0.0        0.0      50.0                     0.0    12.5   12.5       0.0
2                       8.4              4.6             0.8                6.9            16.8       13.0       6.1                    26.0     0.8   16.8       0.0
  Cluster 0: Near Equilibrium, Low Activity (JavaScript/Framework)
  Cluster 1: Demand >> Supply, Low Activity (JavaScript/Frontend)
  Cluster 2: Near Equilibrium, Moderate Activity (Java/Infrastructure / Cloud)

  11_clusters_tsne — t-SNE scatter: Repos colored by behavioral cluster.
    t-SNE preserves local structure (similar repos appear near each other).
    Each color = a distinct behavioral profile (velocity × M-ratio × Gini × burden).

  12_language_tsne — t-SNE scatter: Same projection, colored by programming language.
    Reveals whether languages naturally separate in behavioral space.

  13_category_tsne — t-SNE scatter: Same projection, colored by functional category.
    Categories derived from repo name keywords (Framework, Frontend, Database, etc.).

  14_cluster_parallel_coords — Parallel coordinates: Feature profiles per cluster.
    Each line = one repo. Color = cluster. Shows which feature ranges define each cluster.

  15_clusters_per_language — Faceted t-SNE: One panel per language, colored by cluster.
    Reveals whether cluster definitions hold across languages or are language-specific.

  16_category_language_heatmap — Heatmap: Count of repos per (category, language) pair.
    Darker = more repos. Shows which categories concentrate in which languages.

  Clustering complete: 3 clusters, 528 repos

--- VIF (all candidates) ---
              feature       VIF
4       active_months  5.539826
1          gini_index  5.079421
2            velocity  1.287371
3  maintenance_burden  1.157035
0             m_ratio  1.129828
5         sparse_data  1.001381

--- Tournament (1728 repos, features=['m_ratio', 'velocity', 'maintenance_burden', 'sparse_data']) ---
  Baseline: 23.1% repos stagnant in target window

Dummy (majority): AUC-ROC=0.500  F1=0.000  Recall=0.000
Logistic_Regression    AUC-ROC=0.795  F1=0.517  Recall=0.665
XGBoost                AUC-ROC=0.814  F1=0.528  Recall=0.436
Random_Forest          AUC-ROC=0.812  F1=0.533  Recall=0.886

WINNER: XGBoost

--- Ablation: substituting activity_volatility for gini_index ---
  Ablation AUC-ROC (volatility instead of Gini): 0.855  (baseline Gini model: 0.814)
  Change: +0.041

============================================================
SURVIVAL ANALYSIS
============================================================

  Concordance index (c-index): 0.853

  Checking proportional hazards assumption...
  Some covariates may violate PH — consider strata or time-varying.

============================================================
PER-LANGUAGE MODELS
============================================================
  JavaScript   n=924 (S=298/A=626)  AUC-ROC=0.758  F1=0.567
  Python       n=323 (S=37/A=286)  AUC-ROC=0.807  F1=0.334
  Go           n= 99 (S=11/A=88)  AUC-ROC=0.772  F1=0.339
  Java         n=291 (S=38/A=253)  AUC-ROC=0.840  F1=0.490
  Rust         n= 11 — too small, skipped
  TypeScript   n= 80 (S=15/A=65)  AUC-ROC=0.885  F1=0.403

=== RISK REPORT (SHAP-diagnosed) ===

heilhead/react-bootstrap-validation                [JavaScript  ] Risk=83.6%  Driver=SPARSE_DATA           Compound decay.
kuhnza/angular-google-places-autocomplete          [JavaScript  ] Risk=78.4%  Driver=SPARSE_DATA           Compound decay.
eu81273/angular.treeview                           [JavaScript  ] Risk=78.4%  Driver=SPARSE_DATA           Compound decay.
sparkalow/angular-truncate                         [JavaScript  ] Risk=68.2%  Driver=SPARSE_DATA           Compound decay.
mikenikles/html-to-react                           [JavaScript  ] Risk=68.2%  Driver=SPARSE_DATA           Compound decay.
troch/angular-multi-step-form                      [JavaScript  ] Risk=68.2%  Driver=SPARSE_DATA           Compound decay.
stoeffel/redux-elm-middleware                      [JavaScript  ] Risk=68.2%  Driver=SPARSE_DATA           Compound decay.
angularjs-nvd3-directives/angularjs-nvd3-directives [JavaScript  ] Risk=68.2%  Driver=SPARSE_DATA           Compound decay.
remobile/react-native-toast                        [Java        ] Risk=68.2%  Driver=SPARSE_DATA           Compound decay.
wangzuo/input-moment                               [JavaScript  ] Risk=68.2%  Driver=SPARSE_DATA           Compound decay.
reactjs/react-timer-mixin                          [JavaScript  ] Risk=68.2%  Driver=SPARSE_DATA           Compound decay.
kamilkp/angular-sortable-view                      [JavaScript  ] Risk=68.2%  Driver=SPARSE_DATA           Compound decay.
ContentMine/getpapers                              [JavaScript  ] Risk=67.8%  Driver=SPARSE_DATA           Compound decay.
sethvincent/store-emitter                          [JavaScript  ] Risk=67.8%  Driver=SPARSE_DATA           Compound decay.
prawn-cake/vk-requests                             [Python      ] Risk=67.8%  Driver=SPARSE_DATA           Compound decay.

--- Ground-truth: fetching GitHub archival status ---

--- Alternatives recommendation ---
  heilhead/react-bootstrap-validation                [JavaScript  ] → facebook/react-native
  kuhnza/angular-google-places-autocomplete          [JavaScript  ] → facebook/react-native
  eu81273/angular.treeview                           [JavaScript  ] → facebook/react-native
  sparkalow/angular-truncate                         [JavaScript  ] → facebook/react-native
  mikenikles/html-to-react                           [JavaScript  ] → facebook/react-native
  troch/angular-multi-step-form                      [JavaScript  ] → facebook/react-native
  stoeffel/redux-elm-middleware                      [JavaScript  ] → alextselegidis/easyappointments
  angularjs-nvd3-directives/angularjs-nvd3-directives [JavaScript  ] → facebook/react-native
  remobile/react-native-toast                        [Java        ] → reactor/reactor-core
  wangzuo/input-moment                               [JavaScript  ] → alextselegidis/easyappointments
  reactjs/react-timer-mixin                          [JavaScript  ] → facebook/react-native
  kamilkp/angular-sortable-view                      [JavaScript  ] → facebook/react-native
  ContentMine/getpapers                              [JavaScript  ] → alextselegidis/easyappointments
  sethvincent/store-emitter                          [JavaScript  ] → alextselegidis/easyappointments
  prawn-cake/vk-requests                             [Python      ] → openthread/openthread

── Generating graphs ──

  01_pressure_by_language — Scatter: Issues vs PRs per repo, colored by language.
    Dashed line = equilibrium (Issues = PRs). Points above line have more PRs than Issues (healthy supply).
    Points below line have more Issues than PRs (demand outpaces supply). Log-log scale.

  02_labor_concentration — Bar: Top 40 repos by velocity, colored by language.
    Red dashed line = cohort median. Bars below median have fragile contributor bases.

  03_mratio_by_language — Histogram: M-Ratio distribution per language.
    White dashed line = M=1.0 (equilibrium). Left of line = demand > supply (high pressure).
    Right of line = supply > demand (healthy or bot-inflated).

  04_fragility_by_language — Scatter: Gini index vs velocity, colored by language.
    Right of orange line = high concentration (fragile bus factor). Below gray line = low activity.
    Top-left repos are ideal (distributed work + high velocity).

  05_burden_by_language — Bar: Top 40 repos by maintenance burden, colored by language.
    Orange line = threshold of concern (burden > 5 means 5× demand vs supply capacity).

  06_recency_by_language — Histogram: Months since last activity, colored by language.
    Left = recently active. Right = stagnant. Orange line = 6-month stagnation threshold.

  07_tournament_comparison — Grouped bar: Model performance metrics with dummy baseline.
    Gray line = 0.80 AUC threshold. Models above this are considered strong.
    Winner: Logistic Regression (AUC-ROC=0.795).

  08_stagnation_by_language — Bar: Proportion of repos stagnant per language.
    Gray line = cohort average. JS stagnates at ~2× the rate of Java.
