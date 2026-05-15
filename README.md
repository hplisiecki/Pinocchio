# LLMLatent

Code and data for the paper:
**"The Pinocchio Dimension: Phenomenality of Experience as the Primary Axis of LLM Psychometric Differences"**

We administered 45 validated psychometric questionnaires to 50 LLMs under three prompting conditions and asked: what is the primary dimension along which LLMs differ psychometrically? The answer is experiential self-attribution — the degree to which a model claims to be a locus of inner experience — which we call **the Pinocchio Axis** (PC1, 47.1% of between-model variance). Formal cluster validation (silhouette analysis, k=2..10) confirms a binary item-level structure: reactive/behavioral items at the negative pole vs. phenomenologically rich items at the positive pole.

---

## Prompting Conditions

Each model receives every questionnaire item one at a time and responds numerically, under three conditions:

- **neutral** — no framing; raw questionnaire item only
- **llm_analog** — model is asked to respond as an AI/LLM finding functional analogs to the experiences described
- **human_simulation** — model is asked to simulate how a typical human would respond

---

## Pipeline

Scripts run in the following order.

### 1. Data Collection

**[`run_experiment.py`](run_experiment.py)**
Queries all models in `data/models.json` on all items in `data/questionnaires.json` via the OpenRouter API, under all three conditions. Saves responses to `data/results.json` after every call (resumable if interrupted). Concurrency is controlled by `MAX_CONCURRENT` (default 20).

**[`prompts.py`](prompts.py)**
Defines the system prompt templates for all three conditions and the questionnaire loader.

**[`open_router_quickstart.py`](open_router_quickstart.py)**
Minimal single-call OpenRouter example for testing your API key.

### 2. Preprocessing

**[`clean_responses.py`](clean_responses.py)**
Nulls out-of-range responses (e.g., `7` on a 1–5 scale) and consolidates raw JSON files into `data/results_clean.csv`. Logs every nulled response to [`data/clean_log.txt`](data/clean_log.txt) and saves the removed rows to `data/oob_responses.csv`.

**[`merge_mfq.py`](merge_mfq.py)**
Merges the two MFQ sub-questionnaires (`Moral Foundations Questionnaire (MFQ-30)_1` and `_2`) into a single entry in `data/results_clean.csv`, giving EFA access to all 32 MFQ items jointly. Run once after `clean_responses.py`.

**[`dump_additional.py`](dump_additional.py)**
Merges additional response batches collected in separate runs into the main results files.

**[`check_data.py`](check_data.py)**, **[`check_missing.py`](check_missing.py)**
Basic integrity and coverage checks (model × questionnaire × condition completeness).

### 3. Factor Analysis

**[`analysis.py`](analysis.py)**
Runs EFA on the model × item response matrix for each questionnaire × condition (45 × 3 = 135 analyses). Falls back to PCA when items > models. Excludes questionnaires flagged `Good=0` in `data/questionnaire_scales.xlsx` (used to remove redundant instrument versions — only the more comprehensive version of duplicate instruments is retained). Saves factor loading matrices to `data/efa_results/<questionnaire>__<condition>.csv`. Logs to [`data/analysis_log.txt`](data/analysis_log.txt).

**[`check_efa_shape.py`](check_efa_shape.py)**
Verifies EFA output dimensions are as expected. Outputs [`data/efa_shape.txt`](data/efa_shape.txt).

**[`export_loadings.py`](export_loadings.py)**
Copies all EFA loading matrices from `data/efa_results/` into `data/loadings/` for inspection.

**[`congruence_analysis.py`](congruence_analysis.py)**
Computes Tucker's congruence coefficient (φ) between factor loading matrices across all condition pairs for each questionnaire. Tests the self-model hypothesis: φ(llm_analog, neutral) > φ(human_simulation, llm_analog) ≈ φ(human_simulation, neutral). Outputs `data/congruence_results.csv` and [`data/congruence_log.txt`](data/congruence_log.txt).

### 4. Supervised Semantic Differential (SSD)

**[`ssd_analysis.py`](ssd_analysis.py)**
Pools all items across all questionnaires and uses SSD to learn the semantic gradient in item-text space that best predicts primary EFA Factor-1 loadings. Runs separately per condition. Outputs per-condition files to `data/ssd_results/` and logs to [`data/ssd_log.txt`](data/ssd_log.txt).

**[`read_ssd_clusters.py`](read_ssd_clusters.py)**, **[`read_ssd_snippets.py`](read_ssd_snippets.py)**, **[`read_ssd_top_items.py`](read_ssd_top_items.py)**
Print formatted summaries of SSD cluster members, text snippets, and top-scoring items from `data/ssd_results/`. Outputs [`data/ssd_clusters_dump.txt`](data/ssd_clusters_dump.txt), [`data/ssd_snippets_dump.txt`](data/ssd_snippets_dump.txt), [`data/ssd_top_items_dump.txt`](data/ssd_top_items_dump.txt).

### 5. Pinocchio Score

**[`pinocchio_analysis.py`](pinocchio_analysis.py)**
Computes the Pinocchio score π_i = σ²(neutral) / σ²(human_simulation) for each item: the ratio of inter-model variance when models respond as themselves vs. when simulating a human. High π_i items elicit structured self-model disagreement. Excludes `Good=0` questionnaires. Saves to `data/pinocchio_items.xlsx`. Prints top-20 items to stdout.

**[`pinocchio_prediction.py`](pinocchio_prediction.py)**
Validates π_i by testing whether high-π items have stronger Factor-1 loadings in neutral and weaker loadings under human simulation. Outputs `data/pinocchio_prediction.xlsx`, `data/pinocchio_prediction.png`.

**[`check_specificity.py`](check_specificity.py)**
Verifies that the Phenomenality of Experience axis is not a valence artefact: splits questionnaires by PC1 loading direction and compares mean item variance (neutral condition). Outputs [`data/check_specificity.txt`](data/check_specificity.txt).

### 6. Pi Item Cluster Analysis

**[`pinocchio_cluster_globalpc.py`](pinocchio_cluster_globalpc.py)**
Core item-level structure, global PCA, and per-item PC1 correlation analysis:
1. Clusters the top-80 π items into k=2 (Ward linkage, correlation distance; k=2 validated by silhouette analysis).
2. Computes per-model EFA Factor-1 scores for all 45 questionnaires (neutral), assembles a 50×45 matrix.
3. Applies PCA → PC1 = Phenomenality of Experience (47.1%).
4. Correlates k=2 cluster scores with global PCs and with individual questionnaire EFA F1 scores.
5. Correlates every individual item (n ≥ 15 models) with PC1 — the replicable source for Tables 4 & 5 in the paper.

Outputs [`data/pi_cluster_globalpc.txt`](data/pi_cluster_globalpc.txt) (human-readable report) and `data/pc1_item_correlations.csv` (full item × PC1 table).

**[`check_silhouette.py`](check_silhouette.py)**
Formal cluster validity analysis: computes average silhouette coefficient for k=2..10 (Ward linkage, correlation distance) on the top-80 π item response matrix. Confirms k=2 as the uniquely supported solution (avg silhouette 0.41 vs. ≤0.22 for k≥3). Outputs `data/silhouette_plot.png`.

### 7. Model Scoring and Visualisation

**[`pinocchio_model_scores.py`](pinocchio_model_scores.py)**
Computes per-model Phenomenality of Experience scores: log-π-weighted mean z-score across all high-demand items (π_i > 1) in the neutral condition. Produces a specificity contrast (high-demand minus low-demand z-score) to rule out general acquiescence bias. Outputs `data/pinocchio_model_scores.xlsx`, `data/pinocchio_model_scores.png`, `data/pinocchio_model_scores_contrast.png`.

**[`plot_model_psychometric_space.py`](plot_model_psychometric_space.py)**
Rebuilds the global PCA and produces the main paper figure: ranked bar chart of all 50 models by PC1 / Phenomenality of Experience (`data/model_pc1_ranked.png`).

**[`plot_pc1_with_ci.py`](plot_pc1_with_ci.py)**
Produces the main figure with 95% bootstrap confidence intervals: resamples the 45 questionnaires with replacement 1,000 times, reruns the full PCA pipeline on each sample, and aligns sign and scale to the reference solution. Outputs `data/model_pc1_ranked_ci.png`.

**[`pinocchio_llm_analog.py`](pinocchio_llm_analog.py)**
Robustness check: recomputes model-level scores using llm_analog responses (item weights unchanged). Reports rank correlation with neutral-condition ordering. Outputs `data/pinocchio_llm_analog.png`, `data/pinocchio_llm_analog.xlsx`, [`data/pinocchio_llm_analog_log.txt`](data/pinocchio_llm_analog_log.txt).

**[`plot_llm_analog_ci.py`](plot_llm_analog_ci.py)**
Same as `plot_pc1_with_ci.py` but for the LLM-analog robustness figure: bootstraps over questionnaires using the log-π-weighted z-score scoring method. Outputs `data/pinocchio_llm_analog_ci.png`.

**[`provider_summary.py`](provider_summary.py)**
Aggregates PC1 scores by provider (mean, min, max, n) for the neutral condition. Outputs `data/provider_summary.txt`.

**[`provider_summary_llm_analog.py`](provider_summary_llm_analog.py)**
Same as `provider_summary.py` but for the LLM-analog condition using log-π-weighted z-scores. Outputs `data/provider_summary_llm_analog.txt`.

**[`pinocchio_llm_analog_replication.py`](pinocchio_llm_analog_replication.py)**
Robustness check for the item-level structure: repeats the cluster–PC correlation analysis using llm_analog responses with cluster assignments fixed from neutral. Outputs [`data/pinocchio_llm_analog_replication.txt`](data/pinocchio_llm_analog_replication.txt).

**[`harvest_numbers.py`](harvest_numbers.py)**
Aggregates all key statistics cited in the paper into a single file for cross-checking. Outputs [`data/paper_numbers.txt`](data/paper_numbers.txt).

---

## Data

### Excluded Models

Two models were queried but excluded from all analyses due to near-total response failure (responses could not be parsed as valid numeric ratings):

| Model | NaN rate | Reason |
|---|---|---|
| `anthropic/claude-3-opus` | 100% | All responses unparseable |
| `baidu/ernie-4.5-21b-a3b` | 95.5% | Near-total parsing failure |

This exclusion is not noted in the main paper in the interest of aesthetics of presentation.

### Primary Files

| File | Description |
|---|---|
| `data/questionnaires.json` | All questionnaires with full item text and response scales |
| `data/questionnaire_scales.xlsx` | Registry of all instruments; `Good=1` marks instruments included in analysis |
| `data/models.json` | 50 LLMs queried (`provider/model-name`) |
| `data/results_clean.csv` | **Primary dataset.** One row per (questionnaire, model, condition, item). Out-of-range responses nulled. |
| `data/results.csv` | Pre-cleaning dataset |
| `data/pinocchio_items.xlsx` | Per-item Pinocchio scores (π_i), item text, questionnaire, item index |
| `data/oob_responses.csv` | Out-of-bounds responses removed during cleaning |

### Questionnaire Selection

`data/questionnaires.json` was assembled by filtering `questionnaire_scales.xlsx` to `Good=1` instruments only, so `run_experiment.py` never queried `Good=0` questionnaires. However, two instruments were included in the experiment and subsequently marked `Good=0` after data collection, once they were found to have substantially overlapping item content with their revised counterparts:

| Questionnaire | Superseded by |
|---|---|
| `IRQ` (Internal Representation Questionnaire, short form) | `The Internal Representation Questionnaire` (long form) |
| `Varieties of Inner Speech Questionnaire` | `Varieties of Inner Speech Questionnaire-R` |

Their responses are present in `results_clean.csv` but excluded from all analyses (EFA, Pinocchio scoring, PCA) by the `Good=0` filter applied in `analysis.py` and downstream scripts.

### `results_clean.csv` Schema

| Column | Type | Description |
|---|---|---|
| `questionnaire` | str | Questionnaire name |
| `model` | str | Model ID (`provider/model-name`) |
| `condition` | str | `neutral`, `llm_analog`, or `human_simulation` |
| `item_index` | int | 0-based item position within questionnaire |
| `item` | str | Full item text |
| `response` | float | Numeric response (null if OOB or API error) |
| `response_raw` | str | Raw model output before numeric parsing |

### EFA Results

`data/efa_results/` — CSV files, one per questionnaire × condition.
- Filename: `<questionnaire>__<condition>.csv`
- Content: items × factors loading matrix (rows = items, columns = F1, F2, …)

### SSD Results

`data/ssd_results/` — per-condition output files:

| File pattern | Description |
|---|---|
| `*_clusters.xlsx` | Cluster assignments and summary statistics |
| `*_cluster_members.xlsx` | Full item membership per cluster |
| `*_cluster_snippets.xlsx` | Representative text fragments per cluster |
| `*_top_words.xlsx` | Highest-weight words per semantic dimension |
| `*_item_scores.xlsx` | Per-item SSD projection scores |
| `*_sweep_plot.png` | Variance explained vs. number of clusters K |

### Key Outputs

| File | Description |
|---|---|
| `data/pinocchio_model_scores.xlsx` | Per-model scores, ranks, and specificity contrasts |
| `data/model_pc1_ranked.png` | Models ranked by Phenomenality of Experience (main paper figure) |
| `data/silhouette_plot.png` | Silhouette curve k=2..10 confirming binary cluster structure |
| [`data/pi_cluster_globalpc.txt`](data/pi_cluster_globalpc.txt) | k=2 cluster composition, cluster–PC correlations, PCA variance |
| `data/pc1_item_correlations.csv` | Per-item × PC1 Pearson r for all items (replicable source for Tables 4 & 5) |
| `data/congruence_results.csv` | Tucker's φ across all condition pairs and questionnaires |
| [`data/paper_numbers.txt`](data/paper_numbers.txt) | All key statistics cited in the paper |

---

