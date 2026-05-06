# analysis.py
#
# Runs EFA (or PCA for scales where items > models) on LLM questionnaire responses.
#
# For each questionnaire × prompting condition:
#   - Builds a models × items matrix (each model is one "respondent")
#   - Determines number of factors via parallel analysis
#   - Runs EFA with oblimin rotation if N > p, otherwise PCA
#   - Saves factor loadings and a summary to data/efa_results/
#
# Prerequisites:
#   pip install factor_analyzer scikit-learn pandas numpy
#
# Usage:
#   python analysis.py

import os
import re
import warnings
import numpy as np
import pandas as pd
import scipy.linalg

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_CSV   = "data/results_clean.csv"
OUTPUT_DIR  = "data/efa_results"
CONDITIONS  = ["llm_analog", "neutral", "human_simulation"]

# Parallel analysis settings
PA_ITERATIONS = 200
PA_PERCENTILE = 95


# ---------------------------------------------------------------------------
# Parallel analysis — determines number of factors by comparing real
# eigenvalues to eigenvalues from random data of the same shape
# ---------------------------------------------------------------------------

def safe_eigvalsh(matrix: np.ndarray) -> np.ndarray:
    """Eigenvalues via scipy's more robust driver, with a small ridge for near-singular matrices."""
    ridge = 1e-6 * np.eye(matrix.shape[0])
    eigvals = scipy.linalg.eigh(matrix + ridge, eigvals_only=True, driver="ev")
    return np.sort(eigvals)[::-1]


def parallel_analysis(data: np.ndarray) -> int:
    n, p = data.shape
    corr = np.corrcoef(data.T)
    real_eigvals = safe_eigvalsh(corr)

    random_eigvals = []
    for _ in range(PA_ITERATIONS):
        rand = np.random.normal(size=(n, p))
        rand = (rand - rand.mean(0)) / rand.std(0)
        random_eigvals.append(safe_eigvalsh(np.corrcoef(rand.T)))

    threshold = np.percentile(random_eigvals, PA_PERCENTILE, axis=0)
    n_factors = int(np.sum(real_eigvals > threshold))
    return max(1, n_factors)


# ---------------------------------------------------------------------------
# EFA — oblimin rotation (oblique, standard for psychological constructs)
# ---------------------------------------------------------------------------

def run_efa(data: np.ndarray, items: list, n_factors: int) -> pd.DataFrame:
    fa = FactorAnalyzer(n_factors=n_factors, rotation="oblimin", method="minres")
    fa.fit(data)
    loadings = fa.loadings_
    cols = [f"F{i+1}" for i in range(n_factors)]
    return pd.DataFrame(loadings, index=items, columns=cols)


# ---------------------------------------------------------------------------
# PCA fallback for scales where n_items > n_models
# ---------------------------------------------------------------------------

def run_pca(data: np.ndarray, items: list, n_components: int) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    pca.fit(data)
    cols = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(pca.components_.T, index=items, columns=cols)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    # Drop rows where response couldn't be parsed
    df = df[df["response"].notna() & (df["response"] != "")]
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.dropna(subset=["response"])

    # Exclude questionnaires marked Good=0 in the questionnaire registry
    _qmeta = pd.read_excel("data/questionnaire_scales.xlsx")
    _excluded = frozenset(_qmeta[_qmeta["Good"] != 1]["quest"].str.strip())
    df = df[~df["questionnaire"].isin(_excluded)]

    questionnaires = df["questionnaire"].unique()
    summary_rows = []

    for quest in sorted(questionnaires):
        q_df = df[df["questionnaire"] == quest]

        for condition in CONDITIONS:
            c_df = q_df[q_df["condition"] == condition]
            if c_df.empty:
                continue

            # Pivot to models × items matrix (columns = item text)
            matrix = c_df.pivot_table(
                index="model", columns="item", values="response"
            )

            # Listwise deletion — drop any model with missing responses on any item
            matrix = matrix.dropna()

            # Drop items with zero variance (constant responses across all models),
            # as these produce NaN in the correlation matrix
            matrix = matrix.loc[:, matrix.std() > 0]

            items = list(matrix.columns)

            n, p = matrix.shape

            if n < 2 or p < 2:
                print(f"  SKIP {quest} / {condition}: only {n} complete cases / {p} variable items")
                continue

            data = matrix.values
            n_factors = parallel_analysis(data)
            # Cap factors: can't extract more factors than min(n-1, p-1)
            n_factors = min(n_factors, n - 1, p - 1)

            use_pca = n <= p

            if use_pca:
                method = "PCA"
                loadings_df = run_pca(data, items, n_factors)
            else:
                method = "EFA"
                # Assumption checks
                _, bartlett_p = calculate_bartlett_sphericity(data)
                _, kmo_model = calculate_kmo(data)
                loadings_df = run_efa(data, items, n_factors)

            # Save loadings to CSV
            safe_name = re.sub(r'[\\/*?:"<>|]', "_", quest)
            out_path = os.path.join(OUTPUT_DIR, f"{safe_name}__{condition}.csv")
            loadings_df.to_csv(out_path)

            row = {
                "questionnaire":  quest,
                "condition":      condition,
                "method":         method,
                "n_models":       n,
                "n_items":        p,
                "n_factors":      n_factors,
            }
            if not use_pca:
                row["bartlett_p"] = round(float(bartlett_p), 4)
                row["kmo"]        = round(float(kmo_model), 3)

            summary_rows.append(row)
            print(f"  [{method}] {quest} / {condition}: {n} models, {p} items -> {n_factors} factors")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print(f"\nDone. Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_analysis()
