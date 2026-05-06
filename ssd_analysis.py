"""
ssd_analysis.py

For each of the three conditions (llm_analog, neutral, human_simulation),
pool all items across all questionnaires and predict PC1 loading from item text.

PC1 loading = the loading of each item on the first principal component
within its questionnaire for that condition. Pooling across questionnaires
asks: "what textual features predict whether an item captures the primary
axis of between-model variance?"

Embeddings: GloVe/Dolma 300d (dolma_300_2024_1.2M.100_combined.kv)
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

KV_PATH    = r"D:\Resources\NLP\EN\glove.2024.dolma.300d\dolma_300_2024_1.2M.100_combined.kv"
EFA_DIR    = "data/efa_results"
OUT_DIR    = "data/ssd_results"
CONDITIONS = ["llm_analog", "neutral", "human_simulation"]

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load embeddings + spaCy once
# ---------------------------------------------------------------------------
print("Loading embeddings...")
from ssdiff import load_embeddings, normalize_kv
kv = normalize_kv(load_embeddings(KV_PATH), l2=True, abtt_m=1)
print(f"  Loaded {len(kv)} vectors of dim {kv.vector_size}")

print("Loading spaCy...")
from ssdiff import (load_spacy, load_stopwords, preprocess_texts,
                    build_docs_from_preprocessed, pca_sweep, SSD)
nlp  = load_spacy("en_core_web_lg")
stop = load_stopwords("en")

# ---------------------------------------------------------------------------
# Build pooled dataset per condition
# ---------------------------------------------------------------------------
def load_condition(condition):
    """Return DataFrame with columns: questionnaire, item, pc1_loading."""
    rows = []
    for fname in sorted(os.listdir(EFA_DIR)):
        if not fname.endswith(f"__{condition}.csv") or fname == "summary.csv":
            continue
        quest = fname.replace(f"__{condition}.csv", "")
        efa   = pd.read_csv(os.path.join(EFA_DIR, fname), index_col=0)
        for item_text, loading_row in efa.iterrows():
            rows.append({
                "questionnaire": quest,
                "item":          str(item_text),
                "pc1_loading":   float(loading_row.iloc[0]),
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Run one SSD per condition
# ---------------------------------------------------------------------------
summary_rows = []

for cond in CONDITIONS:
    print(f"\n=== Condition: {cond} ===")
    data = load_condition(cond)
    print(f"  {len(data)} items across {data['questionnaire'].nunique()} questionnaires")

    texts = data["item"].tolist()
    y     = data["pc1_loading"].values.astype(float)

    pre_docs = preprocess_texts(texts, nlp, stop)
    docs     = build_docs_from_preprocessed(pre_docs)

    sel = pca_sweep(
        kv=kv, docs=docs, y=y,
        lexicon=[], use_full_doc=True,
        pca_k_values=list(range(2, 121, 2)),
        window=3, sif_a=1e-3,
        k_min=2, k_max=5,
        auck_radius=3,
        save_figures=True, save_tables=True,
        out_dir=OUT_DIR,
        prefix=cond,
        verbose=True,
    )
    PCA_K = sel.best_k
    print(f"  Selected PCA_K = {PCA_K}")

    ssd = SSD(
        kv=kv, docs=docs, y=y,
        lexicon=[], use_full_doc=True,
        l2_normalize_docs=True, N_PCA=PCA_K,
        use_unit_beta=True, window=3, sif_a=1e-3,
    )

    print(f"  r²={ssd.r2:.4f}  r²_adj={ssd.r2_adj:.4f}  "
          f"F={ssd.f_stat:.2f}  p={ssd.f_pvalue:.4e}  r={ssd.y_corr_pred:.4f}")

    summary_rows.append({
        "condition": cond,
        "n_items":   len(data),
        "PCA_K":     PCA_K,
        "r2":        round(ssd.r2, 4),
        "r2_adj":    round(ssd.r2_adj, 4),
        "f_stat":    round(ssd.f_stat, 3),
        "f_pvalue":  ssd.f_pvalue,
        "r":         round(ssd.y_corr_pred, 4),
    })

    # Top words — returns a DataFrame with columns: side, rank, word, cos
    try:
        tw_df = ssd.top_words(n=40, verbose=True)
        tw_df.to_excel(os.path.join(OUT_DIR, f"{cond}_top_words.xlsx"), index=False)
        print(f"  Top words saved")
    except Exception as e:
        print(f"  top_words failed: {e}")

    # Cluster neighbours — sets ssd.pos_clusters_raw and ssd.neg_clusters_raw
    # which are required before calling cluster_snippets
    try:
        df_clust, df_members = ssd.cluster_neighbors(
            topn=100, k=None, k_min=2, k_max=8, verbose=False)
        df_clust.to_excel(
            os.path.join(OUT_DIR, f"{cond}_clusters.xlsx"), index=False)
        df_members.to_excel(
            os.path.join(OUT_DIR, f"{cond}_cluster_members.xlsx"), index=False)
        print(f"  Clusters saved")
    except Exception as e:
        print(f"  cluster_neighbors failed: {e}")

    # Cluster snippets — item texts closest to each cluster centroid
    # Requires cluster_neighbors (or cluster_neighbors_sign) to have been called first
    # Returns dict: {"pos": DataFrame, "neg": DataFrame}
    # DataFrame columns include: centroid_label, cosine, snippet_anchor, essay_text_surface
    try:
        snips = ssd.cluster_snippets(
            pre_docs=pre_docs, side="both", top_per_cluster=100)
        snip_rows = []
        for side_key, df_side in snips.items():
            if df_side is not None and len(df_side) > 0:
                df_side = df_side.copy()
                df_side.insert(0, "side", side_key)
                snip_rows.append(df_side)
        if snip_rows:
            pd.concat(snip_rows, ignore_index=True).to_excel(
                os.path.join(OUT_DIR, f"{cond}_cluster_snippets.xlsx"), index=False)
            print(f"  Cluster snippets saved")
    except Exception as e:
        print(f"  cluster_snippets failed: {e}")

    # Beta snippets — item texts closest to +β̂ and −β̂ poles
    # Returns dict: {"beta_pos": DataFrame, "beta_neg": DataFrame}
    try:
        beta_snips = ssd.beta_snippets(
            pre_docs=pre_docs, window_sentences=1, top_per_side=40)
        beta_rows = []
        for side_key, df_side in beta_snips.items():
            if df_side is not None and len(df_side) > 0:
                df_side = df_side.copy()
                df_side.insert(0, "side", side_key)
                beta_rows.append(df_side)
        if beta_rows:
            pd.concat(beta_rows, ignore_index=True).to_excel(
                os.path.join(OUT_DIR, f"{cond}_beta_snippets.xlsx"), index=False)
            print(f"  Beta snippets saved")
    except Exception as e:
        print(f"  beta_snippets failed: {e}")

    # Per-item SSD scores — no docs argument; uses internal state
    # Returns DataFrame with: doc_index, kept, cos, yhat_std, yhat_raw, y_true_std, y_true_raw
    try:
        scores_df = ssd.ssd_scores(include_all=True)
        scores_df["item"]          = texts
        scores_df["questionnaire"] = data["questionnaire"].values
        scores_df["pc1_loading"]   = y
        scores_df.to_excel(
            os.path.join(OUT_DIR, f"{cond}_item_scores.xlsx"), index=False)
        print(f"  Item scores saved")
    except Exception as e:
        print(f"  ssd_scores failed: {e}")

# ---------------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------------
pd.DataFrame(summary_rows).to_excel(
    os.path.join(OUT_DIR, "ssd_summary.xlsx"), index=False)
print(f"\nSummary saved → {OUT_DIR}/ssd_summary.xlsx")
