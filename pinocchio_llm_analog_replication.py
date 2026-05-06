"""
Robustness check: replicate Section 4.2 item-level analyses using
llm_analog responses.

Cluster assignments are FIXED from the neutral condition (so the same
constructs are compared); only the responses used to compute scores and
EFA factor projections are swapped to llm_analog.

Outputs:
  data/pinocchio_llm_analog_replication.txt
"""

import os, builtins
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr, spearmanr

_out = open('data/pinocchio_llm_analog_replication.txt', 'w', encoding='utf-8')
def p(*a, **kw): builtins.print(*a, **kw, file=_out)

EFA_DIR  = 'data/efa_results'
PI_FILE  = 'data/pinocchio_items.xlsx'
DATA_CSV = 'data/results_clean.csv'
N_TOP    = 80
N_CLUST  = 2
N_PC     = 5

df = pd.read_csv(DATA_CSV, low_memory=False)
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df = df.dropna(subset=['response'])
neutral  = df[df['condition'] == 'neutral'].copy()
analog   = df[df['condition'] == 'llm_analog'].copy()

pi = pd.read_excel(PI_FILE).sort_values('pinocchio_score', ascending=False)
top_pi = pi.head(N_TOP)

# ---------------------------------------------------------------------------
# 1. Fix cluster assignments from neutral condition
# ---------------------------------------------------------------------------
def build_pivot(cond_df, items):
    records = []
    for _, row in items.iterrows():
        q, idx = row['questionnaire'], row['item_index']
        sub = cond_df[(cond_df['questionnaire'] == q) & (cond_df['item_index'] == idx)]
        if len(sub) < 5: continue
        sub = sub.set_index('model')['response']
        mu, sd = sub.mean(), sub.std()
        if sd == 0: continue
        z = (sub - mu) / sd
        for model, val in z.items():
            records.append({'model': model, 'questionnaire': q,
                            'item_index': idx, 'pi': row['pinocchio_score'], 'z': val})
    zdf = pd.DataFrame(records)
    piv = zdf.pivot_table(index='model', columns=['questionnaire','item_index'], values='z')
    return piv

neutral_pivot = build_pivot(neutral, top_pi)
neutral_pivot = neutral_pivot.dropna(axis=1, thresh=int(0.7*len(neutral_pivot))).fillna(0)
item_keys = list(neutral_pivot.columns)

Z_link = linkage(pdist(neutral_pivot.T.values, metric='correlation'), method='ward')
labels = fcluster(Z_link, N_CLUST, criterion='maxclust')
cluster_map = {k: int(l) for k, l in zip(item_keys, labels)}

NAMES = {1:'C1-Reactive', 2:'C2-Phenomenal'}
p(f"Cluster sizes (neutral, fixed): { {c: sum(1 for k in item_keys if cluster_map[k]==c) for c in range(1,3)} }")

# ---------------------------------------------------------------------------
# 2. Compute cluster scores from llm_analog responses (fixed assignments)
# ---------------------------------------------------------------------------
analog_pivot = build_pivot(analog, top_pi)
# Align to same columns as neutral pivot (drop any missing)
common_keys = [k for k in item_keys if k in analog_pivot.columns]
analog_pivot = analog_pivot[common_keys].fillna(0)

cluster_scores_analog = {}
for c in range(1, N_CLUST+1):
    keys_c = [k for k in common_keys if cluster_map[k] == c]
    w = np.array([pi[(pi['questionnaire']==k[0]) & (pi['item_index']==k[1])
                     ]['pinocchio_score'].values[0] for k in keys_c])
    w = w / w.sum()
    cluster_scores_analog[c] = pd.Series(
        analog_pivot[keys_c].values @ w, index=analog_pivot.index)

p(f"Items available in llm_analog pivot: {len(common_keys)}")

# ---------------------------------------------------------------------------
# 3. EFA F1 scores from llm_analog EFA loadings
# ---------------------------------------------------------------------------
f1_matrix = {}
for fname in sorted(os.listdir(EFA_DIR)):
    if not fname.endswith('.csv') or '__llm_analog' not in fname:
        continue
    quest = fname.replace('__llm_analog.csv', '')
    loadings = pd.read_csv(os.path.join(EFA_DIR, fname), index_col=0)
    f1 = loadings.iloc[:, 0]
    qdf = analog[analog['questionnaire'] == quest]
    if len(qdf) == 0: continue
    qpiv = qdf.pivot_table(index='model', columns='item', values='response')
    common = [i for i in f1.index if i in qpiv.columns]
    if len(common) < 3: continue
    qsub = qpiv[common]
    qsub_z = (qsub - qsub.mean()) / (qsub.std() + 1e-9)
    scores = pd.Series(qsub_z.fillna(0).values @ f1.reindex(common).values,
                       index=qsub_z.index)
    f1_matrix[quest] = scores

f1_df = pd.DataFrame(f1_matrix).dropna(thresh=int(0.8*len(f1_matrix))).fillna(0)
p(f"Questionnaires with F1 scores (llm_analog): {len(f1_matrix)}")
p(f"F1 matrix shape: {f1_df.shape}")

# ---------------------------------------------------------------------------
# 4. Global PCA on llm_analog F1 matrix
# ---------------------------------------------------------------------------
X = StandardScaler().fit_transform(f1_df.values)
pca = PCA(n_components=N_PC)
pca.fit(X)
pc_scores = pd.DataFrame(pca.transform(X), index=f1_df.index,
                          columns=[f'PC{i+1}' for i in range(N_PC)])

p("\n=== PCA variance explained (llm_analog) ===")
for i, v in enumerate(pca.explained_variance_ratio_):
    p(f"  PC{i+1}: {v*100:.1f}%  (cumulative: {pca.explained_variance_ratio_[:i+1].sum()*100:.1f}%)")

loadings_df = pd.DataFrame(pca.components_.T, index=f1_df.columns,
                             columns=[f'PC{i+1}' for i in range(N_PC)])

p("\n=== Top questionnaire loadings on PC1 and PC2 (|loading| > 0.20) ===")
for pc in ['PC1', 'PC2']:
    col = loadings_df[pc].sort_values()
    neg = col[col < -0.20]
    pos = col[col > 0.20]
    p(f"\n  {pc}:")
    if len(neg):
        p(f"    Negative: " + ", ".join(f"{q}({v:.2f})" for q,v in neg.items()))
    if len(pos):
        p(f"    Positive: " + ", ".join(f"{q}({v:.2f})" for q,v in pos.items()))

# ---------------------------------------------------------------------------
# 5. Strand-PC correlations (llm_analog cluster scores x llm_analog PCs)
# ---------------------------------------------------------------------------
p("\n\n=== Pi strand scores x global PCs — llm_analog (Pearson r) ===\n")
header = f"{'':20s}" + "".join(f"{'PC'+str(i+1):>10s}" for i in range(N_PC))
p(header)

for c in range(1, N_CLUST+1):
    cs = cluster_scores_analog[c]
    common = cs.index.intersection(pc_scores.index)
    row_str = f"{NAMES[c]:20s}"
    for pc in [f'PC{i+1}' for i in range(N_PC)]:
        r, pval = pearsonr(cs[common], pc_scores.loc[common, pc])
        stars = '***' if pval < .001 else ('**' if pval < .01 else ('*' if pval < .05 else ''))
        row_str += f"{r:>+7.3f}{stars:>3s}"
    p(row_str)

p("\n=== Spearman rho ===\n")
p(header)
for c in range(1, N_CLUST+1):
    cs = cluster_scores_analog[c]
    common = cs.index.intersection(pc_scores.index)
    row_str = f"{NAMES[c]:20s}"
    for pc in [f'PC{i+1}' for i in range(N_PC)]:
        rho, pval = spearmanr(cs[common], pc_scores.loc[common, pc])
        stars = '***' if pval < .001 else ('**' if pval < .01 else ('*' if pval < .05 else ''))
        row_str += f"{rho:>+7.3f}{stars:>3s}"
    p(row_str)

_out.close()
builtins.print("Done")
