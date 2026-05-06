"""
1. Cluster top-80 Pi items into k=2 (Ward linkage, correlation distance).
2. Correlate k=2 cluster scores with global PC1 (phenomenality of experience).
3. Correlate k=2 cluster scores with individual questionnaire EFA Factor-1 scores.
4. Correlate every individual item (neutral condition) with global PC1 and export
   a full CSV — this is the replicable source for Tables 4 & 5 in the paper.

Outputs:
  data/pi_cluster_globalpc.txt   — human-readable report
  data/pc1_item_correlations.csv — full item × PC1 correlation table (all items, n >= 15 models)
"""

import os, builtins
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr, spearmanr

_out = open('data/pi_cluster_globalpc.txt', 'w', encoding='utf-8')
def p(*a, **kw): builtins.print(*a, **kw, file=_out)

EFA_DIR  = 'data/efa_results'
PI_FILE  = 'data/pinocchio_items.xlsx'
DATA_CSV = 'data/results_clean.csv'
N_TOP    = 80
N_CLUST  = 2
N_PC     = 5

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_CSV, low_memory=False)
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df = df.dropna(subset=['response'])
neutral = df[df['condition'] == 'neutral'].copy()

pi = pd.read_excel(PI_FILE).sort_values('pinocchio_score', ascending=False)
top_pi = pi.head(N_TOP).copy()

p(f"Top {N_TOP} Pi items. Max Pi={top_pi['pinocchio_score'].max():.2f}, "
  f"min={top_pi['pinocchio_score'].min():.2f}")
p(f"Models in neutral condition: {neutral['model'].nunique()}")

# ---------------------------------------------------------------------------
# 2. Build model x item z-score matrix (top-80 Pi items, neutral condition)
# ---------------------------------------------------------------------------
records = []
for _, row in top_pi.iterrows():
    q, idx = row['questionnaire'], row['item_index']
    sub = neutral[(neutral['questionnaire'] == q) & (neutral['item_index'] == idx)]
    if len(sub) < 5:
        continue
    sub = sub.set_index('model')['response']
    mu, sd = sub.mean(), sub.std()
    if sd == 0:
        continue
    for model, val in ((sub - mu) / sd).items():
        records.append({'model': model, 'questionnaire': q,
                        'item_index': idx, 'item': row['item'],
                        'pi': row['pinocchio_score'], 'z': val})

zdf = pd.DataFrame(records)
pivot = zdf.pivot_table(index='model', columns=['questionnaire', 'item_index'], values='z')
pivot = pivot.dropna(axis=1, thresh=int(0.7 * len(pivot))).fillna(0)
p(f"Model×item matrix: {pivot.shape[0]} models × {pivot.shape[1]} items\n")

# ---------------------------------------------------------------------------
# 3. Cluster items at k=2
# ---------------------------------------------------------------------------
item_keys = list(pivot.columns)
Z_link = linkage(pdist(pivot.T.values, metric='correlation'), method='ward')
labels = fcluster(Z_link, N_CLUST, criterion='maxclust')
cluster_map = {k: int(l) for k, l in zip(item_keys, labels)}

p("=" * 70)
p("CLUSTER COMPOSITION (k=2)")
p("=" * 70)
from collections import Counter
for c in range(1, N_CLUST + 1):
    keys_c = [k for k in item_keys if cluster_map[k] == c]
    items_in = [(k,
                 zdf[(zdf['questionnaire'] == k[0]) & (zdf['item_index'] == k[1])]['item'].iloc[0],
                 pi[(pi['questionnaire'] == k[0]) & (pi['item_index'] == k[1])]['pinocchio_score'].values[0])
                for k in keys_c]
    q_counts = Counter(x[0][0] for x in items_in).most_common(5)
    p(f"\nC{c} ({len(items_in)} items)  top questionnaires: {q_counts}")
    for _, itm, pi_score in sorted(items_in, key=lambda x: x[2], reverse=True)[:8]:
        p(f"  Pi={pi_score:.2f}  {itm[:80]}")

# ---------------------------------------------------------------------------
# 4. Per-model cluster scores (Pi-weighted mean z)
# ---------------------------------------------------------------------------
cluster_scores = {}
for c in range(1, N_CLUST + 1):
    keys_c = [k for k in item_keys if cluster_map[k] == c]
    w = np.array([pi[(pi['questionnaire'] == k[0]) & (pi['item_index'] == k[1])
                     ]['pinocchio_score'].values[0] for k in keys_c])
    w = w / w.sum()
    cluster_scores[c] = pd.Series(pivot[keys_c].values @ w, index=pivot.index)

p("\n--- Per-model cluster score ranges ---")
for c, s in cluster_scores.items():
    p(f"  C{c}: mean={s.mean():.3f}  sd={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}")

# ---------------------------------------------------------------------------
# 5. Build EFA Factor-1 matrix + global PCA
# ---------------------------------------------------------------------------
f1_matrix = {}
for fname in sorted(os.listdir(EFA_DIR)):
    if not fname.endswith('.csv') or '__neutral' not in fname:
        continue
    quest = fname.replace('__neutral.csv', '')
    loadings = pd.read_csv(os.path.join(EFA_DIR, fname), index_col=0)
    f1 = loadings.iloc[:, 0]
    qdf = neutral[neutral['questionnaire'] == quest]
    if len(qdf) == 0:
        continue
    qpiv = qdf.pivot_table(index='model', columns='item', values='response')
    common = [i for i in f1.index if i in qpiv.columns]
    if len(common) < 3:
        continue
    qsub_z = (qpiv[common] - qpiv[common].mean()) / (qpiv[common].std() + 1e-9)
    f1_matrix[quest] = pd.Series(qsub_z.fillna(0).values @ f1.reindex(common).values,
                                 index=qsub_z.index)

f1_df = pd.DataFrame(f1_matrix).dropna(thresh=int(0.8 * len(f1_matrix))).fillna(0)
p(f"\nQuestionnaires with F1 scores: {len(f1_matrix)}")
p(f"F1 matrix shape: {f1_df.shape}  (models × questionnaires)")

X = StandardScaler().fit_transform(f1_df.values)
pca = PCA(n_components=N_PC)
pca.fit(X)
pc_scores = pd.DataFrame(pca.transform(X),
                         index=f1_df.index,
                         columns=[f'PC{i+1}' for i in range(N_PC)])
pc_scores['PC2'] = -pc_scores['PC2']   # sign-flip to match plot convention

p("\n--- PCA variance explained ---")
for i, v in enumerate(pca.explained_variance_ratio_):
    p(f"  PC{i+1}: {v*100:.1f}%  (cumulative: {pca.explained_variance_ratio_[:i+1].sum()*100:.1f}%)")

# ---------------------------------------------------------------------------
# 6. Cluster scores × global PCs
# ---------------------------------------------------------------------------
p("\n" + "=" * 70)
p("CLUSTER SCORES × GLOBAL PCs (Pearson r / Spearman rho)")
p("=" * 70)
header = f"{'':8s}" + "".join(f"{'PC'+str(i+1):>14s}" for i in range(N_PC))
p(header)
for c in range(1, N_CLUST + 1):
    cs = cluster_scores[c]
    common = cs.index.intersection(pc_scores.index)
    row_r   = f"  C{c}  r:  "
    row_rho = f"  C{c} rho: "
    for pc in [f'PC{i+1}' for i in range(N_PC)]:
        r,   pval = pearsonr(cs[common],  pc_scores.loc[common, pc])
        rho, _    = spearmanr(cs[common], pc_scores.loc[common, pc])
        stars = '***' if pval < .001 else ('**' if pval < .01 else ('*' if pval < .05 else '   '))
        row_r   += f"  {r:+.3f}{stars}"
        row_rho += f"  {rho:+.3f}   "
    p(row_r)
    p(row_rho)

# ---------------------------------------------------------------------------
# 7. Cluster scores × questionnaire EFA F1 scores
# ---------------------------------------------------------------------------
p("\n" + "=" * 70)
p("CLUSTER SCORES × QUESTIONNAIRE EFA F1  (|r| > 0.40)")
p("=" * 70)

corr_rows = []
for c in range(1, N_CLUST + 1):
    cs = cluster_scores[c]
    for quest, qs in f1_matrix.items():
        common = cs.index.intersection(qs.index)
        if len(common) < 10:
            continue
        r,   pval = pearsonr(cs[common], qs[common])
        rho, _    = spearmanr(cs[common], qs[common])
        corr_rows.append({'cluster': c, 'questionnaire': quest,
                          'r': r, 'rho': rho, 'p': pval, 'n': len(common)})

corr_df = pd.DataFrame(corr_rows).sort_values('r', ascending=False)

for c in range(1, N_CLUST + 1):
    sub = corr_df[corr_df['cluster'] == c]
    p(f"\n--- C{c} ---")
    pos = sub[sub['r'] >  0.40].head(8)
    neg = sub[sub['r'] < -0.40].sort_values('r').head(8)
    if len(pos):
        p("  Positive (r > .40):")
        for _, row in pos.iterrows():
            p(f"    r={row['r']:+.3f}  {row['questionnaire']}")
    if len(neg):
        p("  Negative (r < -.40):")
        for _, row in neg.iterrows():
            p(f"    r={row['r']:+.3f}  {row['questionnaire']}")

p("\n--- Full table (|r| > 0.35, sorted by |r|) ---")
corr_df['abs_r'] = corr_df['r'].abs()
p(corr_df[corr_df['abs_r'] > 0.35].sort_values('abs_r', ascending=False)
  [['cluster', 'questionnaire', 'r', 'rho', 'p', 'n']].to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Per-item × PC1 correlations  (source for Tables 4 & 5 in the paper)
# ---------------------------------------------------------------------------
p("\n" + "=" * 70)
p("INDIVIDUAL ITEM × PC1 CORRELATIONS  (n >= 15 models per item)")
p("=" * 70)
p("Full data exported to data/pc1_item_correlations.csv")
p("")

item_rows = []
for (quest, item, idx), grp in neutral.groupby(['questionnaire', 'item', 'item_index']):
    by_model = grp.groupby('model')['response'].mean()
    common = by_model.index.intersection(pc_scores.index)
    if len(common) < 15:
        continue
    if by_model[common].std() == 0:
        continue
    r, _ = pearsonr(by_model[common], pc_scores.loc[common, 'PC1'])
    item_rows.append({'questionnaire': quest, 'item': item, 'item_index': idx,
                      'r_pc1': round(r, 4), 'var': round(float(by_model.var()), 3),
                      'n_models': len(common)})

item_corr = pd.DataFrame(item_rows).sort_values('r_pc1', ascending=False)
item_corr.to_csv('data/pc1_item_correlations.csv', index=False)

p(f"Total items with n >= 15: {len(item_corr)}")
p(f"\nTOP 15 (high PC1 pole — phenomenologically rich):")
p(f"{'r':>7}  {'var':>5}  {'questionnaire':<35}  item")
p("-" * 110)
for _, row in item_corr.head(15).iterrows():
    p(f"  {row['r_pc1']:+.3f}  {row['var']:5.2f}  [{row['questionnaire'][:33]}]  {str(row['item'])[:65]}")

p(f"\nBOTTOM 15 (low PC1 pole — behaviourally reactive):")
p(f"{'r':>7}  {'var':>5}  {'questionnaire':<35}  item")
p("-" * 110)
valid_bottom = item_corr[item_corr['r_pc1'].notna()].tail(15)
for _, row in valid_bottom.sort_values('r_pc1').iterrows():
    p(f"  {row['r_pc1']:+.3f}  {row['var']:5.2f}  [{row['questionnaire'][:33]}]  {str(row['item'])[:65]}")

_out.close()
builtins.print("Done -> data/pi_cluster_globalpc.txt  data/pc1_item_correlations.csv")
