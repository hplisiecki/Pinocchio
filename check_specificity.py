"""
Verifies the Phenomenality of Experience axis is not a valence artefact:
splits questionnaires by PC1 loading direction and compares mean item variance
(neutral condition). Outputs data/check_specificity.txt.
"""
import os, builtins
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_out = open('data/check_specificity.txt', 'w', encoding='utf-8')
def p(*a, **kw): builtins.print(*a, **kw, file=_out)

EFA_DIR  = 'data/efa_results'
DATA_CSV = 'data/results_clean.csv'

df = pd.read_csv(DATA_CSV, low_memory=False)
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df = df.dropna(subset=['response'])
neutral = df[df['condition'] == 'neutral'].copy()

# ---- rebuild global PC1 scores ----
f1_matrix = {}
for fname in sorted(os.listdir(EFA_DIR)):
    if not fname.endswith('.csv') or '__neutral' not in fname:
        continue
    quest = fname.replace('__neutral.csv', '')
    loadings = pd.read_csv(os.path.join(EFA_DIR, fname), index_col=0)
    f1 = loadings.iloc[:, 0]
    qdf = neutral[neutral['questionnaire'] == quest]
    if len(qdf) == 0: continue
    qpiv = qdf.pivot_table(index='model', columns='item', values='response')
    common = [i for i in f1.index if i in qpiv.columns]
    if len(common) < 3: continue
    qsub = qpiv[common]
    qsub_z = (qsub - qsub.mean()) / (qsub.std() + 1e-9)
    scores = pd.Series(qsub_z.fillna(0).values @ f1.reindex(common).values,
                       index=qsub_z.index)
    f1_matrix[quest] = scores

f1_df = pd.DataFrame(f1_matrix).dropna(thresh=int(0.8*48)).fillna(0)
X = StandardScaler().fit_transform(f1_df.values)
pca = PCA(n_components=2)
pca.fit(X)
loadings_df = pd.DataFrame(pca.components_.T, index=f1_df.columns,
                             columns=['PC1','PC2'])

# ---- Part 1: inter-model variance per item ----
p("=== Inter-model response variance per item (neutral condition) ===")
item_var = (neutral.groupby(['questionnaire','item'])['response']
            .var().reset_index().rename(columns={'response':'var'}))

pos_quests = loadings_df[loadings_df['PC1'] > 0.10].index.tolist()
neg_quests = loadings_df[loadings_df['PC1'] < -0.10].index.tolist()

pos_var = item_var[item_var['questionnaire'].isin(pos_quests)]['var'].mean()
neg_var = item_var[item_var['questionnaire'].isin(neg_quests)]['var'].mean()

p(f"Positive-PC1 questionnaires (n={len(pos_quests)}): mean item variance = {pos_var:.4f}")
p(f"Negative-PC1 questionnaires (n={len(neg_quests)}): mean item variance = {neg_var:.4f}")
p(f"Grand mean: {item_var['var'].mean():.4f}")


_out.close()
builtins.print("Done")
