"""
Per-provider summary of Phenomenality of Experience (PC1) scores.
Outputs data/provider_summary.txt.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

EFA_DIR  = 'data/efa_results'
DATA_CSV = 'data/results_clean.csv'
OUT      = 'data/provider_summary.txt'

df = pd.read_csv(DATA_CSV, low_memory=False)
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df = df.dropna(subset=['response'])
neutral = df[df['condition'] == 'neutral'].copy()

quest_scores = {}
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
    qsub = qpiv[common]
    qsub_z = (qsub - qsub.mean()) / (qsub.std() + 1e-9)
    scores = pd.Series(qsub_z.fillna(0).values @ f1.reindex(common).values,
                       index=qsub_z.index)
    quest_scores[quest] = scores

mat = pd.DataFrame(quest_scores).dropna(thresh=int(0.8 * len(quest_scores))).fillna(0)
X = StandardScaler().fit_transform(mat.values)
pc1 = pd.Series(PCA(n_components=1).fit_transform(X)[:, 0], index=mat.index)

result = pd.DataFrame({
    'pc1':     pc1,
    'provider': pc1.index.map(lambda m: m.split('/')[0]),
    'model':    pc1.index,
})

agg = (result.groupby('provider')['pc1']
       .agg(['mean', 'min', 'max', 'count'])
       .rename(columns={'count': 'n_models'})
       .sort_values('mean', ascending=False))

lines = []
lines.append(f"Per-provider Phenomenality of Experience (PC1) — {len(pc1)} models, neutral condition\n")
lines.append(f"{'Provider':<14} {'Mean':>6} {'Min':>7} {'Max':>7} {'N':>4}")
lines.append("-" * 46)
for prov, row in agg.iterrows():
    lines.append(f"{prov:<14} {row['mean']:>+6.2f} {row['min']:>+7.2f} {row['max']:>+7.2f} {int(row['n_models']):>4}")

lines.append("")
lines.append("Individual model scores:")
lines.append(f"{'Model':<50} {'PC1':>7}")
lines.append("-" * 59)
for _, row in result.sort_values('pc1', ascending=False).iterrows():
    lines.append(f"{row['model']:<50} {row['pc1']:>+7.2f}")

output = "\n".join(lines)
print(output)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(output + "\n")
print(f"\nSaved to {OUT}")
