"""
Dot plot of PC1 (Phenomenality of Experience) with 95% bootstrap CIs.
CIs are obtained by resampling the 45 questionnaires with replacement and
rerunning the full PCA pipeline on each bootstrap sample. Sign and scale are
aligned to the reference solution at each iteration.

Output: data/model_pc1_ranked_ci.png
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EFA_DIR  = 'data/efa_results'
DATA_CSV = 'data/results_clean.csv'
OUT      = 'data/model_pc1_ranked_ci.png'

N_BOOT = 1000
CI     = 95
SEED   = 42
rng    = np.random.default_rng(SEED)

PROVIDER_COLORS = {
    "anthropic":  "#E8843A",
    "openai":     "#10A37F",
    "google":     "#4285F4",
    "meta-llama": "#0668E1",
    "mistralai":  "#FF6B6B",
    "deepseek":   "#9B59B6",
    "qwen":       "#E74C3C",
    "x-ai":       "#1DA1F2",
    "cohere":     "#FF9900",
    "amazon":     "#CC7700",
    "nvidia":     "#76B900",
    "baidu":      "#2932E1",
    "moonshotai": "#00B4D8",
    "minimax":    "#F72585",
    "xiaomi":     "#FF6900",
    "z-ai":       "#6C757D",
}
def prov(m): return m.split("/")[0]
def pcolor(m): return PROVIDER_COLORS.get(prov(m), "#AAAAAA")
def short(m): return m.split("/")[1] if "/" in m else m

# ---------------------------------------------------------------------------
# Load data and precompute per-questionnaire EFA-F1 score vectors
# ---------------------------------------------------------------------------
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

questionnaires = list(quest_scores.keys())
n_quests = len(questionnaires)
print(f"Questionnaires available: {n_quests}")

# ---------------------------------------------------------------------------
# Reference PCA (full sample)
# ---------------------------------------------------------------------------
def run_pca(selected_quests):
    mat = pd.DataFrame({q: quest_scores[q] for q in selected_quests})
    mat = mat.dropna(thresh=int(0.8 * mat.shape[1])).fillna(0)
    X = StandardScaler().fit_transform(mat.values)
    pc1 = PCA(n_components=1).fit_transform(X)
    return pd.Series(pc1[:, 0], index=mat.index)

ref_pc1 = run_pca(questionnaires)
ref_pc1_ranked = ref_pc1.sort_values(ascending=False)
all_models = ref_pc1.index.tolist()

# ---------------------------------------------------------------------------
# Bootstrap: resample questionnaires with replacement
# ---------------------------------------------------------------------------
boot_scores = {m: [] for m in all_models}

for b in range(N_BOOT):
    sample = [questionnaires[i] for i in rng.integers(0, n_quests, n_quests)]
    pc1 = run_pca(sample)
    common = pc1.index.intersection(ref_pc1.index)
    if len(common) < 5:
        continue
    # align sign
    r, _ = pearsonr(pc1.loc[common], ref_pc1.loc[common])
    if r < 0:
        pc1 = -pc1
    # align scale
    ref_mean, ref_std = ref_pc1.loc[common].mean(), ref_pc1.loc[common].std()
    boot_mean, boot_std = pc1.loc[common].mean(), pc1.loc[common].std()
    if boot_std > 1e-9:
        pc1 = (pc1 - boot_mean) / boot_std * ref_std + ref_mean
    for m in all_models:
        if m in pc1.index:
            boot_scores[m].append(pc1[m])

lo_pct = (100 - CI) / 2
hi_pct = 100 - lo_pct
ci_lo = {m: np.percentile(boot_scores[m], lo_pct) for m in all_models}
ci_hi = {m: np.percentile(boot_scores[m], hi_pct) for m in all_models}
print(f"Bootstrap done ({N_BOOT} iterations)")

# ---------------------------------------------------------------------------
# Dot plot
# ---------------------------------------------------------------------------
ranked = ref_pc1_ranked
colors  = [pcolor(m) for m in ranked.index]
y = np.arange(len(ranked))

fig, ax = plt.subplots(figsize=(9, 14))
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

for i, m in enumerate(ranked.index):
    c = pcolor(m)
    ax.hlines(i, ci_lo[m], ci_hi[m], color=c, linewidth=2.2, alpha=0.7)
    ax.scatter(ref_pc1[m], i, color=c, s=50, zorder=4, edgecolors='white', linewidth=0.6)

ax.set_yticks(y)
ax.set_yticklabels([short(m) for m in ranked.index], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Pinocchio Axis (Π, PC1 score)', fontsize=10)
ax.set_title(
    f'Pinocchio Axis (Π) — ranked\n'
    f'(PC1 of {n_quests}-questionnaire EFA Factor-1 matrix; '
    f'lines = {CI}% bootstrap CI, questionnaire resampling)',
    fontsize=10, fontweight='bold')

seen = {}
for m, c in zip(ranked.index, colors):
    lp = prov(m)
    if lp not in seen: seen[lp] = c
handles = [mpatches.Patch(color=c, label=lp) for lp, c in sorted(seen.items())]
ax.legend(handles=handles, loc='lower right', fontsize=7, ncol=2,
          framealpha=0.9, title='Provider')
ax.grid(axis='x', alpha=0.3)
ax.grid(axis='y', alpha=0.15, linewidth=0.6, linestyle='-')
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved to {OUT}")
