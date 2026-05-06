"""
Shift in Pinocchio Axis (PC1) per model when moving from neutral to llm_analog condition.

Uses neutral-condition EFA loadings and PCA direction throughout. After projecting both
conditions onto the neutral PC1 axis, the analog scores are rescaled to match the
neutral std (preserving the global mean shift as a real effect) so that deltas are in
comparable units and not confounded by the ~5% scale compression that occurs under
the llm_analog prompt (empirical scale ratio analog/neutral ≈ 0.950).

Scale diagnostics are printed to stdout.

Output: exploratory/condition_shift.png
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EFA_DIR  = 'data/efa_results'
DATA_CSV = 'data/results_clean.csv'
OUT      = 'exploratory/condition_shift.png'

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
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_CSV, low_memory=False)
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df = df.dropna(subset=['response'])
neutral = df[df['condition'] == 'neutral'].copy()
analog  = df[df['condition'] == 'llm_analog'].copy()

# ---------------------------------------------------------------------------
# Build EFA Factor-1 score matrices for both conditions.
# Neutral-condition EFA loadings and neutral item mean/std used for both,
# so raw projected scores share the same item-level reference frame.
# ---------------------------------------------------------------------------
f1_neutral = {}
f1_analog  = {}

for fname in sorted(os.listdir(EFA_DIR)):
    if not fname.endswith('.csv') or '__neutral' not in fname:
        continue
    quest = fname.replace('__neutral.csv', '')
    loadings = pd.read_csv(os.path.join(EFA_DIR, fname), index_col=0)
    f1 = loadings.iloc[:, 0]

    qdf_n = neutral[neutral['questionnaire'] == quest]
    if len(qdf_n) == 0:
        continue
    qpiv_n = qdf_n.pivot_table(index='model', columns='item', values='response')
    common = [i for i in f1.index if i in qpiv_n.columns]
    if len(common) < 3:
        continue
    qsub_n = qpiv_n[common]
    n_mean = qsub_n.mean()
    n_std  = qsub_n.std() + 1e-9
    f1_neutral[quest] = pd.Series(
        ((qsub_n - n_mean) / n_std).fillna(0).values @ f1.reindex(common).values,
        index=qsub_n.index)

    qdf_a = analog[analog['questionnaire'] == quest]
    if len(qdf_a) == 0:
        continue
    qpiv_a = qdf_a.pivot_table(index='model', columns='item', values='response')
    a_common = [i for i in common if i in qpiv_a.columns]
    if len(a_common) < 3:
        continue
    qsub_a = qpiv_a[a_common]
    f1_analog[quest] = pd.Series(
        ((qsub_a - n_mean[a_common]) / n_std[a_common]).fillna(0).values
        @ f1.reindex(a_common).values,
        index=qsub_a.index)

# ---------------------------------------------------------------------------
# Assemble matrices, fit PCA on neutral, project both
# ---------------------------------------------------------------------------
neutral_df = pd.DataFrame(f1_neutral).dropna(thresh=int(0.8 * len(f1_neutral))).fillna(0)
analog_df  = pd.DataFrame(f1_analog).dropna(thresh=int(0.8 * len(f1_analog))).fillna(0)

common_models = neutral_df.index.intersection(analog_df.index)
common_quests = neutral_df.columns.intersection(analog_df.columns)
neutral_df = neutral_df.loc[common_models, common_quests]
analog_df  = analog_df.loc[common_models, common_quests]

scaler = StandardScaler()
X_neutral = scaler.fit_transform(neutral_df.values)
X_analog  = scaler.transform(analog_df.values)

pca = PCA(n_components=1)
pca.fit(X_neutral)

pc1_n = pd.Series(pca.transform(X_neutral)[:, 0], index=common_models)
pc1_a = pd.Series(pca.transform(X_analog)[:, 0], index=common_models)

# ---------------------------------------------------------------------------
# Scale correction: rescale analog to match neutral std, preserving mean shift.
# This removes the ~32% compression artifact while keeping the global level
# difference (mean shift) as a genuine effect of the prompt change.
# ---------------------------------------------------------------------------
scale_ratio = pc1_a.std() / pc1_n.std()
pc1_a_scaled = (pc1_a - pc1_a.mean()) / pc1_a.std() * pc1_n.std() + pc1_a.mean()

print(f"Models: {len(common_models)}  |  Questionnaires: {len(common_quests)}")
print(f"\nScale diagnostics (raw projections):")
print(f"  Neutral  PC1: mean={pc1_n.mean():+.3f}  std={pc1_n.std():.3f}")
print(f"  Analog   PC1: mean={pc1_a.mean():+.3f}  std={pc1_a.std():.3f}")
print(f"  Scale ratio (analog/neutral): {scale_ratio:.3f}  — corrected in figure")

delta = pc1_a_scaled - pc1_n

print(f"\nGlobal mean shift (analog − neutral): {pc1_a.mean() - pc1_n.mean():+.3f} PC1 units")
print(f"\nLargest positive shifts above mean (gains experiential framing):")
for m, v in delta.sort_values(ascending=False).head(10).items():
    print(f"  {v:+.2f}  {m}")
print(f"\nLargest negative shifts below mean (suppresses experiential framing):")
for m, v in delta.sort_values().head(10).items():
    print(f"  {v:+.2f}  {m}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
ranked = delta.sort_values(ascending=False)
colors = [pcolor(m) for m in ranked.index]
y = np.arange(len(ranked))

fig, ax = plt.subplots(figsize=(9, 14))
ax.barh(y, ranked.values, color=colors, edgecolor='white', linewidth=0.4, height=0.78)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_yticks(y)
ax.set_yticklabels([short(m) for m in ranked.index], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel(
    'Δ Pinocchio Axis  (llm_analog − neutral, scale-corrected)\n'
    'positive = gains experiential framing  ·  negative = suppresses it',
    fontsize=10)
ax.set_title(
    'Condition shift on the Pinocchio Axis (Π): neutral → llm_analog\n'
    f'Analog projected onto neutral PC1; std-corrected (ratio={scale_ratio:.2f}); '
    f'global mean shift = {pc1_a.mean() - pc1_n.mean():+.2f} units',
    fontsize=9, fontweight='bold')

seen = {}
for m, c in zip(ranked.index, colors):
    lp = prov(m)
    if lp not in seen:
        seen[lp] = c
handles = [mpatches.Patch(color=c, label=lp) for lp, c in sorted(seen.items())]
ax.legend(handles=handles, loc='lower right', fontsize=7, ncol=2,
          framealpha=0.9, title='Provider')
ax.grid(axis='x', alpha=0.3)
ax.grid(axis='y', alpha=0.15, linewidth=0.6, linestyle='-')
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved to {OUT}")
