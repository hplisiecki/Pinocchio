"""
Plot per-model positions in the 2D psychometric space (PC1 x PC2).
PC1 = Experiential Depth
PC2 = Cognitive Autonomy (sign flipped: high = spontaneous/exploratory)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EFA_DIR  = 'data/efa_results'
DATA_CSV = 'data/results_clean.csv'
OLD_XLSX = 'data/pinocchio_model_scores.xlsx'
OUTPUT   = 'data/model_psychometric_space.png'
BAR_OUT  = 'data/model_pc1_ranked.png'
BAR_OUT2 = 'data/model_pc2_ranked.png'

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
# Rebuild global PCA
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_CSV, low_memory=False)
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df = df.dropna(subset=['response'])
neutral = df[df['condition'] == 'neutral'].copy()

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

f1_df = pd.DataFrame(f1_matrix).dropna(thresh=int(0.8 * len(f1_matrix))).fillna(0)
X = StandardScaler().fit_transform(f1_df.values)
pca = PCA(n_components=2)
pca.fit(X)
coords = pd.DataFrame(pca.transform(X), index=f1_df.index, columns=['PC1', 'PC2'])
coords['PC2'] = -coords['PC2']   # flip so spontaneous = positive
coords['short'] = [short(m) for m in coords.index]
coords['provider'] = [prov(m) for m in coords.index]

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100

# ---------------------------------------------------------------------------
# Correlation with old log-Pi-weighted score
# ---------------------------------------------------------------------------
old = pd.read_excel(OLD_XLSX, sheet_name='Model Scores').set_index('model')
common_m = coords.index.intersection(old.index)
r_pearson, p_pearson = pearsonr(coords.loc[common_m, 'PC1'], old.loc[common_m, 'pinocchio_score'])
r_spearman, p_spear = spearmanr(coords.loc[common_m, 'PC1'], old.loc[common_m, 'pinocchio_score'])
print(f"PC1 vs log-Pi score: r={r_pearson:.3f} (p={p_pearson:.4g}), rho={r_spearman:.3f} (p={p_spear:.4g})")

# ---------------------------------------------------------------------------
# Determine which labels to show (outliers + notable names)
# Threshold: |PC1| > 4.5 OR |PC2| > 3.0 OR specific interest
# ---------------------------------------------------------------------------
label_mask = (
    (coords['PC1'].abs() > 4.5) |
    (coords['PC2'].abs() > 3.0)
)
label_models = set(coords.index[label_mask])
# Always label a few notable mid-range models
notable = {'anthropic/claude-sonnet-4.6', 'openai/gpt-4o',
           'deepseek/deepseek-r1-0528', 'anthropic/claude-opus-4.7',
           'openai/gpt-5.4'}
label_models |= notable

# ---------------------------------------------------------------------------
# Main scatter plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 9))

ax.axhline(0, color='#cccccc', linewidth=0.7, zorder=0)
ax.axvline(0, color='#cccccc', linewidth=0.7, zorder=0)

# All points
for model, row in coords.iterrows():
    ax.scatter(row['PC1'], row['PC2'],
               color=pcolor(model), s=55, alpha=0.90,
               edgecolors='white', linewidth=0.55, zorder=3)

# Labels only for outliers / notable models
for model, row in coords.iterrows():
    if model not in label_models:
        continue
    name = row['short']
    x, y = row['PC1'], row['PC2']
    # Nudge direction
    ha = 'left' if x > 0 else 'right'
    dx = 0.25 if x > 0 else -0.25
    dy = 0.20
    ax.annotate(name, (x, y), xytext=(x + dx, y + dy),
                fontsize=6.5, color='#111111', alpha=0.9,
                ha=ha, va='bottom',
                arrowprops=dict(arrowstyle='-', color='#aaaaaa',
                                lw=0.6, shrinkA=4, shrinkB=2))

# Axis labels
ax.set_xlabel(
    f"Pinocchio Axis (Π, PC1, {var1:.1f}% variance)\n"
    "◄  behavioral / reactive                                              phenomenally rich  ►",
    fontsize=11, labelpad=8)
ax.set_ylabel(
    f"Phenomenality of Cognition  (PC2, {var2:.1f}% variance)\n"
    "◄  cognition as tool / regulatory                    inner speech as felt experience  ►",
    fontsize=11, labelpad=8)
ax.set_title(
    "LLM Positions in the Psychometric Space\n"
    "Neutral condition  ·  Global EFA Factor-1 PCA  ·  $n=50$ models",
    fontsize=12, fontweight='bold', pad=10)

ax.grid(alpha=0.10, linewidth=0.5)

# Provider legend
seen = {}
for m in coords.index:
    p = prov(m)
    if p not in seen: seen[p] = pcolor(m)
handles = [mpatches.Patch(color=c, label=p) for p, c in sorted(seen.items())]
ax.legend(handles=handles, loc='upper left', fontsize=7, ncol=2,
          framealpha=0.92, title='Provider', title_fontsize=8, borderpad=0.7)

x_pad, y_pad = 1.2, 0.7
ax.set_xlim(coords['PC1'].min() - x_pad, coords['PC1'].max() + x_pad)
ax.set_ylim(coords['PC2'].min() - y_pad, coords['PC2'].max() + y_pad)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=180, bbox_inches='tight')
plt.close()
print(f"Scatter saved to {OUTPUT}")

# ---------------------------------------------------------------------------
# Ranked bar chart of PC1 (Experiential Depth) for Appendix B
# ---------------------------------------------------------------------------
ranked = coords.sort_values('PC1', ascending=False)
fig2, ax2 = plt.subplots(figsize=(9, 14))
colors_bar = [pcolor(m) for m in ranked.index]
y = np.arange(len(ranked))
ax2.barh(y, ranked['PC1'], color=colors_bar, edgecolor='white', linewidth=0.4, height=0.78)
ax2.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax2.set_yticks(y)
ax2.set_yticklabels(ranked['short'], fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel('Pinocchio Axis (Π, PC1 score)', fontsize=10)
ax2.set_title('Pinocchio Axis (Π) — ranked\n(PC1 of the 45-questionnaire EFA Factor-1 matrix)',
              fontsize=11, fontweight='bold')
seen2 = {}
for m, c in zip(ranked.index, colors_bar):
    lp = prov(m)
    if lp not in seen2: seen2[lp] = c
handles2 = [mpatches.Patch(color=c, label=lp) for lp, c in sorted(seen2.items())]
ax2.legend(handles=handles2, loc='lower right', fontsize=7, ncol=2,
           framealpha=0.9, title='Provider')
ax2.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(BAR_OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Bar chart saved to {BAR_OUT}")

# ---------------------------------------------------------------------------
# Ranked bar chart of PC2 (Phenomenality of Cognition) for Appendix B
# ---------------------------------------------------------------------------
ranked2 = coords.sort_values('PC2', ascending=False)
fig3, ax3 = plt.subplots(figsize=(9, 14))
colors_bar2 = [pcolor(m) for m in ranked2.index]
y2 = np.arange(len(ranked2))
ax3.barh(y2, ranked2['PC2'], color=colors_bar2, edgecolor='white', linewidth=0.4, height=0.78)
ax3.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_yticks(y2)
ax3.set_yticklabels(ranked2['short'], fontsize=8)
ax3.invert_yaxis()
ax3.set_xlabel('Phenomenality of Cognition (PC2 score)', fontsize=10)
ax3.set_title('Phenomenality of Cognition — ranked\n(PC2 of the 48-questionnaire EFA Factor-1 matrix)',
              fontsize=11, fontweight='bold')
seen3 = {}
for m, c in zip(ranked2.index, colors_bar2):
    lp = prov(m)
    if lp not in seen3: seen3[lp] = c
handles3 = [mpatches.Patch(color=c, label=lp) for lp, c in sorted(seen3.items())]
ax3.legend(handles=handles3, loc='lower right', fontsize=7, ncol=2,
           framealpha=0.9, title='Provider')
ax3.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(BAR_OUT2, dpi=150, bbox_inches='tight')
plt.close()
print(f"PC2 bar chart saved to {BAR_OUT2}")
