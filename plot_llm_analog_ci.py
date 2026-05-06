"""
Dot plot of LLM-analog Phenomenality of Experience scores with 95% bootstrap CIs.
Mirrors pinocchio_llm_analog.py but adds questionnaire-resampling bootstrap CIs.
Item weights (pi_i) are fixed from the neutral condition throughout.

Output: data/pinocchio_llm_analog_ci.png
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_CSV    = "data/results_clean.csv"
PINOCCHIO_XLSX = "data/pinocchio_items.xlsx"
OUT            = "data/pinocchio_llm_analog_ci.png"

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
# Load data and item weights
# ---------------------------------------------------------------------------
df = pd.read_csv(RESULTS_CSV, low_memory=False)
df["response"] = pd.to_numeric(df["response"], errors="coerce")
df = df.dropna(subset=["response"])
analog = df[df["condition"] == "llm_analog"].copy()

pi_df = pd.read_excel(PINOCCHIO_XLSX)
cap = pi_df["pinocchio_score"].quantile(0.99)
pi_df["weight"] = np.log(pi_df["pinocchio_score"].clip(upper=cap))
pi_df = pi_df[pi_df["weight"] > 0].copy()

# Group items by questionnaire for bootstrap resampling
questionnaires = pi_df["questionnaire"].unique().tolist()
n_quests = len(questionnaires)
print(f"Questionnaires with weighted items: {n_quests}")

# ---------------------------------------------------------------------------
# Scoring function: given a list of (possibly repeated) questionnaires,
# compute log-pi-weighted mean z-score per model
# ---------------------------------------------------------------------------
def score_models(selected_quests):
    subset = pi_df[pi_df["questionnaire"].isin(set(selected_quests))].copy()
    # count duplicates by repeating rows proportionally
    counts = pd.Series(selected_quests).value_counts()
    rows = []
    for q, cnt in counts.items():
        rows.append(pd.concat([pi_df[pi_df["questionnaire"] == q]] * cnt))
    subset = pd.concat(rows, ignore_index=True)

    item_weights = {(r["questionnaire"], r["item"]): r["weight"]
                    for _, r in subset.iterrows()}

    analog["item_key"] = list(zip(analog["questionnaire"], analog["item"]))
    analog_w = analog[analog["item_key"].isin(item_weights)].copy()
    analog_w["item_weight"] = analog_w["item_key"].map(item_weights)

    pivot = analog_w.pivot_table(index="model", columns="item_key",
                                 values="response", aggfunc="mean")
    pivot_z = pivot.apply(stats.zscore, nan_policy="omit", axis=0)
    weights_vec = np.array([item_weights[k] for k in pivot_z.columns])

    def wmean(row):
        valid = ~np.isnan(row.values)
        if valid.sum() == 0:
            return np.nan
        return np.average(row.values[valid], weights=weights_vec[valid])

    return pivot_z.apply(wmean, axis=1)

# ---------------------------------------------------------------------------
# Reference scores (full set of questionnaires)
# ---------------------------------------------------------------------------
ref_scores = score_models(questionnaires).dropna()
ref_ranked = ref_scores.sort_values(ascending=False)
all_models = ref_scores.index.tolist()
print(f"Models scored: {len(all_models)}")

# ---------------------------------------------------------------------------
# Bootstrap: resample questionnaires with replacement
# ---------------------------------------------------------------------------
boot_scores = {m: [] for m in all_models}

for b in range(N_BOOT):
    sample = [questionnaires[i] for i in rng.integers(0, n_quests, n_quests)]
    scores = score_models(sample).dropna()
    common = scores.index.intersection(ref_scores.index)
    if len(common) < 5:
        continue
    # align scale (direction is fixed — weights are always positive)
    ref_mean, ref_std = ref_scores.loc[common].mean(), ref_scores.loc[common].std()
    boot_mean, boot_std = scores.loc[common].mean(), scores.loc[common].std()
    if boot_std > 1e-9:
        scores = (scores - boot_mean) / boot_std * ref_std + ref_mean
    for m in all_models:
        if m in scores.index:
            boot_scores[m].append(scores[m])

lo_pct = (100 - CI) / 2
hi_pct = 100 - lo_pct
ci_lo = {m: np.percentile(boot_scores[m], lo_pct) for m in all_models}
ci_hi = {m: np.percentile(boot_scores[m], hi_pct) for m in all_models}
print(f"Bootstrap done ({N_BOOT} iterations)")

# ---------------------------------------------------------------------------
# Dot plot
# ---------------------------------------------------------------------------
ranked = ref_ranked
colors = [pcolor(m) for m in ranked.index]
y = np.arange(len(ranked))

fig, ax = plt.subplots(figsize=(9, 14))
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

for i, m in enumerate(ranked.index):
    c = pcolor(m)
    ax.hlines(i, ci_lo[m], ci_hi[m], color=c, linewidth=2.2, alpha=0.7)
    ax.scatter(ref_scores[m], i, color=c, s=50, zorder=4,
               edgecolors='white', linewidth=0.6)

ax.set_yticks(y)
ax.set_yticklabels([short(m) for m in ranked.index], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Phenomenality of Experience score (log-$\\pi$-weighted z-score, llm_analog)', fontsize=10)
ax.set_title(
    f'Pinocchio Axis (Π) — LLM-Analog Condition\n'
    f'(log-$\\pi$-weighted z-score; lines = {CI}% bootstrap CI, questionnaire resampling)',
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
