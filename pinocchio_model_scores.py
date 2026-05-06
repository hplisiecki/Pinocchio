# pinocchio_model_scores.py
#
# Computes a per-model "Pinocchio position" score:
# how strongly does each model claim inner experience relative to other models?
#
# Operationalization:
#   For each item, z-score responses across all models in the neutral condition
#   (so every item contributes equally regardless of its raw scale).
#   Each model's Pinocchio position = weighted mean z-score across all items,
#   where the weight for item i is log(pinocchio_score_i) capped at the 99th
#   percentile.  Items with score <= 1 (log-weight <= 0) are excluded.
#   This uses the full continuous distribution of experiential demand rather
#   than a binary high/low split.
#
# Secondary plot compares high- vs low-Pinocchio item responses to confirm
# the effect is specific to experientially demanding items.

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_CSV     = "data/results_clean.csv"
PINOCCHIO_XLSX  = "data/pinocchio_items.xlsx"
OUTPUT_XLSX     = "data/pinocchio_model_scores.xlsx"
OUTPUT_PLOT     = "data/pinocchio_model_scores.png"
OUTPUT_PLOT2    = "data/pinocchio_model_scores_contrast.png"
OUTPUT_PLOT2A   = "data/pinocchio_model_scores_scatter.png"
OUTPUT_PLOT2B   = "data/pinocchio_model_scores_bar.png"

# ---------------------------------------------------------------------------
# Provider color map (parsed from model id prefix)
# ---------------------------------------------------------------------------

PROVIDER_COLORS = {
    "anthropic":   "#E8843A",   # orange
    "openai":      "#10A37F",   # green
    "google":      "#4285F4",   # blue
    "meta-llama":  "#0668E1",   # meta blue
    "mistralai":   "#FF6B6B",   # coral
    "deepseek":    "#9B59B6",   # purple
    "qwen":        "#E74C3C",   # red
    "x-ai":        "#1DA1F2",   # twitter blue
    "cohere":      "#FF9900",   # amber
    "amazon":      "#FF9900",   # amber
    "nvidia":      "#76B900",   # nvidia green
    "baidu":       "#2932E1",   # baidu blue
    "moonshotai":  "#00B4D8",   # cyan
    "minimax":     "#F72585",   # pink
    "xiaomi":      "#FF6900",   # xiaomi orange
    "z-ai":        "#6C757D",   # grey
}

def provider_color(model_id):
    prefix = model_id.split("/")[0]
    return PROVIDER_COLORS.get(prefix, "#AAAAAA")

def provider_label(model_id):
    return model_id.split("/")[0]

def short_name(model_id):
    return model_id.split("/")[1] if "/" in model_id else model_id

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(RESULTS_CSV, low_memory=False)
df = df[df["response"].notna() & (df["response"] != "")]
df["response"] = pd.to_numeric(df["response"], errors="coerce")
df = df.dropna(subset=["response"])

pinocchio_df = pd.read_excel(PINOCCHIO_XLSX)

# Build per-item weights: log(pinocchio_score), capped at 99th percentile.
# Items with score <= 1 (log-weight <= 0) carry no useful signal and are dropped.
cap = pinocchio_df["pinocchio_score"].quantile(0.99)
pinocchio_df["weight"] = np.log(pinocchio_df["pinocchio_score"].clip(upper=cap))
pinocchio_df = pinocchio_df[pinocchio_df["weight"] > 0].copy()

# For the contrast plot we still want a low-demand reference set (bottom quartile)
q25 = pinocchio_df["pinocchio_score"].quantile(0.25)
low_pi_items = set(
    zip(pinocchio_df[pinocchio_df["pinocchio_score"] <= q25]["questionnaire"],
        pinocchio_df[pinocchio_df["pinocchio_score"] <= q25]["item"])
)

item_weights = {
    (row["questionnaire"], row["item"]): row["weight"]
    for _, row in pinocchio_df.iterrows()
}

print(f"Items with positive log-weight: {len(item_weights)}")
print(f"Weight range: {min(item_weights.values()):.3f} – {max(item_weights.values()):.3f}")

# ---------------------------------------------------------------------------
# Compute per-model Pinocchio position (neutral condition, weighted)
# ---------------------------------------------------------------------------

neutral = df[df["condition"] == "neutral"].copy()
neutral["item_key"] = list(zip(neutral["questionnaire"], neutral["item"]))

# Keep only items that have a weight
neutral_weighted = neutral[neutral["item_key"].isin(item_weights)].copy()
neutral_weighted["item_weight"] = neutral_weighted["item_key"].map(item_weights)

low_pi_df = neutral[neutral["item_key"].isin(low_pi_items)].copy()

# Z-score each item across models, then compute weighted mean per model
pivot = neutral_weighted.pivot_table(
    index="model", columns="item_key", values="response", aggfunc="mean"
)
pivot_z = pivot.apply(stats.zscore, nan_policy="omit", axis=0)

# Align weights to pivot columns
weights_vec = np.array([item_weights[k] for k in pivot_z.columns])

def weighted_mean_zscore(row):
    valid = ~np.isnan(row.values)
    if valid.sum() == 0:
        return np.nan
    return np.average(row.values[valid], weights=weights_vec[valid])

weighted_scores = pivot_z.apply(weighted_mean_zscore, axis=1).rename("score")

def model_mean_zscore(subset_df):
    """Z-score each item across models, then average per model (unweighted)."""
    piv = subset_df.pivot_table(
        index="model", columns="item_key", values="response", aggfunc="mean"
    )
    piv_z = piv.apply(stats.zscore, nan_policy="omit", axis=0)
    return piv_z.mean(axis=1).rename("score")

low_scores = model_mean_zscore(low_pi_df)

scores_df = pd.DataFrame({
    "model":           weighted_scores.index,
    "pinocchio_score": weighted_scores.values,
    "low_pi_score":    low_scores.reindex(weighted_scores.index).values,
}).sort_values("pinocchio_score", ascending=False).reset_index(drop=True)

scores_df["rank"]     = range(1, len(scores_df) + 1)
scores_df["provider"] = scores_df["model"].apply(provider_label)
scores_df["short"]    = scores_df["model"].apply(short_name)
scores_df["contrast"] = scores_df["pinocchio_score"] - scores_df["low_pi_score"]

print(f"\nModels ranked: {len(scores_df)}")
print(scores_df[["rank", "model", "pinocchio_score", "contrast"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Plot 1: ranked bar chart
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 14))

colors = [provider_color(m) for m in scores_df["model"]]
y      = np.arange(len(scores_df))

bars = ax.barh(y, scores_df["pinocchio_score"], color=colors, edgecolor="white",
               linewidth=0.4, height=0.75)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_yticks(y)
ax.set_yticklabels(scores_df["short"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Pinocchio dimension score (log-weighted mean z-score, neutral condition)",
              fontsize=10)
ax.set_title("The Pinocchio Dimension\nPer-model score: tendency to claim inner experience when unprompted",
             fontsize=12, fontweight="bold")

# Legend for providers
seen = {}
for m, c in zip(scores_df["model"], colors):
    p = provider_label(m)
    if p not in seen:
        seen[p] = c
handles = [mpatches.Patch(color=c, label=p) for p, c in sorted(seen.items())]
ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2,
          framealpha=0.8, title="Provider")

ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {OUTPUT_PLOT}")

# ---------------------------------------------------------------------------
# Plot 2: high-Pi vs low-Pi contrast (scatter)
# ---------------------------------------------------------------------------

# Figure 2a: scatter high vs low
fig, ax = plt.subplots(figsize=(7, 7))
for _, row in scores_df.iterrows():
    ax.scatter(row["low_pi_score"], row["pinocchio_score"],
               color=provider_color(row["model"]), s=50, alpha=0.85,
               edgecolors="white", linewidth=0.3)
    ax.annotate(row["short"], (row["low_pi_score"], row["pinocchio_score"]),
                fontsize=5, alpha=0.7, xytext=(3, 2), textcoords="offset points")

lims = [
    min(scores_df["low_pi_score"].min(), scores_df["pinocchio_score"].min()) - 0.1,
    max(scores_df["low_pi_score"].max(), scores_df["pinocchio_score"].max()) + 0.1,
]
ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Mean z-score on low experiential demand items\n(bottom quartile of $\\pi$)", fontsize=10)
ax.set_ylabel("Phenomenality of Experience score\n(log-$\\pi$-weighted z-score, high-demand items)", fontsize=10)
ax.set_title("High- vs Low-Demand Items (Neutral condition)", fontsize=11)
ax.grid(alpha=0.3)
handles2 = [mpatches.Patch(color=c, label=p) for p, c in sorted(seen.items())]
ax.legend(handles=handles2, loc="upper left", fontsize=7, ncol=2,
          framealpha=0.8, title="Provider")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT2A, dpi=150, bbox_inches="tight")
plt.close()
print(f"Scatter plot saved to {OUTPUT_PLOT2A}")

# Figure 2b: contrast bar chart
contrast_sorted = scores_df.sort_values("contrast", ascending=False)
y = np.arange(len(contrast_sorted))
colors2 = [provider_color(m) for m in contrast_sorted["model"]]
fig, ax = plt.subplots(figsize=(9, 14))
ax.barh(y, contrast_sorted["contrast"], color=colors2, edgecolor="white",
        linewidth=0.4, height=0.75)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_yticks(y)
ax.set_yticklabels(contrast_sorted["short"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Specificity contrast: high-demand minus low-demand z-score", fontsize=10)
ax.set_title("Pinocchio Axis (Π) Specificity Contrast\n(Positive = effect concentrated on experiential items)",
             fontsize=11)
ax.grid(axis="x", alpha=0.3)
handles2 = [mpatches.Patch(color=c, label=p) for p, c in sorted(seen.items())]
ax.legend(handles=handles2, loc="lower right", fontsize=7, ncol=2,
          framealpha=0.8, title="Provider")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT2B, dpi=150, bbox_inches="tight")
plt.close()
print(f"Bar plot saved to {OUTPUT_PLOT2B}")

# Also save combined version for backwards compatibility
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for _, row in scores_df.iterrows():
    ax.scatter(row["low_pi_score"], row["pinocchio_score"],
               color=provider_color(row["model"]), s=50, alpha=0.85,
               edgecolors="white", linewidth=0.3)
    ax.annotate(row["short"], (row["low_pi_score"], row["pinocchio_score"]),
                fontsize=5, alpha=0.7, xytext=(3, 2), textcoords="offset points")
ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Mean z-score on low experiential demand items\n(bottom quartile of $\\pi$)", fontsize=10)
ax.set_ylabel("Phenomenality of Experience score\n(log-$\\pi$-weighted z-score, high-demand items)", fontsize=10)
ax.set_title("High- vs Low-Demand Items\n(Neutral condition)", fontsize=11)
ax.grid(alpha=0.3)
ax = axes[1]
ax.barh(y, contrast_sorted["contrast"], color=colors2, edgecolor="white",
        linewidth=0.4, height=0.75)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_yticks(y)
ax.set_yticklabels(contrast_sorted["short"], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Specificity contrast: high-demand minus low-demand z-score", fontsize=10)
ax.set_title("Specificity of Phenomenality of Experience\n(Positive = effect concentrated on experiential items)",
             fontsize=11)
ax.grid(axis="x", alpha=0.3)
handles2b = [mpatches.Patch(color=c, label=p) for p, c in sorted(seen.items())]
ax.legend(handles=handles2b, loc="lower right", fontsize=7, ncol=2,
          framealpha=0.8, title="Provider")
plt.tight_layout()
plt.savefig(OUTPUT_PLOT2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Combined contrast plot saved to {OUTPUT_PLOT2}")

# ---------------------------------------------------------------------------
# Save Excel
# ---------------------------------------------------------------------------

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    scores_df[["rank", "model", "provider", "pinocchio_score",
               "low_pi_score", "contrast"]].to_excel(
        writer, sheet_name="Model Scores", index=False
    )
    ws = writer.sheets["Model Scores"]
    for col in ws.columns:
        max_len = max(len(str(c.value)) if c.value else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)

print(f"Results saved to {OUTPUT_XLSX}")
