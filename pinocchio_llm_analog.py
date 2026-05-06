"""
Robustness check: repeat the Pinocchio model-scoring analysis using the
llm_analog condition instead of neutral.

Outputs:
  data/pinocchio_llm_analog.png   — ranked bar chart (llm_analog scores)
  data/pinocchio_llm_analog.xlsx  — scores + rank comparison table
  data/pinocchio_llm_analog_log.txt — Spearman rho and rank table
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_CSV    = "data/results_clean.csv"
PINOCCHIO_XLSX = "data/pinocchio_items.xlsx"
NEUTRAL_XLSX   = "data/pinocchio_model_scores.xlsx"
OUTPUT_PNG     = "data/pinocchio_llm_analog.png"
OUTPUT_XLSX    = "data/pinocchio_llm_analog.xlsx"
LOG_FILE       = "data/pinocchio_llm_analog_log.txt"

PROVIDER_COLORS = {
    "anthropic":   "#E8843A",
    "openai":      "#10A37F",
    "google":      "#4285F4",
    "meta-llama":  "#0668E1",
    "mistralai":   "#FF6B6B",
    "deepseek":    "#9B59B6",
    "qwen":        "#E74C3C",
    "x-ai":        "#1DA1F2",
    "cohere":      "#FF9900",
    "amazon":      "#FF9900",
    "nvidia":      "#76B900",
    "baidu":       "#2932E1",
    "moonshotai":  "#00B4D8",
    "minimax":     "#F72585",
    "xiaomi":      "#FF6900",
    "z-ai":        "#6C757D",
}

def provider_color(model_id): return PROVIDER_COLORS.get(model_id.split("/")[0], "#AAAAAA")
def provider_label(model_id): return model_id.split("/")[0]
def short_name(model_id): return model_id.split("/")[1] if "/" in model_id else model_id

# ---------------------------------------------------------------------------
# Load data and item weights
# ---------------------------------------------------------------------------
df = pd.read_csv(RESULTS_CSV, low_memory=False)
df = df[df["response"].notna() & (df["response"] != "")]
df["response"] = pd.to_numeric(df["response"], errors="coerce")
df = df.dropna(subset=["response"])

pi_df = pd.read_excel(PINOCCHIO_XLSX)
cap = pi_df["pinocchio_score"].quantile(0.99)
pi_df["weight"] = np.log(pi_df["pinocchio_score"].clip(upper=cap))
pi_df = pi_df[pi_df["weight"] > 0].copy()
item_weights = {(r["questionnaire"], r["item"]): r["weight"] for _, r in pi_df.iterrows()}

# ---------------------------------------------------------------------------
# Score models in llm_analog condition
# ---------------------------------------------------------------------------
analog = df[df["condition"] == "llm_analog"].copy()
analog["item_key"] = list(zip(analog["questionnaire"], analog["item"]))
analog_w = analog[analog["item_key"].isin(item_weights)].copy()
analog_w["item_weight"] = analog_w["item_key"].map(item_weights)

pivot = analog_w.pivot_table(index="model", columns="item_key", values="response", aggfunc="mean")
pivot_z = pivot.apply(stats.zscore, nan_policy="omit", axis=0)
weights_vec = np.array([item_weights[k] for k in pivot_z.columns])

def weighted_mean_zscore(row):
    valid = ~np.isnan(row.values)
    if valid.sum() == 0: return np.nan
    return np.average(row.values[valid], weights=weights_vec[valid])

analog_scores = pivot_z.apply(weighted_mean_zscore, axis=1).rename("analog_score")
analog_df = analog_scores.reset_index().sort_values("analog_score", ascending=False).reset_index(drop=True)
analog_df["analog_rank"] = range(1, len(analog_df) + 1)
analog_df["provider"] = analog_df["model"].apply(provider_label)
analog_df["short"] = analog_df["model"].apply(short_name)

# ---------------------------------------------------------------------------
# Compare with neutral scores
# ---------------------------------------------------------------------------
neutral_df = pd.read_excel(NEUTRAL_XLSX, sheet_name="Model Scores")
merged = analog_df.merge(
    neutral_df[["model", "pinocchio_score", "rank"]].rename(
        columns={"pinocchio_score": "neutral_score", "rank": "neutral_rank"}),
    on="model", how="inner"
)
rho, pval = spearmanr(merged["neutral_rank"], merged["analog_rank"])

# ---------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------
with open(LOG_FILE, "w", encoding="utf-8") as f:
    def p(*a): print(*a, file=f)
    p(f"Models scored in llm_analog condition: {len(analog_df)}")
    p(f"Overlap with neutral scores: {len(merged)}")
    p(f"\nSpearman rho (neutral rank vs llm_analog rank) = {rho:.3f}, p = {pval:.4g}")
    p(f"\n{'Rank (analog)':>14s}  {'Rank (neutral)':>15s}  {'Model':<50s}  {'Analog':>8s}  {'Neutral':>8s}")
    p("-"*100)
    for _, row in merged.sort_values("analog_rank").iterrows():
        p(f"{row['analog_rank']:>14.0f}  {row['neutral_rank']:>15.0f}  {row['model']:<50s}  {row['analog_score']:>+8.3f}  {row['neutral_score']:>+8.3f}")

print(f"Spearman rho = {rho:.3f} (p = {pval:.4g})")

# ---------------------------------------------------------------------------
# Plot: ranked bar chart (llm_analog)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 14))
colors = [provider_color(m) for m in analog_df["model"]]
y = np.arange(len(analog_df))
ax.barh(y, analog_df["analog_score"], color=colors, edgecolor="white", linewidth=0.4, height=0.75)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_yticks(y)
ax.set_yticklabels(analog_df["short"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Pinocchio dimension score (log-weighted mean z-score, llm_analog condition)", fontsize=10)
ax.set_title("The Pinocchio Dimension — LLM-Analog Condition\n"
             "Robustness check: per-model score when responding as an AI agent",
             fontsize=12, fontweight="bold")
seen = {}
for m, c in zip(analog_df["model"], colors):
    p2 = provider_label(m)
    if p2 not in seen: seen[p2] = c
handles = [mpatches.Patch(color=c, label=p2) for p2, c in sorted(seen.items())]
ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2, framealpha=0.8, title="Provider")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved to {OUTPUT_PNG}")

# ---------------------------------------------------------------------------
# Save Excel
# ---------------------------------------------------------------------------
merged_out = merged.sort_values("analog_rank")[
    ["analog_rank", "neutral_rank", "model", "provider", "analog_score", "neutral_score"]
]
merged_out["rank_diff"] = (merged_out["analog_rank"] - merged_out["neutral_rank"]).astype(int)
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    merged_out.to_excel(writer, sheet_name="Scores", index=False)
print(f"Results saved to {OUTPUT_XLSX}")
