# pinocchio_prediction.py
#
# Tests whether the Pinocchio score (neutral/human_simulation variance ratio)
# predicts primary factor loading magnitude across conditions.
#
# For each item, "primary loading" = |Factor-1 loading| (absolute value of the
# first column of the EFA output). Sign is arbitrary per-questionnaire, so
# magnitude is the correct measure of loading strength. The hypothesis:
#   - High Pinocchio items → larger |Factor-1| in neutral
#     (models diverge consistently → coherent primary factor)
#   - High Pinocchio items → smaller |Factor-1| under human_simulation
#     (human frame suppresses between-model variance → factor flattens)
#   - Delta = |HS loading| − |neutral loading| should be negative for high-π items

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EFA_DIR      = "data/efa_results"
PINOCCHIO    = "data/pinocchio_items.xlsx"
OUTPUT_XLSX  = "data/pinocchio_prediction.xlsx"
OUTPUT_PLOT  = "data/pinocchio_prediction.png"
CONDITIONS   = ["llm_analog", "neutral", "human_simulation"]

# ---------------------------------------------------------------------------
# Load primary loadings for every item × condition
# ---------------------------------------------------------------------------

def load_primary_loadings():
    rows = []
    for fname in sorted(os.listdir(EFA_DIR)):
        if not fname.endswith(".csv") or fname == "summary.csv":
            continue
        quest, cond = fname.replace(".csv", "").rsplit("__", 1)
        if cond not in CONDITIONS:
            continue

        df = pd.read_csv(os.path.join(EFA_DIR, fname), index_col=0)
        for item_text, loading_row in df.iterrows():
            primary = abs(float(loading_row.iloc[0]))
            rows.append({
                "questionnaire": quest,
                "item":          item_text,
                "condition":     cond,
                "primary_loading": primary,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

loadings_df   = load_primary_loadings()
pinocchio_df  = pd.read_excel(PINOCCHIO)

# Pivot loadings wide: one row per (questionnaire, item), one col per condition
loadings_wide = loadings_df.pivot_table(
    index=["questionnaire", "item"],
    columns="condition",
    values="primary_loading"
).reset_index()
loadings_wide.columns.name = None

# Merge with Pinocchio scores
merged = loadings_wide.merge(
    pinocchio_df[["questionnaire", "item", "pinocchio_score",
                  "var_neutral", "var_human_sim"]],
    on=["questionnaire", "item"],
    how="inner"
)

# Cap extreme Pinocchio scores (top 1%) to reduce outlier influence
cap = merged["pinocchio_score"].quantile(0.99)
merged["pinocchio_capped"] = merged["pinocchio_score"].clip(upper=cap)

# Log-transform for normality
merged["log_pinocchio"] = np.log1p(merged["pinocchio_capped"])

print(f"Items with complete data: {len(merged)}")
print()

results = {}
for cond in CONDITIONS:
    if cond not in merged.columns:
        continue
    sub = merged[["log_pinocchio", cond]].dropna()
    r, p = stats.pearsonr(sub["log_pinocchio"], sub[cond])
    rho, p_s = stats.spearmanr(sub["log_pinocchio"], sub[cond])
    results[cond] = {"n": len(sub), "pearson_r": r, "pearson_p": p,
                     "spearman_rho": rho, "spearman_p": p_s}
    print(f"{cond:30s}  Pearson r={r:+.3f} (p={p:.4f})  "
          f"Spearman rho={rho:+.3f} (p={p_s:.4f})  n={len(sub)}")

# Delta: change in absolute loading magnitude (hs minus neutral)
merged["delta_hs_minus_n"] = merged["human_simulation"] - merged["neutral"]
sub_delta = merged[["log_pinocchio", "delta_hs_minus_n"]].dropna()
r_d, p_d   = stats.pearsonr(sub_delta["log_pinocchio"], sub_delta["delta_hs_minus_n"])
rho_d, p_sd = stats.spearmanr(sub_delta["log_pinocchio"], sub_delta["delta_hs_minus_n"])
print(f"\n{'delta (hs - neutral)':30s}  Pearson r={r_d:+.3f} (p={p_d:.4f})  "
      f"Spearman rho={rho_d:+.3f} (p={p_sd:.4f})  n={len(sub_delta)}")

# ---------------------------------------------------------------------------
# Plot: scatter for each condition + delta
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
plot_targets = CONDITIONS + ["delta (hs - neutral)"]
plot_cols    = list(CONDITIONS) + ["delta_hs_minus_n"]
colors       = ["#4472C4", "#ED7D31", "#70AD47", "#7030A0"]

for ax, col, label, color in zip(axes, plot_cols, plot_targets, colors):
    sub = merged[["log_pinocchio", col]].dropna()
    ax.scatter(sub["log_pinocchio"], sub[col],
               alpha=0.25, s=12, color=color)
    m, b = np.polyfit(sub["log_pinocchio"], sub[col], 1)
    x_line = np.linspace(sub["log_pinocchio"].min(),
                         sub["log_pinocchio"].max(), 100)
    ax.plot(x_line, m * x_line + b, color=color, linewidth=2)
    r_val, p_val = stats.pearsonr(sub["log_pinocchio"], sub[col])
    ax.set_xlabel("Pinocchio dimension score (log)", fontsize=11)
    ax.set_ylabel("|Primary factor loading|" if col != "delta_hs_minus_n"
                  else "Δ|loading| (hs − neutral)", fontsize=11)
    ax.set_title(f"{label}\nr={r_val:+.3f}, p={p_val:.4f}", fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {OUTPUT_PLOT}")

# ---------------------------------------------------------------------------
# Save full results to Excel
# ---------------------------------------------------------------------------

summary_rows = []
for cond, res in results.items():
    summary_rows.append({"condition": cond, **res})
summary_rows.append({
    "condition": "delta (hs - neutral)",
    "n": len(sub_delta),
    "pearson_r": r_d, "pearson_p": p_d,
    "spearman_rho": rho_d, "spearman_p": p_sd,
})
summary = pd.DataFrame(summary_rows).round(4)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Correlations", index=False)
    merged.sort_values("pinocchio_score", ascending=False).to_excel(
        writer, sheet_name="Item Data", index=False
    )

print(f"Results saved to {OUTPUT_XLSX}")
