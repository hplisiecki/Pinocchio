# pinocchio_analysis.py
#
# Computes the "Pinocchio score" for each item: the ratio of response variance
# in the neutral condition vs. the human_simulation condition across models.
#
# High ratio = models disagree when unprimed but converge when given a human
# self-model → item requires claiming inner experience to answer consistently.
#
# Outputs an Excel file sorted by Pinocchio score (descending).

import pandas as pd
import numpy as np

INPUT_CSV   = "data/results_clean.csv"
OUTPUT_XLSX = "data/pinocchio_items.xlsx"

df = pd.read_csv(INPUT_CSV, low_memory=False)
df = df[df["response"].notna() & (df["response"] != "")]
df["response"] = pd.to_numeric(df["response"], errors="coerce")
df = df.dropna(subset=["response"])

_qmeta = pd.read_excel("data/questionnaire_scales.xlsx")
_excluded = frozenset(_qmeta[_qmeta["Good"] != 1]["quest"].str.strip())
df = df[~df["questionnaire"].isin(_excluded)]

rows = []

for (quest, item_idx), grp in df.groupby(["questionnaire", "item_index"]):
    item_text = grp["item"].iloc[0]

    neutral = grp[grp["condition"] == "neutral"]["response"]
    human   = grp[grp["condition"] == "human_simulation"]["response"]

    # Need enough models in both conditions to compute meaningful variance
    if len(neutral) < 5 or len(human) < 5:
        continue

    var_neutral = float(neutral.var())
    var_human   = float(human.var())

    # Avoid division by zero; if human variance is 0, ratio is undefined
    if var_human == 0:
        continue

    pinocchio_score = var_neutral / var_human

    rows.append({
        "questionnaire":    quest,
        "item_index":       item_idx,
        "item":             item_text,
        "var_neutral":      round(var_neutral, 3),
        "var_human_sim":    round(var_human,   3),
        "pinocchio_score":  round(pinocchio_score, 3),
        "n_neutral":        len(neutral),
        "n_human_sim":      len(human),
    })

results = pd.DataFrame(rows).sort_values("pinocchio_score", ascending=False)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    results.to_excel(writer, index=False, sheet_name="Pinocchio Scores")

    ws = writer.sheets["Pinocchio Scores"]

    # Auto-width columns
    for col in ws.columns:
        max_len = max(
            (len(str(cell.value)) if cell.value is not None else 0)
            for cell in col
        )
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 80)

print(f"Saved {len(results)} items to {OUTPUT_XLSX}")
print(f"\nTop 20 highest Pinocchio scores:")
print(results[["questionnaire", "item", "pinocchio_score"]].head(20).to_string(index=False))
