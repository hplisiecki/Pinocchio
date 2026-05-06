import os
import pandas as pd

EFA_DIR    = "data/efa_results"
OUTPUT_DIR = "data/loadings"
CONDITIONS = ["llm_analog", "neutral", "human_simulation"]

for condition in CONDITIONS:
    os.makedirs(os.path.join(OUTPUT_DIR, condition), exist_ok=True)

for fname in sorted(os.listdir(EFA_DIR)):
    if not fname.endswith(".csv") or fname == "summary.csv":
        continue

    # filename format: {questionnaire}__{condition}.csv
    name_part, cond_part = fname.replace(".csv", "").rsplit("__", 1)

    if cond_part not in CONDITIONS:
        continue

    df = pd.read_csv(os.path.join(EFA_DIR, fname))

    # First column is the item text (unnamed in the CSV, becomes index col 0)
    df = df.rename(columns={df.columns[0]: "Item"})

    out_path = os.path.join(OUTPUT_DIR, cond_part, f"{name_part}.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Loadings")

        # Auto-width columns
        ws = writer.sheets["Loadings"]
        for col in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 80)

print("Done.")
for cond in CONDITIONS:
    n = len(os.listdir(os.path.join(OUTPUT_DIR, cond)))
    print(f"  {cond}: {n} files")
