import pandas as pd
scores = pd.read_excel("data/pinocchio_model_scores.xlsx")
df = pd.read_csv("data/results.csv", low_memory=False)
all_models = set(df["model"].unique())
scored_models = set(scores["model"].unique())
print("Missing from scores:", all_models - scored_models)
