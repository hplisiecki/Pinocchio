import builtins, pandas as pd, numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

out = open("data/paper_numbers.txt", "w", encoding="utf-8")
def p(*a, **kw): builtins.print(*a, **kw, file=out)

# ── Questionnaire count ──────────────────────────────────────────────────────
df = pd.read_csv("data/results_clean.csv", low_memory=False)
_qmeta = pd.read_excel("data/questionnaire_scales.xlsx")
_excl = frozenset(_qmeta[_qmeta["Good"] != 1]["quest"].str.strip())
df = df[~df["questionnaire"].isin(_excl)]
n_quests = df["questionnaire"].nunique()
p(f"Questionnaires in analysis: {n_quests}")

# ── Global PCA (replicate plot_model_psychometric_space logic) ───────────────
EFA_DIR = "data/efa_results"
df["response"] = pd.to_numeric(df["response"], errors="coerce")
df = df.dropna(subset=["response"])
neutral = df[df["condition"] == "neutral"].copy()

f1_matrix = {}
for fname in sorted(os.listdir(EFA_DIR)):
    if not fname.endswith(".csv") or "__neutral" not in fname:
        continue
    quest = fname.replace("__neutral.csv", "")
    loadings = pd.read_csv(os.path.join(EFA_DIR, fname), index_col=0)
    f1 = loadings.iloc[:, 0]
    qdf = neutral[neutral["questionnaire"] == quest]
    if len(qdf) == 0: continue
    qpiv = qdf.pivot_table(index="model", columns="item", values="response")
    common = [i for i in f1.index if i in qpiv.columns]
    if len(common) < 3: continue
    qsub = qpiv[common]
    qsub_z = (qsub - qsub.mean()) / (qsub.std() + 1e-9)
    scores = pd.Series(qsub_z.fillna(0).values @ f1.reindex(common).values,
                       index=qsub_z.index)
    f1_matrix[quest] = scores

f1_df = pd.DataFrame(f1_matrix).dropna(thresh=int(0.8 * len(f1_matrix))).fillna(0)
p(f"f1_df shape (models x questionnaires): {f1_df.shape}")

X = StandardScaler().fit_transform(f1_df.values)
pca = PCA(n_components=5)
pca.fit(X)
var = pca.explained_variance_ratio_
p(f"PC1 variance: {var[0]*100:.1f}%")
p(f"PC2 variance: {var[1]*100:.1f}%")
p(f"PC1+PC2: {(var[0]+var[1])*100:.1f}%")

# ── Pinocchio prediction ─────────────────────────────────────────────────────
pred = pd.read_excel("data/pinocchio_prediction.xlsx")
p(f"\nPinocchio prediction sheet columns: {pred.columns.tolist()}")
p(pred.head(3).to_string())

# ── Pi prediction key stats ──────────────────────────────────────────────────
pi = pd.read_excel("data/pinocchio_items.xlsx")
p(f"\nTotal Pi items: {len(pi)}")

# neutral F1 loading ~ pi_score correlation
for cond_tag in ["neutral", "human_sim", "delta"]:
    col = f"f1_{cond_tag}" if f"f1_{cond_tag}" in pred.columns else None
    if col and "pinocchio_score" in pred.columns:
        r, rp = pearsonr(pred["pinocchio_score"], pred[col])
        rs, rsp = spearmanr(pred["pinocchio_score"], pred[col])
        p(f"Pi ~ f1_{cond_tag}: r={r:.3f} p={rp:.4f}  rho={rs:.3f} p={rsp:.4f}  n={len(pred)}")

# ── Cluster-PC correlations ──────────────────────────────────────────────────
p("\n--- pi_cluster_globalpc.txt (first 120 lines) ---")
with open("data/pi_cluster_globalpc.txt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 120: p(line, end="")

# ── Congruence ───────────────────────────────────────────────────────────────
cong = pd.read_csv("data/congruence_results.csv")
p("\n--- Congruence summary ---")
p(f"phi(la,n):  mean={cong['phi_la_n'].mean():.3f}  median={cong['phi_la_n'].median():.3f}  n={cong['phi_la_n'].notna().sum()}")
p(f"phi(hs,la): mean={cong['phi_hs_la'].mean():.3f}  median={cong['phi_hs_la'].median():.3f}")
p(f"phi(hs,n):  mean={cong['phi_hs_n'].mean():.3f}  median={cong['phi_hs_n'].median():.3f}")
pct = (cong["phi_la_n"] > cong["phi_hs_la"]).mean() * 100
p(f"phi(la,n) > phi(hs,la): {pct:.1f}%")
delta = (cong["phi_la_n"] - cong["phi_hs_la"]).dropna()
p(f"Delta phi mean={delta.mean():.3f}  median={delta.median():.3f}")

# ── llm_analog replication ───────────────────────────────────────────────────
p("\n--- pinocchio_llm_analog_replication.txt ---")
with open("data/pinocchio_llm_analog_replication.txt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 80: p(line, end="")

# ── llm_analog Spearman with neutral ────────────────────────────────────────
log = open("data/pinocchio_llm_analog_log.txt", encoding="utf-8").read()
p("\n--- llm_analog log ---")
p(log[:600])

out.close()
print("Done → data/paper_numbers.txt")
