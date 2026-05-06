import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples

df = pd.read_csv("data/results_clean.csv", low_memory=False)
df["response"] = pd.to_numeric(df["response"], errors="coerce")
df = df.dropna(subset=["response"])
_qmeta = pd.read_excel("data/questionnaire_scales.xlsx")
_excl = frozenset(_qmeta[_qmeta["Good"] != 1]["quest"].str.strip())
df = df[~df["questionnaire"].isin(_excl)]
neutral = df[df["condition"] == "neutral"].copy()

pi = pd.read_excel("data/pinocchio_items.xlsx").sort_values("pinocchio_score", ascending=False)
top_pi = pi.head(100)

records = []
for _, row in top_pi.iterrows():
    q, idx = row["questionnaire"], row["item_index"]
    sub = neutral[(neutral["questionnaire"] == q) & (neutral["item_index"] == idx)]
    if len(sub) < 5: continue
    sub = sub.set_index("model")["response"]
    mu, sd = sub.mean(), sub.std()
    if sd == 0: continue
    for model, val in ((sub - mu) / sd).items():
        records.append({"model": model, "q": q, "idx": idx,
                        "pi": row["pinocchio_score"], "z": val})

item_df = pd.DataFrame(records)
pivot = item_df.pivot_table(index="model", columns=["q", "idx"], values="z").fillna(0)

X = pivot.T.values  # items × models
D_condensed = pdist(X, metric="correlation")
D_sq = squareform(D_condensed)
Z = linkage(D_condensed, method="ward")

n_items = X.shape[0]
print(f"Items in matrix: {n_items}")

ks = range(2, 11)
avg_sil  = []
per_item = {}

for k in ks:
    labels = fcluster(Z, t=k, criterion="maxclust")
    sil = silhouette_score(D_sq, labels, metric="precomputed")
    per_item[k] = silhouette_samples(D_sq, labels, metric="precomputed")
    avg_sil.append(sil)
    sizes = {c: int((labels == c).sum()) for c in sorted(set(labels))}
    print(f"  k={k}: avg silhouette={sil:.4f}  sizes={sizes}")

best_k = list(ks)[int(np.argmax(avg_sil))]
print(f"\nBest k: {best_k}")

# ── Figure: silhouette curve only ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(ks), avg_sil, "o-", color="#2c7bb6", linewidth=2, markersize=8)
ax.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"best k={best_k}")
for k, s in zip(ks, avg_sil):
    ax.annotate(f"{s:.3f}", (k, s), textcoords="offset points",
                xytext=(5, 4), fontsize=9)
ax.set_xlabel("k", fontsize=12)
ax.set_ylabel("Avg silhouette", fontsize=12)
ax.set_title("Average silhouette vs k\n(top-80 π items, Ward + correlation distance)", fontsize=11)
ax.set_xticks(list(ks))
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/silhouette_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved data/silhouette_plot.png")

# ── Per-cluster silhouette detail for each k ─────────────────────────────────
items_meta = top_pi[["questionnaire", "item_index", "item", "pinocchio_score"]].reset_index(drop=True)

for k in ks:
    labels = fcluster(Z, t=k, criterion="maxclust")
    sil_vals = per_item[k]
    meta = items_meta.head(len(labels)).copy()
    meta["cluster"]    = labels
    meta["silhouette"] = sil_vals
    print(f"\n{'='*60}\nk={k}  (avg={avg_sil[k-2]:.4f})\n{'='*60}")
    for c in sorted(set(labels)):
        sub = meta[meta["cluster"] == c].sort_values("silhouette")
        neg = (sub["silhouette"] < 0).sum()
        print(f"\n  C{c} (n={len(sub)}, mean={sub['silhouette'].mean():.3f}, n_neg={neg})")
        print(f"  Weakest:")
        for _, row in sub.head(2).iterrows():
            print(f"    {row['silhouette']:+.3f}  [{row['questionnaire']}] {str(row['item'])[:65]}")
        print(f"  Strongest:")
        for _, row in sub.tail(2).iterrows():
            print(f"    {row['silhouette']:+.3f}  [{row['questionnaire']}] {str(row['item'])[:65]}")
