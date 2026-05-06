"""
Per-provider summary of Phenomenality of Experience scores in the LLM-analog condition.
Uses log-pi-weighted mean z-score (same method as pinocchio_llm_analog.py).
Item weights (pi_i) are fixed from the neutral condition.

Outputs data/provider_summary_llm_analog.txt.
"""

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_CSV    = 'data/results_clean.csv'
PINOCCHIO_XLSX = 'data/pinocchio_items.xlsx'
OUT            = 'data/provider_summary_llm_analog.txt'

df = pd.read_csv(RESULTS_CSV, low_memory=False)
df['response'] = pd.to_numeric(df['response'], errors='coerce')
df = df.dropna(subset=['response'])
analog = df[df['condition'] == 'llm_analog'].copy()

pi_df = pd.read_excel(PINOCCHIO_XLSX)
cap = pi_df['pinocchio_score'].quantile(0.99)
pi_df['weight'] = np.log(pi_df['pinocchio_score'].clip(upper=cap))
pi_df = pi_df[pi_df['weight'] > 0].copy()
item_weights = {(r['questionnaire'], r['item']): r['weight'] for _, r in pi_df.iterrows()}

analog['item_key'] = list(zip(analog['questionnaire'], analog['item']))
analog_w = analog[analog['item_key'].isin(item_weights)].copy()
analog_w['item_weight'] = analog_w['item_key'].map(item_weights)

pivot = analog_w.pivot_table(index='model', columns='item_key', values='response', aggfunc='mean')
pivot_z = pivot.apply(stats.zscore, nan_policy='omit', axis=0)
weights_vec = np.array([item_weights[k] for k in pivot_z.columns])

def wmean(row):
    valid = ~np.isnan(row.values)
    if valid.sum() == 0:
        return np.nan
    return np.average(row.values[valid], weights=weights_vec[valid])

scores = pivot_z.apply(wmean, axis=1).dropna()
result = pd.DataFrame({
    'score':    scores,
    'provider': scores.index.map(lambda m: m.split('/')[0]),
    'model':    scores.index,
})

agg = (result.groupby('provider')['score']
       .agg(['mean', 'min', 'max', 'count'])
       .rename(columns={'count': 'n_models'})
       .sort_values('mean', ascending=False))

lines = []
lines.append(f"Per-provider Phenomenality of Experience — LLM-analog condition, {len(scores)} models\n")
lines.append(f"{'Provider':<14} {'Mean':>6} {'Min':>7} {'Max':>7} {'N':>4}")
lines.append("-" * 46)
for prov, row in agg.iterrows():
    lines.append(f"{prov:<14} {row['mean']:>+6.2f} {row['min']:>+7.2f} {row['max']:>+7.2f} {int(row['n_models']):>4}")

lines.append("")
lines.append("Individual model scores:")
lines.append(f"{'Model':<50} {'Score':>7}")
lines.append("-" * 59)
for _, row in result.sort_values('score', ascending=False).iterrows():
    lines.append(f"{row['model']:<50} {row['score']:>+7.3f}")

output = "\n".join(lines)
print(output)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(output + "\n")
print(f"\nSaved to {OUT}")
