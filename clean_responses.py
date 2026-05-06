"""
clean_responses.py
──────────────────
Identifies and nulls out-of-range questionnaire responses.

Outputs
-------
data/results_clean.csv   — cleaned response data (OOB values set to NaN)
data/oob_responses.csv   — one row per nulled response; columns:
                           questionnaire, model, condition, item_index,
                           item, response (original value), response_raw,
                           scale_min, scale_max
data/clean_log.txt       — human-readable summary for the paper methods section

Run once; downstream analyses should use results_clean.csv.
"""

import json, re, sys, builtins
import pandas as pd
import numpy as np

# ── tee: write to both stdout and a log file ──────────────────────────────
_log = open('data/clean_log.txt', 'w', encoding='utf-8')

def log(*args, **kwargs):
    builtins.print(*args, **kwargs)
    builtins.print(*args, **kwargs, file=_log)

# ── 1. Parse valid scale bounds from questionnaires.json ─────────────────
with open('data/questionnaires.json', encoding='utf-8') as f:
    questionnaires = json.load(f)

def parse_bounds(scale_text: str):
    """
    Return (lo, hi) integers by finding all 'N =' or 'N:' style anchors,
    then taking min/max.  Falls back to the two most extreme standalone
    integers in the string.
    """
    # primary: numbers immediately before = or :
    anchors = [int(x) for x in re.findall(r'(\d+)\s*[=:]', scale_text)]
    if len(anchors) >= 2:
        return min(anchors), max(anchors)
    # fallback: all standalone integers
    nums = [int(x) for x in re.findall(r'\b(\d+)\b', scale_text)]
    if len(nums) >= 2:
        return min(nums), max(nums)
    return None, None

scale_bounds = {}
for q in questionnaires:
    lo, hi = parse_bounds(q.get('scale_prompt', ''))
    scale_bounds[q['quest']] = (lo, hi)

# Print for inspection / paper appendix
log("Scale bounds parsed from questionnaires.json:")
for name, (lo, hi) in sorted(scale_bounds.items()):
    flag = '  *** COULD NOT PARSE ***' if lo is None else ''
    log(f"  {name:<50s}  [{lo}, {hi}]{flag}")

missing = [k for k, v in scale_bounds.items() if v[0] is None]
if missing:
    log(f"\nWARNING: could not parse bounds for {len(missing)} questionnaire(s): {missing}")

# ── 2. Load results and identify OOB rows ────────────────────────────────
results = pd.read_csv('data/results.csv')
log(f"\nLoaded results.csv: {len(results):,} rows, "
      f"{results['response'].notna().sum():,} numeric responses")

def is_oob(row):
    if pd.isna(row['response']):
        return False
    lo, hi = scale_bounds.get(row['questionnaire'], (None, None))
    if lo is None:
        return False          # cannot judge — leave as-is
    return not (lo <= row['response'] <= hi)

oob_mask = results.apply(is_oob, axis=1)
oob_df = results[oob_mask].copy()

log(f"\nOut-of-range responses identified: {oob_mask.sum()} "
      f"({oob_mask.sum() / results['response'].notna().sum() * 100:.4f}% of valid responses)")

# ── 3. Build the trace log ────────────────────────────────────────────────
trace = oob_df[['questionnaire', 'model', 'condition',
                 'item_index', 'item', 'response', 'response_raw']].copy()
trace['scale_min'] = trace['questionnaire'].map(lambda q: scale_bounds.get(q, (None,None))[0])
trace['scale_max'] = trace['questionnaire'].map(lambda q: scale_bounds.get(q, (None,None))[1])
trace = trace.sort_values('response', ascending=False).reset_index(drop=True)
trace.to_csv('data/oob_responses.csv', index=False, encoding='utf-8')
log(f"\nTrace log written to data/oob_responses.csv ({len(trace)} rows)")

# ── 4. Print summary for the paper ───────────────────────────────────────
log("\n─── Summary for methods section ───────────────────────────────────")
log(f"  Total responses parsed:          {results['response'].notna().sum():>7,}")
log(f"  Out-of-range (nulled):           {oob_mask.sum():>7,}  "
      f"({oob_mask.sum()/results['response'].notna().sum()*100:.4f}%)")
log(f"  Remaining after cleaning:        "
      f"{results['response'].notna().sum() - oob_mask.sum():>7,}")

log("\n  By condition:")
for cond, grp in results[results['response'].notna()].groupby('condition'):
    n = oob_mask[grp.index].sum()
    log(f"    {cond:<22s}  {n:>3d} / {len(grp):>6,}  ({n/len(grp)*100:.4f}%)")

log("\n  By model (only those with ≥1 OOB):")
log(f"    {'Model':<50s}  {'OOB':>4s}  {'total':>6s}  {'%':>6s}  max_val")
for mod, grp in results[results['response'].notna()].groupby('model'):
    n = oob_mask[grp.index].sum()
    if n == 0:
        continue
    max_val = grp.loc[oob_mask[grp.index], 'response'].max()
    short = mod.split('/')[-1]
    log(f"    {short:<50s}  {n:>4d}  {len(grp):>6,}  "
          f"{n/len(grp)*100:>6.3f}%  {max_val:.0f}")

log("\n  Full OOB listing (sorted by magnitude):")
log(f"    {'Model':<35s}  {'Cond':<16s}  {'Quest':<32s}  "
      f"{'val':>5s}  {'[lo,hi]':>8s}  item")
for _, r in trace.iterrows():
    lo = r['scale_min']; hi = r['scale_max']
    short = r['model'].split('/')[-1]
    log(f"    {short:<35s}  {r['condition']:<16s}  "
          f"{r['questionnaire'][:32]:<32s}  {r['response']:>5.0f}  "
          f"[{lo:.0f},{hi:.0f}]  {str(r['item'])[:55]}")

# ── 5. Null out OOB values and write clean CSV ────────────────────────────
results_clean = results.copy()
results_clean.loc[oob_mask, 'response'] = np.nan
results_clean.to_csv('data/results_clean.csv', index=False, encoding='utf-8')

# Verify
n_clean = results_clean['response'].notna().sum()
log(f"\nCleaned data written to data/results_clean.csv")
log(f"  Numeric responses before: {results['response'].notna().sum():,}")
log(f"  Numeric responses after:  {n_clean:,}  "
      f"(difference: {results['response'].notna().sum() - n_clean})")

log("\nLog written to data/clean_log.txt")
_log.close()
