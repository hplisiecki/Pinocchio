import pandas as pd, sys, builtins
_out = open('data/ssd_top_items_dump.txt', 'w', encoding='utf-8')
def p(*a, **kw): builtins.print(*a, **kw, file=_out)

for cond in ['neutral', 'llm_analog']:
    s = pd.read_excel(f'data/ssd_results/{cond}_cluster_snippets.xlsx')
    p(f'\n=== {cond} top representative items per cluster ===')
    for label in s['centroid_label'].unique():
        sub = s[s['centroid_label'] == label].sort_values('cosine', ascending=False).head(3)
        p(f'\n  {label}:')
        for _, row in sub.iterrows():
            p(f'    cos={row["cosine"]:.3f}  "{row["snippet_anchor"]}"')

_out.close()
builtins.print("Done")
