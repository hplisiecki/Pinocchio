import pandas as pd, sys, builtins
_out = open('data/ssd_clusters_dump.txt', 'w', encoding='utf-8')
def p(*a, **kw): builtins.print(*a, **kw, file=_out)

for cond in ['neutral', 'llm_analog', 'human_simulation']:
    c = pd.read_excel(f'data/ssd_results/{cond}_clusters.xlsx')
    p(f'\n=== {cond} ===')
    p(c.to_string())

_out.close()
builtins.print("Done")
