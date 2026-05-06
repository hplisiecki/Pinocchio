import pandas as pd, sys, builtins
_out = open('data/ssd_snippets_dump.txt', 'w', encoding='utf-8')
def p(*a, **kw): builtins.print(*a, **kw, file=_out)

for cond in ['neutral', 'llm_analog']:
    s = pd.read_excel(f'data/ssd_results/{cond}_cluster_snippets.xlsx')
    p(f'\n=== {cond} snippets ===')
    p(s.to_string())

_out.close()
builtins.print("Done")
