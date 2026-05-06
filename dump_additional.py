import json, builtins
_out = open('data/additional_dump.txt', 'w', encoding='utf-8')
def p(*a, **kw): builtins.print(*a, **kw, file=_out)

with open('data/results_additional.json') as f:
    d = json.load(f)
for k, v in d.items():
    p(f"{k}: {v}")

_out.close()
builtins.print("Done")
