import pandas as pd, os, builtins
_out = open('data/efa_shape.txt', 'w', encoding='utf-8')
def p(*a, **kw): builtins.print(*a, **kw, file=_out)

files = sorted([f for f in os.listdir('data/efa_results') if '__neutral' in f])
for f in files:
    df = pd.read_csv(f'data/efa_results/{f}', index_col=0)
    p(f'{f.replace("__neutral.csv","")}: {df.shape[1]} factors, {df.shape[0]} items')

_out.close()
builtins.print("Done")
