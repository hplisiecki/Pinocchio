import pandas as pd
df = pd.read_csv('data/results.csv', low_memory=False)
print('Conditions:', df['condition'].unique())
print('N models:', df['model'].nunique())
print('Sample models:')
for m in sorted(df['model'].unique()):
    print(' ', m)
