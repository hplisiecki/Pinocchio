"""
Merge MFQ_1 and MFQ_2 into a single 'Moral Foundations Questionnaire (MFQ-30)' entry
in results_clean.csv, then overwrite the file.
"""
import pandas as pd

CSV = 'data/results_clean.csv'
df = pd.read_csv(CSV, low_memory=False)

# Check for item text overlap between the two parts
items1 = set(df[df['questionnaire'] == 'Moral Foundations Questionnaire (MFQ-30)_1']['item'].unique())
items2 = set(df[df['questionnaire'] == 'Moral Foundations Questionnaire (MFQ-30)_2']['item'].unique())
overlap = items1 & items2
print(f'Part 1 items: {len(items1)}, Part 2 items: {len(items2)}, Overlap: {len(overlap)}')
if overlap:
    print('Overlapping items:', overlap)

NEW_NAME = 'Moral Foundations Questionnaire (MFQ-30)'

# Rename both parts to the merged name
df['questionnaire'] = df['questionnaire'].replace({
    'Moral Foundations Questionnaire (MFQ-30)_1': NEW_NAME,
    'Moral Foundations Questionnaire (MFQ-30)_2': NEW_NAME,
})

print(f'Rows now labelled "{NEW_NAME}": {(df["questionnaire"] == NEW_NAME).sum()}')
print(f'Total rows: {len(df)}, Questionnaires: {df["questionnaire"].nunique()}')

df.to_csv(CSV, index=False)
print('Saved.')
