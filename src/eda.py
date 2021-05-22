import pandas as pd

df = pd.read_csv('../input/train.csv')
# print(df)
print("df key :", df.keys())

# for i in df['cleaned_label'].unique():
#     print(i)
print(len(df['dataset_label'].unique()))

# print(df['dataset_label'].unique())
# print()

pd.set_option('display.max_rows', None)
print(df['dataset_label'].value_counts())
