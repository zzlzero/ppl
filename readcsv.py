import pandas as pd

# read csv file
df = pd.read_csv('perplexities_pretrain.csv')
df = pd.read_csv('perplexities_finetune15epoch.csv')
print(df.head())
# 打印第一行
print(df.iloc[0])
print(type(df['ppl'].to_list))

