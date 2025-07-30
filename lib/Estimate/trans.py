import pandas as pd

# 读取 Excel 文件
df = pd.read_csv('demo.csv')

# 将 DataFrame 写入文本文件
df.to_csv('demo.txt', sep='\t', index=False)
