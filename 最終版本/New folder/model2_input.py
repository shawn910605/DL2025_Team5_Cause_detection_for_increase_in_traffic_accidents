import pandas as pd

# 读取step1.csv文件
df = pd.read_csv('step1.csv')

# 查看数据的前几行，确认数据读取正确
print(df.head())

# 增加一列，表示具体日期
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

# 排除 cause_code_individual 中的 43、44 和 67
df = df[~df['cause_code_individual'].isin([43, 44, 67])]

# 按日期和车祸原因分组，计算每种车祸原因的数量
cause_counts = df.groupby(['date', 'cause_code_individual']).size().unstack(fill_value=0)

# 计算每天的总车祸数
cause_counts['total'] = cause_counts.sum(axis=1)

# 查看处理后的数据
print(cause_counts.head())

# 保存结果到step4.csv文件
cause_counts.to_csv('model2.csv')
