import pandas as pd

# 假设CSV文件名为 'data.csv'
# 读取CSV文件
df = pd.read_csv('../data.csv', header=None)

# 统计第三列中3和4的数量
count_3 = (df[2] == 3).sum()
count_4 = (df[2] == 4).sum()
count_1 = (df[2] == 1).sum()
count_2 = (df[2] == 2).sum()
print(f"1的数量: {count_1}")
print(f"2的数量: {count_2}")
print(f"3的数量: {count_3}")
print(f"4的数量: {count_4}")



df = pd.read_csv('../data.csv', header=None, dtype={0: str})
# 筛选出值为1、2、4的行
filtered_df = df[df[2].isin([1, 2, 4])]

# 对每一类（1、2、4）各取121个
result_df = pd.concat([
    filtered_df[filtered_df[2] == 1].head(121),
    filtered_df[filtered_df[2] == 2].head(121),
    filtered_df[filtered_df[2] == 4].head(121)
])

# 保存到新的CSV文件
result_df.to_csv('data_121.csv', index=False, header=False)

print("新的CSV文件已生成：'filtered_data.csv'")