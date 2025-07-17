import pandas as pd

# 1. 读取数据
df = pd.read_csv("files/avocado.csv")

# 2. 查看 price 列的最大值和最小值
min_price = df['AveragePrice'].min()
max_price = df['AveragePrice'].max()

print("最小价格：", min_price)
print("最大价格：", max_price)
