# 创建一个列表
my_list = [50,50,50]

# 使用enumerate和max找到最大值的索引
max_index, max_value = max((i, x) for i, x in enumerate(my_list))

# 打印最大值的索引和值
print("Index of max value:", max_index)
print("Max value:", max_value)